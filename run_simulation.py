
import os
import sys
import json
import argparse as arg
import tables
import numpy as np
import pypan.ui as pan
from numpy.random import RandomState, SeedSequence, MT19937

from build_data import BaseParameters, OU

progname = os.path.basename(sys.argv[0])

if __name__ == '__main__':
    parser = arg.ArgumentParser(description = 'Simulate the IEEE14 network at a fixed value of inertia', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='PAN netlist')
    parser.add_argument('-o', '--output',  default=None,  type=str, help='output file name')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    pan_file = config['pan_file']
    if not os.path.isfile(pan_file):
        print('{}: {}: no such file.'.format(progname, pan_file))
        sys.exit(1)

    if args.output is None:
        output_file = os.path.splitext(os.path.basename(pan_file))[0] + '.h5'
    else:
        output_file = args.output

    if os.path.isfile(output_file) and not args.force:
        print('{}: {}: file exists: use -f to overwrite.'.format(progname, output_file))
        sys.exit(2)

    random_load_buses = config['random_load_buses']
    N_random_loads = len(random_load_buses)
    N_blocks = len(config['tstop'])
    with open('/dev/random', 'rb') as fid:
        rng_seeds = [int.from_bytes(fid.read(4), 'little') % 1000000 for _ in range(N_random_loads + N_blocks)]
        rng_seeds, pan_seeds = rng_seeds[:-N_blocks], rng_seeds[-N_blocks:]
    rnd_states = [RandomState(MT19937(SeedSequence(seed))) for seed in rng_seeds]

    # OU parameters
    alpha = config['OU']['alpha']
    mu = config['OU']['mu']
    c = config['OU']['c']

    # simulation parameters
    frand = config['frand']        # [Hz] sampling rate of the random signal
    tstop = config['tstop'][-1]    # [s]  total simulation duration
    dt = 1 / frand
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size

    mem_vars_map = config['mem_vars_map']
    mem_vars = list(mem_vars_map.keys())
    time_mem_var = mem_vars[['time' in mem_var for mem_var in mem_vars].index(True)]
    time_disk_var = mem_vars_map[time_mem_var]

    ok,libs = pan.load_netlist(pan_file)
    if not ok:
        print('Cannot load netlist from file {}.'.format(pan_file))
        sys.exit(4)

    D = config['damping']
    DZA = config['DZA'] / config['frequency']
    LAMBDA = config['lambda']
    COEFF = config['coeff']

    pan.alter('Altstop', 'TSTOP',  tstop,  annotate=1)
    pan.alter('Alfrand', 'FRAND',  frand,  annotate=1)
    pan.alter('Ald',     'D',      D,      annotate=1)
    pan.alter('Aldza',   'DZA',    DZA,    annotate=1)
    pan.alter('Allam',   'LAMBDA', LAMBDA, annotate=1)
    pan.alter('Alcoeff', 'COEFF',  COEFF,  annotate=1)

    ou = [OU(dt, alpha[i], mu[i], c[i], N_samples, rnd_states[i]) for i in range(N_random_loads)]

    for i,bus in enumerate(random_load_buses):
        exec(f'noise_samples_bus_{bus} = np.vstack((t, ou[i]))')

    get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

    generator_IDs = list(config['inertia'].keys())
    N_generators = len(generator_IDs)
    inertia_values = np.array([config['inertia'][gen_id] for gen_id in generator_IDs])

    class Parameters (BaseParameters):
        generator_IDs  = tables.StringCol(8, shape=(N_generators,))
        rnd_load_buses = tables.Int64Col(shape=(N_random_loads,))
        rng_seeds      = tables.Int64Col(shape=(N_random_loads,))
        pan_seeds      = tables.Int64Col(shape=(N_blocks,))
        inertia        = tables.Float64Col(shape=(N_generators,N_blocks))
        alpha          = tables.Float64Col(shape=(N_random_loads,))
        mu             = tables.Float64Col(shape=(N_random_loads,))
        c              = tables.Float64Col(shape=(N_random_loads,))
        tstop          = tables.Float64Col(shape=(N_blocks,))

    fid = tables.open_file(output_file, 'w', filters=tables.Filters(complib='zlib', complevel=5))
    tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
    params = tbl.row
    params['rng_seeds']      = rng_seeds
    params['pan_seeds']      = pan_seeds
    params['tstop']          = config['tstop']
    params['alpha']          = alpha
    params['mu']             = mu
    params['c']              = c
    params['D']              = D
    params['DZA']            = DZA
    params['LAMBDA']         = LAMBDA
    params['COEFF']          = COEFF
    params['F0']             = config['frequency']
    params['frand']          = frand
    params['inertia']        = inertia_values
    params['generator_IDs']  = generator_IDs
    params['rnd_load_buses'] = random_load_buses
    params.append()
    tbl.flush()

    atom = tables.Float64Atom()
    for disk_var in mem_vars_map.values():
        if disk_var is not None:
            if isinstance(disk_var, str):
                fid.create_carray(fid.root, disk_var, atom, (N_samples,))
            elif isinstance(disk_var, list):
                fid.create_earray(fid.root, disk_var[0], atom, (N_samples,))

    start = 0
    for i,tstop in enumerate(config['tstop']):
        tran_name = f'Tr{i+1}'

        for gen_id, H in config['inertia'].items():
            pan.alter('Alh', 'm', 2 * H[i], instance=gen_id, annotate=1, invalidate=0)

        data = pan.tran(tran_name, tstop, mem_vars, nettype=1, method=1, \
                        timepoints=1/frand, forcetps=1, maxiter=65, annotate=3, \
                        restart=1 if i == 0 else 0)

        for mem_var in mem_vars:
            disk_var = mem_vars_map[mem_var]
            if disk_var is not None:
                if 'omegael' in mem_var:
                    offset = 1.
                else:
                    offset = 0.
                var = get_var(data, mem_vars, mem_var)
                if isinstance(disk_var, str):
                    stop = start + var.size
                    fid.root[disk_var][start : stop] = var + offset
                elif isinstance(disk_var, list):
                    disk_var, orig_time_var, resampled_time_var = disk_var
                    orig_time = get_var(data, mem_vars, orig_time_var)
                    resampled_time = get_var(data, mem_vars, resampled_time_var)
                    try:
                        idx = np.array([np.where(origin_time == tt)[0][0] for tt in resampled_time])
                        stop = start + idx.size
                        fid.root[disk_var][start : stop] = var[idx] + offset
                    except:
                        f = interp1d(orig_time, var)
                        tmp = f(resampled_time)
                        stop = start + tmp.size
                        fid.root[disk_var][start : stop] = tmp + offset
        start = stop

    fid.close()

