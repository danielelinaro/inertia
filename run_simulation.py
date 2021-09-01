
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
    parser.add_argument('-o', '--output',  default=None, type=str, help='output file name')
    parser.add_argument('--overload',  default=None, type=float, help='overload coefficient (overwrites the value in the config file)')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file')
    parser.add_argument('--check-stability', action='store_true',
                        help='check that the system is stable by running a pole-zero analysis')
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

    generator_IDs = list(config['inertia'].keys())
    N_generators = len(generator_IDs)
    N_inertia_values = max(map(len, config['inertia'].values()))
    inertia_values = []
    for gen_id in generator_IDs:
        if len(config['inertia'][gen_id]) == 1:
            inertia_values.append([config['inertia'][gen_id][0] for _ in range(N_inertia_values)])
        elif len(config['inertia'][gen_id]) == N_inertia_values:
            inertia_values.append(config['inertia'][gen_id])
        else:
            raise Exception(f'Wrong number of inertia values for generator {gen_id}')
    inertia_values = np.array(inertia_values)

    try:
        integration_mode = config['integration_mode'].lower()
    except:
        integration_mode = 'trapezoidal'

    if integration_mode not in ('trapezoidal', 'gear'):
        print('{}: integration_mode must be one of "trapezoidal" or "Gear".')
        sys.exit(2)

    random_load_buses = config['random_load_buses']
    N_random_loads = len(random_load_buses)
    N_blocks = len(config['tstop'])

    try:
        rng_seeds = config['seeds']
    except:
        with open('/dev/urandom', 'rb') as fid:
            rng_seeds = [int.from_bytes(fid.read(4), 'little') % 1000000 for _ in range(N_random_loads + N_blocks)]

    if integration_mode == 'trapezoidal':
        rng_seeds = rng_seeds[:N_random_loads]
        pan_seeds = np.nan + np.zeros(N_blocks)
    else:
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
    if args.overload is not None:
        LAMBDA = args.overload
    else:
        LAMBDA = config['lambda']
    COEFF = config['coeff']

    pan.alter('Altstop', 'TSTOP',  tstop,  libs, annotate=1)
    pan.alter('Alfrand', 'FRAND',  frand,  libs, annotate=1)
    pan.alter('Ald',     'D',      D,      libs, annotate=1)
    pan.alter('Aldza',   'DZA',    DZA,    libs, annotate=1)
    pan.alter('Allam',   'LAMBDA', LAMBDA, libs, annotate=1)
    pan.alter('Alcoeff', 'COEFF',  COEFF,  libs, annotate=1)

    ou = [OU(dt, alpha[i], mu[i], c[i], N_samples, rnd_states[i]) for i in range(N_random_loads)]

    for i,bus in enumerate(random_load_buses):
        exec(f'noise_samples_bus_{bus} = np.vstack((t, ou[i]))')

    get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

    class Parameters (BaseParameters):
        generator_IDs  = tables.StringCol(8, shape=(N_generators,))
        rnd_load_buses = tables.Int64Col(shape=(N_random_loads,))
        rng_seeds      = tables.Int64Col(shape=(N_random_loads,))
        pan_seeds      = tables.Float64Col(shape=(N_blocks,)) # these must be floats because they might be NaN's
        inertia        = tables.Float64Col(shape=(N_generators,N_blocks))
        alpha          = tables.Float64Col(shape=(N_random_loads,))
        mu             = tables.Float64Col(shape=(N_random_loads,))
        c              = tables.Float64Col(shape=(N_random_loads,))
        tstop          = tables.Float64Col(shape=(N_blocks,))

    if args.output is None:
        import subprocess
        name_max = int(subprocess.check_output('getconf NAME_MAX /', shell=True))
        output_file = os.path.splitext(os.path.basename(pan_file))[0] + '_' + \
            '_'.join(['-'.join(map(lambda h: f'{h:.3f}', H)) for H in inertia_values])
        if len(output_file) > name_max:
            output_file = os.path.splitext(os.path.basename(pan_file))[0] + '_' + \
                '_'.join(['-'.join(map(lambda h: f'{h:.3f}', np.unique(H))) for H in inertia_values])
        if args.overload is not None:
            output_file += f'_lambda={LAMBDA:.3f}'
        output_file += '.h5'
    else:
        output_file = args.output

    try:
        # we check whether the file exists in this way because if it doesn't
        # pan crashes due to errno being somehow set to 2 ("No such file or directory"
        # error).
        import pathlib
        pathlib.Path(output_file).touch(mode=0o644, exist_ok=args.force)
    except FileExistsError as file_error:
        print('{}: {}: file exists: use -f to overwrite.'.format(progname, output_file))
        sys.exit(2)

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

    if integration_mode == 'trapezoidal':
        array_shape = N_samples,
    else:
        array_shape = N_samples - 1,

    for disk_var in mem_vars_map.values():
        if disk_var is not None:
            if isinstance(disk_var, str):
                fid.create_carray(fid.root, disk_var, atom, array_shape)
            elif isinstance(disk_var, list):
                fid.create_earray(fid.root, disk_var[0], atom, array_shape)

    if 'save_OU' in config and config['save_OU']:
        fid.create_array(fid.root, 'OU', np.array(ou), atom=atom)

    start = 0
    for i, tstop in enumerate(config['tstop']):
        tran_name = f'Tr{i+1}'

        for j, gen_id in enumerate(generator_IDs):
            pan.alter('Alh', 'm', 2 * inertia_values[j,i], libs, instance=gen_id, annotate=1, invalidate=0)

        kwargs = {'nettype': 1, 'annotate': 3, 'restart': 1 if i == 0 else 0}

        if integration_mode == 'trapezoidal':
            kwargs['method']     = 1
            kwargs['timepoints'] = 1 / frand
            kwargs['forcetps']   = 1
            kwargs['maxiter']    = 65
            kwargs['saman']      = 'yes'
            kwargs['sparse']     = 2
        else:
            kwargs['method']     = 2
            kwargs['maxord']     = 2
            kwargs['noisefmax']  = frand / 2
            kwargs['noiseinj']   = 2
            kwargs['seed']       = pan_seeds[i]
            kwargs['iabstol']    = 1e-6
            kwargs['devvars']    = 1
            kwargs['tmax']       = 0.1

        if args.check_stability:
            poles = pan.PZ('Pz', mem_vars=['poles'], libs=libs, nettype=1, annotate=0)[0]
            # sort the poles in descending order and convert them to Hz
            poles = poles[np.argsort(poles.real)[::-1]] / (2 * np.pi)
            n_unstable = np.sum(poles.real > 1e-6)
            print(f'The system has {n_unstable} poles with real part > 1e-6.')
            # save the poles to file
            if i == 0:
                fid.create_carray(fid.root, 'poles', atom=tables.ComplexAtom(16), shape=(len(config['tstop']), poles.size))
            fid.root.poles[i,:] = poles

        try:
            data = pan.tran(tran_name, tstop, mem_vars, libs, **kwargs)
        except:
            fid.close()
            os.remove(output_file)
            sys.exit(-1)

        for mem_var in mem_vars:
            disk_var = mem_vars_map[mem_var]
            if disk_var is not None:
                if 'omegael' in mem_var:
                    offset = 1.
                else:
                    offset = 0.
                var = get_var(data, mem_vars, mem_var)
                if i > 0 and integration_mode == 'trapezoidal':
                    var = var[1:]
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

