
import os
import sys
import argparse as arg
import tables
import numpy as np
import pypan.ui as pan

from build_data import BaseParameters#, generator_ids

progname = os.path.basename(sys.argv[0])


def OU(dt, alpha, mu, c, N, seed = None):
    if seed is not None:
        np.random.seed(seed)
    coeff = np.array([alpha * mu * dt, 1 / (1 + alpha * dt)])
    rnd = c * np.sqrt(dt) * np.random.normal(size=N)
    ou = np.zeros(N)
    for i in range(N-1):
        ou[i+1] = (ou[i] + coeff[0] + rnd[i]) * coeff[1]
    return ou



if __name__ == '__main__':
    parser = arg.ArgumentParser(description = 'Simulate the IEEE14 network at a fixed value of inertia', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('pan_file', type=str, action='store', help='PAN netlist')
    parser.add_argument('-H', '--inertia',  type=str, required=True, help='inertia value')
    parser.add_argument('-G', '--gen-ids',  type=str, required=True, help='generator id(s)')
    parser.add_argument('-d', '--dur', default=300, type=float, help='simulation duration in seconds')
    parser.add_argument('--alpha',  default=0.5,  type=float, help='alpha parameter of the OU process')
    parser.add_argument('--mu',  default=0.0,  type=float, help='mu parameter of the OU process')
    parser.add_argument('-c',  default=0.5,  type=float, help='c parameter of the OU process')
    parser.add_argument('--frand',  default=10,  type=float, help='sampling rate of the random signal')
    parser.add_argument('-D', '--damping', default=0, type=int, help='damping coefficient')
    parser.add_argument('--DZA', default=0.036, type=float, help='deadband amplitude')
    parser.add_argument('-F', '--frequency', default=60, type=float, help='baseline frequency of the system')
    parser.add_argument('-o', '--output',  default=None,  type=str, help='output file name')
    parser.add_argument('-s', '--seed',  default=None, type=int, help='seed of the random number generator')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file')
    args = parser.parse_args(args=sys.argv[1:])

    pan_file = args.pan_file
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

    inertia_values = np.array([float(h) for h in args.inertia.split(',')])
    if np.any(inertia_values <= 0):
        print('{}: inertia values must be > 0'.format(progname))
        sys.exit(3)

    generator_IDs = args.gen_ids.split(',')
    N_generators = len(generator_IDs)
    if len(inertia_values) != N_generators:
        print('The number of inertia values must match the number of generator IDs.')
        sys.exit(4)

    if args.seed is None:
        with open('/dev/random', 'rb') as fid:
            rng_seed = int.from_bytes(fid.read(4), 'little') % 1000000
    else:
        rng_seed = args.seed

    # OU parameters
    alpha = args.alpha
    mu = args.mu
    c = args.c

    # simulation parameters
    frand = args.frand  # [Hz] sampling rate of the random signal
    tstop = args.dur    # [s]  simulation duration
    dt = 1 / frand
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size

    if 'ieee14' in pan_file.lower():
        mem_vars_map = {
	    'time:noise': 'time',
	    'omegacoi:noise': 'omega_coi',
	    'omega01:noise': 'omega_G1',
	    'omega02:noise': 'omega_G2',
	    'G3:omega:noise': 'omega_G3',
	    'G6:omega:noise': 'omega_G6',
	    'G8:omega:noise': 'omega_G8',
	    'omegael01:noise': 'omegael_G1',
	    'omegael02:noise': 'omegael_G2',
	    'omegael03:noise': 'omegael_G3',
	    'omegael06:noise': 'omegael_G6',
	    'omegael08:noise': 'omegael_G8',
	    'DevTime': null,
	    'G1:pe': ['Pe_G1', 'DevTime', 'time:noise'],
	    'G1:qe': ['Qe_G1', 'DevTime', 'time:noise'],
	    'G2:pe': ['Pe_G2', 'DevTime', 'time:noise'],
	    'G2:qe': ['Qe_G2', 'DevTime', 'time:noise'],
	    'G3:pe': ['Pe_G3', 'DevTime', 'time:noise'],
	    'G3:qe': ['Qe_G3', 'DevTime', 'time:noise'],
	    'G6:pe': ['Pe_G6', 'DevTime', 'time:noise'],
	    'G6:qe': ['Qe_G6', 'DevTime', 'time:noise'],
	    'G8:pe': ['Pe_G8', 'DevTime', 'time:noise'],
	    'G8:qe': ['Qe_G8', 'DevTime', 'time:noise']
        }
    elif 'two' in pan_file.lower() and 'area' in pan_file.lower():
        mem_vars_map = {
	    'time:noise': 'time',
	    'omegael07:noise': 'omegael_bus7',
	    'omegael09:noise': 'omegael_bus9',
	    'pe7:noise': 'Pe_bus7',
	    'qe7:noise': 'Qe_bus7',
	    'pe9:noise': 'Pe_bus9',
	    'qe9:noise': 'Qe_bus9'
        }
    mem_vars = list(mem_vars_map.keys())
    time_mem_var = mem_vars[['time' in mem_var for mem_var in mem_vars].index(True)]
    time_disk_var = mem_vars_map[time_mem_var]

    ok,libs = pan.load_netlist(pan_file)
    if not ok:
        print('Cannot load netlist from file {}.'.format(pan_file))
        sys.exit(4)

    D = args.damping
    DZA = args.DZA / args.frequency
    pan.alter('Altstop', 'TSTOP', tstop, annotate=1)
    pan.alter('Alfrand', 'FRAND', frand, annotate=1)
    pan.alter('Ald',     'D',     D,     annotate=1)
    pan.alter('Aldza',   'DZA',   DZA,   annotate=1)
    for gen_id, H in zip(generator_IDs, inertia_values):
        pan.alter('Alh', 'm', 2 * H, instance=gen_id, annotate=1)

    np.random.seed(rng_seed)
    pan_seed = np.random.randint(low=0, high=1000000)
    ou = OU(dt, alpha, mu, c, N_samples)

    noise_samples = np.vstack((t, ou))
    tran_name = 'Tr'

    data = pan.tran(tran_name, tstop, mem_vars, nettype=1, method=2, maxord=2, \
                    noisefmax=frand/2, noiseinj=2, seed=pan_seed, \
                    iabstol=1e-6, devvars=1, tmax=0.1, annotate=3)

    # save the results to file
    get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

    compression_filter = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Float64Atom()

    class Parameters (BaseParameters):
        generator_IDs = tables.StringCol(8, shape=(N_generators,))
        inertia = tables.Float64Col(shape=(N_generators,))
        pan_seed = tables.Float64Col()

    fid = tables.open_file(output_file, 'w', filters=compression_filter)
    tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
    params = tbl.row
    params['hw_seed']  = rng_seed
    params['pan_seed'] = pan_seed
    params['alpha']    = alpha
    params['mu']       = mu
    params['c']        = c
    params['D']        = D
    params['DZA']      = DZA
    params['F0']       = args.frequency
    params['frand']    = frand
    params['inertia']  = inertia_values
    params['generator_IDs'] = generator_IDs
    params.append()
    tbl.flush()

    for k,mem_var in enumerate(mem_vars):
        disk_var = mem_vars_map[mem_var]
        if 'omegael' in mem_var:
            offset = 1.
        else:
            offset = 0.
        if disk_var is not None:
            if isinstance(disk_var, str):
                fid.create_array(fid.root, disk_var, get_var(data, mem_vars, mem_var) + offset)
            elif isinstance(disk_var, list):
                disk_var, orig_time_var, resampled_time_var = disk_var
                var = get_var(data, mem_vars, mem_var)
                orig_time = get_var(data, mem_vars, orig_time_var)
                resampled_time = get_var(data, mem_vars, resampled_time_var)
                try:
                    idx = np.array([np.where(origin_time == tt)[0][0] for tt in resampled_time])
                    fid.create_array(fid.root, disk_var, var[idx] + offset)
                except:
                    f = interp1d(orig_time, var)
                    fid.create_array(fid.root, disk_var, f(resampled_time) + offset)

    fid.close()

