
import os
import sys
import argparse as arg
import tables
import numpy as np
import pypan.ui as pan

from build_data import Parameters, generator_ids

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


class ParametersExt (Parameters):
    pan_seed = tables.Float64Col()


if __name__ == '__main__':
    parser = arg.ArgumentParser(description = 'Simulate the IEEE14 network at a fixed value of inertia', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('pan_file', type=str, action='store', help='PAN netlist')
    parser.add_argument('-H', '--inertia',  type=float, required=True, help='inertia value')
    parser.add_argument('-G', '--gen-id',  type=int, default=1, help='generator id')
    parser.add_argument('-d', '--dur', default=300, type=float, help='simulation duration in seconds')
    parser.add_argument('--alpha',  default=0.5,  type=float, help='alpha parameter of the OU process')
    parser.add_argument('--mu',  default=0.0,  type=float, help='mu parameter of the OU process')
    parser.add_argument('-c',  default=0.5,  type=float, help='c parameter of the OU process')
    parser.add_argument('--frand',  default=10,  type=float, help='sampling rate of the random signal')
    parser.add_argument('-D', '--damping', default=0, type=int, help='damping coefficient')
    parser.add_argument('--DZA', default=0.036, type=float, help='deadband amplitude')
    parser.add_argument('-F', '--frequency', default=60, type=float, help='baseline frequency of the system')
    parser.add_argument('-o', '--output',  default='ieee14.h5',  type=str, help='output file name')
    parser.add_argument('-s', '--seed',  default=None, type=int, help='seed of the random number generator')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file')
    args = parser.parse_args(args=sys.argv[1:])

    if not os.path.isfile(args.pan_file):
        print('{}: {}: no such file.'.format(progname, args.pan_file))
        sys.exit(1)

    if os.path.isfile(args.output) and not args.force:
        print('{}: {}: file exists: use -f to overwrite.'.format(progname, args.output))
        sys.exit(2)

    if args.inertia <= 0:
        print('{}: the inertia value must be > 0'.format(progname))
        sys.exit(3)

    if not args.gen_id in generator_ids:
        print('{}: generator id must be one of {}'.format(progname, generator_ids))
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

    # value of inertia
    H = args.inertia

    mem_vars = ['time:noise', 'omega01:noise', 'omega02:noise', 'G3:omega:noise', \
                'G6:omega:noise', 'G8:omega:noise', 'omegacoi:noise', \
                'omegael01:noise', 'omegael02:noise', 'omegael03:noise', \
                'omegael06:noise', 'omegael08:noise']
    mem_vars.append('DevTime')
    for gen_id in generator_ids:
        mem_vars.append(f'G{gen_id}:pe')
        mem_vars.append(f'G{gen_id}:qe')

    disk_vars = ['^omega', '^G.*omega$']#, '^G[0-9]+[pq]$']

    ok,libs = pan.load_netlist(args.pan_file)
    if not ok:
        print('Cannot load netlist from file {}.'.format(args.pan_file))
        sys.exit(4)

    D = args.damping
    DZA = args.DZA / args.frequency
    pan.alter('Altstop', 'TSTOP', tstop, annotate=1)
    pan.alter('Alfrand', 'FRAND', frand, annotate=1)
    pan.alter('Ald',     'D',     D,     annotate=1)
    pan.alter('Aldza',   'DZA',   DZA,   annotate=1)
    pan.alter('Alh',     'm',     2 * H, instance=f'G{args.gen_id}', annotate=1)

    np.random.seed(rng_seed)
    pan_seed = np.random.randint(low=0, high=1000000)
    ou = OU(dt, alpha, mu, c, N_samples)

    noise_samples = np.vstack((t, ou))
    tran_name = 'Tr'

    data = pan.tran(tran_name, tstop, mem_vars, nettype=1, method=2, maxord=2, \
                    noisefmax=frand/2, noiseinj=2, seed=pan_seed, \
                    iabstol=1e-6, devvars=1, tmax=0.1, annotate=3, \
                    savelist='["' + '","'.join(disk_vars) + '"]')


    # save the results to file
    get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

    compression_filter = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Float64Atom()

    fid = tables.open_file(args.output, 'w', filters=compression_filter)
    tbl = fid.create_table(fid.root, 'parameters', ParametersExt, 'parameters')
    params = tbl.row
    params['hw_seed']  = rng_seed
    params['pan_seed'] = pan_seed
    params['alpha']    = alpha
    params['mu']       = mu
    params['c']        = c
    params['inertia']  = H
    params['D']        = D
    params['DZA']      = DZA
    params['F0']       = args.frequency
    params['frand']    = frand
    params['gen_id']   = args.gen_id
    params.append()
    tbl.flush()

    time = get_var(data, mem_vars, 'time:noise')
    fid.create_array(fid.root, 'time', time)
    fid.create_array(fid.root, 'omega_coi', get_var(data, mem_vars, 'omegacoi:noise'))

    for gen_id in (1,2):
        key = f'omega_G{gen_id}'
        fid.create_array(fid.root, key, get_var(data, mem_vars, f'omega{gen_id:02d}:noise'))

    for gen_id in (3,6,8):
        key = f'omega_G{gen_id}'
        fid.create_array(fid.root, key, get_var(data, mem_vars, f'G{gen_id}:omega:noise'))

    for gen_id in generator_ids:
        # the electrical omega in PAN has zero mean, so it needs to
        # be shifted at 1 p.u.
        key = f'omegael_G{gen_id}'
        fid.create_array(fid.root, key, 1.0 + get_var(data, mem_vars, f'omegael{gen_id:02d}:noise'))

    dev_time = get_var(data, mem_vars, 'DevTime')
    try:
        idx = np.array([np.where(dev_time == tt)[0][0] for tt in time])
        is_subset = True
    except:
        is_subset = False
    for gen_id in generator_ids:
        for lbl in ('p','q'):
            var = get_var(data, mem_vars, f'G{gen_id}:{lbl}e')
            key = f'{lbl.upper()}e_G{gen_id}'
            if is_subset:
                fid.create_array(fid.root, key, var[idx])
            else:
                f = interp1d(dev_time, var)
                fid.create_array(fid.root, key, f(time))

    fid.close()

