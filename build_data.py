
import os
import sys
import json
import argparse as arg
import numpy as np

import pypan.ui as pan

progname = os.path.basename(sys.argv[0])


if __name__ == '__main__':

    parser = arg.ArgumentParser(description = 'Build data for inertia estimation with deep neural networks', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-d', '--dur', default=None, type=float, help='simulation duration in seconds')
    parser.add_argument('-n', '--n-trials',  default=None,  type=int, help='number of trials')
    parser.add_argument('-s', '--suffix',  default='',  type=str, help='suffix to add to the output files')
    parser.add_argument('-o', '--output-dir',  default=None,  type=str, help='output directory')
    args = parser.parse_args(args=sys.argv[1:])
    
    with open('/dev/random', 'rb') as fid:
        hw_seed = int.from_bytes(fid.read(4), 'little') % 1000000
    np.random.seed(hw_seed)

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))
    
    # OU parameters
    alpha = config['alpha']
    mu = config['mu']
    c = config['c']

    # simulation parameters
    frand = config['frand']  # [Hz] sampling rate of the random signal
    if args.dur is not None:
        tstop = args.dur     # [s]  simulation duration
    else:
        tstop = config['dur']

    # inertia values
    H_min = config['Hmin']
    H_max = config['Hmax']
    dH = config['Hstep']
    H = np.r_[H_min : H_max + dH/2 : dH]
    N_H = H.size

    # how many trials per inertia value
    if args.n_trials is not None:
        N_trials = args.n_trials
    else:
        N_trials = config['Ntrials']

    random_seeds = np.random.randint(low=0, high=1000000, size=(N_H, N_trials))

    dt = 1 / frand
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size
    coeff = np.array([alpha * mu * dt, 1 / (1 + alpha * dt)])

    pan_file = config['netlist']
    if not os.path.isfile(pan_file):
        print('{}: {}: no such file.'.format(progname, pan_file))
        sys.exit(1)
    
    ok,libs = pan.load_netlist(pan_file)
    if not ok:
        print('Cannot load netlist from file {}.'.format(pan_file))
        sys.exit(2)

    # reference frequency of the system
    F0 = config['F0']
    # damping coefficient
    D = config['D']
    # dead-band width
    DZA = config['DZA'] / F0
    
    pan.alter('Altstop', 'TSTOP', tstop, annotate=1)
    pan.alter('Alfrand', 'FRAND', frand, annotate=1)
    pan.alter('Ald',     'D',     D,     annotate=1)
    pan.alter('Aldza',   'DZA',   DZA,   annotate=1)

    omega = {'G{}'.format(i): np.zeros((N_trials, N_samples - 1)) for i in (1,2,3,6,8)}
    omega['coi'] = np.zeros((N_trials, N_samples - 1))
    omegael = {'G{}'.format(i): np.zeros((N_trials, N_samples - 1)) for i in (1,2,3,6,8)}
    noise     = np.zeros((N_trials, N_samples))

    mem_vars = ['time:noise', 'omega01:noise', 'omega02:noise', 'G3:omega:noise', \
                'G6:omega:noise', 'G8:omega:noise', 'omegacoi:noise', \
                'omegael01:noise', 'omegael02:noise', 'omegael03:noise', \
                'omegael06:noise', 'omegael08:noise']
    disk_vars = ['omega*']

    if args.output_dir is None:
        from time import strftime, localtime
        output_dir = strftime('%Y%m%d-%H%M%S', localtime())
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isfile(output_dir + '/' + config_file):
        config['dur'] = tstop
        config['Ntrials'] = N_trials
        json.dump(config, open(output_dir + '/' + config_file, 'w'), indent=4)

    suffix = args.suffix
    if suffix != '':
        if suffix[0] != '_':
            suffix = '_' + suffix
        if suffix[-1] == '_':
            suffix = suffix[:-1]

    for i in range(N_H):

        pan.alter('Al{}'.format(i), 'm', 2 * H[i], instance='G1', invalidate='false')

        for j in range(N_trials):

            np.random.seed(random_seeds[i,j])
            rnd = c * np.sqrt(dt) * np.random.normal(size=N_samples)

            for k in range(N_samples-1):
                noise[j, k+1] = (noise[j, k] + coeff[0] + rnd[k]) * coeff[1]

            noise_samples = np.vstack((t, noise[j,:]))

            tran_name = 'Tr_{}_{}'.format( i, j)
            data = pan.tran(tran_name, tstop, mem_vars, nettype=1, method=2, maxord=2, \
                            noisefmax=frand/2, noiseinj=2, seed=random_seeds[i,j], \
                            iabstol=1e-6, devvars=0, tmax=0.1, annotate=1, \
                            savelist='["' + '","'.join(disk_vars) + '"]')

            for k in (1,2):
                var_name = 'omega{:02d}:noise'.format(k)
                idx = mem_vars.index(var_name)
                omega['G{}'.format(k)][j,:] = data[idx,:]

            for k in (3,6,8):
                var_name = 'G{}:omega:noise'.format(k)
                idx = mem_vars.index(var_name)
                omega['G{}'.format(k)][j,:] = data[idx,:]

            idx = mem_vars.index('omegacoi:noise')
            omega['coi'][j,:] = data[idx,:]

            for k in (1,2,3,6,8):
                var_name = 'omegael{:02d}:noise'.format(k)
                idx = mem_vars.index(var_name)
                # the electrical omega in PAN has zero mean, so it needs to
                # be shifted at 1 p.u.
                omegael['G{}'.format(k)][j,:] = data[idx,:] + 1.0

        time = data[0,:]
        seeds = random_seeds[i,:]
        inertia = H[i]

        out_file = '{}/{}{}_H_{:.3f}'.format(output_dir, os.path.splitext(os.path.basename(pan_file))[0], suffix, H[i])

        kwargs = {'time': time}
        for i in (1,2,3,6,8):
            kwargs['omega_G{}'.format(i)] = omega['G{}'.format(i)]
        kwargs['omega_coi'] = omega['coi']
        for i in (1,2,3,6,8):
            kwargs['omegael_G{}'.format(i)] = omegael['G{}'.format(i)]
        kwargs['seeds'] = seeds
        kwargs['inertia'] = inertia
        kwargs['noise'] = noise
        kwargs['hw_seed'] = hw_seed
        kwargs['alpha'] = alpha
        kwargs['mu'] = mu
        kwargs['c'] = c

        np.savez_compressed(out_file, **kwargs)

