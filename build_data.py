
import os
import sys
import json
import glob
import shutil
import argparse as arg
import numpy as np
from scipy.interpolate import interp1d
from tempfile import NamedTemporaryFile

import pypan.ui as pan

progname = os.path.basename(sys.argv[0])
generator_ids = (1,2,3,6,8)

if __name__ == '__main__':

    parser = arg.ArgumentParser(description = 'Build data for inertia estimation with deep neural networks', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('--with-power', action='store_true', help='save also power traces')
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

    if args.with_power:
        with_power = True
    elif 'with_power' in config:
        with_power = config['with_power']
    else:
        with_power = False

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

    with NamedTemporaryFile(prefix = os.path.splitext(os.path.basename(pan_file))[0] + '_', \
                            suffix = '.pan', delete = False) as fid:
        for va_file in glob.glob(os.path.dirname(pan_file) + '/*.va'):
            shutil.copy(va_file, '/tmp')
        shutil.copyfile(pan_file, fid.name)
        pan_file = fid.name

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

    omega = {'G{}'.format(i): np.zeros((N_trials, N_samples - 1)) for i in generator_ids}
    omega['coi'] = np.zeros((N_trials, N_samples - 1))
    omegael = {'G{}'.format(i): np.zeros((N_trials, N_samples - 1)) for i in generator_ids}
    noise     = np.zeros((N_trials, N_samples))

    disk_vars = ['omega*']
    mem_vars = ['time:noise', 'omega01:noise', 'omega02:noise', 'G3:omega:noise', \
                'G6:omega:noise', 'G8:omega:noise', 'omegacoi:noise', \
                'omegael01:noise', 'omegael02:noise', 'omegael03:noise', \
                'omegael06:noise', 'omegael08:noise']

    if with_power:
        Pe = {'G{}'.format(i): np.zeros((N_trials, N_samples - 1)) for i in generator_ids}
        Qe = {'G{}'.format(i): np.zeros((N_trials, N_samples - 1)) for i in generator_ids}
        mem_vars.append('DevTime')
        for i in generator_ids:
            mem_vars.append('G{}:pe'.format(i))
            mem_vars.append('G{}:qe'.format(i))

    if args.output_dir is None:
        from time import strftime, localtime
        output_dir = strftime('%Y%m%d-%H%M%S', localtime())
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isfile(output_dir + '/' + os.path.basename(config_file)):
        config['dur'] = tstop
        config['Ntrials'] = N_trials
        json.dump(config, open(output_dir + '/' + os.path.basename(config_file), 'w'), indent=4)

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

            tran_name = 'Tr_{}_{}'.format(i, j)
            data = pan.tran(tran_name, tstop, mem_vars, nettype=1, method=2, maxord=2, \
                            noisefmax=frand/2, noiseinj=2, seed=random_seeds[i,j], \
                            iabstol=1e-6, devvars=1, tmax=0.1, annotate=1, \
                            savelist='["' + '","'.join(disk_vars) + '"]')

            get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

            time = get_var(data, mem_vars, 'time:noise')

            for k in (1,2):
                omega['G{}'.format(k)][j,:] = get_var(data, mem_vars, 'omega{:02d}:noise'.format(k))

            for k in (3,6,8):
                omega['G{}'.format(k)][j,:] = get_var(data, mem_vars, 'G{}:omega:noise'.format(k))

            omega['coi'][j,:] = get_var(data, mem_vars, 'omegacoi:noise')

            for k in generator_ids:
                # the electrical omega in PAN has zero mean, so it needs to
                # be shifted at 1 p.u.
                omegael['G{}'.format(k)][j,:] = 1.0 + get_var(data, mem_vars, 'omegael{:02d}:noise'.format(k))

            if with_power:
                dev_time = get_var(data, mem_vars, 'DevTime')
                try:
                    idx = np.array([np.where(dev_time == tt)[0][0] for tt in time])
                    is_subset = True
                except:
                    is_subset = False
                for k in generator_ids:
                    key = 'G{}'.format(k)
                    for lbl,dic in zip( ('p','q'), (Pe,Qe) ):
                        var = get_var(data, mem_vars, 'G{}:{}e'.format(k, lbl))
                        if is_subset:
                            dic[key][j,:] = var[idx]
                        else:
                            f = interp1d(dev_time, var)
                            dic[key][j,:] = f(time)
                
        seeds = random_seeds[i,:]
        inertia = H[i]

        out_file = '{}/{}{}_H_{:.3f}'.format(output_dir, os.path.splitext(os.path.basename(pan_file))[0], suffix, H[i])

        kwargs = {'time': time, 'omega_coi': omega['coi']}

        for i in generator_ids:
            gen = 'G{}'.format(i)
            kwargs['omega_' + gen] = omega[gen]
            kwargs['omegael_' + gen] = omegael[gen]
            if with_power:
                kwargs['Pe_' + gen] = Pe[gen]
                kwargs['Qe_' + gen] = Qe[gen]

        kwargs['seeds'] = seeds
        kwargs['inertia'] = inertia
        kwargs['noise'] = noise
        kwargs['hw_seed'] = hw_seed
        kwargs['alpha'] = alpha
        kwargs['mu'] = mu
        kwargs['c'] = c

        np.savez_compressed(out_file, **kwargs)

