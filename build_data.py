
import os
import sys
import json
import glob
import shutil
import argparse as arg
import numpy as np
from scipy.interpolate import interp1d
from tempfile import NamedTemporaryFile
import tables

import pypan.ui as pan

__all__ = ['Parameters', 'generator_ids']

progname = os.path.basename(sys.argv[0])
generator_ids = (1,2,3,6,8)


class Parameters (tables.IsDescription):
    hw_seed = tables.Float64Col()
    alpha   = tables.Float64Col()
    mu      = tables.Float64Col()
    c       = tables.Float64Col()
    inertia = tables.Float64Col()
    D       = tables.Float64Col()
    DZA     = tables.Float64Col()
    F0      = tables.Float64Col()
    frand   = tables.Float64Col()
    gen_id  = tables.Int32Col()


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
    noise_samples = np.zeros((2, N_samples))
    noise_samples[0,:] = t
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

    disk_vars = ['omega*']
    mem_vars = ['time:noise', 'omega01:noise', 'omega02:noise', 'G3:omega:noise', \
                'G6:omega:noise', 'G8:omega:noise', 'omegacoi:noise', \
                'omegael01:noise', 'omegael02:noise', 'omegael03:noise', \
                'omegael06:noise', 'omegael08:noise']

    if with_power:
        mem_vars.append('DevTime')
        for gen_id in generator_ids:
            mem_vars.append('G{}:pe'.format(gen_id))
            mem_vars.append('G{}:qe'.format(gen_id))

    if args.output_dir is None:
        from time import strftime, localtime
        output_dir = strftime('%Y%m%d-%H%M%S', localtime())
    else:
        output_dir = args.output_dir

    try:
        os.mkdir(output_dir)
    except:
        pass

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

    compression_filter = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Float64Atom()

    # generator whose inertia is varied
    gen_inst = 'G{}'.format(config['gen_id'])

    for i in range(N_H):

        pan.alter('Al', 'm', 2 * H[i], instance=gen_inst, invalidate='false')

        out_file = '{}/H_{:.3f}{}.h5'.format(output_dir, H[i], suffix)

        if os.path.isfile(out_file):
            continue

        ### first of all, write to file all the data and parameters that we already have
        fid = tables.open_file(out_file, 'w', filters=compression_filter)
        tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
        params = tbl.row
        params['hw_seed'] = hw_seed
        params['alpha']   = alpha
        params['mu']      = mu
        params['c']       = c
        params['inertia'] = H[i]
        params['D']       = D
        params['DZA']     = DZA
        params['F0']      = F0
        params['frand']   = frand
        params['gen_id']  = config['gen_id']
        params.append()
        tbl.flush()
        fid.create_array(fid.root, 'seeds', random_seeds[i,:])
        fid.create_earray(fid.root, 'noise', atom, (0, N_samples))
        fid.create_earray(fid.root, 'omega_coi', atom, (0, N_samples-1))
        for gen_id in generator_ids:
            gen = 'G{}'.format(gen_id)
            for lbl in ('omega_', 'omegael_', 'Pe_', 'Qe_'):
                key = lbl + gen
                if lbl not in ('Pe_','Qe_'):
                    fid.create_earray(fid.root, key, atom, (0, N_samples-1))
                elif with_power:
                    fid.create_earray(fid.root, key, atom, (0, N_samples-1))
        # close the file so that other programs can read it
        fid.close()

        for j in range(N_trials):

            # build the noisy samples
            np.random.seed(random_seeds[i,j])
            rnd = c * np.sqrt(dt) * np.random.normal(size=N_samples)
            noise_samples[1, 0] = 0.
            for k in range(N_samples-1):
                noise_samples[1, k+1] = (noise_samples[1, k] + coeff[0] + rnd[k]) * coeff[1]

            # run a transient analysis
            data = pan.tran('Tr', tstop, mem_vars, nettype=1, method=2, maxord=2, \
                            noisefmax=frand/2, noiseinj=2, seed=random_seeds[i,j], \
                            iabstol=1e-6, devvars=1, tmax=0.1, annotate=1, \
                            savelist='["' + '","'.join(disk_vars) + '"]')

            # save the results to file
            get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

            fid = tables.open_file(out_file, 'a')

            fid.root.noise.append(noise_samples[1, np.newaxis, :])
            fid.root.omega_coi.append(get_var(data, mem_vars, 'omegacoi:noise')[np.newaxis,:])

            if j == 0:
                time = get_var(data, mem_vars, 'time:noise')
                fid.create_array(fid.root, 'time', time)

            for gen_id in (1,2):
                gen = 'G{}'.format(gen_id)
                key = 'omega_' + gen
                fid.root[key].append(get_var(data, mem_vars, 'omega{:02d}:noise'.format(gen_id))[np.newaxis,:])

            for gen_id in (3,6,8):
                gen = 'G{}'.format(gen_id)
                key = 'omega_' + gen
                fid.root[key].append(get_var(data, mem_vars, 'G{}:omega:noise'.format(gen_id))[np.newaxis,:])

            for gen_id in generator_ids:
                # the electrical omega in PAN has zero mean, so it needs to
                # be shifted at 1 p.u.
                key = 'omegael_G{}'.format(gen_id)
                fid.root[key].append(1.0 + get_var(data, mem_vars, 'omegael{:02d}:noise'.format(gen_id))[np.newaxis,:])

            if with_power:
                dev_time = get_var(data, mem_vars, 'DevTime')
                try:
                    idx = np.array([np.where(dev_time == tt)[0][0] for tt in time])
                    is_subset = True
                except:
                    is_subset = False
                for gen_id in generator_ids:
                    for lbl in ('p','q'):
                        var = get_var(data, mem_vars, 'G{}:{}e'.format(gen_id, lbl))
                        key = '{}e_G{}'.format(lbl.upper(), gen_id)
                        if is_subset:
                            fid.root[key].append(var[np.newaxis,idx])
                        else:
                            f = interp1d(dev_time, var)
                            fid.root[key].append(f(time)[np.newaxis,:])

            fid.close()

