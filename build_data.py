
import os
import sys
import json
import glob
import shutil
import argparse as arg
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
from scipy.interpolate import interp1d
from tempfile import NamedTemporaryFile
import tables

import pypan.ui as pan

__all__ = ['BaseParameters', 'OU', 'generator_ids']

progname = os.path.basename(sys.argv[0])
generator_ids = {'IEEE14': (1,2,3,6,8), 'two-area': (1,2,3,4)}


class BaseParameters (tables.IsDescription):
    D        = tables.Float64Col()
    DZA      = tables.Float64Col()
    F0       = tables.Float64Col()
    frand    = tables.Float64Col()
    LAMBDA   = tables.Float64Col()
    COEFF    = tables.Float64Col()


def OU(dt, alpha, mu, c, N, random_state = None):
    coeff = np.array([alpha * mu * dt, 1 / (1 + alpha * dt)])
    if random_state is not None:
        rnd = c * np.sqrt(dt) * random_state.normal(size=N)
    else:
        rnd = c * np.sqrt(dt) * np.random.normal(size=N)
    ou = np.zeros(N)
    ou[0] = mu
    for i in range(N-1):
        ou[i+1] = (ou[i] + coeff[0] + rnd[i]) * coeff[1]
    return ou


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
    
    config_file = args.config_file
    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))
    
    # OU parameters
    alpha = config['OU']['alpha']
    mu = config['OU']['mu']
    c = config['OU']['c']

    # simulation parameters
    frand = config['frand']  # [Hz] sampling rate of the random signal
    if args.dur is not None:
        tstop = args.dur     # [s]  simulation duration
    else:
        tstop = config['dur']

    # generator IDs
    generator_IDs = config['generator_IDs']
    N_generators = len(generator_IDs)

    # inertia values
    inertia = config['inertia']
    inertia_values = []
    for gen_id in generator_IDs:
        inertia_values.append(inertia[gen_id])
    H = np.meshgrid(*inertia_values)
    inertia_values = {}
    for i,gen_id in enumerate(generator_IDs):
        inertia_values[gen_id] = H[i].flatten()
    N_inertia = inertia_values[generator_IDs[0]].size

    # how many trials per inertia value
    if args.n_trials is not None:
        N_trials = args.n_trials
    else:
        N_trials = config['Ntrials']

    # buses where the stochastic loads are connected
    random_load_buses = config['random_load_buses']
    N_random_loads = len(random_load_buses)

    # seeds for the random number generators
    with open('/dev/random', 'rb') as fid:
        hw_seeds = [int.from_bytes(fid.read(4), 'little') % 1000000 for _ in range(N_random_loads + 1)]
    loads_random_states = [RandomState(MT19937(SeedSequence(seed))) for seed in hw_seeds[:-1]]
    pan_random_state = RandomState(MT19937(SeedSequence(hw_seeds[-1])))
    loads_random_seeds = [loads_random_state.randint(low=0, high=1000000, size=(N_inertia, N_trials)) \
                          for loads_random_state in loads_random_states]
    pan_random_seeds = pan_random_state.randint(low=0, high=1000000, size=(N_inertia, N_trials))

    dt = 1 / frand
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size
    for i,bus in enumerate(random_load_buses):
        exec(f'noise_samples_bus_{bus} = np.zeros((2, N_samples))')
        exec(f'noise_samples_bus_{bus}[0,:] = t')

    pan_file = config['netlist']
    if not os.path.isfile(pan_file):
        print('{}: {}: no such file.'.format(progname, pan_file))
        sys.exit(1)

    with NamedTemporaryFile(prefix = os.path.splitext(os.path.basename(pan_file))[0] + '_', \
                            suffix = '.pan', delete = False) as fid:
        for va_file in glob.glob(os.path.dirname(pan_file) + '/*.va'):
            try:
                shutil.copy(va_file, '/tmp')
            except:
                print(f'Cannot copy {va_file} to /tmp: trying to continue anyway...')
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
    # overload coefficient
    LAMBDA = config['lambda']
    # load scaling coefficient
    COEFF = config['coeff']
    
    pan.alter('Altstop', 'TSTOP',  tstop,  annotate=1)
    pan.alter('Alfrand', 'FRAND',  frand,  annotate=1)
    pan.alter('Ald',     'D',      D,      annotate=1)
    pan.alter('Aldza',   'DZA',    DZA,    annotate=1)
    pan.alter('Allam',   'LAMBDA', LAMBDA, annotate=1)
    pan.alter('Alcoeff', 'COEFF',  COEFF,  annotate=1)

    mem_vars_map = config['mem_vars']
    mem_vars = list(mem_vars_map.keys())
    time_mem_var = mem_vars[['time' in mem_var for mem_var in mem_vars].index(True)]
    time_disk_var = mem_vars_map[time_mem_var]

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

    compression_filter = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Float64Atom()

    class Parameters (BaseParameters):
        hw_seeds       = tables.Int64Col(shape=(N_random_loads+1,))
        pan_seeds      = tables.Int64Col(shape=(N_trials,))
        loads_seeds    = tables.Int64Col(shape=(N_random_loads,N_trials))
        count          = tables.Int64Col()
        alpha          = tables.Float64Col(shape=(N_random_loads,))
        mu             = tables.Float64Col(shape=(N_random_loads,))
        c              = tables.Float64Col(shape=(N_random_loads,))
        rnd_load_buses = tables.Int64Col(shape=(N_random_loads,))
        generator_IDs  = tables.StringCol(8, shape=(N_generators,))
        inertia        = tables.Float64Col(shape=(N_generators,))

    for i in range(N_inertia):

        out_file = ''
        for gen_id in generator_IDs:
            pan.alter('Al', 'm', 2 * inertia_values[gen_id][i], instance=gen_id, invalidate='false')
            out_file += f'_{inertia_values[gen_id][i]:.3f}'
        out_file = '{}/inertia{}{}.h5'.format(output_dir, out_file, suffix)

        if os.path.isfile(out_file):
            continue

        ### first of all, write to file all the data and parameters that we already have
        fid = tables.open_file(out_file, 'w', filters=compression_filter)
        tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
        params = tbl.row
        params['hw_seeds']       = hw_seeds
        params['pan_seeds']      = pan_random_seeds[i,:]
        params['loads_seeds']    = np.array([load_random_seeds[i,:] for load_random_seeds in loads_random_seeds])
        params['count']          = i
        params['alpha']          = alpha
        params['mu']             = mu
        params['c']              = c
        params['D']              = D
        params['DZA']            = DZA
        params['LAMBDA']         = LAMBDA
        params['COEFF']          = COEFF
        params['F0']             = F0
        params['frand']          = frand
        params['rnd_load_buses'] = random_load_buses
        params['generator_IDs']  = config['generator_IDs']
        params['inertia']        = [inertia_values[gen_id][i] for gen_id in generator_IDs]
        params.append()
        tbl.flush()

        fid.create_array(fid.root, 'seeds', pan_random_seeds[i,:])
        for bus in random_load_buses:
            fid.create_earray(fid.root, f'noise_bus_{bus}', atom, (0, N_samples))
        for disk_var in mem_vars_map.values():
            if disk_var is not None:
                if isinstance(disk_var, str) and disk_var != time_disk_var:
                    fid.create_earray(fid.root, disk_var, atom, (0, N_samples-1))
                elif isinstance(disk_var, list):
                    fid.create_earray(fid.root, disk_var[0], atom, (0, N_samples-1))
        # close the file so that other programs can read it
        fid.close()

        for j in range(N_trials):

            # build the noisy samples
            for k,bus in enumerate(random_load_buses):
                state = RandomState(MT19937(SeedSequence(loads_random_seeds[k][i,j])))
                exec(f'noise_samples_bus_{bus}[1, ] = OU(dt, alpha[k], mu[k], c[k], N_samples, state)')

            # run a transient analysis
            data = pan.tran('Tr', tstop, mem_vars, nettype=1, method=2, maxord=2, \
                            noisefmax=frand/2, noiseinj=2, seed=pan_random_seeds[i,j], \
                            iabstol=1e-6, devvars=1, tmax=0.1, annotate=1)

            # save the results to file
            get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

            fid = tables.open_file(out_file, 'a')

            for bus in random_load_buses:
                exec(f'fid.root.noise_bus_{bus}.append(noise_samples_bus_{bus}[1, np.newaxis, :])')

            if j == 0:
                time = get_var(data, mem_vars, time_mem_var)
                fid.create_array(fid.root, time_disk_var, time)

            for k,mem_var in enumerate(mem_vars):
                disk_var = mem_vars_map[mem_var]
                if 'omegael' in mem_var:
                    offset = 1.
                else:
                    offset = 0.
                if disk_var is not None:
                    if isinstance(disk_var, str) and disk_var != time_disk_var:
                        fid.root[disk_var].append(data[k][np.newaxis, :] + offset)
                    elif isinstance(disk_var, list):
                        disk_var, orig_time_var, resampled_time_var = disk_var
                        var = get_var(data, mem_vars, mem_var)
                        orig_time = get_var(data, mem_vars, orig_time_var)
                        resampled_time = get_var(data, mem_vars, resampled_time_var)
                        try:
                            idx = np.array([np.where(origin_time == tt)[0][0] for tt in resampled_time])
                            fid.root[disk_var].append(var[np.newaxis, idx] + offset)
                        except:
                            f = interp1d(orig_time, var)
                            fid.root[disk_var].append(f(resampled_time)[np.newaxis, :] + offset)

            fid.close()

