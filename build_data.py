
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

__all__ = ['BaseParameters', 'OU', 'OU_TH', 'generator_ids',
           'optimize_compensators_set_points', 'save_compensators_info']

progname = os.path.basename(sys.argv[0])
generator_ids = {'IEEE14': (1,2,3,6,8), 'two-area': (1,2,3,4)}


class BaseParameters (tables.IsDescription):
    F0       = tables.Float64Col()
    srate    = tables.Float64Col()
    D        = tables.Float64Col()
    DZA      = tables.Float64Col()
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


def OU_TH(dt, alpha, mu, c, N, random_state = None):
    t = np.arange(N) * dt
    ex = np.exp(-alpha * t)
    if random_state is not None:
        rnd = random_state.normal(size=N-1)
    else:
        rnd = np.random.normal(size=N-1)
    ou0 = 0
    tmp = np.cumsum(np.r_[0, np.sqrt(np.diff(np.exp(2 * alpha * t) - 1)) * rnd])
    ou = ou0 * ex + mu * (1 - ex) + c * ex * tmp / np.sqrt(2 * alpha);
    return ou


def optimize_compensators_set_points(compensators, pan_libs, Qmax=50, verbose=True):
    def cost(set_points, compensators, pan_libs):
        mem_vars = []
        for (name,bus),vg in zip(compensators.items(), set_points):
            pan.alter('Alvg', 'vg', vg, pan_libs, instance=name, annotate=0, invalidate=0)
            mem_vars.append(bus + ':d')
            mem_vars.append(bus + ':q')
            mem_vars.append(name + ':id')
            mem_vars.append(name + ':iq')
        Q = np.zeros(len(compensators))
        lf = pan.DC('Lf', mem_vars=mem_vars, libs=pan_libs, nettype=1, annotate=0)
        for i,(Vd,Vq,Id,Iq) in enumerate(zip(lf[0::4], lf[1::4], lf[2::4], lf[3::4])):
            V = Vd[0] + 1j * Vq[0]
            I = Id[0] + 1j * Iq[0]
            S = V * I.conj()
            Q[i] = -S.imag * 1e-6
        return Q

    from scipy.optimize import fsolve
    Vgopt = fsolve(cost, np.ones(len(compensators)), args=(compensators, pan_libs))
    Q = cost(Vgopt, compensators, pan_libs) * 1e6 # [VAR]
    if np.any(np.abs(Q) > Qmax):
        raise Exception('Optimization of compensators set points failed')
    if verbose:
        print('Setting optimal compensators set points:')
    for name,vg in zip(compensators, Vgopt):
        pan.alter('Alvg', 'vg', vg, pan_libs, instance=name, annotate=3 if verbose else 0, invalidate=0)
    return Vgopt, Q


def save_compensators_info(fid, compensators, set_points, Q):
    N_compensators = len(compensators)
    names = list(compensators.keys())
    buses = list(compensators.values())
    name_len = max(map(len, names))
    bus_len = max(map(len, buses))
    class Compensators (tables.IsDescription):
        names      = tables.StringCol(name_len, shape=(N_compensators,))
        buses      = tables.StringCol(bus_len, shape=(N_compensators,))
        set_points = tables.Float64Col(shape=(N_compensators,))
        Q          = tables.Float64Col(shape=(N_compensators,))
    tbl = fid.create_table(fid.root, 'compensators', Compensators, 'compensators info')
    comps = tbl.row
    comps['names']      = names
    comps['buses']      = buses
    comps['set_points'] = set_points
    comps['Q']          = Q
    comps.append()
    tbl.flush()


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
    
    if 'compensators' in config and isinstance(config['compensators'], dict):
        # these are just placeholder variables, they will be overwritten in the following
        n = 10
        t = np.arange(n)
        x = np.random.uniform(size=n)
        for bus in config['variable_load_buses']:
            exec(f'load_samples_bus_{bus} = np.vstack((t, x))')
        # I do this here because doing it later causes an instability in the simulation
        # I haven't figured out why, but the problem seems to be the call to pan.DC in
        # the function optimize_compensators_set_points
        _,libs = pan.load_netlist(config['netlist'])
        compensators = {}
        compensators['vg'], compensators['Q'] = optimize_compensators_set_points(config['compensators'], libs)

    # OU parameters
    alpha = config['OU']['alpha']
    mu = config['OU']['mu']
    c = config['OU']['c']

    # simulation parameters
    srate = config['srate']  # [Hz] sampling rate of the random signal
    decimation = config['decimation'] if 'decimation' in config else 1
    if args.dur is not None:
        tstop = args.dur     # [s]  simulation duration
    else:
        tstop = config['dur']

    # generator IDs
    generator_IDs = list(config['inertia'].keys())
    N_generators = len(generator_IDs)

    # inertia values
    try:
        inertia_mode = config['inertia_mode']
    except:
        inertia_mode = 'combinatorial'
    if inertia_mode == 'combinatorial':
        inertia = config['inertia']
        inertia_values = []
        for gen_id in generator_IDs:
            inertia_values.append(inertia[gen_id])
        H = np.meshgrid(*inertia_values)
        inertia_values = {}
        for i,gen_id in enumerate(generator_IDs):
            inertia_values[gen_id] = H[i].flatten()
        N_inertia = inertia_values[generator_IDs[0]].size
    elif inertia_mode == 'sequential':
        inertia_values = config['inertia'].copy()
        N_inertia = max(map(len, inertia_values.values()))
        for k in inertia_values:
            N_values = len(inertia_values[k])
            if N_values == 1:
                inertia_values[k] = inertia_values[k][0] + np.zeros(N_inertia)
            elif N_values != N_inertia:
                raise Exception(f'The number of inertia values for generator "{k}" does not match the other generators')
    else:
        print(f'Unknown value for inertia_mode: "{inertia_mode}".')
        print('Accepted values are: "combinatorial" and "sequential".')
        sys.exit(1)

    # how many trials per inertia value
    if args.n_trials is not None:
        N_trials = args.n_trials
    else:
        N_trials = config['Ntrials']

    # buses where the stochastic loads are connected
    variable_load_buses = config['variable_load_buses']
    N_variable_loads = len(variable_load_buses)

    # seeds for the random number generators
    with open('/dev/urandom', 'rb') as fid:
        hw_seeds = [int.from_bytes(fid.read(4), 'little') % 1000000 for _ in range(N_variable_loads)]
    random_states = [RandomState(MT19937(SeedSequence(seed))) for seed in hw_seeds]
    seeds = [rs.randint(low=0, high=1000000, size=(N_inertia, N_trials)) for rs in random_states]

    dt = 1 / srate
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size
    earray_shape = (0, t[::decimation].size)
    for i,bus in enumerate(variable_load_buses):
        exec(f'load_samples_bus_{bus} = np.zeros((2, N_samples))')
        exec(f'load_samples_bus_{bus}[0,:] = t')

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
    F0 = config['frequency']
    pan.alter('Altstop', 'TSTOP',  tstop,  libs, annotate=1)
    pan.alter('Alsrate', 'SRATE',  srate,  libs, annotate=1)

    # damping coefficient
    if 'damping' in config:
        D = config['damping']
        pan.alter('Ald', 'D', D, libs, annotate=1)
    else:
        D = np.nan

    # dead-band width
    if 'DZA' in config:
        DZA = config['DZA'] / F0
        pan.alter('Aldza', 'DZA', DZA, libs, annotate=1)
    else:
        DZA = np.nan

    # overload coefficient
    if 'lambda' in config:
        LAMBDA = config['lambda']
        pan.alter('Allam', 'LAMBDA', LAMBDA, libs, annotate=1)
    else:
        LAMBDA = np.nan

    # load scaling coefficient
    if 'coeff' in config:
        COEFF = config['coeff']
        pan.alter('Alcoeff', 'COEFF',  COEFF,  libs, annotate=1)
    else:
        COEFF = np.nan

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
        hw_seeds       = tables.Int64Col(shape=(N_variable_loads,))
        seeds          = tables.Int64Col(shape=(N_variable_loads,N_trials))
        count          = tables.Int64Col()
        decimation     = tables.Int64Col()
        alpha          = tables.Float64Col(shape=(N_variable_loads,))
        mu             = tables.Float64Col(shape=(N_variable_loads,))
        c              = tables.Float64Col(shape=(N_variable_loads,))
        var_load_buses = tables.Int64Col(shape=(N_variable_loads,))
        generator_IDs  = tables.StringCol(8, shape=(N_generators,))
        inertia        = tables.Float64Col(shape=(N_generators,))

    for i in range(N_inertia):

        out_file = ''
        for gen_id in generator_IDs:
            pan.alter('Al', 'm', 2 * inertia_values[gen_id][i], libs, instance=gen_id, invalidate='false')
            out_file += f'_{inertia_values[gen_id][i]:.3f}'
        out_file = '{}/inertia{}{}.h5'.format(output_dir, out_file, suffix)

        if os.path.isfile(out_file):
            continue

        # run a poles/zeros analysis to save data about the stability of the system
        poles = pan.PZ('Pz', mem_vars=['poles'], libs=libs, nettype=1, annotate=0)
        # sort the poles in descending order and convert them to Hz
        poles = poles[:, [i for i in np.argsort(poles.real)[0][::-1]]] / (2 * np.pi)

        ### first of all, write to file all the data and parameters that we already have
        fid = tables.open_file(out_file, 'w', filters=compression_filter)
        tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
        params = tbl.row
        params['hw_seeds']       = hw_seeds
        params['seeds']          = np.array([s[i,:] for s in seeds])
        params['count']          = i
        params['decimation']     = decimation
        params['alpha']          = alpha
        params['mu']             = mu
        params['c']              = c
        params['D']              = D
        params['DZA']            = DZA
        params['LAMBDA']         = LAMBDA
        params['COEFF']          = COEFF
        params['F0']             = F0
        params['srate']          = srate
        params['var_load_buses'] = variable_load_buses
        params['generator_IDs']  = generator_IDs
        params['inertia']        = [inertia_values[gen_id][i] for gen_id in generator_IDs]
        params.append()
        tbl.flush()

        try:
            save_compensators_info(fid, config['compensators'], compensators['vg'], compensators['Q'])
        except:
            pass

        fid.create_array(fid.root, 'poles', poles)

        for bus in variable_load_buses:
            fid.create_earray(fid.root, f'load_samples_bus_{bus}', atom, earray_shape)
        for disk_var in mem_vars_map.values():
            if disk_var is not None:
                if isinstance(disk_var, str) and disk_var != time_disk_var:
                    fid.create_earray(fid.root, disk_var, atom, earray_shape)
                elif isinstance(disk_var, list):
                    fid.create_earray(fid.root, disk_var[0], atom, earray_shape)
        # close the file so that other programs can read it
        fid.close()

        for j in range(N_trials):

            # build the noisy samples
            for k,bus in enumerate(variable_load_buses):
                state = RandomState(MT19937(SeedSequence(seeds[k][i,j])))
                exec(f'load_samples_bus_{bus}[1,:] = OU(dt, alpha[k], mu[k], c[k], N_samples, state)')

            # run a transient analysis
            data = pan.tran('Tr', tstop, mem_vars, libs, nettype=1,
                            method=1, timepoints=1/srate, forcetps=1,
                            maxiter=65, saman=1, sparse=2)

            # save the results to file
            get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

            fid = tables.open_file(out_file, 'a')

            for bus in variable_load_buses:
                exec(f'fid.root.load_samples_bus_{bus}.append(load_samples_bus_{bus}[1, np.newaxis, ::decimation])')

            if j == 0:
                time = get_var(data, mem_vars, time_mem_var)
                fid.create_array(fid.root, time_disk_var, time[::decimation])

            for k,mem_var in enumerate(mem_vars):
                disk_var = mem_vars_map[mem_var]
                if 'omegael' in mem_var:
                    offset = 1.
                else:
                    offset = 0.
                if disk_var is not None:
                    if isinstance(disk_var, str) and disk_var != time_disk_var:
                        fid.root[disk_var].append(data[k][np.newaxis, ::decimation] + offset)
                    elif isinstance(disk_var, list):
                        disk_var, orig_time_var, resampled_time_var = disk_var
                        var = get_var(data, mem_vars, mem_var)
                        orig_time = get_var(data, mem_vars, orig_time_var)
                        resampled_time = get_var(data, mem_vars, resampled_time_var)
                        try:
                            idx = np.array([np.where(origin_time == tt)[0][0] for tt in resampled_time])
                            var = var[idx][::decimation]
                            fid.root[disk_var].append(var[np.newaxis, :] + offset)
                        except:
                            f = interp1d(orig_time, var)
                            fid.root[disk_var].append(f(resampled_time)[np.newaxis, ::decimation] + offset)

            fid.close()

