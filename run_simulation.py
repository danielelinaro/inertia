
import os
import sys
import json
import argparse as arg
import tables
import numpy as np
import scipy
import pypan.ui as pan
from numpy.random import RandomState, SeedSequence, MT19937

from build_data import BaseParameters, OU, save_compensators_info
from optimize_compensators_set_points import optimize_compensators_set_points

progname = os.path.basename(sys.argv[0])

if __name__ == '__main__':
    parser = arg.ArgumentParser(description = 'Simulate a power network at a fixed value of inertia', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='PAN netlist')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    parser.add_argument('-O', '--outdir', default=None, type=str, help='output directory')
    parser.add_argument('-S', '--suffix', default=None, type=str, help='suffix to prepend to file extension')
    parser.add_argument('--overload', default=None, type=float,
                        help='overload coefficient (overwrites the value in the config file)')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file')
    parser.add_argument('--save-ou-to-mat', action='store_true', help='save the OU traces to a MAT file')
    parser.add_argument('--check-stability', action='store_true',
                        help='check that the system is stable by running a pole-zero analysis')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
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

    pan_file = config['netlist']
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

    variable_load_buses = config['variable_load_buses']
    N_variable_loads = len(variable_load_buses)
    N_blocks = len(config['tstop'])

    # simulation parameters
    srate = config['srate']        # [Hz] sampling rate
    sim_dur = config['tstop'][-1]  # [s]  total simulation duration
    decimation = config['decimation'] if 'decimation' in config else 1
    dt = 1 / srate
    t = dt + np.r_[0 : sim_dur + dt/2 : dt]
    N_samples = t.size

    if 'OU' in config:
        try:
            rng_seeds = config['seeds']
        except:
            with open('/dev/urandom', 'rb') as fid:
                rng_seeds = [int.from_bytes(fid.read(4), 'little') % 1000000 for _ in range(N_variable_loads + N_blocks)]

        if integration_mode == 'trapezoidal':
            rng_seeds = rng_seeds[:N_variable_loads]
            pan_seeds = np.nan + np.zeros(N_blocks)
        else:
            rng_seeds, pan_seeds = rng_seeds[:-N_blocks], rng_seeds[-N_blocks:]

        rnd_states = [RandomState(MT19937(SeedSequence(seed))) for seed in rng_seeds]

        # OU parameters
        alpha = config['OU']['alpha']
        mu = config['OU']['mu']
        c = config['OU']['c']

        var_loads = [OU(dt, alpha[i], mu[i], c[i], N_samples, rnd_states[i]) for i in range(N_variable_loads)]
    elif 'PWL' in config:
        PWL = [np.array(pwl) for pwl in config['PWL']]
        var_loads = [np.zeros(N_samples) for _ in range(N_variable_loads)]
        for var_load,pwl in zip(var_loads, PWL):
            N_steps = pwl.shape[0]
            for i in range(N_steps - 1):
                idx = (t >= pwl[i, 0]) & (t < pwl[i+1, 0])
                var_load[idx] = pwl[i, 1]
            idx = t >= pwl[-1, 0]
            var_load[idx] = pwl[-1, 1]
    else:
        var_loads = [np.zeros(N_samples) for _ in range(N_variable_loads)]

    mem_vars_map = config['mem_vars_map']
    mem_vars = list(mem_vars_map.keys())
    time_mem_var = mem_vars[['time' in mem_var for mem_var in mem_vars].index(True)]
    time_disk_var = mem_vars_map[time_mem_var]

    ok,libs = pan.load_netlist(pan_file)
    if not ok:
        print('Cannot load netlist from file {}.'.format(pan_file))
        sys.exit(4)

    if args.output is None:
        import subprocess
        name_max = int(subprocess.check_output('getconf NAME_MAX /', shell=True))
        output_file = os.path.splitext(os.path.basename(pan_file))[0] + '_' + \
            '_'.join(['-'.join(map(lambda h: f'{h:.3f}', H)) if np.any(H != H[0]) else f'{H[0]:.3f}' for H in inertia_values])
        if len(output_file) > name_max:
            output_file = os.path.splitext(os.path.basename(pan_file))[0] + '_' + \
                '_'.join(['-'.join(map(lambda h: f'{h:.3f}', np.unique(H))) for H in inertia_values])
        if args.overload is not None:
            output_file += f'_lambda={LAMBDA:.3f}'
        if args.suffix is not None:
            output_file += '_' + args.suffix.lstrip('_')
        output_file += '.h5'
        if args.outdir is not None:
            output_file = os.path.join(args.outdir, output_file)
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

    pan.alter('Altstop', 'TSTOP',  sim_dur, libs, annotate=1)
    pan.alter('Alsrate', 'SRATE',  srate,   libs, annotate=1)

    try:
        # gen_a is the ''original'' generator, gen_b is the additional one
        for gen_b,(gen_a,power_frac) in config['split_gen'].items():
            pg = pan.get_var(gen_a + '.pg')
            pan.alter('Alpg', 'pg', pg[0] * (1-power_frac), libs, instance=gen_a, annotate=1, invalidate=0)
            pg = pan.get_var(gen_b + '.pg')
            pan.alter('Alpg', 'pg', pg[0] * power_frac, libs, instance=gen_b, annotate=1, invalidate=0)
        with_split_gen = True
    except:
        with_split_gen = False

    if 'VSGs' in config:
        with_VSGs = True
        for vsg,val in config['VSGs'].items():
            if len(val) > 0:
                gen, power_frac = val
                pg = pan.get_var(gen + '.pg')
                pan.alter('Alpg', 'pg', pg[0] * (1-power_frac), libs, instance=gen, annotate=1, invalidate=0)
                pg = pan.get_var(vsg + '.PG')
                pan.alter('Alpg', 'PG', pg[0] * power_frac, libs, instance=vsg, annotate=1, invalidate=0)
    else:
        with_VSGs = False

    if args.save_ou_to_mat:
        data = {}
    for i,bus in enumerate(variable_load_buses):
        exec(f'load_samples_bus_{bus} = np.vstack((t, var_loads[i]))')
        if args.save_ou_to_mat:
            data[f'load_samples_bus_{bus}'] = np.vstack((t, var_loads[i])).T
    if args.save_ou_to_mat:
        scipy.io.savemat('OU.mat', data)

    get_var = lambda data, mem_vars, name: data[mem_vars.index(name)]

    class Parameters (BaseParameters):
        generator_IDs  = tables.StringCol(8, shape=(N_generators,))
        var_load_buses = tables.Int64Col(shape=(N_variable_loads,))
        inertia        = tables.Float64Col(shape=(N_generators,N_blocks))
        tstop          = tables.Float64Col(shape=(N_blocks,))

    if 'OU' in config:
        Parameters.__dict__['columns']['rng_seeds'] = tables.Int64Col(shape=(N_variable_loads,))
        Parameters.__dict__['columns']['pan_seeds'] = tables.Float64Col(shape=(N_blocks,)) # these must be floats because they might be NaN's
        for key in 'alpha','mu','c':
            Parameters.__dict__['columns'][key] = tables.Float64Col(shape=(N_variable_loads,))
    elif 'PWL' in config:
        for bus,pwl in zip(variable_load_buses, PWL):
            m,n = pwl.shape
            Parameters.__dict__['columns'][f'PWL_bus_{bus}'] = tables.Float64Col(shape=(m,n))

    fid = tables.open_file(output_file, 'w', filters=tables.Filters(complib='zlib', complevel=5))
    tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
    params = tbl.row
    params['tstop']          = config['tstop']
    params['F0']             = config['frequency']
    params['srate']          = srate
    params['inertia']        = inertia_values
    params['generator_IDs']  = generator_IDs
    params['var_load_buses'] = variable_load_buses

    if 'OU' in config:
        params['rng_seeds']  = rng_seeds
        params['pan_seeds']  = pan_seeds
        params['alpha']      = alpha
        params['mu']         = mu
        params['c']          = c
    elif 'PWL' in config:
        for bus,pwl in zip(variable_load_buses, PWL):
            params[f'PWL_bus_{bus}'] = pwl

    if 'damping' in config:
        D = config['damping']
        pan.alter('Ald', 'D', D, libs, annotate=1)
        params['D'] = D
    else:
        params['D'] = np.nan

    if 'DZA' in config:
        DZA = config['DZA'] / config['frequency']
        pan.alter('Aldza', 'DZA', DZA, libs, annotate=1)
        params['DZA'] = DZA
    else:
        params['DZA'] = np.nan

    if args.overload is not None:
        LAMBDA = args.overload
        pan.alter('Allam', 'LAMBDA', LAMBDA, libs, annotate=1)
        params['LAMBDA'] = LAMBDA
    elif 'lambda' in config:
        LAMBDA = config['lambda']
        pan.alter('Allam', 'LAMBDA', LAMBDA, libs, annotate=1)
        params['LAMBDA'] = LAMBDA
    else:
        params['LAMBDA'] = np.nan

    if 'coeff' in config:
        COEFF = config['coeff']
        pan.alter('Alcoeff', 'COEFF', COEFF, libs, annotate=1)
        params['COEFF'] = COEFF
    else:
        params['COEFF'] = np.nan

    params.append()
    tbl.flush()

    try:
        save_compensators_info(fid, config['compensators'], compensators['vg'], compensators['Q'])
        for (name,bus),vg in zip(config['compensators'].items(), compensators['vg']):
            pan.alter('Alvg', 'vg', vg, libs, instance=name, annotate=1, invalidate=0)
    except:
        pass

    atom = tables.Float64Atom()

    if integration_mode == 'trapezoidal':
        array_shape = t[::decimation].size,
    else:
        array_shape = t[::decimation].size - 1,

    for disk_var in mem_vars_map.values():
        if disk_var is not None:
            if isinstance(disk_var, str):
                fid.create_carray(fid.root, disk_var, atom, array_shape)
            elif isinstance(disk_var, list):
                fid.create_earray(fid.root, disk_var[0], atom, array_shape)

    if 'save_var_loads' in config and config['save_var_loads']:
        fid.create_array(fid.root, 'var_loads', np.array(var_loads)[:,::decimation], atom=atom)

    start = 0
    for i, tstop in enumerate(config['tstop']):

        for j, gen_id in enumerate(generator_IDs):
            if with_VSGs:
                if gen_id in config['VSGs'].keys():
                    pan.alter('Alh', 'TA', 2*inertia_values[j,i], libs, instance=gen_id, annotate=1, invalidate=0)
                else:
                    for vsg,val in config['VSGs'].items():
                        if len(val) > 0:
                            gen,power_frac = val
                            if gen == gen_id:
                                pan.alter('Alh', 'h',    inertia_values[j,i] * (1-power_frac), libs, instance=gen, annotate=1, invalidate=0)
                                pan.alter('Alh', 'TA', 2*inertia_values[j,i] * power_frac, libs, instance=vsg, annotate=1, invalidate=0)
            elif with_split_gen:
                # gen_a is the ''original'' generator, gen_b is the additional one
                for gen_b,(gen_a,power_frac) in config['split_gen'].items():
                    if gen_a == gen_id:
                        pan.alter('Alh', 'h', inertia_values[j,i] * (1-power_frac), libs, instance=gen_a, annotate=1, invalidate=0)
                        pan.alter('Alh', 'h', inertia_values[j,i] * power_frac, libs, instance=gen_b, annotate=1, invalidate=0)
            else:
                pan.alter('Alh', 'h', inertia_values[j,i], libs, instance=gen_id, annotate=1, invalidate=0)

        kwargs = {'nettype': 1, 'annotate': 3, 'restart': 1 if i == 0 else 0}

        if integration_mode == 'trapezoidal':
            kwargs['method']     = 1
            kwargs['timepoints'] = 1 / srate
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
            if with_VSGs and i == 0:
                kwargs_reduced = kwargs.copy()
                kwargs_reduced.pop('forcetps')
                pan.tran('TrA', 10/srate, None, libs, **kwargs_reduced)
                kwargs['restart'] = 0
                data = pan.tran(f'Tr{i+1}', tstop, mem_vars, libs, **kwargs)
            else:
                data = pan.tran(f'Tr{i+1}', tstop, mem_vars, libs, **kwargs)
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
                    stop = start + var[::decimation].size
                    fid.root[disk_var][start : stop] = var[::decimation] + offset
                elif isinstance(disk_var, list):
                    disk_var, orig_time_var, resampled_time_var = disk_var
                    orig_time = get_var(data, mem_vars, orig_time_var)
                    resampled_time = get_var(data, mem_vars, resampled_time_var)
                    try:
                        idx = np.array([np.where(origin_time == tt)[0][0] for tt in resampled_time])
                        stop = start + idx.size
                        fid.root[disk_var][start : stop] = var[idx][::decimation] + offset
                    except:
                        f = interp1d(orig_time, var)
                        tmp = f(resampled_time)
                        stop = start + tmp.size
                        fid.root[disk_var][start : stop] = tmp[::decimation] + offset
        start = stop

    fid.close()

