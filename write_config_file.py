
import os
import sys
import json
import numpy as np

default_config = {
    'inertia': {'G01': [5], 'G02': [4.33], 'G03': [4.47], 'G04': [3.57],
	        'G05': [4.33], 'G06': [4.35], 'G07': [3.77], 'G08': [3.47],
	        'G09': [3.45], 'G10': [4.2], 'Comp11': [0.1], 'Comp21': [0.1],
                'Comp31': [0.1]
    },
    'inertia_mode': 'sequential',
    'compensators': {'Comp11': 'bus8', 'Comp21': 'bus24', 'Comp31': 'bus27'},
    'variable_load_buses': [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 39],
    'OU': {
	'alpha': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
	'mu': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	'c': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    },
    'dur': 300.0,
    'Ntrials': 100,
    'frequency': 60.0,
    'srate': 400.0,
    'decimation': 10,
    'netlist': 'pan/ieee39_PF_stoch_loads_compensators.pan',
    'mem_vars': {
	'time': 'time',
	'id1': 'Id_line_1_39',
	'id3': 'Id_line_3_4',
	'id14': 'Id_line_14_15',
	'id16': 'Id_line_16_17',
	'iq1': 'Iq_line_1_39',
	'iq3': 'Iq_line_3_4',
	'iq14': 'Iq_line_14_15',
	'iq16': 'Iq_line_16_17',
        'bus3:d': 'Vd_bus3',
        'bus3:q': 'Vq_bus3',
        'bus14:d': 'Vd_bus14',
        'bus14:q': 'Vq_bus14',
        'bus17:d': 'Vd_bus17',
        'bus17:q': 'Vq_bus17',
        'bus39:d': 'Vd_bus39',
        'bus39:q': 'Vq_bus39',
        'omega32': 'omega_ref'
    }
}

prog_name = os.path.basename(sys.argv[0])


def usage():
    print(f'usage: {prog_name} [<options>] inertia')
    print('')
    print(f'    -d, --dur               simulation duration (default: {default_config["dur"]} s)')
    print(f'    -n, --n-trials          number of trials (default: {default_config["Ntrials"]})')
    print(f'    -F, --sampling-rate     sampling rate (default: {default_config["srate"]} Hz)')
    print(f'    -d, --decimation        decimation (default: {default_config["decimation"]})')
    print( '    -o, --output            output file name pattern (default: config.json)')
    print( '    -g, --gen               generator name(s)')
    print( '    -f, --force             force overwrite of existing file(s)')
    print( '    -h, --help              print this help message and exit')
    print('')
    print('inertia can be a (comma-separated list of) scalar(s) or a ' +
          '(semicolon-separated list of) Python expression(s) producing a list of values')


if __name__ == '__main__':

    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)

    i = 1
    n_args = len(sys.argv)

    gen_name = None
    force = False
    output_dir = '.'
    output_file = 'config'
    ext = '.json'
    dur = default_config['dur']
    n_trials = default_config['Ntrials']
    sampling_rate = default_config['srate']
    decimation = default_config['decimation']

    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-d', '--dur'):
            dur =  float(sys.argv[i+1])
            i += 1
            if dur <= 0:
                print('Simulation duration must be > 0')
                sys.exit(1)
        elif arg in ('-n', '--n-trials'):
            n_trials = int(sys.argv[i+1])
            i += 1
            if n_trials <= 0:
                print('Number of trials must be > 0')
                sys.exit(2)
        elif arg in ('-F', '--sampling-rate'):
            sampling_rate = float(sys.argv[i+1])
            i += 1
            if sampling_rate <= 0:
                print('Sampling rate must be > 0')
                sys.exit(3)
        elif arg in ('-d', '--decimation'):
            decimation = int(sys.argv[i+1])
            i += 1
            if decimation <= 0:
                print('Decimation must be > 0')
                sys.exit(4)
        elif arg in ('-o', '--output'):
            output_dir, output_file = os.path.split(sys.argv[i+1])
            output_file, ext = os.path.splitext(output_file)
            i += 1
        elif arg in ('-f', '--force'):
            force = True
        elif arg in ('-g', '--gen'):
            gen_names = sys.argv[i+1].split(',')
            i += 1
            for gen_name in gen_names:
                if gen_name not in default_config['inertia']:
                    print(f'Generator `{gen_name}` not present.')
                    sys.exit(5)
        else:
            break
        i += 1

    if i == n_args:
        usage()

    exprs = sys.argv[i].split(';')
    if len(gen_names) != len(exprs):
        print('The number of generators must match the values of inertia')
        sys.exit(6)

    H = [eval(expr) for expr in exprs]
    X = [x.flatten() for x in np.meshgrid(*H)]
    n,m = len(X), X[0].size
    H = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            H[i,j] = X[j][i]

    if H.size == 1:
        H = H[0,0]
        with_suffix = False
    else:
        with_suffix = True
        fmt = '{}_{:0' + str(int(np.ceil(np.log10(H.shape[0])))) + 'd}{}'

    config = default_config.copy()
    config['dur'] = dur
    config['Ntrials'] = n_trials
    config['srate'] = sampling_rate
    config['decimation'] = decimation

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i,h in enumerate(H):
        for j in range(n):
            config['inertia'][gen_names[j]] = [h[j]]
        if with_suffix:
            outfile = os.path.join(output_dir, fmt.format(output_file, i+1, ext))
        else:
            outfile = os.path.join(output_dir, output_file + ext)
        if os.path.isfile(outfile) and not force:
            print(f'{prog_name}: {outfile}: file exists, use -f to force overwrite.')
            sys.exit(1)
        json.dump(config, open(outfile, 'w'), indent=4)
