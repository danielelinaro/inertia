
import os
import sys
import json
import numpy as np

IEEE39_config_template = {
    "inertia": {"G01": [5], "G02": [4.33], "G03": [4.47], "G04": [3.57], "G05": [4.33],
	        "G06": [4.35], "G07": [3.77], "G08": [3.47], "G09": [3.45], "G10": [4.2],
                "Comp11": [0.1], "Comp21": [0.1], "Comp31": [0.1]
    },
    "inertia_mode": "sequential",
    "compensators": {"Comp11": "bus8", "Comp21": "bus24", "Comp31": "bus27"},
    "variable_load_buses": [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 39],
    "OU": {
	"alpha": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
	"mu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	"c": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    },
    "dur": 300.0,
    "Ntrials": 400,
    "frequency": 60.0,
    "srate": 400.0,
    "decimation": 10,
    "netlist": "pan/ieee39_PF_stoch_loads_compensators.pan",
    "mem_vars": {"time": "time", "omegael03": "omegael_bus3", "omegael14": "omegael_bus14", "omegael17": "omegael_bus17",
	         "omegael39": "omegael_bus39", "pe1": "Pe_line_1_39", "pe3": "Pe_line_3_4", "pe14": "Pe_line_14_15",
	         "pe16": "Pe_line_16_17", "qe1": "Qe_line_1_39", "qe3": "Qe_line_3_4", "qe14": "Qe_line_14_15",
                 "qe16": "Qe_line_16_17", "id1": "Id_line_1_39", "id3": "Id_line_3_4", "id14": "Id_line_14_15",
                 "id16": "Id_line_16_17", "iq1": "Iq_line_1_39", "iq3": "Iq_line_3_4", "iq14": "Iq_line_14_15",
	         "iq16": "Iq_line_16_17", "bus3:d": "Vd_bus3", "bus3:q": "Vq_bus3", "bus14:d": "Vd_bus14",
                 "bus14:q": "Vq_bus14", "bus17:d": "Vd_bus17", "bus17:q": "Vq_bus17", "bus39:d": "Vd_bus39",
                 "bus39:q": "Vq_bus39", "omega32": "omega_ref"
    }
}

prog_name = os.path.basename(sys.argv[0])


def usage():
    print(f'usage: {prog_name} [<options>] <target_momentum>')
    print('')
    print('    -S                       apparent power of the generators')
    print('    --H-fixed, --S-fixed     inertia and apparent power of the generators')
    print('                             that should not be changed')
    print('    -F, --freq               operating frequency of the network (default: 60 Hz)')
    print('    -o, --output             output file name')
    print('    --gen-names              names of the generators whose inertia is computed')
    print('                             (only used when -o is given)')
    print('    -f, --force              force overwrite of existing file')
    print('    -h, --help               print this help message and exit')
    print('')
    print('Default values for the IEEE 39-bus network:')
    print('    Area 1 (G2, G3): S = [700,800], H = [4.33,4.47]')
    print('    Area 2 (G4, G5, G6, G7): S = [800,300,800,700], H = [3.57,4.33,4.35,3.77]')
    print('    Area 3 (G8, G9, G10): S = [700,1000,1000], H = [3.47,3.45,4.2])')
    print('')


if __name__ == '__main__':

    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)

    i = 1
    n_args = len(sys.argv)

    S = None
    fixed_S = None
    fixed_H = None
    fn = 60
    output_file = None
    force = False
    gen_names = []

    to_array = lambda arg: np.array(list(map(float, arg.split(','))))
    while i < n_args:
        arg = sys.argv[i]
        if arg == '-S':
            S = to_array(sys.argv[i+1])
            i += 1
        elif arg == '--S-fixed':
            fixed_S = to_array(sys.argv[i+1])
            i += 1
        elif arg == '--H-fixed':
            fixed_H = to_array(sys.argv[i+1])
            i += 1
        elif arg in ('-F', '--freq'):
            fn = float(sys.argv[i+1])
            i += 1
        elif arg in ('-o', '--output'):
            output_file = sys.argv[i+1]
            i += 1
        elif arg in ('-f', '--force'):
            force = True
        elif arg == '--gen-names':
            gen_names = sys.argv[i+1].split(',')
            i += 1
        else:
            break
        i += 1

    if i == n_args:
        usage()

    if S is None:
        print('You must provide values of apparent power.')
        sys.exit(1)

    if fn <= 0:
        print('Frequency should be > 0')
        sys.exit(2)

    if fixed_H is not None and fixed_S is None:
        fixed_S = 100. + np.zeros(fixed_H.shape)

    if fixed_S is not None and fixed_H is None:
        print('The inertia values of the fixed generators should be provided.')
        sys.exit(3)

    target_M = float(sys.argv[i])
    if target_M <= 0:
        print('Target momentum should be > 0.')
        sys.exit(4)

    if output_file is not None and os.path.isfile(output_file) and not force:
        print(f'{prog_name}: {output_file}: file exists, use -f to force overwrite')
        sys.exit(5)

    momentum = lambda H, S, fn: 2 * H@S / fn * 1e-3
    
    n = S.size
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0,:] = S * 1e-3
    b[0] = target_M * fn / 2
    if fixed_S is not None:
        b[0] -= fixed_S @ fixed_H * 1e-3
    for i in range(1, n):
        A[i,0] = S[i]
        A[i,i] = -S[0]

    H = np.linalg.solve(A, b)
    if fixed_H is not None:
        H = np.concatenate((H, fixed_H))
        S = np.concatenate((S, fixed_S))

    print(f'         S: {S}')
    if fixed_H is not None:
        print(f'   Fixed S: {fixed_S}')
        print(f'   Fixed H: {fixed_H}')
    print(f'  Target M: {target_M}')
    print(f'Computed H: {H}')
    print(f'Computed M: {momentum(H, S, fn)}')

    if output_file is not None:
        if len(gen_names) != S.size:
            print('Not enough generator names provided')
            sys.exit(6)
        for gen_name,h in zip(gen_names, H):
            IEEE39_config_template['inertia'][gen_name] = [h]
        json.dump(IEEE39_config_template, open(output_file, 'w'), indent=4)

        target_gen = 'G02'
        variable = gen_names.index(target_gen)
        others = [i for i in range(len(gen_names) - len(fixed_H)) if gen_names[i] != target_gen]
        if len(others) > 1:
            raise Exception('Do not know how to deal with multiple other generators')

        other = others[0]
        n_steps = 10
        H_variable = np.linspace(0.5 * H[variable], 1.5 * H[variable], n_steps)
        output_file = os.path.splitext(output_file)[0]
        for i,h_variable in enumerate(H_variable):
            h_other = (target_M * 1000 * fn/2 - h_variable * S[variable] - fixed_H @ fixed_S) / S[other]
            H[variable] = h_variable
            H[other] = h_other
            print(f'{target_M}: {H} -> {momentum(H, S, fn)}')
            json.dump(IEEE39_config_template, open(f'{output_file}_{i+1:02d}.json', 'w'), indent=4)


