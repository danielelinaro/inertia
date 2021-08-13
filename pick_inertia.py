
import os
import sys
import numpy as np


if __name__ == '__main__':
    def usage(exit_code=0):
        print(f'usage: {os.path.basename(sys.argv[0])} [area_id]')
        sys.exit(exit_code)

    area_id = 1
    if len(sys.argv) > 1:
        if sys.argv[1] in ('-h','--help'):
            usage(0)

        try:
            area_id = int(sys.argv[1])
        except:
            usage(1)

        if area_id not in (1,2,3):
            print('Area id must be 1, 2 or 3.')
            sys.exit(2)

    # area 1: Pg31, Pg32, Pg39
    N = 15
    H1  = 20 + 2 * np.arange(N)
    H2  = 24 + 2 * np.arange(N)
    H3  = 32 + 2 * np.arange(N)
    # area 2: Pg33, Pg34, Pg35, Pg36
    N = 15
    H4  = 18 + 2 * np.arange(N)
    H5  = 16 + 2 * np.arange(N)
    H6  = 24 + 2 * np.arange(N)
    H7  = 16 + 2 * np.arange(N)
    # area 3: Pg37, Pg38, Pg30
    N = 20
    H8  = 10 + 2 * np.arange(N)
    H9  = 20 + 2 * np.arange(N)
    H10 = 500 + np.zeros(N)

    generator_names = {
        1: ['Pg31', 'Pg32', 'Pg39'],
        2: ['Pg33', 'Pg34', 'Pg35', 'Pg36'],
        3: ['Pg37', 'Pg38', 'Pg30'],
    }

    all_inertias = {
        1: np.array([[h1, h2, h3, (h1 + h2 + h3) * 1e-1] for h1 in H1 for h2 in H2 for h3 in H3]),
        2: np.array([[h4, h5, h6, h7, (h4 + h5 + h6 + h7) * 1e-1] for h4 in H4 for h5 in H5 for h6 in H6 for h7 in H7]),
        3: np.array([[h8, h9, h10, (h8 + h9 + h10) * 1e-1] for h8 in H8 for h9 in H9 for h10 in H10]),
    }

    desired_values = {
        1: {
            'training': np.arange(8.0, 14.2, 0.6),
            'test': np.arange(8.2, 14.4, 0.6),
            'validation': np.arange(8.4, 14.6, 0.6)
        },
        2: {
            'training': np.arange(8.0, 14.2, 0.6),
            'test': np.arange(8.2, 14.4, 0.6),
            'validation': np.arange(8.4, 14.6, 0.6)
        },
        3: {
            'training': np.arange(53.0, 59.2, 0.6),
            'test': np.arange(53.2, 59.4, 0.6),
            'validation': np.arange(53.4, 59.6, 0.6)
        }
    }

    idx = np.argsort(all_inertias[area_id][:,-1])
    inertia = all_inertias[area_id][idx,:]

    for set_name in 'training', 'test', 'validation':
        H = []
        for val in desired_values[area_id][set_name]:
            jdx = np.argmin(np.abs(inertia[:,-1] - val))
            H.append(inertia[jdx, :])
        print(f'{set_name.capitalize()} set:')
        H = np.array(H)
        H_str = ' '.join([f'{h:<5.1f}' for h in H[:,-1]])
        print(f'Area H:  {H_str}')
        for col,gen_name in enumerate(generator_names[area_id]):
            s = repr(H[:,col]).replace('array(','').replace(')','').replace('.,','.0,').replace('.]','.0]')
            print(f'"{gen_name}": {s}')
        print()

