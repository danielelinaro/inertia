
import os
import sys
import numpy as np


prog_name = os.path.basename(sys.argv[0])


def usage():
    print(f'usage: {prog_name} [<options>] <target_momentum>')
    print('')
    print('    -S                       apparent power of the generators')
    print('    --fixed-H, --fixed-S     inertia and apparent power of the generators')
    print('                             that should not be changed')
    print('    -f, --freq               operating frequency of the network (default: 60 Hz)')
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

    i = 1
    n_args = len(sys.argv)

    S = None
    fixed_S = None
    fixed_H = None
    fn = 60

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
        elif arg in ('-f', '--freq'):
            fn = float(sys.argv[i+1])
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
    
