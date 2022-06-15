
import numpy as np
import sys

if __name__ == '__main__':
    low = float(sys.argv[1])
    high = float(sys.argv[2])
    N = int(sys.argv[3])
    rnd = np.linspace(low, high, N)
    for r in rnd:
        sys.stdout.write(f'{r:.3f} ')
    sys.stdout.write('\n')
