
import numpy as np
import sys

if __name__ == '__main__':
    mu = float(sys.argv[1])
    sigma = float(sys.argv[2])
    N = int(sys.argv[3])
    rnd = np.random.normal(mu, sigma, N)
    for r in rnd:
        sys.stdout.write(f'{r} ')
    sys.stdout.write('\n')
