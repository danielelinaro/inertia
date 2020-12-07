
import os
import sys
import argparse as arg
import numpy as np

import pypan.ui as pan

progname = os.path.basename(sys.argv[0])


if __name__ == '__main__':

    parser = arg.ArgumentParser(description = 'Build data for inertia estimation with deep neural networks', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('pan_file', type=str, action='store', nargs='?', default='ieee14.pan', help='PAN netlist')
    parser.add_argument('-d', '--dur', default=300, type=float, help='simulation duration in seconds')
    parser.add_argument('--H-min',  default=2,  type=float, help='minimum value of inertia')
    parser.add_argument('--H-max',  default=10, type=float, help='maximum value of inertia')
    parser.add_argument('--H-step', default=1,  type=float, help='step of inertia')
    parser.add_argument('-H', '--inertia',  default=None, type=str, help='inertia values in the form "Hmin:Hmax:Hstep"')
    parser.add_argument('--alpha',  default=0.5,  type=float, help='alpha parameter of the OU process')
    parser.add_argument('--mu',  default=0.0,  type=float, help='mu parameter of the OU process')
    parser.add_argument('-c',  default=0.5,  type=float, help='c parameter of the OU process')
    parser.add_argument('--frand',  default=10,  type=float, help='sampling rate of the random signal')
    parser.add_argument('-N', '--n-trials',  default=100,  type=int, help='number of trials')
    parser.add_argument('-s', '--suffix',  default='',  type=str, help='suffix to add to the output files')
    parser.add_argument('-O', '--output-dir',  default='.',  type=str, help='output directory')
    args = parser.parse_args(args=sys.argv[1:])
    
    with open('/dev/random', 'rb') as fid:
        hw_seed = int.from_bytes(fid.read(4), 'little') % 1000000
    np.random.seed(hw_seed)
    
    # OU parameters
    alpha = args.alpha
    mu = args.mu
    c = args.c

    # simulation parameters
    frand = args.frand  # [Hz] sampling rate of the random signal
    tstop = args.dur    # [s]  simulation duration

    # inertia values
    H_min = args.H_min
    H_max = args.H_max
    dH = args.H_step
    if args.inertia is not None:
        try:
            H_min, H_max, dH = map(float, args.inertia.split(':'))
        except:
            H_min, H_max = map(float, args.inertia.split(':'))
            dH = 1.0
    H = np.r_[H_min : H_max + dH/2 : dH]
    N_H = H.size

    # how many trials per inertia value
    N_trials = args.n_trials
    random_seeds = np.random.randint(low=0, high=1000000, size=(N_H, N_trials))

    dt = 1 / frand
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size
    coeff = np.array([alpha * mu * dt, 1 / (1 + alpha * dt)])

    pan_file = args.pan_file
    if not os.path.isfile(pan_file):
        print('{}: {}: no such file.'.format(progname, pan_file))
        sys.exit(1)
    
    ok,libs = pan.load_netlist(pan_file)
    if not ok:
        print('Cannot load netlist from file {}.'.format(pan_file))
        sys.exit(2)

    pan.alter('Altstop', 'TSTOP', tstop, annotate=1)
    pan.alter('Alfrand', 'FRAND', frand, annotate=1)

    omega_G1  = np.zeros((N_trials, N_samples - 1))
    omega_G2  = np.zeros((N_trials, N_samples - 1))
    omega_G3  = np.zeros((N_trials, N_samples - 1))
    omega_G6  = np.zeros((N_trials, N_samples - 1))
    omega_G8  = np.zeros((N_trials, N_samples - 1))
    omega_coi = np.zeros((N_trials, N_samples - 1))
    noise     = np.zeros((N_trials, N_samples))

    mem_vars = ['time:noise', 'omega01:noise', 'omega02:noise', \
                'G3:omega:noise', 'G6:omega:noise', \
                'G8:omega:noise', 'omegacoi:noise']
    #disk_vars = ['agc02','drift14','omega*','pm','bus13']
    disk_vars = ['pm']

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    suffix = args.suffix
    if suffix[0] != '_':
        suffix = '_' + suffix
    if suffix[-1] == '_':
        suffix = suffix[:-1]

    for i in range(N_H):

        pan.alter('Al{}'.format(i), 'm', 2 * H[i], instance='G1', invalidate='false')

        for j in range(N_trials):

            np.random.seed(random_seeds[i,j])
            rnd = c * np.sqrt(dt) * np.random.normal(size=N_samples)

            for k in range(N_samples-1):
                noise[j, k+1] = (noise[j, k] + coeff[0] + rnd[k]) * coeff[1]

            noise_samples = np.vstack((t, noise[j,:]))

            tran_name = 'Tr_{}_{}'.format( i, j)
            data = pan.tran(tran_name, tstop, mem_vars, nettype=1, method=2, maxord=2, \
                            noisefmax=frand/2, noiseinj=2, seed=random_seeds[i,j], \
                            iabstol=1e-6, devvars=1, tmax=0.1, annotate=1, \
                            savelist='["' + '","'.join(disk_vars) + '"]')

            omega_G1[j,:]  = data[1,:]
            omega_G2[j,:]  = data[2,:]
            omega_G3[j,:]  = data[3,:]
            omega_G6[j,:]  = data[4,:]
            omega_G8[j,:]  = data[5,:]
            omega_coi[j,:] = data[6,:]

        time = data[0,:]
        seeds = random_seeds[i,:]
        inertia = H[i]

        out_file = '{}/{}{}_H_{:.3f}'.format(output_dir, os.path.splitext(pan_file)[0], suffix, H[i])

        np.savez_compressed(out_file, time=time, omega_G1=omega_G1, omega_G2=omega_G2, \
                            omega_G3=omega_G3, omega_G6=omega_G6, omega_G8=omega_G8, \
                            omega_coi=omega_coi, seeds=seeds, inertia=inertia, noise=noise, \
                            hw_seed=hw_seed, alpha=alpha, mu=mu, c=c)

