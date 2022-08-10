
import os
import sys
import json
import pickle
import numpy as np
import pypan.ui as pan


def optimize_compensators_set_points(compensators, pan_libs, Qmax=50, verbose=False):
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

if __name__ == '__main__':

    progname = os.path.basename(sys.argv[0])
    try:
        config = json.load(open(sys.argv[1]))
    except:
        print('usage: {} config_file [pan_file]'.format(progname))
        print('')
        print('       if pan_file is not provided, the one in config_file will be used')
        sys.exit(0)

    if len(sys.argv) > 2:
        pan_file = sys.argv[2]
    else:
        pan_file = config['netlist']
    if not os.path.isfile(pan_file):
        print('{}: {}: no such file.'.format(progname, pan_file))
        sys.exit(1)

    ok,libs = pan.load_netlist(pan_file)
    if not ok:
        print('Cannot load netlist from file {}.'.format(pan_file))
        sys.exit(2)

    # buses where the stochastic loads are connected
    variable_load_buses = config['variable_load_buses']
    N_variable_loads = len(variable_load_buses)
    N_samples = 10
    t = np.arange(N_samples)
    for bus in variable_load_buses:
        exec(f'load_samples_bus_{bus} = np.zeros((2, N_samples))')
        exec(f'load_samples_bus_{bus}[0,:] = t')

    vg, Q = optimize_compensators_set_points(config['compensators'], libs)
    try:
        pickle.dump({'vg': vg, 'Q': Q}, open(config['compensators_opt_file'], 'wb'))
    except:
        pass
    print('vg =', vg)
    print('Q =', Q)

