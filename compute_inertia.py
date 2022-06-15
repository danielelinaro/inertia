
import numpy as np
from scipy.optimize import fsolve

if __name__ == '__main__':

    def func(H, S, fn, M, Hfixed=None, Sfixed=None):
        if Hfixed is not None and Sfixed is not None:
            ret = [fn / 2 * M - (S@H + Sfixed@Hfixed) * 1e-3]
        else:
            ret = [fn / 2 * M - S@H * 1e-3]
        for h,s in zip(H[1:],S[1:]):
            ret.append(S[0]/s - H[0]/h)
        return np.array(ret)

    momentum = lambda H, S, fn: 2 * H@S / fn * 1e-3

    low_mom = False
    low_comp_H = True
    set_name = 'validation'
    fn = 60
    S = np.array([700, 800])
    Sfixed = np.array([100])
    Hdefault = np.array([4.33, 4.47])
    if low_mom:
        if low_comp_H:
            Hfixed = np.array([0.1])
        else:
            Hfixed = np.array([2])
        Mtarget = 0.15
    else:
        if low_comp_H:
            Hfixed = np.array([0.1])
        else:
            Hfixed = np.array([6])
        Mtarget = 0.3

    dM = 0.005
    dH = 1/3
    if set_name == 'test':
        Mtarget += dM
        Hfixed = Hfixed + dH if not low_comp_H else Hfixed
    elif set_name == 'validation':
        Mtarget += 2 * dM
        Hfixed = Hfixed + 2 * dH if not low_comp_H else Hfixed

    Hopt = fsolve(func, Hdefault, args=(S, fn, Mtarget, Hfixed, Sfixed))
    if Hfixed is not None:
        H = np.concatenate((Hopt, Hfixed))
        S = np.concatenate((S, Sfixed))
    else:
        H = Hopt

    print(f'  Target M: {Mtarget}')
    print(f'Computed H: {H}')
    print(f'Computed M: {momentum(H,S,fn)}')
    print('')
    
    
