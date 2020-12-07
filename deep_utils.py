
import numpy as np
import tensorflow as tf


__all__ = ['load_one_block', 'load_data', 'slide_window']


def load_one_block(filename, varname, trial_dur = 60, verbose = False):
    """
    This function loads a single data file containing the results of the simulations for one value of inertia
    """
    data = np.load(filename)
    orig_n_trials, orig_n_samples = data[varname].shape
    dt = np.diff(data['time'][:2])[0]
    n_samples = int(trial_dur / dt)
    n_trials = int(orig_n_trials * orig_n_samples / n_samples)
    var = np.reshape(data[varname], [n_trials, n_samples], order='C')
    time = data['time'][:n_samples]
    if verbose:
        print('There are {} trials, each of which contains {} samples.'.\
              format(n_trials, n_samples))
    return tf.constant(time, dtype=tf.float32), \
           tf.constant(var, dtype=tf.float32), \
           tf.constant([float(data['inertia']) for _ in range(n_trials)], \
                       shape=(n_trials,1), dtype=tf.float32)


def load_data(folder, inertia):
    if folder[-1] != '/':
        folder += '/'
    x = {}
    y = {}
    for key,H in inertia.items():
        for h in H:
            time, omega, inertia = load_one_block(folder + \
                                                  'ieee14_{}_set_H_{:.3f}.npz'.format(key, h), \
                                                  'omega_coi', 60)
            try:
                x[key] = tf.concat([x[key], omega], axis=0)
                y[key] = tf.concat([y[key], inertia], axis=0)
            except:
                x[key] = omega
                y[key] = inertia
    return time, x, y

            
def slide_window(X, window_size, overlap=None, window_step=None, N_windows=-1):
    if overlap is not None and window_step is not None:
        raise 'Only one of "overlap" and "window_step" should be passed'
    if overlap is None and window_step is None:
        raise 'One of "overlap" and "window_step" must be passed'
    if window_step is None:
        window_step = int(window_size * overlap)
    if N_windows <= 0:
        N_windows = int((X.size - window_size) / window_step)
    idx = np.zeros((N_windows, window_size), dtype=int)
    Y = np.zeros((N_windows, window_size))
    for i in range(N_windows):
        idx[i,:] = i * window_step + np.r_[0 : window_size]
        try:

            Y[i,:] = X[idx[i,:]]
        except:
            print('>>> {:04d}/{:04d}'.format(i, N_windows))
            break
    return Y, idx
