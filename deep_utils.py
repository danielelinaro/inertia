
import os
import re
import tables
import numpy as np
import tensorflow as tf


__all__ = ['load_one_block', 'load_data', 'slide_window']


def load_one_block(filename, var_names, trial_dur = 60, verbose = False):
    """
    This function loads a single data file containing the results of the simulations for one value of inertia
    """
    ext = os.path.splitext(filename)[1]

    fid = tables.open_file(filename, 'r')
    time = fid.root.time.read()
    X = [fid.root[var_name].read() for var_name in var_names]
    pars = fid.root.parameters.read()
    inertia = pars['inertia'][0]
    fid.close()

    dt = np.diff(time[:2])[0]
    orig_n_trials, orig_n_samples = X[0].shape
    n_samples = int(trial_dur / dt)
    n_trials = int(orig_n_trials * orig_n_samples / n_samples)
    time = time[:n_samples]
    X = [np.reshape(x, [n_trials, n_samples], order='C') for x in X]

    if verbose:
        print('There are {} trials, each of which contains {} samples.'.\
              format(n_trials, n_samples))

    return tf.constant(time, dtype=tf.float32), \
           tf.constant(X, dtype=tf.float32), \
           tf.constant([inertia for _ in range(n_trials)], shape=(n_trials,1), dtype=tf.float32)


def load_data(folder, inertia, var_names=('omega_coi',)):
    if folder[-1] != '/':
        folder += '/'
    X = {}
    Y = {}
    for key,H in inertia.items():
        for h in H:
            filename = folder + 'H_{:.3f}_{}_set.h5'.format(h, key)
            time, x, inertia = load_one_block(filename, var_names, 60)
            try:
                X[key] = tf.concat([X[key], x], axis=1)
                Y[key] = tf.concat([Y[key], inertia], axis=0)
            except:
                X[key] = x
                Y[key] = inertia
    return time, X, Y

            
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
