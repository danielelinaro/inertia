
import os
import re
import tables
import numpy as np
import tensorflow as tf


__all__ = ['load_one_block', 'load_data', 'slide_window']

default_H = {
    'IEEE14': {
        1: 10.296 / 2,
        2: 13.08 / 2,
        3: 13.08 / 2,
        6: 10.12 / 2,
        8: 10.12 / 2
    }
}

def load_one_block(filename, var_names, trial_dur = 60, max_num_rows = np.inf, verbose = False):
    """
    This function loads a single data file containing the results of the simulations for one value of inertia
    """
    ext = os.path.splitext(filename)[1]

    fid = tables.open_file(filename, 'r')
    time = fid.root.time.read()
    X = [fid.root[var_name].read(stop=np.min([fid.root[var_name].shape[0], max_num_rows])) for var_name in var_names]
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


def load_data(folders, generator_IDs, inertia_values, var_names, max_block_size = np.inf):
    X = {}
    Y = {}
    if isinstance(inertia_values, dict):
        inertias = [inertia_values for _ in folders]
    else:
        inertias = inertia_values
    n_outputs = len(generator_IDs)
    for i, (folder, inertia) in enumerate(zip(folders, inertias)):
        if folder[-1] != '/':
            folder += '/'
        for key,H in inertia.items():
            for h in H:
                filename = folder + 'H_{:.3f}_{}_set.h5'.format(h, key)
                time, x, y_tmp = load_one_block(filename, var_names, 60, max_block_size)
                y_tmp = np.squeeze(y_tmp.numpy())
                y = np.zeros([y_tmp.size, n_outputs])
                y[:,i] = y_tmp
                for j in range(n_outputs):
                    if j != i:
                        y[:,j] = default_H['IEEE14'][generator_IDs[j]]
                y = tf.constant(y)
                try:
                    X[key] = tf.concat([X[key], x], axis=1)
                    Y[key] = tf.concat([Y[key], y], axis=0)
                except:
                    X[key] = x
                    Y[key] = y
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
