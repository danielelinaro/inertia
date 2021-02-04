
import os
import re
import tables
import numpy as np


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

def load_one_block(filename, var_names, trial_dur = 60, max_num_rows = np.inf, dtype = np.float32):
    """
    This function loads a single data file containing the results of the simulations for one value of inertia
    """
    ext = os.path.splitext(filename)[1]

    fid = tables.open_file(filename, 'r')
    # do not convert time to dtype here because that gives problems when computing n_samples below
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
    X = np.array([np.reshape(x, [n_trials, n_samples], order='C') for x in X], dtype=dtype)

    return time.astype(dtype), X, inertia


def load_data(folders, generator_IDs, inertia_values, var_names, max_block_size = np.inf, dtype = np.float32, use_tf = True):
    # Note: dtype should be a NumPy type, even if use_tf = True
    if use_tf:
        import tensorflow as tf

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
                time, x, _ = load_one_block(filename, var_names, 60, max_block_size, dtype)
                y = np.zeros((x.shape[-2], n_outputs), dtype=dtype)
                y[:,i] = h
                for j in range(n_outputs):
                    if j != i:
                        y[:,j] = default_H['IEEE14'][generator_IDs[j]]
                try:
                    X[key] = np.concatenate((X[key], x), axis=1)
                    Y[key] = np.concatenate((Y[key], y), axis=0)
                except:
                    X[key] = x
                    Y[key] = y
    if use_tf:
        time = tf.constant(time, dtype=tf.dtypes.as_dtype(dtype))
        for key in X:
            X[key] = tf.constant(X[key], dtype=tf.dtypes.as_dtype(dtype))
            Y[key] = tf.constant(Y[key], dtype=tf.dtypes.as_dtype(dtype))
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
