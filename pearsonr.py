
import os
import time
import numpy as np
from scipy.stats import pearsonr
import ctypes

if __name__ == '__main__':
    lib = ctypes.CDLL(os.path.join('.', 'libcorr.so'))
    lib.pearsonr.argtypes = [ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double)]

    N = 36
    x = np.random.uniform(size=N)
    y = np.random.uniform(size=N)

    ### python
    R_py,p_py = pearsonr(x, y)

    ### C
    pointer = ctypes.POINTER(ctypes.c_double)
    X = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    Y = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    R = pointer(ctypes.c_double(0.0))
    p = pointer(ctypes.c_double(0.0))
    lib.pearsonr(X, Y, N, R, p)
    R_C,p_C = R[0], p[0]

    N_iter = 10000
    start = time.time_ns()
    [pearsonr(x, y) for _ in range(N_iter)]
    stop = time.time_ns()
    dur_py = ((stop - start) / N_iter) * 1e-6
    start = time.time_ns()
    [lib.pearsonr(X, Y, N, R, p) for _ in range(N_iter)]
    stop = time.time_ns()
    dur_C = ((stop - start) / N_iter) * 1e-6
    
    print(f'Py: R = {R_py:.4f}, p = {p_py:.3e}. Execution time: {dur_py:.5f} ms')
    print(f' C: R = {R_C:.4f}, p = {p_C:.3e}. Execution time: {dur_C:.5f} ms')
