import numpy as np

def type_1(x,rcut,fs):
    xp = (x-rcut)/fs

    y = xp**4 / (1+xp**4)

    idx_zero = np.nonzero(x>rcut)

    y[idx_zero] = 0.0

    return y
