"""
module to return list of atomic numbers
"""
import numpy as np

def atomic_number(gips):
    zvals = []
    for _gip in gips:
        for _s in _gip:
            zvals += list(_s["atomic_number"])
    return np.asarray(zvals)
