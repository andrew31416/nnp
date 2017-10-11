"""
observables associated with semi-empirical dispersion which one
may want to regress
"""
import numpy as np

def vdw_radius(gips):
    obs = np.asarray([],dtype=np.float64)
    for _gip in gips:
        for _s in _gip:
            obs = np.hstack((obs, _s["vdw_radius"]["environment"] - _s["vdw_radius"]["vacuum"] ))
    return np.reshape(obs,(-1,1))
