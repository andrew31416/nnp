import numpy as np

def energies(gip):
    """
    return np.ndarray of energies for all structures in gip
    """
    return np.asarray([_s["energy"] for _s in gip.supercells],dtype=np.float64)

def forces(gip):
    forces = []
    for _s in gip:
        forces += list(_s["forces"])
    return np.asarray(forces)
