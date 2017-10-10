"""
atom centered symmetry functions

References
----------

Ref [1] : Jorg Behler, Perspective: Machine learning potentials
for atomistic simualations, Journal of Chemical Physics, (2016)
"""

import parsers
from nn_potential.util.taper import type_1 as taper1
from nn_potential.util.pbc import get_ultracell
from nn_potential.fortran import interface
from scipy import spatial
import numpy as np

def acsf(gips,functional,form,rcut,parameters,usefortran=True):
    """
    Input
    -----
        gips - list of parsers.GeneralInputParser() instances 
    """

    if form.lower() not in ['isotropic','anisotropic']:
        raise Behler_usererror('{} is an unsupported form'.format(form))


    if usefortran:
        if functional == 'behler':
            raise NotImplementedError
        elif functional == 'normal':
            if form == 'isotropic':
                feature = interface.gamma_featuref90(gips=gips,rcut=rcut,mean=np.asarray([parameters["mean"]]),\
                        precision=np.asarray([parameters["precision"]]),normal_zk=parameters["zk_power"])
            elif form == 'anisotropic':
                feature = interface.anisotropic_featuref90(gips=gips,rcut=rcut,\
                        means=np.asarray([parameters["mean"]]),precisions=np.asarray([parameters["precision"]]),\
                        normal_zk=parameters["zk_power"])
    else:

        if form.lower() == 'isotropic':
            eta = parameters["eta"]
            rs = parameters["rs"]
            fs = parameters["fs"]
        else:
            xi = parameters["xi"]
            lambd = parameters["lambda"]
            eta = paramteters["eta"]
            fs = parameters["fs"]

        feature = []

        for _gip in gips:
            for _s in _gip:
                ultracart,ultra_species,ultra_idx = get_ultracell(fpos=_s["positions"],\
                        cell=_s["cell"],r_cut=rcut,species=_s["species"],verbose=False)
        
                # need atomic number
                ultraz = np.asarray([parsers.atomic_data.atomic_number.atomic_number[_s_] for _s_ \
                        in ultra_species])
                localz = np.asarray([parsers.atomic_data.atomic_number.atomic_number[_s_] for _s_ \
                        in _s["species"]])

                skd = spatial.cKDTree(ultracart)
                idx = skd.query_ball_point(np.dot(_s["positions"],_s["cell"]),rcut) 
                
                structure_feature = np.zeros(len(_s["species"]),dtype=np.float64)

                for atom,_r in enumerate(np.dot(_s["positions"],_s["cell"])):
                    # vector of displacements
                    dr_vec = np.asarray([ultracart[_idx]-_r for _idx in idx[atom]])

                    # magnitude of displacements
                    dr = np.linalg.norm(dr_vec,axis=0)

                    # atomic number
                    zatom = localz[atom]

                    # do not want interactions with oneself
                    tmp_idx = np.nonzero(np.invert(np.isclose(dr,0.0)))[0]
                    
                    nonzero_idx = np.asarray(idx[atom],dtype=int)[np.nonzero(np.invert(np.isclose(dr,0.0)))[0]]
                    dr = dr[tmp_idx]
                    dr_vec = dr_vec[tmp_idx]
                    atomic_num = ultraz[idx[atom][tmp_idx]]

                    if form.lower() == 'isotropic':
                        if category == 'behler':
                            structure_feature[atom] = np.sum(taper1(x=dr,rcut=rcut,fs=fs)*np.exp(-eta*(dr-rs)**2))
                        elif category == 'normal':
                            structure_feature[atom] = np.sum(taper1(x=dr,rcut=rcut,fs=fs)*\
                                    normal(x=dr,mu=mu,Lambd=Lambd)*atomic_num)*zatom
                    elif form.lower() == 'anisotropic':
                        for jj in range(len(idx[atom])):
                            for kk in range(len(idx[atom])):
                                if jj==kk:
                                    continue

                                drij_vec = dr_vec[jj]
                                drik_vec = dr_vec[kk]
                                drjk_vec = ultracart[idx[atom][kk]] - ultracart[idx[atom][jj]]

                                drij_mag = np.linalg.norm(drij_vec)
                                drik_mag = np.linalg.norm(drik_vec)
                                drjk_mag = np.linalg.norm(drjk_vec)

                                if any(np.asarray([drij_mag,drik_mag,drjk_mag])>rcut) or \
                                    any(np.isclose(np.asarray([drij_mag,drik_mag,drjk_mag]),0.0)):
                                    continue
                           
                                costheta = np.dot(drij_vec,drik_vec)/(drij_mag*drik_mag)

                                structure_feature[atom] += (1.0+lambd*costheta)**xi * np.exp(-eta*(rij_mag**2 +\
                                        rik_mag**2 + rjk_mag**2)) * taper1(x=rij_mag,rcut=rcut,fs=fs) *\
                                        taper1(x=rik_mag,rcut=rcut,fs=fs) * taper1(x=rjk_mag,rcut=rcut,fs=fs)

                        structure_feature[atom] *= 2**(1.0-xi)

                    feature += list(structure_feature)

    return np.asarray(feature)

def acsf_info(gips,rcut,form):
    if form == 'isotropic':
        info = interface.atomatomdistances(gips=gips,rcut=rcut)
    elif form == 'anisotorpic':
        info = interface.angular_infof90(gips=gips,rcut=rcut)

    return info

class Behler_usererror(Exception):
    pass

