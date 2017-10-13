import nn_potential.nn.fortran.nn_f95 as f95
import numpy as np

def train(features,energies,forces,slice_indices,num_nodes,nlf_type):
    if nlf_type.lower() == 'sigmoid':
        nlf = 1
    elif nlf_type.lower() == 'tanh':
        nlf = 2
    else:
        raise NotImplementedError

    settype = {"train":1,"test":2}

    # feature dimension
    D = features["test"].shape[1]

    f95.f90wrap_initialise_net(np.asarray(num_nodes,dtype=np.int32),nlf,D)
    
    for _set in features:
        # account for py = 0 offset, f90 = 1 offset
        slice_indices[_set] += 1

        f95.f90wrap_initialise_set(settype[_set],energies[_set].shape[0],\
                features[_set].shape[0],np.asarray(slice_indices[_set].T,dtype=np.int32),\
                np.asarray(features[_set].T,order='F',dtype=np.float64),\
                np.asarray(forces[_set].T,order='F',dtype=np.float64),\
                np.asarray(energies[_set],dtype=np.float64))

    
    f95.f90wrap_info_net()
    f95.f90wrap_info_set(1)
    f95.f90wrap_info_set(2)
