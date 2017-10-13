import nn_f95 as f90

def train(features,energies,forces,slice_indices,num_nodes,nlf_type):
    if nlf_type.lower() == 'sigmoid':
        nlf = 1
    elif nlf_type.lower() == 'tanh':
        nlf = 2
    else:
        raise NotImplementedError

    # feature dimension
    D = features.shape[1]

    f90.initialise_net(np.asarray(num_nodes,dtype=np.int32),nlf,D)
