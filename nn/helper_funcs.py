"""
Helping functions, mostly for fortran interface. Put here to tidy up main Mlpp
class a bit.
"""
import numpy as np

def get_node_distribution(weights,input_type='a',set_type="train"):
    """
    For the given set (already in fortran), return either the distribution of
    node values before or after activation functions are applied
    
    Parameters
    ----------
    weights : np.ndarray
        A 1d array of weights for the neural network. Must be the correct size.
    """
    import nnp.nn.fortran.nn_f95 as f95_api
    import nnp.util
    
    if input_type not in ['a','z']:
        raise GeneralHelperError("input type {} not in 'a','z'".format(input_type))

    total_num_atoms = nnp.util.misc.total_atoms_in_set(set_type)

    layer_sizes = get_num_nodes()

    node_distribution = {"layer1":0,"layer2":1}
    for _layer in node_distribution: 
        num_nodes = layer_sizes[node_distribution[_layer]] 

        node_distribution[_layer] = np.zeros((num_nodes,total_num_atoms),order='F',\
                dtype=np.float64)

    getattr(f95_api,"f90wrap_get_node_distribution")(flat_weights=weights,\
            set_type={"test":2,"train":1}[set_type],input_type={'a':1,'z':2}[input_type],\
            layer_one=node_distribution["layer1"],layer_two=node_distribution["layer2"])

    return np.asarray(node_distribution["layer1"],order='C'),\
        np.asarray(node_distribution["layer2"],order='C')

def get_num_configs(set_type):
    """
    Number of configurations in the given set type in fortran memory

    Parameters
    ----------
    set_type : String, allowed values = 'test','train'
        Test or train set
    """
    import nnp.nn.fortran.nn_f95 as f95_api
    if set_type not in ['train','test']:
        raise GeneralHelperError("set type {} not supported. User error.".format(set_type))
    
    return getattr(f95_api,"f90wrap_get_nconf")(set_type={"train":1,"test":2}[set_type])

def get_atoms_per_conf(set_type):
    """
    Return 1d array of number of atoms in each configuration for given set type

    Parameters
    ----------
    set_type : String, allowed values = 'test','train'
        Test or train set
    """    
    import nnp.nn.fortran.nn_f95 as f95_api
    if set_type not in ['train','test']:
        raise GeneralHelperError("set type {} not supported. User error.".format(set_type))
    
    Nconf = get_num_configs(set_type)

    Natms = np.zeros(Nconf,dtype=np.int16)

    for ii in range(1,Nconf+1):
        Natms[ii-1] = getattr(f95_api,"f90wrap_get_natm")(\
                set_type={"train":1,"test":2}[set_type],conf=ii)
    return Natms

def get_num_nodes():
    """
    Return the number of nodes in each hidden layer from fortran
    """
    import nnp.nn.fortran.nn_f95 as f95_api
    num_nodes = np.zeros(2,dtype=np.int32,order='F')
    
    getattr(f95_api,"f90wrap_get_num_nodes")(num_nodes)
    
    return np.asarray(num_nodes,order='C')

def get_reference_energies(set_type):
    """
    return list of reference energies from configurations in given set
    
    Parameters
    ----------
    set_type : String, allowed values = 'test','train'
        Test or train set
    """
    import nnp.nn.fortran.nn_f95 as f95_api
    energies = np.zeros(get_num_configs(set_type=set_type),dtype=np.float64,order='F')

    getattr(f95_api,"f90wrap_get_ref_energies")(set_type={"train":1,"test":2}[set_type],\
            ref_energies=energies)

    return np.asarray(energies,order='C')

def reduced_second_layer_distribution(weights,set_type="train"):
    """
    Return distribution of sum_k^atoms z_{jk}^{(2)}
    """
    import nnp.nn.fortran.nn_f95 as f95_api
    import nnp.util
    
    total_num_atoms = nnp.util.misc.total_atoms_in_set(set_type)


    layer_sizes = get_num_nodes()

    node_distribution = {"layer1":0,"layer2":1}
    for _layer in node_distribution:
        num_nodes = layer_sizes[node_distribution[_layer]] 

        node_distribution[_layer] = np.zeros((num_nodes,total_num_atoms),order='F',\
                dtype=np.float64)

    getattr(f95_api,"f90wrap_get_node_distribution")(flat_weights=weights,\
            set_type={"test":2,"train":1}[set_type],input_type={"a":1,"z":2}['z'],\
            layer_one=node_distribution["layer1"],layer_two=node_distribution["layer2"])

    # atoms in each configuration
    atoms_per_conf = get_atoms_per_conf(set_type=set_type)

    reduced_distribution = np.zeros((atoms_per_conf.shape[0],layer_sizes[1]),dtype=np.float64)

    cntr = 0
    for _conf in range(reduced_distribution.shape[0]):
        natm = atoms_per_conf[_conf]

        reduced_distribution[_conf,:] = np.sum(node_distribution["layer2"][:,cntr:cntr+natm],\
                axis=1)[:]
        #reduced_distribution[_conf,:] = np.average(node_distribution["layer2"][:,cntr:cntr+natm],\
        #        axis=1)[:]
        
        # book keeping
        cntr += natm
    
    return np.asarray(reduced_distribution).T

class GeneralHelperError(Exception):
    pass
