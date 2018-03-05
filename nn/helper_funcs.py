"""
Helping functions, mostly for fortran interface. Put here to tidy up main Mlpp
class a bit.
"""
import nnp.nn.fortran.nn_f95 as f95_api
import nnp.util

def get_node_distribution(weights,input_type='a',set_type="train"):
    """
    For the given set (already in fortran), return either the distribution of
    node values before or after activation functions are applied
    
    Parameters
    ----------
    weights : np.ndarray
        A 1d array of weights for the neural network. Must be the correct size.
    """
    if input_type not in ['a','z']:
        raise GeneralHelperError("input type {} not in 'a','z'".format(input_type))

    total_num_atoms = nnp.util.misc.total_atoms_in_set(set_type)

    node_distribution = {"layer1":0,"layer2":1}
    for _layer in node_distribution: 
        num_nodes = self.hidden_layer_sizes[node_distribution[_layer]] 

        node_distribution[_layer] = np.zeros((num_nodes,total_num_atoms),order='F',\
                dtype=np.float64)

    getattr(f95_api,"f90wrap_get_node_distribution")(flat_weights=weights,\
            set_type={"test":2,"train":1}[set_type],input_type={'a':1,'z':2}[input_type],\
            layer_one=node_distribution["layer1"],layer_two=node_distribution["layer2"])

    return np.asarray(node_distribution["layer1"],order='C'),\
        np.asarray(node_distribution["layer2"],order='C')


class GeneralHelperError(Exception):
    pass
