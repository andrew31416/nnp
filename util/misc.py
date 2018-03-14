import nnp.nn.fortran.nn_f95 as f95_api
import os
import numpy as np

def num_threads():
    """
    Return the number of threads running in parallel sections

    export omp_num_threads=X to change

    Examples
    --------
    >>> import nnp
    >>> print('running on {} threads'.format(nnp.util.misc.num_threads()))
    """

    return getattr(f95_api,"f90wrap_num_threads")()


def total_atoms_in_set(set_type):
    """
    Return the total number of atoms in a given set

    Parameters
    ----------
    set_type : String, alowed valued = "train","test"

    Examples
    --------
    >>> import nnp
    >>> import parsers
    >>>
    >>> # parse data
    >>> gip = parsers.GeneralInputParser()
    >>> gip.parse_all('./train_sets')
    >>>
    >>> # generate features
    >>> _features = nnp.features.types.features(gip)
    >>> _features.generate_gmm_features()
    >>> _mlpp = nnp.mlpp.MultiLayerPerceptronPotential()
    >>> _mlpp.set_features(_features)
    >>>
    >>> # write data to fortran
    >>> _mlpp._prepare_data_structures(gip,"train")
    >>> print('number of atoms in train set = {}'.format(nnp.util.total_atoms_in_set("train")))
    """
    if set_type not in ["train","test"]:
        raise GeneralUtilError("Set {} no in 'train','test'".format(set_type))
    
    return getattr(f95_api,"f90wrap_get_total_natm")({"test":2,"train":1}[set_type])

def get_num_configs(set_type):
    """
    Return number of configurations in given set type

    Parameters
    ----------
    set_type : String, allowed values = "train","test"
        Which set type to consider
    """
    if set_type not in ["train","test"]:
        raise GeneralUtilError("Set {} no in 'train','test'".format(set_type))
    
    return getattr(f95_api,"f90wrap_get_nconf")({"train":1,"test":2}[set_type])

def split_sets(gip,train_fraction,seed=None):
    """
    split a single parsers.GeneralInputParser() into
    test and train parsers.GeneralInputParser()'s
   
    Parameters
    ----------
    gip : parsers.GeneralInputParser()
        The class object containing a number of configurations

    train_fraction : float
        The fraction of structures to include in the training set
    
    seed : int, default value = None
        If not None, initialise numpy random numbers with this seed

    Examples
    --------
    >>> import nnp
    >>> import parsers
    >>> gip = parsers.GeneralInputParser()
    >>> gip.parse_all('./test_train_sets/')
    >>>
    >>> # split all configurations into 2 sets
    >>> sets = nnp.util.misc.split_sets(gip,0.1)
    >>>
    >>> # 0.1% of all configurations are now included in training set
    >>> features = nnp.features.types.features(train_data=sets["train"])
    """
    import parsers
    num_structures = len(gip.supercells)

    train = parsers.GeneralInputParser()
    test = parsers.GeneralInputParser()
    train.supercells = []
    test.supercells = []

    # pick randomly
    train_idx = np.random.choice(np.arange(num_structures),\
            size=round(train_fraction*num_structures),replace=False)
    
    test_idx = set(np.arange(num_structures)).difference(set(train_idx))

    for ii,_s in enumerate(gip.supercells):
        if ii in train_idx:
            train.supercells.append(_s)
        elif ii in test_idx:
            test.supercells.append(_s)

    assert len(test.supercells)+len(train.supercells)==num_structures,'error in test/train splitting'           

    return {"train":train,"test":test}


class GeneralUtilError(Exception):
    pass    
        
