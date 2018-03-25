import nnp.nn.fortran.nn_f95 as f95_api
import os
import numpy as np

_set_map = {"train":1,"holdout":2,"test":3}

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
    if set_type not in ["train","holdout","test"]:
        raise GeneralUtilError("Set {} no in 'train','test'".format(set_type))
    
    return getattr(f95_api,"f90wrap_get_total_natm")(_set_map[set_type])

def get_num_configs(set_type):
    """
    Return number of configurations in given set type

    Parameters
    ----------
    set_type : String, allowed values = "train","test"
        Which set type to consider
    """
    if set_type not in ["train","holdout","test"]:
        raise GeneralUtilError("Set {} no in 'train','test'".format(set_type))
    
    return getattr(f95_api,"f90wrap_get_nconf")(_set_map[set_type])

def split_sets(gip,train_fraction,holdout_fraction=0.2,seed=None):
    """
    split a single parsers.GeneralInputParser() into
    test and train parsers.GeneralInputParser()'s
   
    Parameters
    ----------
    gip : parsers.GeneralInputParser()
        The class object containing a number of configurations

    train_fraction : float
        The fraction of structures to include in the training set
   
    holdout_fraction : float
        The fraction of structures from train+holdout to put into holdout set
    
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
    
    if train_fraction < 0.0 or train_fraction > 1.0:
        raise GeneralUtilError("invalid train_fraction : {}".format(train_fraction))
    if holdout_fraction < 0.0 or holdout_fraction > 1.0:
        raise GeneralUtilError("invalid holdout_fraction : {}".format(holdout_fraction))

    num_structures = len(gip.supercells)

    train = parsers.GeneralInputParser()
    test = parsers.GeneralInputParser()
    if not np.isclose(holdout_fraction,0.0,atol=1e-50,rtol=1e-50):
        holdout = parsers.GeneralInputParser()
        holdout.supercells = []
    else:
        holdout = None
    train.supercells = []
    test.supercells = []

    # pick randomly
    trainholdout_idx = np.random.choice(np.arange(num_structures),\
            size=round(train_fraction*num_structures),replace=False)
    
    test_idx = set(np.arange(num_structures)).difference(set(trainholdout_idx))

    if holdout is not None:
        holdout_idx = np.random.choice(trainholdout_idx,size=round(holdout_fraction*\
                len(trainholdout_idx)),replace=False)

        train_idx = set(trainholdout_idx).difference(set(holdout_idx))
    else:
        train_idx = trainholdout_idx
        holdout_idx = []

    for ii,_s in enumerate(gip.supercells):
        if ii in train_idx:
            train.supercells.append(_s)
        elif ii in test_idx:
            test.supercells.append(_s)
        elif ii in holdout_idx:
            holdout.supercells.append(_s)

    if holdout is not None:
        nconf = len(holdout.supercells)
    else:
        nconf = 0
    nconf += len(train.supercells)+len(test.supercells)

    if nconf!=num_structures:
        raise GeneralUtilError('error in test/train splitting')         

    return {"train":train,"test":test,"holdout":holdout}


class GeneralUtilError(Exception):
    pass    
        
