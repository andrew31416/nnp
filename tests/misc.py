import nnp.nn.mlpp
import numpy as np
import parsers
import copy
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
import warnings

def generate_random_structure():
    from ase.calculators.emt import EMT

    cell = np.random.normal(loc=0,scale=1.0,size=(3,3))
    for ii in range(3):
        cell[ii,ii] = np.random.normal(loc=10.0,scale=1.0,size=1)

    N_atoms = 30
    fractional_coordinates = np.random.random(size=(N_atoms,3))
    species = ["C" for ii in range(N_atoms)]
    
    supercell = parsers.supercell()
    supercell["cell"] = cell
    supercell["positions"] = fractional_coordinates
    supercell["species"] = species

    # set calc to ase atoms to gen. ref. forces
    ase_atoms = supercell.get_ase_atoms()
    ase_atoms.set_calculator(EMT())

    supercell["energy"] = ase_atoms.get_potential_energy()
    supercell["forces"] = ase_atoms.get_forces()

    return supercell

def random_gip(num_configs=2):
    gip = parsers.GeneralInputParser()
    gip.supercells = []
    for ii in range(num_configs):
        gip.supercells.append(generate_random_structure())
    return gip

def check_jacobian():
    def _check_weight(weight_idx,original_weights,mlpp,anl_jac,num_iters=10):
        # run through all appropriate finite differences for given weight

        if np.isclose(original_weights[weight_idx],0.0,rtol=1e-50,atol=1e-50):
            use_abs = True
        else:
            use_abs = False

        abs_lim = [1e-8,1e-4]
        frc_lim = [1e-6,1e-1]
        
        if use_abs:
            finite_differences = np.logspace(np.log10(abs_lim[0]),np.log10(abs_lim[1]),num_iters)
        else:
            finite_differences = np.logspace(np.log10(frc_lim[0]),np.log10(frc_lim[1]),num_iters)

        numerical_jac = np.zeros(finite_differences.shape,dtype=np.float64)
        for ii,_fd in enumerate(finite_differences):
            if use_abs:
                epsilon = _fd
            else:
                epsilon = original_weights[weight_idx]*_fd 

            epsilon_array = np.zeros(original_weights.shape,dtype=np.float64)
            
            # only take finite difference along one coordinate at a time
            epsilon_array[weight_idx] = epsilon

            jac_array = approx_fprime(original_weights,mlpp._loss,\
                    epsilon_array,"train")

            # check all other elements are zero
            for jj in range(original_weights.shape[0]):
                if jj!=weight_idx:
                    if not np.isnan(jac_array[jj]):
                        raise SeriousImplementationError("Jacobian element {} should be nan".\
                                format(jj))


            numerical_jac[ii] = jac_array[weight_idx]
            if weight_idx in [original_weights.shape[0]-1,original_weights.shape[0]-2]:
                pass
                #print('dx = {} num = {} anl = {}'.format(_fd,numerical_jac[ii],anl_jac[weight_idx]))

        # lets see if any finite differences approximate the jac accurately
        success = np.isclose(numerical_jac,anl_jac[weight_idx],rtol=1e-4,atol=1e-5).any()
        
        if not success:
            for ii,_fd in enumerate(finite_differences):
                print('dx = {} num = {} anl = {}'.format(_fd,numerical_jac[ii],anl_jac[weight_idx]))
        
        return success

    training_data = random_gip(num_configs=2)

    mlpp = nnp.nn.mlpp.MultiLayerPerceptronPotential(hidden_layer_sizes=[5,5],parallel=False)
    mlpp.hyper_params["energy"] = 1.0
    mlpp.hyper_params["forces"] = 1.0
    mlpp.hyper_params["regularization"] = 1.0
    
    # use default Behler features (4x G-2, 4x G-4)
    mlpp.set_features(nnp.features.defaults.Behler(training_data))

    mlpp._prepare_data_structures(X=training_data,set_type="train")
    mlpp._init_random_weights()
    mlpp._njev = 0
   
    # just to be safe 
    initial_weights = copy.deepcopy(mlpp.weights)

    # compute analytical jacobian
    anl_jac = mlpp._loss_jacobian(initial_weights,"train")

    weight_test = [_check_weight(_weight,initial_weights,mlpp,anl_jac,10) for \
            _weight in range(initial_weights.shape[0])]

    return np.all(weight_test)

def better_than_random():
    """
    check can overfit better than random energy distribution
    """

    training_data = random_gip(num_configs=6)
    ref_energies = np.asarray([_s["energy"] for _s in training_data])
    ref_forces = np.asarray([_s["forces"] for _s in training_data]).flatten()

    mlpp = nnp.nn.mlpp.MultiLayerPerceptronPotential(hidden_layer_sizes=[5,10],parallel=False,\
            precision_update_interval=0,max_precision_update_number=0)
    mlpp.hyper_params["energy"] = 1.0
    mlpp.hyper_params["forces"] = 0.0
    mlpp.hyper_params["regularization"] = 0.0

    # use default Behler features (4x G-2, 4x G-4)
    mlpp.set_features(nnp.features.defaults.Behler(training_data))
    
    # fit
    fit_loss,fit_gip = mlpp.fit(training_data)
    fit_energies = np.asarray([_s["energy"] for _s in fit_gip])
    fit_forces = np.asarray([_s["forces"] for _s in fit_gip]).flatten()
 
    # check RMSE < 0.01 [max(y)-min(y)]
    return mean_squared_error(ref_energies,fit_energies) < 1e-4*(np.max(ref_energies)-\
            np.min(ref_energies))**2

def check_openmp():
    """
    Run loss and jacobian computations in parallel 
    """
    # check OMP_NUM_THREADS>1
    if nnp.util.misc.num_threads() < 2:
        raise UserError("OMP_NUM_THREADS = {}. export to be > 1".\
                format(nnp.util.misc.num_threads()))

    gip = random_gip(num_configs=6)

    features = nnp.features.defaults.Behler(gip)
   
    mlpp = nnp.nn.mlpp.MultiLayerPerceptronPotential(parallel=False)
    for _attr in ["energy","forces","regularization"]:
        mlpp.set_hyperparams(_attr,1.0)
    mlpp.set_features(features)
    mlpp._njev = 0
    mlpp._prepare_data_structures(gip,"train")
    mlpp._init_random_weights() 

    loss = {True:None,False:None}
    jacb = {True:None,False:None}

    for _parallel in [True,False]:
        mlpp.set_parallel(_parallel)
        loss[_parallel] = mlpp._loss(mlpp.weights,"train")
        jacb[_parallel] = mlpp._loss_jacobian(mlpp.weights,"train")

    if not np.isclose(loss[False],loss[True]):
        raise SeriousImplementationError("serial and parallel loss are not equal : {} != {}".\
                format(loss[False],loss[True]))
         
    if not np.allclose(jacb[False],jacb[True]):
        raise SeriousImplementationError("serial and parallel jac. are not equal")

    return True

def check_features(set_type):
    """
    Check features for invalid floating point numbers
    
    Parameters
    ----------
    set_type : String, allowed values = 'train','test'
    """
    import nnp.nn.fortran.nn_f95 as f95_api
    
    if set_type not in ['train','test']:
        raise UserError("set type {} is not 'train' or 'test'".format(set_type))

    getattr(f95_api,"f90wrap_check_features")(set_type={"train":1,"test":2}[set_type])

def check_feature_derivatives(set_type):
    """
    Check feature derivatives for invalid floating point numbers
    
    Parameters
    ----------
    set_type : String, allowed values = 'train','test'
    """
    import nnp.nn.fortran.nn_f95 as f95_api

    if set_type not in ['train','test']:
        raise UserError("set type {} is not 'train' or 'test'".format(set_type))

    getattr(f95_api,"f90wrap_check_feature_derivatives")(set_type={"train":1,"test":2}[set_type])
         
def run_all():
    """
    Run all unit tests contained in Python API
    """
    # disable warnings
    warnings.simplefilter("ignore",RuntimeWarning)

    np.random.seed(0) 
    unit_tests = [check_jacobian(),better_than_random(),check_openmp()]

    return all(unit_tests)

class UserError(Exception):
    pass

class SeriousImplementationError(Exception):
    pass

if __name__ == "__main__":
    if run_all():
        print('unit tests successful')
    else:
        print('unit tests unsuccessful')
