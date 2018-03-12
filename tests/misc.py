import nnp.nn.mlpp
import numpy as np
import parsers
import copy
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
        frc_lim = [1e-8,1e-1]
        
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
        success = np.isclose(numerical_jac,anl_jac[weight_idx],rtol=1e-5,atol=1e-5).any()
        
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

    mlpp = nnp.nn.mlpp.MultiLayerPerceptronPotential(hidden_layer_sizes=[5,5],parallel=False)
    mlpp.hyper_params["energy"] = 1.0
    mlpp.hyper_params["forces"] = 0.0
    mlpp.hyper_params["regularization"] = 0.0

    # use default Behler features (4x G-2, 4x G-4)
    mlpp.set_features(nnp.features.defaults.Behler(training_data))
    
    mlpp._prepare_data_structures(X=training_data,set_type="train")
    mlpp._init_random_weights()

    # get initial loss
    init_loss,init_gip = mlpp.predict(training_data)
    init_energies = np.asarray([_s["energy"] for _s in init_gip])

    # fit
    fit_loss,fit_gip = mlpp.fit(training_data)
    fit_energies = np.asarray([_s["energy"] for _s in fit_gip])
    
    plt.plot(ref_energies,init_energies,alpha=0.5,\
            label='initial',linestyle='none',marker='o')
    plt.plot(ref_energies,fit_energies,alpha=0.5,\
            label='after fit',linestyle='none',marker='o')
    plt.plot([np.min(ref_energies),np.max(ref_energies)],[np.min(ref_energies),\
            np.max(ref_energies)]) 
    plt.legend()
    plt.show()


    natm = fit_gip.supercells[0]["positions"].shape[0]
    print('init loss = {} final loss = {}'.format(init_loss,fit_loss))
    print('mse init = {} mse final = {}'.format(mean_squared_error(ref_energies/natm,\
            init_energies/natm),\
            mean_squared_error(ref_energies/natm,fit_energies/natm)))

    
def run_all():
    """
    Run all unit tests contained in Python API
    """
    np.random.seed(0) 
    unit_tests = [check_jacobian()]

    return all(unit_tests)

class SeriousImplementationError(Exception):
    pass

if __name__ == "__main__":
    if run_all():
        print('unit tests successful')
    else:
        print('unit tests unsuccessful')
