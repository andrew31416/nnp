from nnp.util.log import NnpOptimizeResult
import nnp.nn.helper_funcs
import numpy as np
import nnp.features
from scipy import optimize
import time

class MultiLayerPerceptronPotential():
    """Fully connected feed forward multi layer perceptron for empirical 
    potentials

    This network follows the architecture of Behler using 3 weight layers [1]. 
    The net output is the energy per atom and forces are computed from 
    derivatives of the net with respect to input features and these feature 
    derivatives with respect to atom positions. Periodic boundaries are treated 
    in full and not in the nearest image convention employed for molecular 
    dynamics (when large unit cells are assumed). 


    Parameters
    ----------
    hidden_layer_sizes : {int, array-like}, shape = [2,]
        An array [n1,n2] giving the number of nodes in the 1st and 2nd hidden 
        layer.

    activation : string
        The type of non-linear activation function to apply to all nodes

    solver : string
        The method to minimise loss 

    hyper_params : dict , keys = 'loss_energy','loss_forces',
                  'loss_regularization'
        A dictionary of network hyper parameters including coefficients to the
        energy, force and regularization parts of the loss function

    Attributes
    ----------
    weights : {float,array-like}, shape = [(D+1)*n1 + (n1+1)*n2 + n2+1,]
        A flattened array of the current network weights. 

    activation : String, allowed values = 'sigmoid','tanh'
        The type of non-linear activation function, same for all nodes

    loss_norm : String, allowed values = 'l1','l2'
        The norm type to use for the loss function

    solver : String
        The type of scipy.optimize.minimize solver to use for minimisation of
        the loss function. First order derivatives (Jacobian methods) only
        can be used.
    == 0 
    hyper_params : dict, keys = 'loss_energy','loss_forces',
                  'loss_regularization'
        Key,value pairs of regression hyper parameters. loss_* are the 
        constants multiplying each term in the loss function.

    hidden_layer_sizes : np.ndarray, dtype=np.int32, shape=(2,)
        Two element 1d array of number of nodes in each hidden layer of the
        multilayer perceptron
    
    features : nnp.features.types.features
        Features class containing a list of features to employ in the net

    weights : np.ndarray, dtype=np.float64, shape=(num_weights,)
        A 1d array of all neural net weights including all biases        
    
    jacobian : np.ndarray, dtype=np.float64, shape=(num_weights,)
        A 1d array of derivative of loss with respect to network weights

    num_weights : int
        Total number of weights in the multilayer perceptron. This includes
        bias weights.

    References
    ----------
    [1] J. Behler, J. Phys. : Condensed Matter, 26, (2014)

    Examples
    --------
    >>> from nnp import mlpp
    >>> _mlpp = mlpp.MultiLayerPerceptronPotential(hidden_layer_size=[10,5])
    """

    def __init__(self,hidden_layer_sizes=[10,5],activation='sigmoid',solver='bfgs',\
            hyper_params={'energy':1.0,'forces':1.0,'regularization':0.0},\
            solver_kwargs={"maxiter":15000,"gtol":1e-12},precision_update_interval=0,\
            max_precision_update_number=1,parallel=True):
            
            self.activation = activation
            self.loss_norm = 'l2'
            self.hyper_params = {}
            self.activation_variance = 1.0
            self.hidden_layer_sizes = None
            self.features = None
            self.weights = None
            self.jacobian = None
            self.num_weights = None
            self.D = None
            self.OptimizeResult = None
            self.computed_features = None
            self.parallel = parallel
            self.scale_features = True # super large numerical instabilities without this
            self.set_weight_init_scheme("general")
            self.set_solver(solver) 
            self.set_layer_size(hidden_layer_sizes)
            for _key in hyper_params:
                self.set_hyperparams(_key,hyper_params[_key]) 
            self._opt_callback_cntr = 0 
            self._loss_callback_freq = 10 # frequency with which callback function is performed 
            self.set_precision_update_interval(precision_update_interval)
            self.max_precision_update_number = max_precision_update_number

            # set map from Python string to Fortran int
            self._set_map = {"train":1,"holdout":2,"test":3}

            # activation function map (String->Int)
            self._activation_map = {"sigmoid":1,"tanh":2}

            # loss norma map (String->Int)
            self._loss_norm_map = {"l1":1,"l2":2}

            # list for holdout set loss, used for early stopping if holdout present                
            self._holdout_loss = []

            # data for optimization log 
            self._njev = 0 
            self._loss_log = []
   
            self.solver_kwargs = solver_kwargs

    def _init_optimization_log(self):
        """
        Perform operations necessary to initialise log data for optimization
        """

        self._njev = 0
        self._loss_log = []
        self._holdout_loss = []
    
    def _update_num_weights(self):
        if self.D is None:
            raise MlppError("Cannot update weights while dimension of features is unknown")
        
        # biases included in weights array
        self.num_weights = (self.D+1)*self.hidden_layer_sizes[0]
        self.num_weights += (self.hidden_layer_sizes[0]+1)*self.hidden_layer_sizes[1]
        self.num_weights += self.hidden_layer_sizes[1] + 1
        
        # update buffer for weights
        #self._init_random_weights()

        # update buffer for jacobian
        self.jacobian = np.zeros(self.num_weights,dtype=np.float64,order='F')

    def _zero_weight_biases(self,weights=None):
        bias_idx = [ii for ii in range(self.hidden_layer_sizes[0])]
        bias_idx += [self.hidden_layer_sizes[0]*(self.D+1)+ii for \
                ii in range(self.hidden_layer_sizes[1])]
        bias_idx.append(self.hidden_layer_sizes[0]*(self.D+1)+(self.hidden_layer_sizes[0]+1)*\
                self.hidden_layer_sizes[1])
        if weights is not None:
            weights[np.asarray(bias_idx,dtype=np.int32)] = 0.0
            return weights
        else:
            self.weights[np.asarray(bias_idx,dtype=np.int32)] = 0.0

    def _init_random_weights(self):
        import nnp.nn.fortran.nn_f95 as f95_api 
        
        if self.D is None:
            raise MlppError("Cannot initialise weights before dimension of features is known")

        if self.weight_init_scheme == 'xavier':
            self.weights = np.asarray(np.random.normal(loc=0.0,scale=np.sqrt(1.0/self.D),\
                    size=self.hidden_layer_sizes[0]*(self.D+1)),order='F',dtype=np.float64)
            
            self.weights = np.hstack( (self.weights,np.asarray(np.random.normal(loc=0.0,\
                    scale=np.sqrt(1.0/self.hidden_layer_sizes[0]),\
                    size=self.hidden_layer_sizes[1]*(self.hidden_layer_sizes[0]+1)))) )

            self.weights = np.hstack( (self.weights,np.asarray(np.random.normal(loc=0.0,\
                    scale=np.sqrt(1.0/self.hidden_layer_sizes[1]),\
                    size=self.hidden_layer_sizes[1]+1))) ) 

            self._zero_weight_biases()
        elif self.weight_init_scheme == "glorot":
            self.weights = (np.asarray(np.random.random(size=self.hidden_layer_sizes[0]*\
                    (self.D+1)),order='F',\
                    dtype=np.float64)*2.0 - 1.0)*np.sqrt(6.0/(self.D+self.hidden_layer_sizes[0]))
            
            self.weights = np.hstack(  ( self.weights,\
                    (np.asarray(np.random.random(size=self.hidden_layer_sizes[1]*\
                    (self.hidden_layer_sizes[0]+1)),order='F',\
                    dtype=np.float64)*2.0 - 1.0)*np.sqrt(6.0/(self.hidden_layer_sizes[1]+\
                    self.hidden_layer_sizes[0])) )  )

            self.weights = np.hstack(  ( self.weights,\
                    (np.asarray(np.random.random(size=self.hidden_layer_sizes[1]+1),\
                    order='F',dtype=np.float64)*2.0 - 1.0)*\
                    np.sqrt(6.0/(self.hidden_layer_sizes[1]+1)) )  )

            self._zero_weight_biases()
        elif self.weight_init_scheme == "general":
            # allows for <x_k> != 0, variance(x_k)!=1
            #
            # 1. compute empirical sample variance and mean for each 
            #    coordinate
            # 2. forward propagate to compute empirical variance and mean for
            #    z^1
            # 3. forward propagate to compute correct energy variance

            if nnp.util.misc.get_num_configs("train") < 2:
                # we take variance over config energies later
                raise MlppError("Must have more than 1 configuration in training set")

            # 1. empirical sample mean and variance for each component
            data_mean = np.average(self.computed_features,axis=1)
            data_vari = np.std(self.computed_features,axis=1)**2

            # idx of bad features
            bad_feature = np.isinf(1.0/np.average(self.computed_features**2,axis=1))

            if np.any(bad_feature):
                raise MlppError("Features {} are 0! Please remove.".\
                        format(np.nonzero(bad_feature)[0]))
                

            # variance(w^1) = variance(a^1) / [\sum_d^D variance(x_d) + mean(x_d)**2] 
            #w1_variance = self.activation_variance / np.sum(data_vari + data_mean**2)
            w1_variance = self.activation_variance / (self.D*\
                    np.average(self.computed_features**2,axis=1))
           
            self.weights = np.zeros(self.hidden_layer_sizes[0]*(self.D+1),\
                    order='F',dtype=np.float64)
            cntr = self.hidden_layer_sizes[0]
            for _dimension in range(self.D):
                self.weights[cntr:cntr+self.hidden_layer_sizes[0]] = np.random.normal(\
                        loc=0.0,scale=np.sqrt(w1_variance[_dimension]),\
                        size=self.hidden_layer_sizes[0])
                cntr += self.hidden_layer_sizes[0]

            #self.weights = np.asarray(np.random.normal(loc=0.0,scale=np.sqrt(w1_variance),\
            #        size=self.hidden_layer_sizes[0]*(self.D+1)),order='F',dtype=np.float64)

            # weights in level 2 and 3 (including biases)
            num_remaining_weights = self.hidden_layer_sizes[1]*\
                    (self.hidden_layer_sizes[0]+1) + self.hidden_layer_sizes[1]+1

            partial_weights = np.hstack((self.weights,np.zeros(num_remaining_weights,\
                    dtype=np.float64,order='F'))) 
           
            # need to zero bias weights
            partial_weights = self._zero_weight_biases(weights=partial_weights)
            
            z1,_ = nnp.nn.helper_funcs.get_node_distribution(weights=partial_weights,\
                    input_type='z',set_type="train")

            # 2.
            z1_mean = np.average(z1,axis=1)
            z1_vari = np.std(z1,axis=1)**2

            #w2_variance = self.activation_variance / np.sum(z1_vari + z1_mean**2)
            #self.weights = np.hstack(( self.weights,\
            #        np.asarray(np.random.normal(loc=0.0,scale=np.sqrt(w2_variance),\
            #        size=self.hidden_layer_sizes[1]*(self.hidden_layer_sizes[0]+1)),\
            #        order='F',dtype=np.float64) ))
            w2_variance = self.activation_variance / (self.hidden_layer_sizes[0]*\
                    np.average(z1**2,axis=1))
            
            layer2_weights = np.zeros(self.hidden_layer_sizes[1]*\
                    (self.hidden_layer_sizes[0]+1),order='F',dtype=np.float64)
            cntr = self.hidden_layer_sizes[1]
            for _dimension in range(self.hidden_layer_sizes[0]):
                layer2_weights[cntr:cntr+self.hidden_layer_sizes[1]] = np.random.normal(\
                        loc=0.0,scale=np.sqrt(w2_variance[_dimension]),\
                        size=self.hidden_layer_sizes[1])
                cntr += self.hidden_layer_sizes[1]

            self.weights = np.hstack(( self.weights, layer2_weights ))
            
            #----#
            # 3. #
            #----#
       
            # remaining number of free weights
            num_remaining_weights = self.hidden_layer_sizes[1] + 1 
    
            partial_weights = np.hstack((self.weights , np.zeros(num_remaining_weights,\
                    dtype=np.float64,order='F') ))
            
            # need to zero bias weights
            partial_weights = self._zero_weight_biases(weights=partial_weights)
            
            _,z2 = nnp.nn.helper_funcs.get_node_distribution(weights=partial_weights,\
                    input_type='z',set_type="train")
            

            # train set ref. energies
            ref_energies = nnp.nn.helper_funcs.get_reference_energies(set_type="train")

            # need energy per atom
            atoms_per_conf = nnp.nn.helper_funcs.get_atoms_per_conf(set_type="train")

            # empirical stats from training data
            ref_total_energy_variance = np.var(ref_energies)
            ref_energy_per_atom_mean = np.average(ref_energies/atoms_per_conf)
            atoms_per_conf = nnp.nn.helper_funcs.get_atoms_per_conf("train")
            per_atom_variance = np.var(ref_energies)/np.mean(atoms_per_conf)

            weight_means = np.zeros(self.hidden_layer_sizes[1],dtype=np.float64)
            weight_variances = per_atom_variance/( self.hidden_layer_sizes[1] * \
                    np.mean(z2**2,axis=1) )

            layer3_weights = np.zeros(self.hidden_layer_sizes[1]+1,dtype=np.float64,order='F')
            for ii in range(self.hidden_layer_sizes[1]):
                layer3_weights[ii+1] = np.random.normal(loc=weight_means[ii],\
                        scale=np.sqrt(weight_variances[ii]),size=1)
            
            # propagate to get total energy distribution
            _ = getattr(f95_api,"f90wrap_loss")(flat_weights=np.hstack((self.weights,\
                    layer3_weights)),set_type=1,\
                    parallel=self.parallel,squared_errors=np.zeros(3,dtype=np.float64,order='F'))
          
            gip = nnp.util.io._parse_configs_from_fortran("train")
            total_energies = np.asarray([_s["energy"] for _s in gip])
          
            # scale weights to reproduce exact total energ variance 
            weight_magnitude_correction = np.sqrt(ref_total_energy_variance/np.var(total_energies))

            layer3_weights[1:] *= weight_magnitude_correction 

            # forward prop to adjust final layer bias for perfect average
            per_atom_energies = np.dot(layer3_weights[1:],z2)
            
            weight_bias = ref_energy_per_atom_mean - np.average(per_atom_energies)
            
            self.weights = np.hstack(( self.weights, layer3_weights ))
            self._zero_weight_biases()

            w3_bias_idx = self.hidden_layer_sizes[0]*(self.D+1)+(self.hidden_layer_sizes[0]+1)*\
                    self.hidden_layer_sizes[1]
            
            self.weights[w3_bias_idx] = weight_bias

        else:
            raise MlppError("Weight initialization scheme not supported")
            
        if self.weights.shape[0]!=self.num_weights:
            raise MlppError("Severe implementation error")

    def set_hyperparams(self,key,value):
        """
        Set coefficients for energy,force,regularization loss terms

        Paramaters
        ---------
        key : String, allowed values = 'energy','forces','regularization'
            Which loss term coefficient is being set

        value : float
            Corresponding value
        """

        if key.lower() not in ['energy','forces','regularization']:
            raise MlppError("Hyper parameter {} not recognised".format(key.lower()))
        
        self.hyper_params[key.lower()] = value
            
    def set_precision_update_interval(self,precision_update_interval):
        """
        Set the number of optimizer iterations between updating the energy,
        force,regularization loss coefficients. If initial loss hyper parameters
        are 0, these will not be updated
        """
        if not isinstance(precision_update_interval,(int,np.int16,np.int32,np.int64)):
            raise MlppError("precision update interval must be an int")
        
        self.precision_update_interval = precision_update_interval

    def set_weight_init_scheme(self,scheme):
        """
        Set the scheme to use when initialisation neural net weights

        Parameters
        ----------
        scheme : String, allowed values = xavier
            The scheme to use when initialising weights before optimization
        """
        if scheme.lower() not in ["xavier","glorot","general"]:
            raise MlppError("weight initialisation scheme {} not supported".format(scheme.lower()))

        self.weight_init_scheme = scheme.lower()

    def set_layer_size(self,hidden_layer_sizes):
        """
        Set the size of hidden layers. Allocate memory for jacobian and set 
        num_weights if features have already been set
        
        Parameters
        ----------
        hidden_layer_sizes : array-like, int args, shape = (2,) 
            An array of the number of nodes in each hidden layer
        """

        self.hidden_layer_sizes = np.asarray(hidden_layer_sizes,dtype=np.int32)

        if self.features is not None:
            self._update_num_weights()
            
    def set_parallel(self,parallel):
        """
        Set whether to use multithreading in loss and jacobian calculation
        """
        self.parallel = parallel

    def set_solver(self,solver):
        solver = solver.lower()

        if solver not in ['nelder-mead','powell','cg','bfgs','l-bfgs-b','tnc','cobyla',\
                'slsqp','dogleg','trust-ncg','trust-krylov','trust-exact','adam','cma']:
            raise MlppError("solver {} not supported".format(solver))
        self.solver = solver

    def set_features(self,features):
        """
        Set features to calculate during fitting
        
        Parameters
        ----------
        features : nnp.features.types.features()
            Class of configuration features
        
        Examples
        --------
        >>> import parsers
        >>> import nnp
        >>> gip = parsers.GeneralInputParser()
        >>> gip.parse_all('./training_data')
        >>> features = nnp.features.types.features(gip)
        >>>
        >>> mlpp = nnp.nn.mlpp.MultiLayerPerceptronPotential()
        >>> mlpp.set_features(features)
        """
        if isinstance(features,nnp.features.types.features)!=True:
            raise MlppError("Features type {} != nnp.features.types.features()".format(type(features))) 

        self.features = features

        # dimension of feature vector - CHANGE FOR PCA
        self.D = len(self.features.features)

        if self.hidden_layer_sizes is not None:
            # update weights and jacobian buffer
            self._update_num_weights()

    def _initialise_net(self):
        """
        Initialise all fortran data structures ready for training
        weights
        """
        import nnp.nn.fortran.nn_f95 as f95_api 

        # initialise weights and node variables
        getattr(f95_api,"f90wrap_initialise_net")(num_nodes=self.hidden_layer_sizes,\
                nlf_type=self._activation_map[self.activation],feat_d=self.D)
     
        # set loss function parameters
        getattr(f95_api,"f90wrap_init_loss")(k_energy=self.hyper_params["energy"],\
                k_forces=self.hyper_params["forces"],\
                k_reglrn=self.hyper_params["regularization"],
                norm_type=self._loss_norm_map[self.loss_norm])

    def check_node_distribution(self,X):
        """
        For both layers, forward propagate self.weights to compute the input 
        value to all nodes (value before activation function is applied). 
        
        For reasonable weights, we want values to be distributed about 0 with a 
        standard deviation of ~ 1 for all nodes. If this is not the case, poor
        performance may be seen as the gradient of many activation functions
        (sigmoid for eg.) may be almost 0.
        
        Parameters
        ----------
        X : parsers.GeneralInputParser()
            A structure class of reference structures
        """
        import nnp.nn.fortran.nn_f95 as f95_api 
        
        set_type = "train"

        # parse data to fortran and compute features
        self._prepare_data_structures(X=X,set_type=set_type)
       
        # forward propagate               
        return nnp.nn.helper_funcs.get_node_distribution(weights=self.weights,input_type='a',\
                set_type=set_type)

    def _update_net_weights(self):        
        if self.solver in ['adam','cma']:
            self.OptimizeResult = nnp.optimizers.stochastic.minimize(fun=self._loss,\
                    jac=self._loss_jacobian,x0=self.weights,solver=self.solver,\
                    args=("train"),**self.solver_kwargs)
        else:
            if self.precision_update_interval != 0:
                try:
                    self.solver_kwargs["maxiter"] = self.precision_update_interval
                except KeyError:
                    self.solver_kwargs.update({"maxiter":self.precision_update_interval})
                # SCIPY's l-bfgs-b default
                maxiter = 15000
            else:
                # use SCIPY's bfgs default
                try:
                    self.solver_kwargs["maxiter"] = 15000
                except KeyError:
                    self.solver_kwargs.update({"maxiter":15000})
            self._holdout_loss = []

            self.OptimizeResult = optimize.minimize(fun=self._loss,x0=self.weights,\
                    method=self.solver,args=("train"),jac=self._loss_jacobian,\
                    options=self.solver_kwargs,callback=self._loss_callback)
        
        self.weights = self.OptimizeResult["x"]
    
    def _update_precision(self):
        """
        Called after each optimizer iteration with weights as arg
        
        Set precision_update_interval to 0 if you never want loss hyper params
        to be updated.

        Parameters
        ----------
        weights : np.ndarray, shape = (self.Nwght,)
            1d array of neural net weights
        """
        import nnp.nn.fortran.nn_f95 as f95_api 
        
        _ = self._loss(weights=self.weights,set_type="train")

  
        ses = {"energy":self._squared_errors[0],"forces":self._squared_errors[1],\
                "regularization":self._squared_errors[2]}
       
        num_weights_no_bias = self.hidden_layer_sizes[0]*self.D + self.hidden_layer_sizes[0]*\
                self.hidden_layer_sizes[1] + self.hidden_layer_sizes[1]

        const = {"energy":nnp.util.misc.get_num_configs("train"),\
                "forces":nnp.util.misc.total_atoms_in_set("train")*3,"regularization":num_weights_no_bias}

        for _loss_term in ses:
            if not np.isclose(self.hyper_params[_loss_term],0.0,rtol=1e-50,atol=1e-50):
                if np.isclose(ses[_loss_term],0.0,rtol=1e-50,atol=1e-50):
                    raise MlppError("loss squared error for {} is 0 when coefficient != 0".\
                            format(ses[_loss_term]))
            
                self.set_hyperparams(_loss_term,const[_loss_term]/ses[_loss_term])
        
        # send to fortran
        getattr(f95_api,"f90wrap_init_loss")(k_energy=self.hyper_params["energy"],\
                k_forces=self.hyper_params["forces"],\
                k_reglrn=self.hyper_params["regularization"],
                norm_type={"l1":1,"l2":2}[self.loss_norm])
            

               
    def _update_loss_log(self,loss):
        """
        Store most recent value of objective function during optimization
        """
        self._loss_log.append(loss)

    def _loss(self,weights,set_type,log_loss=True):
        import nnp.nn.fortran.nn_f95 as f95_api 
        
        if np.isnan(weights).any():
            print('{} Nan found in weights array'.format(np.sum(np.isnan(weights))))
       
        # energy,forces,regularization 
        squared_errors = np.zeros(3,dtype=np.float64,order='F')

        tmp = getattr(f95_api,"f90wrap_loss")(flat_weights=weights,set_type=self._set_map[set_type],\
                parallel=self.parallel,squared_errors=squared_errors)
        
        if log_loss:
            # book keeping
            self._update_loss_log(tmp)
        self._squared_errors = squared_errors
        
        if np.isnan(tmp):
            raise MlppError("Nan returned from loss calculation")
        return tmp

    def _loss_callback(self,weights):
        """
        callback function for SCIPY optimizer, called after each iteration

        Terminate optimization when the train energy rmse (and force rmse) drop
        below a given tolerance OR (if the holdout set is in use), when the
        holdout set loss is greater than a reference for so many iterations in
        turn.
        """
        import nnp.nn.fortran.nn_f95 as f95_api 
       
        # early stopping flags 
        terminate_opt = False
        train_holdout_divergence = False

        holdout_loss_mem = 10

        Ermse_per_atom_meV_tol = 0.1
        Frmse_eVA_tol = 0.1

        if np.mod(self._opt_callback_cntr,self._loss_callback_freq) == 0:
            if nnp.util.misc.get_num_configs("holdout") != 0:

                _loss = getattr(f95_api,"f90wrap_loss")(flat_weights=weights,\
                        set_type=self._set_map["holdout"],parallel=self.parallel,\
                        squared_errors=np.zeros(3,dtype=np.float64,order='F'))

                self._holdout_loss.append(_loss)
                
                if len(self._holdout_loss)>holdout_loss_mem: 
                    loss0 = self._holdout_loss.pop(0)

                    dloss = [_l - loss0 for _l in self._holdout_loss]
                    if all([_l>0.0 for _l in dloss]):
                        # every iteration since loss0 has increased holdout loss
                        #print('terminating because of holdout set')
                        train_holdout_divergence = True

            # number of configurations
            Nconf = nnp.util.misc.get_num_configs("train")

            # average total energy error per atom (meV)
            Ermse_per_atom_meV = 1000.0 * np.sqrt(self._squared_errors[0] / Nconf)
           
            # energy convergence criterion 
            converged  = [Ermse_per_atom_meV < Ermse_per_atom_meV_tol]
          
            if not np.isclose(self.hyper_params["forces"],0.0,rtol=1e-50,atol=1e-50):
                # total number of atoms in set
                Ntot = nnp.util.misc.total_atoms_in_set("train")
               
                # rmse for force observations (eV/A)
                Frmse_eVA = np.sqrt(self._squared_errors[1] / Ntot)
               
                # force convergence criterion
                converged.append(Frmse_eVA < Frmse_eVA_tol)
                
                #print(Ermse_per_atom_meV,Frmse_eVA)        
            
            terminate_opt = all(converged) or train_holdout_divergence
        self._opt_callback_cntr += 1 

        return terminate_opt

    def _loss_jacobian(self,weights,set_type):
        """
        Compute the jacobian of the loss function with respect to net weights 
        for the data set_type
        
        Parameters
        ----------
        weights : np.ndarray, dtype=np.float64,order='F'
            A 1d array of net weights

        set_type : String, allowed values = 'train','test'
            Identifier of which set type to compute jacobian for
        """
        import nnp.nn.fortran.nn_f95 as f95_api 
        
        if np.isnan(weights).any():
            print('{} Nan found in weights array'.format(np.sum(np.isnan(weights))))
    
        tmp_jac = np.zeros(weights.shape,dtype=np.float64,order='F')
        
        getattr(f95_api,"f90wrap_loss_jacobian")(flat_weights=weights,\
                set_type=self._set_map[set_type],\
                parallel=self.parallel,jacobian=tmp_jac)
      
        # count number of jacobian evaluations 
        self._njev += 1
        
        if np.isnan(self.jacobian).any():
            raise MlppError("Nan computed for jacobian of loss")
        
        #self.

        return tmp_jac
    
    def _prepare_data_structures(self,X,set_type):
        """
        Initialise all fortran data structures for given set

        Parameters
        ----------
        X : parsers.GeneralInputParser
            All atomic configurations to included in this 
            data set
        set_type : String, allowed values = 'train','test'
            The type of data set
        """
        # write configurations to fortran data structures
        self.features.set_configuration(gip=X,set_type=set_type)
    
        if set_type == "train" and self.scale_features:
            # need to compute pre conditioning coefficients
            self.features.calculate_precondition()
        
        # initialise feature mem and compute for set_type
        self.computed_features = self.features.calculate(set_type=set_type,\
                derivatives=True,scale=self.scale_features)
    
        # initialise neural net data structures
        self._initialise_net()


    def fit(self,train,holdout=None):
        """
        Learn neural net weights for given loss function

        Parameters
        ----------
        X : parsers.GeneralInputParser()
            A Python data structure of configurations
        
        Examples
        --------
        >>> import parsers
        >>> import nnp
        >>> training_data = parsers.GeneralInputParser('./training_data')
        >>>
        >>> _features = nnp.features.types.features(training_data)
        >>> # Gaussian features
        >>> _features.generate_gmm_features()
        >>> # atomic number 
        >>> _features.add(nnp.features.types.feature('atomic_number'))
        >>>
        >>> mlpp = nnp.nn.mlpp.MultiLayerPerceptron()
        >>> mlpp.set_features(_features)
        >>> # regress net weights
        >>> training_loss = mlpp.fit(training_data,holdout_data)
        """
        if self.features is None:
            raise MlppError("set_features() must be called before fit()")
        
        # Scipy log class lacks objective function with iteration
        self._init_optimization_log()

        # move necessary info to fortran via disk
        self._prepare_data_structures(X=train,set_type="train")
        if holdout is not None:
            self._prepare_data_structures(X=holdout,set_type="holdout")
        
        # compute initial weights based upon training data
        self._init_random_weights()

        if self.max_precision_update_number == 0:
            self._update_net_weights()
        else:
            for ii in range(self.max_precision_update_number):
                self._update_precision()
                self._update_net_weights() 

        # book keeping
        self.OptimizeResult.njev = self._njev

        # use custom OptimizeResult class
        self.OptimizeResult = NnpOptimizeResult(self.OptimizeResult)

        # store loss with optimization iterations
        self.OptimizeResult.loss_log = np.asarray(np.asarray(self._loss_log))

        # store optimized weights 
        self.weights = self.OptimizeResult.get("x")

        # predicted energies and forces
        predicted_gip = nnp.util.io._parse_configs_from_fortran("train")

        # return loss
        return self.OptimizeResult.get("fun"),predicted_gip

    def predict(self,X):
        """
        Calculate the total energy and atomic forces

        Parameters
        ----------
        X : parsers.GeneralInputParser
            A cofigurations data structure containing the atomic configurations 
            to regress the PES of.

        Returns
        -------
        parsers.GeneralInputParser
            A configurations data structure containing the predicted total 
            energy and atom forces
        
        Examples
        --------
        >>> import parsers
        >>> import nnp
        >>> training_data = parsers.GeneralInputParser('./training_data')
        >>>
        >>> _features = nnp.features.types.features(training_data)
        >>> # Gaussian features
        >>> _features.generate_gmm_features()
        >>> # atomic number 
        >>> _features.add(nnp.features.types.feature('atomic_number'))
        >>>
        >>> mlpp = nnp.nn.mlpp.MultiLayerPerceptron()
        >>> mlpp.set_features(_features)
        >>> # regress net weights
        >>> training_loss = mlpp.fit(training_data)
        >>>
        >>> test_data = parsers.GeneralInputParser('./test_data')
        >>> testing_loss = mlpp.predict(test_data)
        """
        import nnp.nn.fortran.nn_f95 as f95_api 
       
        self._prepare_data_structures(X=X,set_type="test")

        # forward and backward propagate to calc. all forces
        loss = self._loss(weights=self.weights,set_type="test")
        
        # predicted energies and forces
        predicted_gip = nnp.util.io._parse_configs_from_fortran("test")

        # return loss
        return loss,predicted_gip

class MlppError(Exception):
    pass    
