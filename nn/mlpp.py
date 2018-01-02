from nnp.util.log import NnpOptimizeResult
import numpy as np
import nnp
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

    def __init__(self,hidden_layer_sizes=[10,5],activation='sigmoid',solver='l-bfgs-b',\
            hyper_params={'loss_energy':1.0,'loss_forces':1.0,'loss_regularization':1.0}):
            
            self.activation = activation
            self.loss_norm = 'l1'
            self.solver = solver
            self.hyper_params = hyper_params
            self.init_weight_var = 1e-5
            self.hidden_layer_sizes = None
            self.features = None
            self.weights = None
            self.jacobian = None
            self.num_weights = None
            self.D = None
            self.OptimizeResult = None
            self.parallel = False
            self.scale_features = False
           
            self.set_layer_size(hidden_layer_sizes)
  
            # data for optimization log 
            self._njev = None
            self._loss_log = None
   
    def _init_optimization_log(self):
        """
        Perform operations necessary to initialise log data for optimization
        """

        self._njev = 0
        self._loss_log = []
    
    def _update_num_weights(self):
        # biases included in weights array
        self.num_weights = (self.D+1)*self.hidden_layer_sizes[0]
        self.num_weights += (self.hidden_layer_sizes[0]+1)*self.hidden_layer_sizes[1]
        self.num_weights += self.hidden_layer_sizes[1] + 1
        
        # update buffer for weights
        self._init_random_weights()

        # update buffer for jacobian
        self.jacobian = np.zeros(self.num_weights,dtype=np.float64,order='F')

    def _init_random_weights(self):
        # update buffer for weights
        self.weights = np.asarray(np.random.normal(loc=0.0,scale=self.init_weight_var,\
                size=self.num_weights),dtype=np.float64,order='F')

        bias_idx = [ii for ii in range(self.hidden_layer_sizes[0])]
        bias_idx += [self.hidden_layer_sizes[0]*(self.D+1)+ii for \
                ii in range(self.hidden_layer_sizes[1])]
        bias_idx.append(self.hidden_layer_sizes[0]*(self.D+1)+(self.hidden_layer_sizes[0]+1)*\
                self.hidden_layer_sizes[1])
        self.weights[np.asarray(bias_idx,dtype=np.int32)] = 0.0

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
            

    def set_solver(self,solver):
        solver = solver.lower()

        if solver not in ['nelder-mead','powell','cg','bfgs','l-bfgs-b','tnc','cobyla',\
                'slsqp','dogleg','trust-ncg','trust-krylov','trust-exact']:
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

    def _initialise_net(self,set_type):
        """
        Initialise all fortran data structures ready for training
        weights
        """
        import nnp.nn.fortran.nn_f95 as f95_api 

        _map = {"sigmoid":1,"tanh":2}
        # initialise weights and node variables
        getattr(f95_api,"f90wrap_initialise_net")(num_nodes=self.hidden_layer_sizes,\
                nlf_type=_map[self.activation],feat_d=self.D)
      
        if self.loss_norm == 'l2':
            # BUGGY FOR ENERGY LOSS TERM
            raise NotImplementedError
       
        _map = {"l1":1,"l2":2}
        # set loss function parameters
        getattr(f95_api,"f90wrap_init_loss")(k_energy=self.hyper_params["loss_energy"],\
                k_forces=self.hyper_params["loss_forces"],k_reglrn=self.hyper_params["loss_regularization"],
                norm_type=_map[self.loss_norm])

    def _update_loss_log(self,loss):
        """
        Store most recent value of objective function during optimization
        """
        self._loss_log.append(loss)

    def _loss(self,weights,set_type):
        import nnp.nn.fortran.nn_f95 as f95_api 
        if np.isnan(weights).any():
            print('{} Nan found in weights array'.format(np.sum(np.isnan(weights))))
        
        _map = {"train":1,"test":2} 
        tmp = getattr(f95_api,"f90wrap_loss")(flat_weights=weights,set_type=_map[set_type],\
                parallel=self.parallel)
        
        # book keeping
        self._update_loss_log(tmp)
        
        if np.isnan(tmp):
            raise MlppError("Nan returned from loss calculation")
        return tmp

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
        
        _map = {"train":1,"test":2} 
        getattr(f95_api,"f90wrap_loss_jacobian")(flat_weights=weights,set_type=_map[set_type],\
                parallel=self.parallel,jacobian=self.jacobian)
      
        # count number of jacobian evaluations 
        self._njev += 1
        
        if np.isnan(self.jacobian).any():
            raise MlppError("Nan computed for jacobian of loss")
        return self.jacobian
    
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
        
        # initialise feature mem and compute for set_type
        self.features.calculate(set_type=set_type,derivatives=True,scale=self.scale_features)
    
        # initialise neural net data structures
        self._initialise_net(set_type=set_type)


    def fit(self,X):
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
        >>> training_loss = mlpp.fit(training_data)
        """
        if self.features is None:
            raise MlppError("set_features() must be called before fit()")
        
        # Scipy log class lacks objective function with iteration
        self._init_optimization_log()

        self._prepare_data_structures(X=X,set_type="train")

        self.OptimizeResult = optimize.minimize(fun=self._loss,x0=self.weights,\
                method=self.solver,args=("train"),jac=self._loss_jacobian)

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
        predicted_gip = nnp.util.io._parse_configs_from_fortran("train")

        # return loss
        return loss,predicted_gip

class MlppError(Exception):
    pass    
