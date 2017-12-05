import numpy as np

class MultiLayerPerceptronPotential():
    """Full connected feed forward multi layer perceptron for empirical potentials

    This network follows the architecture of Behler using 3 weight layers [1]. The
    net output is the energy per atom and forces are computed from derivatives of
    the net with respect to input features and these feature derivatives with
    respect to atom positions. Periodic boundaries are treated in full and not in
    the nearest image convention employed for molecular dynamics (when large unit
    cells are assumed). 


    Parameters
    ----------
    hidden_layer_sizes : {int, array-like}, shape = [2,]
        An array [n1,n2] giving the number of nodes in the 1st and 2nd hidden 
        layer.

    activation : string
        The type of non-linear activation function to apply to all nodes

    solver : string
        The method to minimise loss 

    hyper_params : dict , keys = 'loss_energy','loss_forces','loss_regularization'
        A dictionary of network hyper parameters including coefficients to the
        energy, force and regularization parts of the loss function

    Attributes
    ----------
    weights : {float,array-like}, shape = [(D+1)*n1 + (n1+1)*n2 + n2+1,]
        A flattened array of the current network weights. 

    References
    ----------
    [1] J. Behler, J. Phys. : Condensed Matter, 26, (2014)

    Examples
    --------
    >>> from nnp import mlpp
    >>> _mlpp = mlpp.MultiLayerPerceptronPotential(hidden_layer_size=[10,5])
    """

    def __init__(self,hidden_layer_sizes=[10,5],activation='sigmoid',solver='lbfgs',\
            hyper_params={'loss_energy':1.0,'loss_forces':1.0,'loss_regularization':1.0}):
            
            self.activation = activation
            self.loss_norm = 'l2'
            self.solver = solver
            self.hyper_params = hyper_params
            self.hidden_layer_sizes = None
            self.features = None
            self.weights = None
            self.jacobian = None
            self.num_weights = None
            self.D = None
            
            self.set_layer_size(hidden_layer_sizes)
   
    def _update_num_weights(self):
        # biases included in weights array
        self.num_weights = (self.D+1)*self.hidden_layer_sizes[0]
        self.num_weights += (self.hidden_layer_sizes[0]+1)*self.hidden_layer_sizes[1]
        self.num_weights += self.hidden_layer_sizes[1] + 1
        
        # update buffer for weights
        self.weights = np.zeros(self.num_weights,dtpye=np.float64,order='F')

        # update buffer for jacobian
        self.jacobian = np.zeros(self.num_weights,dtype=np.float64,order='F')

    def set_layer_size(self,hidden_layer_sizes):
        """
        Set the size of hidden layers. Allocate memory for jacobian 
        and set num_weights if features have already been set
        
        Parameters
        ----------
        hidden_layer_sizes : array-like, int args, shape = (2,) 
            An array of the number of nodes in each hidden layer
        """

        self.hidden_layer_sizes = np.asarray(hidden_layer_sizes,dtype=np.int32)

        if self.features is not None:
            self._update_num_weights()
            

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

        _map = {"sigmoid":1,"tanh":2}
        # initialise weights and node variables
        getattr(f95_api,"f90wrap_initialise_net")(num_nodes=self.hiddenlayer_sizes,\
                mlf_type=_map[self.acitvation],feat_D=self.D)
       
        _map = {"l1":1,"l2":2}
        # set loss function parameters
        getattr(f95_api,"f90wrap_init_loss")(k_energy=self.hyper_params["loss_energy"],\
                k_forces=self.hyper_params["loss_forces"],k_reglrn=self.hyper_params["loss_reglrn"],
                norm_type=_map[self.loss_norm])

    def _loss(self,weights,set_type):
        return getattr(f95_api,"f90wrap_loss")(flat_weights=weights,set_type=set_type)

    def _loss_jacobian(self,weights,set_type):
        """
        Compute the jacobian of the loss function with respect to 
        net weights for the data set_type
        
        Parameters
        ----------
        weights : np.ndarray, dtype=np.float64,order='F'
            A 1d array of net weights

        set_type : int,np.int16,np.int32 , allowed values = 1,2
            Which data set to produce the jacobian of loss for
        """

        getattr(f95_api,"f90wrap_loss_jacobian")(flat_weights=weights,set_type=set_type,\
                jacobian=self.jacobian)

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
        self.features.calculate(set_type=set_type,derivatives=True)
    
        # initialise neural net data structures
        self._initialise_net(set_type=set_type)


    def fit(self,X):
        """
        Learn neural net weights for given loss function

        Parameters
        ----------
        X : parsers.GeneralInputParser()
            A Python data structure of configurations
        """

        self._prepare_data_structures(X=X,set_type="train")

        _map = {"train":1,"test":2}
        
        opt_result = optimize.minimize(fun=self._loss,x0=self.weights,\
                method=self.solver,args=(_map["train"]),jac=self._loss_jacobian)

    def predict(self,X):
        """
        Calculate the total energy and atomic forces

        Parameters
        ----------
        X : parsers.GeneralInputParser
            A cofigurations data structure containing the
            atomic configurations to regress the PES of.

        Returns
        -------
        predicted_gip : parsers.GeneralInputParser
            A configurations data structure containing the
            predicted total energy and atom forces
        """ 
        self._prepare_data_structures(X=X,set_type="test")

        # forward propagate
        loss = self._loss(weights=self.weights,set_type="test")

        _map = {"train":1,"test":2}
        # backward propagate and calculate forces
        getattr(f95_api,"f90wrap_backprop_all_forces")(set_type=_map[set_type])

class MlppError(Exception):
    pass    
