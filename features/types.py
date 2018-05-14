"""
Python types for feature information
"""
import nnp.util.io
import nnp.nn.fortran.nn_f95 as f95_api
from nnp.features import pca as features_pca
import numpy as np
import tempfile
import shutil
from copy import deepcopy
from io import TextIOWrapper
from os import listdir
from sklearn import mixture
from scipy import optimize
import parsers
import warnings
import time

class feature():
    """feature
    
    Single feature class holding all free parameter values

    Parameters
    ----------
    feature_type : String
        Key defining type of feature. 

    Attributes
    ----------
    supp_types : [String]
        List of supported feature types

    Examples
    --------
    >>> import nnp
    >>> _feature1 = nnp.features.types.feature('atomic_number')
    >>> _feature2 = nnp.features.types.feature('acsf_behler-g2')
    >>> _feature2.set_params({'rcut':5.2,'fs':0.5,'eta':1.2,'rs':4.5,'za':1.0,'zb':0.6})
    """
    def __init__(self,feature_type,params=None):
        self.supp_types = ['atomic_number',
                           'acsf_behler-g1',
                           'acsf_behler-g2',
                           'acsf_behler-g4',
                           'acsf_behler-g5',
                           'acsf_normal-b2',
                           'acsf_normal-b3',
                           'acsf_fourier-b2',
                           'devel_iso']
        
        self.type = feature_type
        self.params = {}

        # pre condition x -> a*x + b so that x \in [-1,1]
        self.precondition = {"times":1.0,"add":0.0}

        if self.type not in self.supp_types:
            raise FeatureError('{} not in {}'.format(self.type,','.join(self.supp_types)))
        elif self.type == 'devel_iso':
            print('Warning : feature {} has no unit test. Use with extreme caution'.format(\
                    self.type))

        if params is not None:
            self.set_params(params)

    def set_param(self,key,value):
        """
        Set a single attribute for feature

        Parameters
        ----------
        param : dict, len = 1
            Length 1 dictionary containing key,value pair for single parameter
        """
        
        if self.type == 'atomic_number':
            raise FeatureError("atomic_number features have no params")
        elif self.type == 'acsf_behler-g1':
            _keys = ['rcut','fs','za','zb']
        elif self.type == 'acsf_behler-g2':
            _keys = ['rcut','fs','eta','rs','za','zb']
        elif self.type in ['acsf_behler-g4','acsf_behler-g5']:
            _keys = ['rcut','fs','xi','lambda','za','zb','eta']
        elif self.type in ['acsf_normal-b2','acsf_normal-b3']:
            _keys = ['rcut','fs','prec','mean','za','zb']
        elif self.type in ['devel_iso']:
            _keys = ['rcut','fs','mean','const','std']
        elif self.type in ['acsf_fourier-b2']:
            _keys = ['rcut','fs','weights','za','zb']

        if key not in _keys:
            raise FeatureError("Parameter {} not supported".format(key))
        
        _ftype = (int,np.int16,np.int32,np.int64,float,np.float32,np.float64)
        _atype = (list,np.ndarray,int,float,np.float32,np.float64)
        _types = {'rcut':_ftype,'fs':_ftype,'eta':_ftype,'za':_ftype,\
                'zb':_ftype,'xi':_ftype,'lambda':_ftype,'prec':_atype,\
                'mean':_atype,'rs':_ftype,'const':_atype,'std':_atype,'weights':_atype}
        # constraints on bound parameters
        _constraints_ok = {"xi":lambda x: x>=1,\
                           "rcut":lambda x: x>0,\
                           "lambda":lambda x: x in [-1.0,1.0],\
                           "eta":lambda x: x>=0}

        # check param type
        if type(value) not in _types[key]:
            raise FeatureError("param type {} != {}".format(type(value),_types[key]))

        if key in _constraints_ok.keys():
            # check any constraints are OK
            if not _constraints_ok[key](value):
                raise FeatureError("param type {} has invalid value {}".format(key,value))
        
        if self.type in ["acsf_normal-b3"]:
            # check length of arrays
            if self.type == "acsf_normal-b2":
                _length = {"mean":1,"prec":1}
            elif self.type == "acsf_normal-b3":
                _length = {"mean":3,"prec":9}
            for _attr_ in ["mean","prec"]:
                if _attr_ == key:
                    value = np.asarray(value,dtype=np.float64)
                    if value.flatten().shape[0] != _length[_attr_]:
                        raise FeatureError("arrays lengths are not as expected")
            if "prec" == key:
                try:
                    # check that matrix is symmetric
                    np.testing.assert_array_almost_equal(value,value.T)
                except AssertionError:
                    raise FeatureError("Supplied precision matrix for 3-body gaussian is not \
                            symmetric")
        elif self.type in ["acsf_fourier-b2"]:
            if key == "weights":
                value = np.asarray(value,dtype=np.float64)
        try:
            self.params[key] = deepcopy(value)
        except KeyError:
            self.params.update({key:copy.deepcopy(value)})

    def set_params(self,params):
        """
        check parameters for correct args and then set

        Parameters
        ----------
        params : dict
            Key,value pairs of parameters for specific feature type
        """

        # DO NOT CHANGE ORDER OF VARIABLES _UPDATE_ME !! 
        if self.type == 'atomic_number':
            raise FeatureError("atomic_number features have no params")
        elif self.type == 'acsf_behler-g1':
            _keys = ['rcut','fs','za','zb']
            _update_me = []
        elif self.type == 'acsf_behler-g2':
            _keys = ['rcut','fs','eta','rs','za','zb']
            _update_me = ['eta','rs']
        elif self.type in ['acsf_behler-g4','acsf_behler-g5']:
            _keys = ['rcut','fs','xi','lambda','za','zb','eta']
            _update_me = ['xi','eta']
        elif self.type in ['acsf_normal-b2','acsf_normal-b3']:
            _keys = ['rcut','fs','prec','mean','za','zb']
            _update_me = ['prec','mean']
        elif self.type in ['acsf_fourier-b2']:
            _keys = ['rcut','fs','weights','za','zb']
            _update_me = ['weights']
        elif self.type in ['devel_iso']:
            _keys = ['rcut','fs','mean','const','std']
        # list of parameters that optimized
        self.update_keys = _update_me
            
        if set(params.keys())!=set(_keys):
            # check keys
            raise FeatureError("supplied keys {} != {}".format(params.keys(),_keys))
        for _attr in params:
            self.set_param(_attr,params[_attr])

    def get_bounds(self,key):
        """
        Return bounds of parameters as constrained opt. of params is necessary.
        For feature params of scalar length > 1, a list of 2 elements lists is
        returned, eg [ [None,10] , [-1,3] ]
        
        Parameters
        ----------
        key : String, allowed vales = self.update_keys
            Parameter key name as in self.params
        """
        if key not in self.update_keys:
            raise FeatureError("feature parameter {} is not in {}".format(key,self.update_keys))
        
        value = None
        if key == "xi":
            value = [[1,None]]
        elif key == "eta":
            value = [[0,None]]
        elif key == "rs":
            value = [[None,None]]
        elif key == "mean":
            if self.type == "acsf_normal-b2":
                value = [[None,None]]
            elif self.type == "acsf_normal-b3":
                value = [[None,None] for ii in range(3)]
            else:
                raise FeatureError("Implementation Error")
        elif key == "prec":
            if self.type == "acsf_normal-b2":
                value = [[0,None]]
            elif self.type == "acsf_normal-b3":
                value = [[None,None] for ii in range(6)]
        elif key == 'weights':
            value = [[None,None] for ii in range(self.params['weights'].shape[0])]
        else:
            raise FeatureError("Need to write get_bounds for key {}".format(key))
        return value 

    def _write_to_disk(self,file_object):
        """
        Format feature info to disk

        Parameters
        ----------
        file_object : io.TextIOWrapper
            The file object of an open file

        Examples
        --------
        >>> f = open('newfile.features','w')
        >>> feature._write_to_disk(f)
        """
        
        if isinstance(file_object,TextIOWrapper)!=True:
            raise FeatureError("type {} is not io.TextIOWrapper.".format(type(file_object))) 

        if self.type == 'atomic_number':
            file_object.write('{} {:<20} {:<20}\n'.format(self.type,self.precondition["times"],\
                    self.precondition["add"]))
        elif self.type == 'acsf_behler-g1':
            file_object.write('{} {} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','za','zb']]),\
                    ' '.join(['{:<20}'.format(self.precondition[_a]) for _a in ['times','add']]) ))
        elif self.type == 'acsf_behler-g2':
            file_object.write('{} {} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in \
                    ['rcut','fs','eta','rs','za','zb']]) , \
                    ' '.join(['{:<20}'.format(self.precondition[_a]) for _a in ['times','add']]) ))
        elif self.type in ['acsf_behler-g4','acsf_behler-g5']:
            file_object.write('{} {} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','xi',\
                    'lambda','eta','za','zb']]) , \
                    ' '.join(['{:<20}'.format(self.precondition[_a]) for _a in ['times','add']]) ))
        elif self.type == 'acsf_normal-b2':
            file_object.write('{} {} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','za','zb',\
                    'mean','prec']]) ,\
                    ' '.join(['{:<20}'.format(self.precondition[_a]) for _a in ['times','add']])  ))
        elif self.type == 'acsf_normal-b3':
            file_object.write('{} {} {} {} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','za','zb']]),\
                    ' '.join(['{:<20}'.format(_m) for _m in self.params["mean"].flatten()]),\
                    ' '.join(['{:<20}'.format(_m) for _m in self.params["prec"].flatten()]) ,\
                    ' '.join(['{:<20}'.format(self.precondition[_a]) for _a in ['times','add']]) ))
        elif self.type == 'devel_iso':
            file_object.write('{} {} {:<20} {:<20}\n'.format(self.type,' '.join(['{:<20}'.format(\
                    self.params[_a]) for _a in ['rcut','fs','mean','const','std']]),\
                    self.precondition["times"],self.precondition["add"] ))
        elif self.type == 'acsf_fourier-b2':
            file_object.write('{} {} {} {} {} {}\n'.format(self.type,len(self.params["weights"]),\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','za','zb']]),\
                    ' '.join(['{:<20}'.format(_w) for _w in self.params['weights']]),\
                    self.precondition["times"],self.precondition["add"] ))

        else: raise FeatureError("Implementation error")
            
class features():
    """features

    Class for computation of atomic features 

    Attributes
    ----------
    gip : parsers.GeneralInputParser
        Data structure for all configurations

    maxrcut : dic, keys = 'twobody','threebody', default = {'twobody':6.0,'threebody':4.0}
        Maximum cut off radii of two and three body terms

    sample_rate : dic, keys = 'twobody','threebody', default = {'twobody':0.5,'threebody':0.1}
        Probability of sampling a bond towards the computed distribution
    
    
    Examples
    --------
    >>> import nnp
    >>> import parsers
    >>> gip_train = parsers.GeneralInputParser()
    >>> gip_test  = parsers.GeneralInputParser()
    >>> gip_train.parse_all('./training_data')
    >>> gip_test.parse_all('./testing_data')
    >>> _features = nnp.features.types.features(gip_train)
    >>> _features.set_configuration(gip_test,"test")
    """

    def __init__(self,train_data,PCA='linear'):
        # atomic configuraitons
        self.data = {"train":train_data,"test":None}

        # maximum cut off radii for two and three body terms
        self.maxrcut = {'twobody':7.0,'threebody':5.0}

        # fraction of bonds sampled for bond distribution
        self.sample_rate = {'twobody':0.5,'threebody':0.1}

        # parse data into fortran structure
        self.set_configuration(gip=self.data["train"],set_type="train")
        
        self.features = []

        # internal fortran mem. buffer size (GB)
        self.buffer_size = 0.1

        # internal parallelism 
        self.set_parallel(True)

        # set type of PCA to use
        self.set_pca(PCA) 

        # set type map (String->Int)
        self._set_map = {"train":1,"holdout":2,"test":3}

        # this improves regression by tuning features to be [-1,1]
        self.scale_features = True

    def set_parallel(self,parallel):
        """
        Set feature calculation to perform in parallel

        Parameters
        ----------
        parallel : boolean
        """
        if isinstance(parallel,bool)!=True:
            raise FeaturesError("must supply boolean variable")
        self.parallel = parallel

    def set_configuration(self,gip,set_type):
        """
        Parse configurations into fortran data structure
        
        Parameters
        ----------
        gip : parsers.GeneralInputParser
            A data structure with all configurations that are 
            to be stored in fortran

        set_type : String, allowed values = ['train','test']
            The data set type 
        """

        if set_type.lower() not in ['test','train','holdout']:
            raise FeaturesError('{} not in {}'.format(set_type,'train,test','holdout'))
        if not isinstance(gip,parsers.GeneralInputParser):
            raise FeaturesError('training data must be parsers.GeneralInputParser object not {}'.\
                    format(type(gip)))

        nnp.util.io._parse_configs_to_fortran(gip,set_type.lower())
        self.data[set_type.lower()] = gip

        if set_type == "train":
            # pre conditioning coefficients need recomputing
            self.precondition_computed = False

    def set_pca(self,pca_type):
        if pca_type not in [None,'linear']:
            raise FeaturesError('{} is not a supported PCA type')

        self.pca_type = pca_type

    def set_rcut(self,rcuts):
        """
        Set the maximum interaction cut off radii for two and 
        three body terms

        Parameters
        ----------
        rcuts : dict, keys = 'twobody','threebody'
            Key,value pairs of two and/or three body max cut
            off radius
        """

        if isinstance(rcuts,dict)!=True:
            raise FeaturesError("argument to set_rcut must be a dict")
        for _key in rcuts:
            if _key not in ['twobody','threebody']:
                raise FeaturesError("{} not a supported key in maxrcut".format(_key))
   
            self.maxrcut[_key] = rcuts[_key]
   
    def add_feature(self,feature):
        self.features.append(feature)
    
        # update maxrcut
        if feature.type in ['atomic_number']:
            pass
        elif feature.type in ['acsf_behler-g1','acsf_behler-g2','acsf_normal-b2','acsf_fourier-b2']:
            # two body feature
            if feature.params["rcut"] > self.maxrcut["twobody"]:
                self.set_rcut({"twobody":feature.params["rcut"]})
        else:
            # three body feature
            if feature.params["rcut"] > self.maxrcut["threebody"]:
                self.set_rcut({"threebody":feature.params["rcut"]})
        self.precondition_computed = False

    def _threebody_feature_present(self):
        if len(self.features)==0:
            return False
        present = False
        for _feature in self.features:
            if _feature.type in ['acsf_behler-g4','acsf_behler-g5','acsf_normal-b3']:
                present = True
        return present

    def bond_distribution(self,set_type="train"):
        """
        Calculate the two body and three body distributions
        
        Parameters
        ----------
        set_type : String, allowed values = ['test','train'], default = train
            The data set type
        """
        set_type = set_type.lower()
        if self.data[set_type] is None:
            raise FeaturesError("data for the data set: {} has not been set yet")

        remove_features = 0
        if len(self.features)==0:
            # create toy features
           
            self.add_feature(feature('acsf_normal-b2',{'rcut':self.maxrcut["twobody"],'fs':0.1,'za':4.1,\
                    'zb':4.2,'mean':2,'prec':3.0}))

            self.add_feature(feature("acsf_normal-b3",{'rcut':self.maxrcut["threebody"],'fs':0.1,'za':4.1,\
                    'zb':4.2,'mean':np.ones(3),'prec':np.ones((3,3))}))
            remove_features = 2
        elif self._threebody_feature_present()!=True:
            self.add_feature(feature("acsf_normal-b3",{'rcut':self.maxrcut["threebody"],'fs':0.1,'za':4.1,\
                    'zb':4.2,'mean':np.ones(3),'prec':np.ones((3,3))}))
            remove_features = 1 


        # populate fortran data structures
        self._parse_features_to_fortran()
      
        # convert buffer size (GB) to # of 64B floats
        num64_floats = int(np.floor(10e9*self.buffer_size/64.0))
       
        twobody = np.zeros((num64_floats),dtype=np.float64,order='F') 
        threebody = np.zeros((3,int(num64_floats/3.)),dtype=np.float64,order='F') 
        _sample_rate = np.asarray([self.sample_rate["twobody"],self.sample_rate["threebody"]],\
                dtype=np.float64)
                
        # calculate 2&3 body distributions
        n2,n3 = getattr(f95_api,"f90wrap_calculate_distance_distributions")(\
                set_type=self._set_map[set_type],sample_rate=_sample_rate,twobody_dist=twobody,\
                threebody_dist=threebody)

        if remove_features:
            for ii in range(remove_features):
                _ = self.features.pop()
        return np.asarray(twobody[:n2],order='C'),np.asarray(threebody.T[:n3],order='C')
  
    def calculate_precondition(self,updating_features=False):
        """
        Compute a and b : x-> a*x + b and x \in [-1,1] and set these values for
        all features in self.features
        """
        feature_list = self.calculate(set_type="train",derivatives=False,scale=False,safe=True,\
                updating_features=updating_features)
   
        xmax = np.max(feature_list,axis=1)
        xmin = np.min(feature_list,axis=1)

        if np.isclose(xmax,xmin,rtol=1e-128,atol=1e-128).any():
            warnings.warn("Feature found with possibly no support in training set: {} {}".\
                    format(xmin,xmax),Warning)
            # descriptors without support
            all_zero = np.where(np.logical_and(np.isclose(xmin,0.0,rtol=1e-256,atol=1e-256) , \
                    np.isclose(xmax,0.0,rtol=1e-256,atol=1e-256)))[0]

        else:
            all_zero = np.asarray([])
            

        for _feature in range(len(self.features)):
            if _feature in all_zero:
                self.features[_feature].precondition["times"] = 1.0
                self.features[_feature].precondition["add"] = 0.0
            else:
                self.features[_feature].precondition["times"] = 2.0/(xmax[_feature]-\
                        xmin[_feature])
                self.features[_feature].precondition["add"] = 1.0 -2.0*xmax[_feature]/\
                        (xmax[_feature]-xmin[_feature])
        self.precondition_computed = True
        del feature_list                    

    def calculate(self,set_type="train",derivatives=False,scale=False,safe=True,\
    updating_features=False):
        """
        Compute the value of all features for the given set type
        
        Parameters
        ----------
        set_type : String, default = 'train'
            The data set to calculate feature vectors for

        derivatives : Boolean, default = False
            Whether or not to calculate feature derivatives as 
            well
    
        Examples
        --------
        >>> import parsers
        >>> import nnp
        >>>
        >>> gip = parsers.GeneralInputParser()
        >>> gip.parse_all('./training_data')
        >>>
        >>> features = nnp.features.types.features(gip)
        >>>
        >>> # use automatic GMM-based feature generation
        >>> features.generate_gmm_features()
        >>>
        >>> # compute feature vectors for training set
        >>> features.calculate()
        """
        if len(self.features)==0:
            raise FeaturesError("There are no features to compute")
        elif set_type.lower() not in ['train','test','holdout']:
            raise FeaturesError("set type {} not supported".format(set_type))
        if scale and not self.precondition_computed:
            raise FeaturesError("attempting to calculate features with scaling before \
                    self.calculate_precondition() has been called") 

        set_type = set_type.lower()

        t1 = time.time()

        # parse features to fortran data structures
        self._parse_features_to_fortran()
        
        # initialise feature vector mem. and derivatives wrt. atoms
        getattr(f95_api,"f90wrap_init_feature_vectors")(init_type=self._set_map[set_type])
        
        t2 = time.time()
        
        # compute features (and their derivatives wrt. atoms) 
        getattr(f95_api,"f90wrap_calculate_features_singleset")(set_type=self._set_map[set_type],\
                derivatives=derivatives,scale_features=scale,parallel=self.parallel,\
                updating_features=updating_features)
        
        t3 = time.time()
        #print('feature comp. time : {}s parallel = {}'.format(t3-t2,self.parallel))

        if safe:
            # abort if Nan found in features or their derivatives
            getattr(f95_api,"f90wrap_check_features")(set_type=self._set_map[set_type])
            if derivatives:
                getattr(f95_api,"f90wrap_check_feature_derivatives")(\
                        set_type=self._set_map[set_type])

        # no PCA
        non_pca_features = self.get_features(set_type=set_type)

        if self.pca_type is not None:
            pca_instance = features_pca.pca_class(pca_type=self.pca_type,data=non_pca_features)

            final_features = pca_instance.perform_pca()
        else:
            final_features = non_pca_features

        # return [NxD] array of features
        return final_features

    def _parse_features_to_fortran(self):
        """
        Parse formatted list of features to fortran
        """

        tmpfile = tempfile.NamedTemporaryFile(mode='w',prefix='nnp-',suffix='.features',\
                delete=False) 
        
        for _feature in self.features:
            _feature._write_to_disk(tmpfile.file)

        tmpfile.file.close()

        # fortrans really fussy about chars
        filepath = np.asarray('{:<1024}'.format(tmpfile.name),dtype='c')

        # parse features from disk into fortran types
        getattr(f95_api,"f90wrap_init_features_from_disk")(filepath)
        
        shutil.os.remove(tmpfile.name)

    def generate_gmm_features(self,n_comp_two=None,n_comp_three=None,set_type="train"):
        """
        Generate two and three body features using a Guassian Mixture Model
        for inference of component means and precisions. 

        Automatic selection of number of components for each distribution
        is available using Bayesian GMM and a data-heavy weight prior for
        the three body distribution (drij,drik,dtheta_ijk) and use of the
        bic measure for model comparison for the two body feature 
        (drij,drik).

        Parameters
        ----------
        n_comp_two : int, default = None
            The number of components to use for the two body distribution

        n_comp_three : int, default = None
            The number of components to use for the three body 
            distribution

        set_type : String, default = 'train'
            The data set to fit the GMMs to
        
        Examples
        --------
        >>> import parsers
        >>> import nnp
        >>> training_data = parsers.GeneralInputParser()
        >>> training_data.parse_all('./training_data/')
        >>>
        >>> features = nnp.features.types.features(training_data)
        >>>
        >>> # Generate x40 two body features only from training set
        >>> features.generate_gmm_features(n_comp_two=40,n_comp_three=0)
        """

        self._generate_twobody_gmm(num_components=n_comp_two,set_type=set_type)
        self._generate_threebody_gmm(num_components=n_comp_three,set_type=set_type)

    def _generate_twobody_gmm(self,num_components=None,num_samples=1e5,set_type="train"):
        """
        Generate a Gaussian Mixture Model from twobody (distance-distance)
        distribution of given set.
        
        Parameters
        ----------
        num_components : int, default = None
            Number of components to use in GMM. If None, automatic selection
            is attempted by independant classical (EM) inferences using
            gic measure for model comparison. Bounded local minima solver 
            used to approximate true number of components for iid. data

        set_type : String, allowed values = 'train','test', default = 'train'
            Data set to fit GMM to
        """
        def _bic_fun_wrapper(_K):
            """
            retrun bic measure for model comparison for classical (EM) GMM
            using _K components
            """
            gmm = mixture.GaussianMixture(n_components=int(_K),covariance_type='full',\
                    tol=1e-3,max_iter=200)

            # do EM on mean,precision,mixing coeffs.
            gmm.fit(X=np.reshape(twobody_dist,(-1,1)))
            
            return gmm.bic(X=np.reshape(twobody_dist,(-1,1)))

        set_type = set_type.lower()
        if self.data[set_type] is None:
            raise FeaturesError("data for the data set: {} has not been set yet")
      
        automatic_selection = False 
        if num_components is None:
            automatic_selection = True
            num_components = 100 
        elif num_components == 0:
            return
        if isinstance(num_components,(int,np.int16,np.int32))!=True:
            raise FeaturesError("num_components is not an int : {}".format(type(num_components)))
       
        
        # collect atom-atom distance distribution
        twobody_dist_original,_ = self.bond_distribution(set_type=set_type)
     
        if True:
            # account for p(dr) ~ dr^2, non uniform prior 
            twobody_dist = np.zeros(int(num_samples),dtype=np.float64)
            cntr = 0
            while cntr<twobody_dist.shape[0]:
                generating_sample = True
                while generating_sample:
                    idx = np.random.choice(twobody_dist_original.shape[0],1)

                    if 1.0/(twobody_dist_original[idx]**2) > np.random.random():
                        # rejection sampling
                        generating_sample = False
                twobody_dist[cntr] = twobody_dist_original[idx]
                cntr += 1
        else:
            twobody_dist = twobody_dist_original

        if automatic_selection:
            # minimize bic measure wrt. n_components - Bayesian GMM doesn't converge 
            opt_result = optimize.minimize_scalar(fun=_bic_fun_wrapper,\
                    bounds=[1,num_components],method='bounded',options={'xatol':1.0})

            if opt_result.success!=True:
                raise FeaturesError("Optimize of n_components for classical (EM) gmm \
                        has not converged")

            num_components = int(opt_result.x)


        gmm = mixture.GaussianMixture(n_components=num_components,covariance_type='full',\
                tol=1e-3,max_iter=200)

        # do EM on mean,precision,mixing coeffs.
        gmm.fit(X=np.reshape(twobody_dist,(-1,1)))

        means = gmm.means_
        precisions = gmm.precisions_

        for _component in range(means.shape[0]):
            self.add_feature(feature(feature_type="acsf_normal-b2",\
                    params={"rcut":self.maxrcut["twobody"],\
                    "fs":0.1,"mean":means[_component,0],"prec":precisions[_component,0,0],\
                    "za":1.0,"zb":1.0}))
    
    def _generate_threebody_gmm(self,num_components=None,sample_num=1e5,set_type="train"):
        """
        Generate a Gaussian Mixture Model from three body distribution
        (drij,drik,dtheta_ijk) of given set.
        
        Parameters
        ----------
        num_components : int, default = None
            Number of components to use in GMM

        set_type : String, allowed values = 'train','test', default = 'train'
            Data set to fit GMM to
        """
        set_type = set_type.lower()
        if self.data[set_type] is None:
            raise FeaturesError("data for the data set: {} has not been set yet")
        
        automatic_selection = False
        if num_components is None:
            # attempt automatic component selection using 
            num_components = 150
            automatic_selection = True
        elif num_components == 0:
            return
        if isinstance(num_components,(int,np.int16,np.int32))!=True:
            raise FeaturesError("num_components is not an int : {}".format(type(num_components)))
       
        # initialise mixture
        gmm = mixture.BayesianGaussianMixture(n_components=num_components,covariance_type='full',\
                weight_concentration_prior=1.0/num_components)
        
        # collect atom-atom distance distribution
        _,unsymmetric_dist = self.bond_distribution(set_type=set_type)

        # distribution is only for \sum_i \sum_j \sum_{k>j}
        tmp1 = np.hstack((unsymmetric_dist[:,0],unsymmetric_dist[:,1]))
        tmp2 = np.hstack((unsymmetric_dist[:,1],unsymmetric_dist[:,0]))
        tmp3 = np.hstack((unsymmetric_dist[:,2],unsymmetric_dist[:,2]))

        # mirror plane down [:,0]=[:,1]
        symmetric_dist = np.vstack((np.vstack((tmp1,tmp2)),tmp3)).T

        # take lower diagonal
        symmetric_dist = np.asarray(list(filter(lambda x : True if x[0]>=x[1] else False, \
                symmetric_dist)))
      

        # re-sample taking into account p(dr) ~ dr^2
        random_sample = np.zeros((int(sample_num),3),dtype=np.float64)
        cntr = 0
        while cntr<random_sample.shape[0]:
            generating_sample = True
            while generating_sample:
                idx = np.random.choice(symmetric_dist.shape[0],1)
                if 1.0/(symmetric_dist[idx,0]*symmetric_dist[idx,1])**2 > np.random.random():
                    generating_sample = False
            random_sample[cntr,:] = symmetric_dist[idx,:]
            cntr += 1

        # do EM on mean,precision,mixing coeffs.
        gmm.fit(random_sample)


        means = gmm.means_
        precisions = gmm.precisions_


        if automatic_selection:
            # remove components by mixture coefficient magnitude
            
            threshold = 1e-3
            idx = np.nonzero(gmm.weights_ > np.max(gmm.weights_)*threshold)[0]

            # crop components with mixing coefficient less than threshold*max_mix_coeff.
            means = means[idx]
            precisions = precisions[idx]

            if abs(means.shape[0]-num_components)/num_components<0.1:
                # check that more than 10% of components are being pruned
                raise FeaturesError("{} of {} components selected automatically, increse initial \
                        number of components in mixture and rerun.".\
                        format(means.shape[0],num_components))

        for _component in range(means.shape[0]):
            self.add_feature(feature(feature_type="acsf_normal-b3",\
                    params={"rcut":self.maxrcut["threebody"],\
                    "fs":0.2,"mean":means[_component,:],"prec":precisions[_component,:,:],\
                    "za":1.0,"zb":1.0}))


    def get_features(self,set_type="train"):
        """
        Return features once computed in fortran
        
        Example
        -------
        >>> import parsers
        >>> import nnp
        >>>
        >>> # parse data
        >>> gip = parsers.GeneralInputParser()
        >>> gip.parse_all('./training_data/')
        >>>
        >>> features = nnp.features.types.features(gip)
        >>> # generate descriptors
        >>> features.generate_gmm_features()
        >>> features.add_feature(nnp.features.types.feature("atomic_number"))
        >>> # compute feature values
        >>> _ = features.calculate()
        >>>
        >>> # return [N,D] array of computed features
        >>> computed_features = features.get_features()
        """
        if set_type not in ["train","holdout","test"]:
            raise FeaturesError("{} not in {}".format(set_type,"train,test"))
        
        # total number of atoms in set
        #tot_num_atoms = np.sum([_s["positions"].shape[0] for _s in self.data[set_type]])
        tot_num_atoms = nnp.util.misc.total_atoms_in_set(set_type=set_type)

        # dimension of feature vector
        num_dim = len(self.features)
    
        # array for all computed features in set
        all_features = np.zeros((num_dim,tot_num_atoms),dtype=np.float64,order='F')

        getattr(f95_api,"f90wrap_get_features")(self._set_map[set_type],all_features)
        
        return np.asarray(all_features,order='C')
        
    def fit(self,X,feature_save_interval=0,only_energy=False,maxiter=None,search_scope="local",\
            verbose=False,global_maxiter=500):
        """
        Fine tune basis function parameters using PES 
        
        Parameters
        ----------
        feature_save_interval : int, default value = 0
            If not 0, write current features to disk (as in callback to feature
            loss), every feature_save_interval iterations of minimizer
        """ 
        import nnp.nn.mlpp

        # set new training data
        self.data["train"] = X

        # interval between writing features to disk during optimization
        self.feature_save_interval = feature_save_interval

        # whether or not to include forces in loss
        self.update_only_energy = only_energy

        # output verbosity
        self.fit_verbosity = verbose

        # number of iterations of CMA (global search)
        self.global_maxiter = global_maxiter

        if maxiter is None:
            # max number of bfgs iterations
            self.local_maxiter = 20000
        else:
            self.local_maxiter = maxiter
        
        if search_scope == "global":
            # do initial coarse search over features
            self._fit_global_search()
            
            self.save('best_feat_from_global_search.pckl')
        # do local optimization of NN and current basis func params 
        self._fit_local_search()

    def _fit_local_search(self):
        """
        Perform local gradient descent of NN weights and basis params
        """
        
        # write configs to disk
        self.set_configuration(gip=self.data["train"],set_type="train")

        # instance of neural net class
        self.mlpp = nnp.nn.mlpp.MultiLayerPerceptronPotential(hidden_layer_sizes=[10,10],\
                max_precision_update_number=0)
        if self.update_only_energy:
            # only consider energy term
            for _key,_value in {"energy":1.0,"forces":0.0,"regularization":0.0}.items():
                self.mlpp.set_hyperparams(key=_key,value=_value)
        else:
            # include forces too
            for _key,_value in {"energy":1.0,"forces":1e-3,"regularization":0.0}.items():
                self.mlpp.set_hyperparams(key=_key,value=_value)
        
        # for forward prop to initialise weights
        self.mlpp.set_features(features=self)
        
        # compute pre conditioning and initialise net
        self.mlpp._prepare_data_structures(X=self.data["train"],set_type="train",\
                derivatives=False,updating_features=True)
      
        # compute initial weights and concacenate weights with feature params 
        x0 = self._init_concacenation()
       
        #if not only_energy:
        #    # use automatic scaling of energy and force noise precision
        #    self.mlpp._update_precision()
        
        # log for loss with optimization
        self._loss_log = []

        # do optimization 
        self.OptimizeResult = optimize.minimize(fun=self._feature_loss,\
                jac=self._feature_loss_jacobian,x0=x0,\
                method='l-bfgs-b',options={"gtol":1e-8,"maxiter":self.local_maxiter},\
                bounds=self.concacenated_bounds,callback=self._feature_opt_callback)
   
        # write final parameters to feature class instances 
        self._parse_param_array_to_class(parameters=self.OptimizeResult["x"])

        # recompute scaling constants for preconditioning
        self.calculate_precondition()

    def _fit_global_search(self):
        """
        Perform coarse global optimization of features. For each feature set,
        do a coarse minimization of NN weights, returning minimum loss as 
        objective function
        """
        # necessary for parameter concacenation
        self.mlpp = toy_mlpp_class()
     
        # get initial basis parameters
        x0 = self._concacenate_parameters()

        # summary
        opt_res = nnp.optimizers.stochastic.minimize(\
                fun=self._objective_function_coarse_search,jac=None,\
                x0=x0,solver="cma",**{"max_iter":self.global_maxiter,\
                "bounds":self.concacenated_bounds,\
                "sigma":1.0,"verbose":self.fit_verbosity})

        # store best indivual as current basis params
        self._parse_param_array_to_class(opt_res["x"])

    def _feature_loss(self,parameters):
        """
        Objective function for feature parameters regression - energy squared 
        error only.

        Parameters
        ----------
        : np.ndarray
        of neural net weights and basis function parameters

        Returns
        -------
        Energy squared error : float
        """
        # parse new params to feature instances
        self._parse_param_array_to_class(parameters)
        
        if np.isclose(self.mlpp.hyper_params["forces"],0.0,1e-20,1e-20):
            force_derivatives = False
        else:
            force_derivatives = True

        # write new features to fortran and compute feature values (no derivs)
        self.calculate(set_type="train",derivatives=force_derivatives,scale=True,safe=True,\
                updating_features=True)
    
        loss = self.mlpp._loss(weights=parameters[:self.mlpp.num_weights],set_type="train",\
                log_loss=False)

        if np.isnan(loss) or np.isinf(loss):
            raise FeaturesError("Nan or Inf returned from fortran loss")

        # keep loss during optimization
        self._loss_log.append(loss)
        
        return loss

    def _feature_loss_jacobian(self,parameters):
        """
        Jacobian of energy squared error with respect to neural net weights and
        basis function parameters

        Parameters
        ----------
        parameters : np.ndarray
            Concacenation of neural net weights and basis function parameters
        """
        import nnp.nn.fortran.nn_f95 as f95_api 

        # parse features to Python format
        self._parse_param_array_to_class(parameters)

        # write new features to fortran and compute X for loss jacobian
        #self.calculate(set_type="train",derivatives=False,scale=True,safe=True,\
        #        updating_features=True)

        # check not trying to use forces or regularization
        for _hyperparam in ['forces','regularization']:
            if not np.isclose(self.mlpp.hyper_params[_hyperparam],0.0):
                if _hyperparam == 'regularization':
                    raise FeaturesError("Basis function parameter optimization only supported for \
                            energy squared error, not forces or regularization")
   
        net_weights = parameters[:self.mlpp.num_weights]
        
        # neural net weights
        nn_weight_jac = self.mlpp._loss_jacobian(weights=net_weights,set_type="train")

        if np.isnan(nn_weight_jac).any() or np.isinf(nn_weight_jac).any():
            raise FeaturesError("Nan or Inf raised in loss jacobian wrt. net weights")

        # basis function parameter jacobian
        basis_func_jac = np.zeros(parameters.shape[0]-self.mlpp.num_weights,dtype=np.float64)

        getattr(f95_api,"f90wrap_loss_feature_jacobian")(flat_weights=net_weights,\
                set_type=self._set_map["train"],parallel=self.parallel,\
                scale_features=self.scale_features,jacobian=basis_func_jac) 
       
        if np.isnan(basis_func_jac).any() or np.isinf(basis_func_jac).any():
            raise FeaturesError("Nan or Inf raised in loss jacobian wrt. basis func. params")
        
        return np.hstack((nn_weight_jac,basis_func_jac))

    def _feature_opt_callback(self,parameters):
        """
        Function called during SCIPY optimization, returning True will 
        terminate optimization
        """
        terminate_opt = False

        iter_num = len(self._loss_log)

        if self.feature_save_interval > 0:
            # write features to disk
            if np.mod(iter_num,self.feature_save_interval)==0:
                self.save('feature_opt-{}'.format(iter_num))

        return terminate_opt
    
    def _parse_param_array_to_class(self,parameters):
        """
        Parse 1d array of parameters to feature instance parameters

        Parameters
        ----------
        parameters : np.ndarray
            Concacenation of neural net weights and feature parameters
        """

        for _ft_idx in range(len(self.features)):
            for _key in self.features[_ft_idx].update_keys:
                # updatable attributes for given feature

                # location of attribute in parameters np.ndarray
                (id_start,id_end) = self._parse_idx_map[(_ft_idx,_key)]

                attribute_value = self._parse_attributes_from_array(_key,\
                        parameters[id_start:id_end])

                # format used to write to fortran
                self.features[_ft_idx].set_param(_key,attribute_value)

    def _concacenate_parameters(self):
        """
        Concacenate neural net weights with current feature params for feture
        class instances

        Returns
        -------
        weights and feature params concacenated, np.ndarray, shape = (N,)
        """

        # need constrained minimization for some basis func. params
        self.concacenated_bounds = [[None,None] for ii in range(self.mlpp.weights.shape[0])]

        offset = self.mlpp.weights.shape[0]
        
        x0 = list(self.mlpp.weights)

        # generate mapping from feature idx and attribute key to array indices
        self._parse_idx_map = {}

        for _ft_idx in range(len(self.features)):
            _ftype = self.features[_ft_idx].type

            for _key in self.features[_ft_idx].update_keys:
                # iterate over attributes that are updatable
                
                #if _key not in ['mean','prec','weights'] or _ftype != "acsf_normal-b3":
                if _ftype not in ["acsf_normal-b3","acsf_fourier-b2"]:
                    length_of_param = 1
                    _value = [self.features[_ft_idx].params[_key]]
                elif _key == "mean":
                    length_of_param = 3
                    _value = self.features[_ft_idx].params[_key]
                elif _key == "prec":
                    length_of_param = len(np.triu_indices(3,0)[0])
                    
                    idx = np.triu_indices(3,0)
                    _value = self.features[_ft_idx].params[_key][idx] 
                elif _key == "weights":
                    length_of_param = self.features[_ft_idx].params[_key].shape[0]
                    _value = self.features[_ft_idx].params[_key]
                else:
                    raise FeaturesError("Implementation error for {}".format(_ftype))

                # fetch bounds for basis func param type 
                self.concacenated_bounds += self.features[_ft_idx].get_bounds(_key)
                
                # append params to list
                x0 += list(_value)
                
                # store location of feature attribute in params list
                self._parse_idx_map.update({(_ft_idx,_key):(offset,offset+length_of_param)})

                # book keeping
                offset = len(x0)

        if len(self.concacenated_bounds)!=len(x0):
            raise FeaturesError("shape mismatch between concatenated parameters and bounds {}!= {}"\
                    .format(len(self.concacenated_bounds),len(x0)))
        
        return np.asarray(x0)

    def _init_concacenation(self):
        """
        Initialise net weights and return concacenated array of neural net 
        weights and appropriate feature parameters
        """
        # forward prop to get initial weights
        self.mlpp._init_random_weights()

        # parse feature class instances to 1d array with weights
        return self._concacenate_parameters()

    def _parse_attributes_from_array(self,key,value):
        """ 
        Format attributes from parameters array, to feature instances. For all
        floats this is simply returning the correct element of the 1d 
        concacenation of neural net and feature parameters. 

        For gaussian feature means, this involes casting specified elements as 
        a np.ndarray. For precision matrices, this involes forming a symmetric
        matrix from the upper triangular elements given 
        """
        if key in ['prec'] and len(value)!=1:
            # have upper diagonal of a 3x3 matrix (inclusive of diagonals)
           
            # indices of upper triangular elements for 3x3 matrix (include diag) 
            idx = np.triu_indices(3,0)

            if len(idx[0]) != len(value):
                raise FeaturesError("Error passing upper triangular matrix elements {}".\
                        format(key))
            
            symmetric_matrix = np.zeros((3,3),dtype=np.float64)
            
            # upper triangular (including diagonal)
            symmetric_matrix[idx] = value

            # lower triangular indices (not including diagonals)
            idx = np.tril_indices(3,-1)
            symmetric_matrix[idx] = symmetric_matrix.T[idx]
            
            return symmetric_matrix
        elif key in ['prec','mean'] and len(value)!=1:
            return np.asarray(value)
        elif key in ["weights"]:
            return np.asarray(value)
        else:
            # we pass in an array slice
            return value[0]
    
    def _objective_function_coarse_search(self,basis_params,args=()):
        """
        For a given set of basis function parameters, do a quick optimization 
        of neural net weights and return the minimum loss

        Parameters
        ----------
        basis_params : np.ndarray
            A concacenated array of basis function parameters
        """
        # parse new params to feature instances
        self._parse_param_array_to_class(basis_params)
        
        mlpp = nnp.mlpp.MultiLayerPerceptronPotential(hidden_layer_sizes=[10,10],\
                activation='sigmoid',precision_update_interval=100,\
                max_precision_update_number=0)
        mlpp.set_features(self)
        for _key,_value in {"energy":1.0,"forces":1.0,"regularization":0.0}.items():
            mlpp.set_hyperparams(_key,_value)
        t1 = time.time()
        loss,_ = mlpp.fit(self.data["train"])
        t2 = time.time()
        #print('{} steps in {}s with L={}'.format(len(mlpp._loss_log),t2-t1,loss))
        return loss

    def save(self,sysname=None):
        """
        Save self.features to a pckl file sysname.features
        
        Parameters
        ----------
        sysname : String, default value = year-month-day.features
            Save self.features to sysname.features

        Examples
        --------
        >>> import nnp
        >>> import parsers
        >>> gip = parsers.GeneralInputParser()
        >>> gip.parse_all('./training_set-graphene/')
        >>> features = nnp.features.types.features(gip)
        >>>
        >>> # this may take some time
        >>> features.generate_gmm_features()
        >>> 
        >>> # save generated list of nnp.features.types.feature() instances
        >>> features.save('graphene')
        """
        import datetime
        import pickle

        if sysname is None:
            sysname = datetime.datetime.today().split()[0]

        with open(sysname+'.features','wb') as f:
            pickle.dump({'features':self.features},f)
        f.close()

    def load(self,sysname=None):
        """
        Load a previously computed set of descriptors from pckl file

        Parameters
        ----------
        sysname : String, default value = None
            Load sysname.features as a dictionary with key 'features', 
            attributing this to self.features. If sysname=None, any .features 
            files present in the working directory will be read. 
        
        Examples
        --------
        >>> import nnp
        >>> import parsers
        >>> gip = parsers.GeneralInputParser()
        >>> gip.parse_all('./training_set-graphene/')
        >>>
        >>> features = nnp.features.types.features()
        >>>
        >>> # load features from previous run
        >>> features.load(sysname='graphene')
        """
        import pickle

        if sysname is None:
            files = listdir('.')

            pckl_files = [_file.split('.')[-1]=='features' for _file in files]

            if np.sum(pckl_files)!=1:
                raise FeaturesError("Could not automatically load .features file")         

            sysname = '.'.join(files[np.nonzero(pckl_files)[0][0]].split('.')[:-1])

        with open(sysname+'.features','rb') as f:
            try:
                self.features = pickle.load(f)['features']
            except KeyError or TypeError:
                raise FeaturesError("features pckl file {} appears to be corrupted".\
                        format(sysname+'.features'))
        f.close()

class toy_mlpp_class():
    """
    Necessary to trick parameter concacenation into thinking this is an instance
    of mlpp.MultiLayerPerceptronPotential()
    """
    def __init__(self):
        self.weights = np.zeros(0)

class FeatureError(Exception):
    pass
class FeaturesError(Exception):
    pass
