"""
Python types for feature information
"""
import nnp.util.io
import nnp.nn.fortran.nn_f95 as f95_api
import numpy as np
import tempfile
import shutil
from copy import deepcopy
from io import TextIOWrapper
from os import listdir
from sklearn import mixture
from scipy import optimize


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
                           'acsf_normal-b3']
        
        self.type = feature_type
        self.params = None

        if self.type not in self.supp_types:
            raise FeatureError('{} not in {}'.format(self.type,','.join(self.supp_types)))

        if params is not None:
            self.set_params(params)

    def set_params(self,params):
        """
        check parameters for correct args and then set

        Parameters
        ----------
        params : dict
            Key,value pairs of parameters for specific feature type
        """
        if isinstance(params,dict)!=True:
            raise FeatureError("{} is not dict".format(type(params)))

        _ftype = (int,float,np.float32,np.float64)
        _atype = (list,np.ndarray,int,float,np.float32,np.float64)
        _types = {'rcut':_ftype,'fs':_ftype,'eta':_ftype,'za':_ftype,\
                'zb':_ftype,'xi':_ftype,'lambda':_ftype,'prec':_atype,\
                'mean':_atype,'rs':_ftype}

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
            
        if set(params.keys())!=set(_keys):
            # check keys
            raise FeatureError("supplied keys {} != {}".format(params.keys(),_keys))
        for _attr in params:
            # check param type
            if type(params[_attr]) not in _types[_attr]:
                raise FeatureError("param type {} != {}".format(type(params[_attr]),_types[_attr]))
        if self.type in ["acsf_normal-b3"]:
            # check length of arrays
            if self.type == "acsf_normal-b2":
                _length = {"mean":1,"prec":1}
            elif self.type == "acsf_normal-b3":
                _length = {"mean":3,"prec":9}
            for _attr in ["mean","prec"]:
                params[_attr] = np.asarray(params[_attr],dtype=np.float64)
                if params[_attr].flatten().shape[0] != _length[_attr]:
                    raise FeatureError("arrays lengths are not as expected")

            try:
                # check that matrix is symmetric
                np.testing.assert_array_almost_equal(params["prec"],params["prec"].T)
            except AssertionError:
                raise FeatureError("Supplied precision matrix for threebody gaussian is not symmetric")

        self.params = deepcopy(params)

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
            file_object.write('{}\n'.format(self.type))
        elif self.type == 'acsf_behler-g1':
            file_object.write('{} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','za','zb']])))
        elif self.type == 'acsf_behler-g2':
            file_object.write('{} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','eta','rs','za','zb']])))
        elif self.type in ['acsf_behler-g4','acsf_behler-g5']:
            file_object.write('{} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','xi','lambda','eta',\
                    'za','zb']])))
        elif self.type == 'acsf_normal-b2':
            file_object.write('{} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','za','zb',\
                    'mean','prec']])))
        elif self.type == 'acsf_normal-b3':
            file_object.write('{} {} {} {}\n'.format(self.type,\
                    ' '.join(['{:<20}'.format(self.params[_a]) for _a in ['rcut','fs','za','zb']]),\
                    ' '.join(['{:<20}'.format(_m) for _m in self.params["mean"].flatten()]),\
                    ' '.join(['{:<20}'.format(_m) for _m in self.params["prec"].flatten()])))
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

    def __init__(self,train_data):
        # atomic configuraitons
        self.data = {"train":train_data,"test":None}

        # maximum cut off radii for two and three body terms
        self.maxrcut = {'twobody':6.0,'threebody':4.0}

        # fraction of bonds sampled for bond distribution
        self.sample_rate = {'twobody':0.5,'threebody':0.1}

        # parse data into fortran structure
        self.set_configuration(gip=self.data["train"],set_type="train")

        self.features = []

        # internal fortran mem. buffer size (GB)
        self.buffer_size = 0.1

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

        if set_type.lower() not in ['test','train']:
            raise FeaturesError('{} not in {}'.format(set_type,'train,test'))

        nnp.util.io._parse_configs_to_fortran(gip,set_type.lower())
        self.data[set_type.lower] = gip

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
        elif feature.type in ['acsf_behler-g1','acsf_behler-g2','acsf_normal-b2']:
            # two body feature
            if feature.params["rcut"] > self.maxrcut["twobody"]:
                self.set_rcut({"twobody":feature.params["rcut"]})
        else:
            # three body feature
            if feature.params["rcut"] > self.maxrcut["threebody"]:
                self.set_rcut({"threebody":feature.params["rcut"]})

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
        _sample_rate = np.asarray([self.sample_rate["twobody"],self.sample_rate["threebody"]],dtype=np.float64)
        
        _map = {"train":1,"test":2}
        # calculate 2&3 body distributions
        n2,n3 = getattr(f95_api,"f90wrap_calculate_distance_distributions")(set_type=_map[set_type],\
                sample_rate=_sample_rate,twobody_dist=twobody,threebody_dist=threebody)

        if remove_features:
            for ii in range(remove_features):
                _ = self.features.pop()
        return np.asarray(twobody[:n2],order='C'),np.asarray(threebody.T[:n3],order='C')
   
    def calculate(self,set_type="train",derivatives=False,scale=False):
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
        elif set_type.lower() not in ['train','test']:
            raise FeaturesError("set type {} not supported".format(set_type))

        set_type = set_type.lower()

        # parse features to fortran data structures
        self._parse_features_to_fortran()

        _map = {"train":1,"test":2}
        
        # initialise feature vector mem. and derivatives wrt. atoms
        getattr(f95_api,"f90wrap_init_feature_vectors")(init_type=_map[set_type])
        
        # compute features (and their derivatives wrt. atoms) 
        getattr(f95_api,"f90wrap_calculate_features_singleset")(set_type=_map[set_type],\
                derivatives=derivatives,scale_features=scale)

        # return [NxD] array of features
        return self.get_features(set_type=set_type)

    def _parse_features_to_fortran(self):
        """
        Parse formatted list of features to fortran
        """

        tmpfile = tempfile.NamedTemporaryFile(mode='w',prefix='nnp-',suffix='.features',delete=False) 
        
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

    def _generate_twobody_gmm(self,num_components=None,set_type="train"):
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
        twobody_dist,_ = self.bond_distribution(set_type=set_type)
       
        if automatic_selection:
            # minimize bic measure wrt. n_components - Bayesian GMM doesn't converge 
            opt_result = optimize.minimize_scalar(fun=_bic_fun_wrapper,bounds=[1,num_components],\
                    method='bounded',options={'xatol':1.0})

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
                    "fs":0.2,"mean":means[_component,0],"prec":precisions[_component,0,0],\
                    "za":1.0,"zb":1.0}))
    
    def _generate_threebody_gmm(self,num_components=None,set_type="train"):
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
        symmetric_dist = np.asarray(list(filter(lambda x : True if x[0]>=x[1] else False, symmetric_dist)))
      

        # do EM on mean,precision,mixing coeffs.
        gmm.fit(symmetric_dist)


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
        if set_type not in ["train","test"]:
            raise FeaturesError("{} not in {}".format(set_type,"train,test"))
        
        # total number of atoms in set
        tot_num_atoms = np.sum([_s["positions"].shape[0] for _s in self.data[set_type]])

        # dimension of feature vector
        num_dim = len(self.features)
    
        # array for all computed features in set
        all_features = np.zeros((num_dim,tot_num_atoms),dtype=np.float64,order='F')

        _map = {"train":1,"test":2}
        getattr(f95_api,"f90wrap_get_features")(_map[set_type],all_features)
        
        return np.asarray(all_features.T,order='C')

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


class FeatureError(Exception):
    pass
class FeaturesError(Exception):
    pass
