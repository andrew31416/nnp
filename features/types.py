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
    >>> _feature = nnp.features.types.feature('atomic_number')
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

    def bond_distribution(self,set_type):
        """
        Calculate the two body and three body distributions
        
        Parameters
        ----------
        set_type : String, allowed values = ['test','train']
            The data set type
        """
        set_type = set_type.lower()
        if self.data[set_type] is None:
            raise FeaturesError("data for the data set: {} has not been set yet")

        if len(self.features)==0:
            # create toy features
           
            self.add_feature(feature('acsf_normal-b2',{'rcut':self.maxrcut["twobody"],'fs':0.1,'za':4.1,\
                    'zb':4.2,'mean':2,'prec':3.0}))

            self.add_feature(feature("acsf_normal-b3",{'rcut':self.maxrcut["threebody"],'fs':0.1,'za':4.1,\
                    'zb':4.2,'mean':np.ones(3),'prec':np.ones((3,3))}))

        print(self.features)
        self._parse_features_to_fortran()


    def calculate(self):
        raise NotImplementedError

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

class FeatureError(Exception):
    pass
class FeaturesError(Exception):
    pass
