"""
Python types for feature information
"""
import nnp.util.io

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
    def __init__(self,feature_type):
        self.supp_types = ['atomic_number',
                           'acsf_behler-g1',
                           'acsf_behler-g2',
                           'acsf_behler-g4',
                           'acsf_behler-g5',
                           'acsf_normal-b2',
                           'acsf_normal-b3']
        
        self.type = feature_type

        if self.type not in self.supp_types:
            raise FeatureError('{} not in {}'.format(self.type,','.join(self.supp_types)))


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

    def set_configuration(self,gip,set_type):
        """
        Parse configurations into fortran data structure
        
        Parameters
        ----------
        gip : parsers.GeneralInputParser
            A data structure with all configurations that are to be stored in fortran

        set_type : String, allowed values = ['train','test']
            The data set type 
        """

        if set_type.lower() not in ['test','train']:
            raise FeaturesError('{} not in {}'.format(set_type,'train,test'))

        nnp.util.io._parse_configs_to_fortran(gip,set_type.lower())
        self.data[set_type.lower] = gip

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

        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError


class FeatureError(Exception):
    pass
class FeaturesError(Exception):
    pass
