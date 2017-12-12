from scipy.optimize import OptimizeResult

class NnpOptimizeResult(OptimizeResult):
    """
    Inherit OptimizeResult to add objective and jacobian function values with
    time

    Additional Attributes
    ---------------------
    fun_all : np.ndarray
        A np array of the objective function with optimisation iteration
    """

    def __init__(self,result):
        for _key in result.keys():
            setattr(self,_key,result.get(_key)) 

