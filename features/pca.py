class pca_class():
    def __init__(data):
        self.set_data(data)

    def set_data(self,data):
        if not isinstance(data,np.ndarray):
            raise PCAError("data type {} not supported".format(type(data)))
        self.data = data

    def perform_pca(self):
        """
        Do
        """

class PCAError(Exception):
    pass        
