import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class pca_class():
    """
    Class to perform PCA on data

    Parameters
    ----------
    pca_type : String, allowed values = 'linear'
        Type of PCA to perform

    data : [NxD] np.ndarray
        N data points of D dimension

    Attributes
    ----------
    pca_type : String

    data : np.ndarrary

    Examples
    --------
    >>> import parsers
    >>> import nnp
    >>> # read data
    >>> gip = parsers.GeneralInputParser()
    >>> gip.parse_all('./training_data')
    >>> # initialise features class 
    >>> features = nnp.features.types.features(gip,pca_type=None)
    >>> features.generate_gmm_features()
    >>> # compute GMM features
    >>> computed_features = features.calculate()
    >>>
    >>> # perform pca on computed features
    >>> reduced_features = nnp.features.pca.pca_class(pca_type='linear',\
    >>>         data=computed_features) 
    """
    
    def __init__(self,pca_type,data):
        # number of bins to use for eigen val. distr.
        self._nbins = 40
       
        # minimum fractional difference for convergence of distr. fit
        self._ftol = 1e-4
        
        # initialise data
        self.set_pca_type(pca_type)
        self.set_data(data) 

        self.perform_pca()

    def set_pca_type(self,pca_type):    
        """
        Set type of PCA to use
        """ 
        self.supp_types = ['linear']

        if pca_type.lower() not in self.supp_types:
            raise PCAError("{} not a supported PCA type : {}".format(pca_type,self.supp_types))
            
        self.pca_type = pca_type.lower()

    def set_data(self,data):
        """
        Set data for pca class
        """
        if not isinstance(data,np.ndarray):
            raise PCAError("data of type {} not supported".format(type(data)))
        self.data = data

    def perform_pca(self):
        """
        Perform PCA for specified type
        """

        if self.pca_type == 'linear':
            print('computing covariance matrix...')
            self._calculate_covariance_matrix()

            print('diagonalizing {}...'.format(self.data.shape))
            self._diagonalize_covariance()

            # find optimal Q,sigma value for Marcenko Pastur distribution
            popt = self._fit_MarcenkoPastur()

            # max eigenvalue of random distribution
            _,randmax = self._MarcenkoPastur_minmax_eigenvalues(*popt)

            # indices of nonrandom eigenvectors
            nonrandom = np.where(self._eigenvalues>randmax)[0]
            print('ids',nonrandom)
            print('max eig',randmax)
            print('eigs',self._eigenvalues)
            W_transpose = np.asarray(self._eigenvalues[nonrandom[0],:])

            for _component in nonrandom[1:]:
                W_tranpose = np.vstack((W_transpose,self._eigenvalues[_component,:]))
            
            # new_ni = \sum_j^D W_{ij} old_nj        
            new_projection = np.dot(self.data - np.tile(self._sample_mean,(self.data.shape[0],1)) ,\
                    W_transpose)
                
            return new_projection

    def _calculate_covariance_matrix(self):
        """
        Compute the covariance matrix of self.data
        """
        self._sample_mean = np.average(self.data,axis=0)
        H = self.data - np.tile(self._sample_mean,(self.data.shape[0],1))
        self._covariance_matrix = np.dot(H.T,H)/float(self.data.shape[0])

    def _diagonalize_covariance(self):
        eigenvalues,eigenvectors = np.linalg.eig(self._covariance_matrix)

        # sort by magnitude descending
        idx = np.argsort(eigenvalues)[::-1]

        self._eigenvalues = eigenvalues[idx]
        self._eigenvectors = eigenvectors.T[idx]

    def _fit_MarcenkoPastur(self):
        """
        Compute Q,sigma free params to Marcenko Pastur distribution for sample
        eigvenalues distribution
        """

        hist,edges = np.histogram(self._eigenvalues,bins=self._nbins,normed=True)
        centres = np.asarray([0.5*(edges[ii]+edges[ii+1]) for ii in range(len(hist))])

        plt.plot(centres,hist)
        plt.show() 
        for _ftol in np.logspace(-10,np.log10(self._ftol),5):
            # fit Q,sigma values
            try:
                popt,pcov = curve_fit(f=self._MarcenkoPastur_distribution,xdata=centres,ydata=hist,\
                        maxfev=1000,ftol=_ftol,method='trf',bounds=([1,0],[np.inf,np.inf]))
                fail = False
                break
            except RuntimeError:
                fail = True

        if fail:
            raise PCAError("optimisation of Marcenko Pastur distribution params has failed")

        return popt

    def _MarcenkoPastur_minmax_eigenvalues(self,Q,sigma):
        eig_max = (sigma*(1+np.sqrt(1.0/Q)))**2
        eig_min = (sigma*(1-np.sqrt(1.0/Q)))**2
        return eig_min,eig_max
    
    def _MarcenkoPastur_distribution(self,eigenvalue,Q,sigma):
        eig_min,eig_max = self._MarcenkoPastur_minmax_eigenvalues(Q=Q,sigma=sigma)

        nonzero_idx = np.nonzero( np.logical_and(eigenvalue >= eig_min,\
                eigenvalue <= eig_max)  )[0] 

        val = np.zeros(eigenvalue.shape[0],dtype=np.float64)
        
        val[nonzero_idx] = Q/(2.0*np.pi*sigma**2 * eigenvalue[nonzero_idx]) * \
                np.sqrt( (eig_max-eigenvalue[nonzero_idx])*(-eig_min+eigenvalue[nonzero_idx]) )
                
        return val                


class PCAError(Exception):
    pass 
