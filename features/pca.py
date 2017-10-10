"""
routines for performing principal component analysis on 
atomic environment features
"""

def perform_pca(X,q)
    """
    Input
    -----
        - X : X.shape = (N,D)
        - N : number of data points
        - D : original number of dimensions
        - q : final number of dimensions
    """

    (N,D) = X.shape


    tmp = (X - np.tile(np.average(X,axis=0),(D,1))) 

    # covariance matrix
    cov_matrix = np.inner(tmp,tmp) / float(N)

    # find eigen values and vectors
    eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)

    # sort into ascending list of eigen values
    idx = np.argsort(eigen_values)[::-1]

    for ii in range(q):
        print(eigen_values[ii])
