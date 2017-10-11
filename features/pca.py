"""
routines for performing principal component analysis on 
atomic environment features
"""
import numpy as np

def perform_pca(X,q):
    """
    Input
    -----
        - X : X.shape = (N,D)
        - N : number of data points
        - D : original number of dimensions
        - q : final number of dimensions
    """

    (N,D) = X.shape


    tmp = (X - np.tile(np.average(X,axis=0),(N,1))) 

    # sample covariance matrix
    cov_matrix = np.dot(tmp.T,tmp) / float(N)

    # find eigen values and vectors
    eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)

    # sort into ascending list of eigen values
    idx = np.argsort(eigen_values)[::-1]

    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[idx]

    # principal projections
    W_transpose = np.asarray(eigen_vectors[0,:])

    for ii in range(1,q):
        W_transpose = np.vstack((W_transpose,eigen_vectors[ii,:]))
    W = W_transpose.T

    principal_X = np.dot(W_transpose,X.T).T 
    
    return principal_X,W


def project_into_pc(X,W)
    """
    project the data points X into their principal components
    """

    return np.dot(W.T,X.T).T
