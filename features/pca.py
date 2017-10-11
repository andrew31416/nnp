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


    print('shape of average : {}'.format(np.average(X,axis=0).shape))

    tmp = (X - np.tile(np.average(X,axis=0),(N,1))) 

    print('shape of tmp : {}'.format(tmp.shape))

    # sample covariance matrix
    cov_matrix = np.dot(tmp.T,tmp) / float(N)

    # find eigen values and vectors
    eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)

    # sort into ascending list of eigen values
    idx = np.argsort(eigen_values)[::-1]

    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors.T[idx]

    # principal projections
    W_transpose = np.asarray(eigen_vectors[0,:])

    for ii in range(1,q):
        W_transpose = np.vstack((W_transpose,eigen_vectors[ii,:]))
    W = W_transpose.T

    principal_X = np.dot(W_transpose,tmp.T).T 

    #if True:
    #    tmp = (principal_X - np.tile(np.average(principal_X,axis=0),(N,1))) 
    #    
    #    new_cov = np.dot(tmp.T,tmp)/float(N)
    #    
    #    for ii in range(new_cov.shape[0]):
    #        print(new_cov[ii,:])
    #    print('\neigenvalues : ')
    #    for ii in range(q):
    #        print(eigen_values[ii])
    return principal_X,W,np.average(X,axis=0)


def project_into_pc(X,W,average_x):
    """
    project the data points X into their principal components
    """
  
    tmp = X - np.tile(average_x,(X.shape[0],1)) 

    return np.dot(W.T,tmp.T).T
