"""
==================
 Sparse filtering
==================

Scikit-learn compativle Python port of the sparse filtering matlab code by Jiquan Ngiam.

Requires numpy and scipy installed.

Based on Marting Singh-Blom code: https://github.com/martinblom/py-sparse-filtering
"""

import numpy as np
from sklearn import base
from scipy.optimize import minimize
import sparseFiltering

def l2row(X):
	"""
	L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
	"""
	N = np.sqrt((X**2).sum(axis=1)+1e-8)
	Y = (X.T/N).T
	return Y,N


def l2rowg(X,Y,N,D):
	"""
	Backpropagate through Normalization.

	Parameters
	----------

	X = Raw (possibly centered) data.
	Y = Row normalized data.
	N = Norms of rows.
	D = Deltas of previous layer. Used to compute gradient.

	Returns
	-------

	L2 normalized gradient.
	"""
	return (D.T/N - Y.T * (D*X).sum(axis=1) / N**2).T

def SparseFilteringObj(W, N, X):
    # Feed forward
    W = W.reshape((N,X.shape[0]))
    F = W.dot(X)
    Fs = np.sqrt(F**2 + 1e-8)
    NFs, L2Fs = l2row(Fs)
    Fhat, L2Fn = l2row(NFs.T)
    
    # Compute objective function
    Obj = Fhat.sum()

    # Backprop through each feedforward step
    DeltaW = l2rowg(NFs.T, Fhat, L2Fn, np.ones(Fhat.shape))
    DeltaW = l2rowg(Fs, NFs, L2Fs, DeltaW.T)
    DeltaW = (DeltaW*(F/Fs)).dot(X.T)
    return Fhat.sum(), DeltaW.flatten()

class SparseFilter(base.BaseEstimator, base.TransformerMixin):
	""" Sparse Filtering Algorithm

    Parameters
    ----------

	X : input data (examples in row, which is the oposite of what would we do in 
    Ngiam's matlab code)
    
    N : number of features to be extracted by the filter 
    maxiter : (integer) maximum number of iterations of L-BFGS-B algorithm.
    
    """
        def __init__(self, N=256, maxiter=200):
            self.W = 0.
            self.N = N
            self.maxiter = maxiter

        def fit(self, X, y):
            X = X.T
            self.W = np.random.randn(self.N, X.shape[0])
            w,g = SparseFilteringObj(self.W, self.N, X)
            res = minimize(SparseFilteringObj, self.W, args=(self.N, X), \
                    method='L-BFGS-B', jac = True, \
                    options = {'maxiter':self.maxiter})
            self.W = res.x.reshape(self.N, X.shape[0])
            return self


        def transform(self, X):
            X = X.T
            return self.feedForwardSF(X)

        def feedForwardSF(self,X):
            "Feed-forward"
            F = self.W.dot(X)
            Fs = np.sqrt(F**2 + 1e-8)
            NFs = l2row(Fs)[0]
            return l2row(NFs.T)[0]#.T
