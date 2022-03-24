import numpy as np
from numpy.linalg import pinv, eig
import pandas as pd

def compute_mashup_updated(As, reduced_dim = 1000, const_param=2):
    """
    Returns the MASHUP embedding. As is a list of adjacency matrices with the same dimension; obtained using the function As[0].shape
    """

    n, _ = As[0].shape
    
    def A_to_Q(A):
        e = np.ones((n, 1))
        d = A @ e
        return A / d
    
    R_f  = np.zeros((n,n))
    for A in As:
        Q = A_to_Q(A)
        R = np.log(Q + const_param / n)
        R_f += R @ R.T
    e, W = eig(R_f)
    # sort in descending order
    ids  = np.argsort(-e)
    ids  = ids.astype(int)
    print(ids)
    e    = e[ids[:reduced_dim]]
    W    = W[:, ids[:reduced_dim]]
    e    = np.diag(np.sqrt(np.sqrt(e)))
    return W @ e.T


def compute_mashup_dsd(Ds, reduced_dim = 1000):
    """
    Returns the MASHUP embeddings on the distance matrices.
    """

    def normalize_dist(D):
        """
        Normalizing the distance matrix: M^{-1/2}D M^{-1/2}
        """
        dim, _ = D.shape
        d      = D @ np.ones((dim, 1))
        d_sqrt = np.sqrt(d)
        return ((D / d_sqrt) / d_sqrt.T)

    n, _ = Ds[0].shape
    R = np.zeros((n, n))
    for D in Ds:
        R += normalize_dist(D)
    
    e, W = eig(R)
    # sort in descending order
    ids  = np.argsort(-e)
    ids  = ids.astype(int)
    print(ids)
    e    = e[ids[:reduced_dim]]
    W    = W[:, ids[:reduced_dim]]
    e    = np.diag(np.sqrt(np.sqrt(e)))
    return W @ e.T
