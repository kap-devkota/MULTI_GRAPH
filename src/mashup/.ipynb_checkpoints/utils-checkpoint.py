import numpy as np
from numpy.linalg import pinv, eig
import pandas as pd

def compute_mashup(As, reduced_dim = 1000, restart_prob = 0.5):
    """
    Returns the MASHUP embedding. As is a list of adjacency matrices with the same dimension; obtained using the function As[0].shape
    """
    def compute_rwr(P, restart_prob = 0.5):
        """
        Computing RWR matrix.
        """
        n, _ = P.shape
        return pinv(np.identity(n) - restart_prob * P) * (1 - restart_prob)
    
    def compute_p(A):
        """
        Computing markov matrix.
        """
        n, _ = A.shape
        e = np.ones((n, 1))
        d = A @ e
        return A / d

    n, _ = As[0].shape
    print(n)
    R_f  = np.zeros((n,n))
    for A in As:
        Q    = compute_rwr(compute_p(A), restart_prob = restart_prob)
        R    = np.log(Q + 1 / n)
        R_f += R @ R.T
        
    e, W = eig(R_f)
    # sort in descending order
    ids  = np.argsort(-e)
    ids  = ids.astype(int)
    e    = e[ids[:reduced_dim]]
    W    = W[:, ids[:reduced_dim]]
    e    = np.diag(np.sqrt(np.sqrt(e)))
    return W @ e.T

