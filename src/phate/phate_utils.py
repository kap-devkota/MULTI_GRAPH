import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import Isomap

def compute_potential_dist(A, t, n_components, small_val=0.000001):
    """
    Given the diffusion and timestep `t`, compute the potential distance. 
    """
    # Compute the Markov Transition Matrix
    d = A @ np.ones((A.shape[0], 1)) # n x 1
    P = A / d                        # Equivalent to doing D^{-1}A

    # Compute the matrix power upto t timestep
    P_t   = np.linalg.matrix_power(P, t) + small_val
    # Compute the corresponding log value
    logPt = np.log(P_t)
    embedding = Isomap(n_components = n_components)
    return embedding.fit_transform(logPt)
    
    
