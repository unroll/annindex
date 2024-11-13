import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean, cosine
from numpy.typing import ArrayLike


dist_funcs = {'euclidean':euclidean, 
              'cosine':cosine, 'inner': 
              lambda x,y: -np.inner(x,y) # return negative inner product to return distance, not similarity
              }

def medoid(vectors: ArrayLike, dist_func: str = 'euclidean') -> int:
    """
    Return medoid of the dataset - the point that minimizes the sum of distances to all other points.    
    
    Supports Euclidean distance, cosine distance, and inner product.
    For inner product, it returns the point that maximizes the sum.

    
    Parameters
    ----------
    vectors : ArrayLike
        An n by d array of n vectors in d dimensions.
    dist_func : str, optional
        Distance metric: 'euclidean', 'cosine', or 'inner'. By default 'euclidean'. 

    Returns
    -------
    int
        index of medoid.
    """ 
    if dist_func not in dist_funcs:
        raise ValueError(f'Unkown distance function {dist_func}. Supported: {list(dist_funcs.keys())}')

    # Compute symmetric distance matrix    
    if dist_func=='inner':
        X = np.array(vectors)
        # Invert sign since for inner prod we want argmax
        d = -(X @ X.T) 
    else:
        d = squareform(pdist(vectors, metric='euclidean'))
    # Return vector with minimum distance
    return d.sum(axis=1).argmin()
        

