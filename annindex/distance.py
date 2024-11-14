import numpy as np
from scipy.spatial.distance import pdist, squareform, sqeuclidean, cosine
from numpy.typing import ArrayLike, NDArray


# Use squared Euclidean distance, not Euclidean.
# euclidean distance function calls sqrt which causes numpy to check there are no infintes in the array
dist_funcs = {'euclidean':sqeuclidean,  # we actually use squared e
              'cosine':cosine, 'inner': 
              lambda x,y: -np.inner(x,y) # return negative inner product to return distance, not similarity
              }

def symmetric_distance_matrix(vectors: ArrayLike, dist_func: str = 'euclidean') -> NDArray:
    """
    Return the distance matrix of all vectors.    
    
    Supports Euclidean distance, cosine distance, and inner product.
        
    Parameters
    ----------
    vectors : ArrayLike
        An n by d array of n vectors in d dimensions.
    dist_func : str, optional
        Distance metric: 'euclidean', 'cosine', or 'inner'. By default 'euclidean'. 

    Returns
    -------
    out : ndarray of shape (n, n).
        Distances between vectors: out[i,j] = out[j,i] = distance(vectors[i], vectors[j]) 
    """

    if dist_func not in dist_funcs:
        raise ValueError(f'Unkown distance function {dist_func}. Supported: {list(dist_funcs.keys())}')

    if dist_func=='inner':
        X = np.array(vectors)
        return X @ X.T     
    
    # Use squared Euclidean distance, not Euclidean.
    if dist_func == 'euclidean':
        dist_func = 'sqeuclidean'
    
    return squareform(pdist(vectors, metric=dist_func))    


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
    D = symmetric_distance_matrix(vectors, dist_func)
    # Invert sign since for inner prod we want argmax
    if dist_func=='inner':
        D *= -1.0
    # Return vector with minimum distance
    return D.sum(axis=1).argmin()
        
