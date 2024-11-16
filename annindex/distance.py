import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform, sqeuclidean, cosine
from numpy.typing import ArrayLike, NDArray


# Use squared Euclidean distance, not Euclidean.
# euclidean distance function calls sqrt which causes numpy to check there are no infintes in the array
dist_funcs = {'euclidean':sqeuclidean,  # we actually use squared e
              'cosine':cosine, 'inner': 
              lambda x,y: -np.inner(x,y) # return negative inner product to return distance, not similarity
              }

def one_to_many(x: ArrayLike, vectors: ArrayLike, dist_func: str = 'euclidean') -> NDArray:
    """
    Return the distances from vector x to other vectors.    
    
    Supports Euclidean distance, cosine distance, and inner product.
        
    Parameters
    ----------
    x : ArrayLike
        An vector of length d.
    vectors : ArrayLike
        An (n, d)-array containing n vectors in d dimensions.
    dist_func : str, optional
        Distance metric: 'euclidean', 'cosine', or 'inner'. By default 'euclidean'. 

    Returns
    -------
    out : ndarray of length n
        Distances between `x` to each vector in `vectors`.
    """

    if dist_func not in dist_funcs:
        raise ValueError(f'Unkown distance function {dist_func}. Supported: {list(dist_funcs.keys())}')

    if dist_func=='inner':
        V = np.array(vectors)
        return V.dot(x)
    
    # Use squared Euclidean distance, not Euclidean.
    if dist_func == 'euclidean':
        dist_func = 'sqeuclidean'
    
    return cdist([x], vectors, metric=dist_func)[0]

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
        # Quick hack: fast multi implementation with linear algebra.
        # TODO: fix this mess.
        X = np.asarray(vectors)
        tmp = np.multiply(X, X).sum(axis=1)
        res = -2 * np.dot(X, X.T)
        res += tmp[:, None]
        res += tmp[None, :]
        return res
    
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
        
