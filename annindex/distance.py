from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform, sqeuclidean, cosine
import sklearn.metrics.pairwise as skmp
from numpy.typing import ArrayLike, NDArray


Vector = NDArray
Matrix = NDArray

@dataclass
class DistanceFunction:
    name: str
    allpairs: Callable[[Matrix], float]
    pairwise: Callable[[Matrix, Matrix], float]
    one2many: Callable[[Vector, Matrix], float]
    distance: Callable[[Vector, Vector], float]
    allpairs_compressed: Callable[[Matrix], float] = None

    def __post_init__(self):
        if self.allpairs_compressed is None:
            self.allpairs_compressed = lambda X: squareform(self.allpairs(X), checks=False)


def _squared_euclidean_distance_matrix(X, Y=None):
    # Fast multi-core implementation that uses linear algebra.
    XX = np.multiply(X, X).sum(axis=1)
    if Y is None:
        YY = XX
        Y = X
    else:
        YY = np.multiply(Y, Y).sum(axis=1)
    res = -2 * np.dot(X, Y.T)
    res += XX[:, None]
    res += YY[None, :]
    return res    

_distance_functions = [ 
    DistanceFunction('euclidean', 
                     allpairs = skmp.euclidean_distances,
                     pairwise = skmp.euclidean_distances,
                     one2many = lambda x, Y: cdist([x], Y, metric='euclidean')[0], 
                     distance = lambda x, y: np.sqrt(((x-y)**2).sum())
    ),
    DistanceFunction('sqeuclidean', 
                     allpairs = lambda X: skmp.euclidean_distances(X, squared=True),
                     pairwise = lambda X, Y: skmp.euclidean_distances(X, Y, squared=True),
                     one2many = lambda x, Y: cdist([x], Y, metric='sqeuclidean')[0], 
                     distance = lambda x, y: ((x-y)**2).sum()
    ),
    DistanceFunction('inner',  # invert inner product sign so it sorts like a distance
                     allpairs = lambda X: -(X @ X.T),
                     pairwise = lambda X, Y: -(X @ Y.T),
                     one2many = lambda x, Y: -(Y.dot(x)),
                     distance = lambda x, y: -(x.dot(y))
    ),
    DistanceFunction('cosine', 
                     allpairs = skmp.cosine_distances,
                     pairwise = skmp.cosine_distances,
                     one2many = lambda x, Y: cdist([x], Y, metric='cosine')[0],
                     distance = lambda x, y: 1 - (x.dot(y) / np.sqrt(x.dot(x)*y.dot(y)))
    )                                  
]


distance_funcs = { f.name:f for f in _distance_functions }

def get_distance_func(dist_name: str, internal_use: bool = False) -> DistanceFunction:
    if internal_use and dist_name == 'euclidean':
        dist_name = 'sqeuclidean'
    f = distance_funcs.get(dist_name, None)
    if f is None:
        raise ValueError(f'Unkown distance function {dist_name}. Supported: {list(distance_funcs.keys())}')
    return f

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
    dist = get_distance_func(dist_func)
    D = dist.allpairs(vectors)    
    # Return vector with minimum distance
    return D.sum(axis=1).argmin()
        
if __name__ == '__main__':
    from timeit import repeat
    np.random.seed(39)
    n, d = 100000, 128
    X = np.random.rand(n,d)
    Y = np.random.rand(n//10,d)
    Z = np.random.rand(n//1000,d)
    ONE = Y[0][None,:]
    print(f'Timing Euclidean distance computation with n={n}, d={d}')
    print()
    print('one to many:')
    print(f'sklearn:  {min(repeat(lambda: skmp.euclidean_distances(ONE, X, squared=True), repeat=3, number=100 )):.3f}' )
    print(f'homebrew: {min(repeat(lambda: _squared_euclidean_distance_matrix(ONE, X)[0], repeat=3, number=100 )):.3f}' )
    print(f'scipy:    {min(repeat(lambda: cdist(ONE, X, metric="sqeuclidean")[0], repeat=3, number=100 )):.3f}' )
    print()
    print('pairwise 10:1')
    print(f'sklearn:  {min(repeat(lambda: skmp.euclidean_distances(X, Y, squared=True), repeat=3, number=5 )):.3f}' )
    print(f'homebrew: {min(repeat(lambda: _squared_euclidean_distance_matrix(X, Y)[0], repeat=3, number=5 )):.3f}' )
    print(f'scipy:    {min(repeat(lambda: cdist(X, Y, metric="sqeuclidean")[0], repeat=3, number=5 )):.3f}' )
    print()
    print('pairwise 1:10')
    print(f'sklearn:  {min(repeat(lambda: skmp.euclidean_distances(Y, X, squared=True), repeat=3, number=5 )):.3f}' )
    print(f'homebrew: {min(repeat(lambda: _squared_euclidean_distance_matrix(Y, X)[0], repeat=3, number=5 )):.3f}' )
    print(f'scipy:    {min(repeat(lambda: cdist(Y, X, metric="sqeuclidean")[0], repeat=3, number=5 )):.3f}' )
    print()
    print('pairwise 100:1')
    print(f'sklearn:  {min(repeat(lambda: skmp.euclidean_distances(X, Z, squared=True), repeat=3, number=5 )):.3f}' )
    print(f'homebrew: {min(repeat(lambda: _squared_euclidean_distance_matrix(X, Z)[0], repeat=3, number=5 )):.3f}' )
    print(f'scipy:    {min(repeat(lambda: cdist(X, Z, metric="sqeuclidean")[0], repeat=3, number=5 )):.3f}' )
    print()
    print('pairwise 1:100')
    print(f'sklearn:  {min(repeat(lambda: skmp.euclidean_distances(Z, X, squared=True), repeat=3, number=5 )):.3f}' )
    print(f'homebrew: {min(repeat(lambda: _squared_euclidean_distance_matrix(Z, X)[0], repeat=3, number=5 )):.3f}' )
    print(f'scipy:    {min(repeat(lambda: cdist(Z, X, metric="sqeuclidean")[0], repeat=3, number=5 )):.3f}' )
    print()
    print('allpairs')
    print(f'sklearn:  {min(repeat(lambda: skmp.euclidean_distances(X, squared=True), repeat=3, number=5 )):.3f}' )
    print(f'homebrew: {min(repeat(lambda: _squared_euclidean_distance_matrix(X), repeat=3, number=5 )):.3f}' )
