from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform, sqeuclidean, cosine
import sklearn.metrics.pairwise as skmp
from numpy.typing import ArrayLike, NDArray


# Distance functions in annindex are modeled as a data class whose attributes
# are standalone functions (not class methods; there is no ``self``). The C/C++
# equivalent is a struct of function pointers.
#
# One might reasonably argue that they are in-fact a kind of class hierarchy,
# and there should have been a BaseDistanceFunction interface / ABC from which
# other distance functions are derived. Indeed it could be argued that this
# is merely re-implementing inheritence.
#
# The reasons DistanceFunction is modeled as a dataclass with functions: 
# 1. More concise to declare and implement them (see _distance_functions).
# 2. Easier to generate new DistanceFunctions on the fly at runtime, for 
#    example for distances between PQ-compressed vectors.
# 3. Easier to use the function as there is no need to use self. Moreover, some
#    libraries even expect such function (e.g., scipy's `pdist``).
# 4. Avoids creating yet another class hierarchy to keep track of.
# 5. scipy does it too, although their case everything is more static.
#
# The downside is that sometimes a distance implementation needs to access 
# precalculated data, or store data in some other way. In such cases, a lambda
# (closure) can reference and access the precalculated data, and we can also
# store it in a new attribute. See `ProductQuantization.get_distance_function`
# for an example. 

# Type variables aliases are here for readability, and since currently there
# is no way to declare a different numpy type for a matrix or a vector.
Vector = NDArray
Matrix = NDArray

@dataclass
class DistanceFunction:
    """
    A collection of functions (not methods) that implement different variations
    of a particular distance function.

    Attributes
    ----------
    name : str
        Name of the distance function being implemented. Built-in functions are
        ``euclidean``, ``sqeuclidean``, ``inner``, etc (see `distance_funcs`). Some 
        runtime generated function may have different names with multiple words.
    allpairs : function 
        Given an array of vectors, return a distance matrix between all pairs
        of vectors: ouj[i,j] = distance(X[i], X[j]) .
    pairwise : function 
        Given an arrays of vectors X and Y, return a distance matrix between
        all vectors in X and all vectors in Y: ouj[i,j] = distance(X[i], Y[j]) .
    one2many : function
        Given a vector x and an array of vectors Y, return a vector with 
        distances of x to all vectors in Y: ouj[j] = distance(x, Y[j]) .
    distance : function
        Given two vectorx x and y, return the distance(x, y).
    allpairs_nonsquare:
        Similar to allpairs except returns a flattened version of the distance
        matrix that contains only entires for i > j, hence using less memory.
        See `scipy.spatial.distance.pdist` documentation.
    paired:
        Given two vector arrays X and Y of same shape, returns the distance
        between vector pairs in X and Y: out[i] = distance(X[i], Y[i]) .
    
    See also
    --------
    get_distance_func : get a distance function by name
    distance_funczs : dictionary of built-in distance function    

    Notes
    ----- 
    1. Some indexes create their own distance functions that have additional
       attributes.
    2. `allpairs_nonsquare` and `paired` can be omitted when instantiating, in
       which case a straight-forward, less efficient implementations are
       created using `allpairs` or `distance`.
    3. Some distance functions (specifically inner product distance ``inner``)
       are not true distance functions in the formal sense: distance(., .) 
       could be negative and distance (x, x) is not zero.
    4. Note that `allpairs_nonsquare` does not return distance(X[i], X[i]), 
       even for ``inner`` where it is not zero.
    """    
    name: str
    allpairs: Callable[[Matrix], Matrix]
    pairwise: Callable[[Matrix, Matrix], Matrix]
    one2many: Callable[[Vector, Matrix], Vector]
    distance: Callable[[Vector, Vector], float]
    allpairs_nonsquare: Callable[[Matrix], Vector] = None
    paired:   Callable[[Matrix, Matrix], float] = None

    def __post_init__(self):
        if self.allpairs_nonsquare is None:
            self.allpairs_nonsquare = lambda X: squareform(self.allpairs(X), checks=False)
        if self.paired is None:
            self.paired = lambda X, Y: np.array([ self.distance(x,y) for x, y in zip(X,Y) ])


def _squared_euclidean_distance_matrix(X, Y=None):
    """
    Fast implentation of pairwise/allpair squared Euclidean distance using numpy.
    """    
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

def _squared_euclidean_distance_paired(X, Y):
    """
    Fast implentation of paired squared Euclidean distance using numpy.
    """    
    # Fast multi-core implementation that uses linear algebra.
    n = min(len(X), len(Y))
    return ((X[:n] - Y[:n])**2).sum(axis=1)
    

# List of built-in distance functions
_distance_functions = [ 
    DistanceFunction('euclidean', 
                     allpairs = skmp.euclidean_distances,
                     pairwise = skmp.euclidean_distances,
                     one2many = lambda x, Y: cdist([x], Y, metric='euclidean')[0], 
                     distance = lambda x, y: np.sqrt(((x-y)**2).sum()),
                     paired   = lambda X, Y: np.sqrt(_squared_euclidean_distance_paired(X, Y))
    ),
    DistanceFunction('sqeuclidean', 
                     allpairs = lambda X: skmp.euclidean_distances(X, squared=True),
                     pairwise = lambda X, Y: skmp.euclidean_distances(X, Y, squared=True),
                     one2many = lambda x, Y: cdist([x], Y, metric='sqeuclidean')[0], 
                     distance = lambda x, y: ((x-y)**2).sum(),
                     paired   = _squared_euclidean_distance_paired
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

# Dictionary of built-in distance functions
distance_funcs = { f.name:f for f in _distance_functions }

def get_distance_func(dist_name: str, internal_use: bool = False) -> DistanceFunction:
    """
    Given distance function name, return DistanceFunction implementing it.
    Internally most indexes replace Euclidean distances with squared Euclidean.

    Parameters
    ----------
    dist_name : str
        Name of distance function to use (see `distance_funcs`).
    internal_use : bool, optional
        Replace ``euclidean`` with ``sqeuclidean``, by default False.

    Returns
    -------
    out : DistanceFunction
        A DistanceFunction object implementing the requested distance function.
    """    
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
        

# When running the module as a script, it times various implementation choices.
# TODO: move this to tests or somewhere more appropriate
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
    print('paired')
    print(f'sklearn:  {min(repeat(lambda: skmp.paired_euclidean_distances(X, X[::-1])**2, repeat=3, number=5 )):.3f}' )
    print(f'homebrew: {min(repeat(lambda: _squared_euclidean_distance_paired(X, X[::-1]), repeat=3, number=5 )):.3f}' )
    print(f'loop: {min(repeat(lambda: np.array([ ((x-y)**2).sum() for x, y in zip(X, X[::-1]) ]), repeat=3, number=5 )):.3f}' )
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
