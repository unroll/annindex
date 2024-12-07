from typing import Any, Optional
import numpy as np
from heapq import merge

from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans

from .distance import medoid
from .base import BaseIndex
from .utils import arg_ksmallest

   

class IVF(BaseIndex):
    """
    The IVF index clusters vectors and  lists all vectors in the list belonging
    to the cluster with the nearest centroid. When quering, it searchers the
    lists of the `n_probe` centroids nearest to the query. 

    Currently only ``euclidean`` and ``sqeuclidean` distances are supported.

    Parameters
    ----------
    d : int
        Vector dimension
    dist_name : str
        Distance function to use. By default, `euclidean`.
    n_clusters : int
        Number of clusters to use. By default, 32.
    n_probe : int
        How many cluster lists to search when querying. By default, 4.
    """           
    def __init__(self, 
                 d: int, dist_name: str = 'sqeuclidean', 
                 n_clusters : int = 32,
                 n_probe : int = 1) -> None:
        super().__init__(d, dist_name)

        if n_clusters < 1:
            raise ValueError(f'Number of clusters {n_clusters} should be at'
                             f' least 1')
        if n_probe > n_clusters:
            raise ValueError(f'Number of probed clusters during query {n_probe}' 
                             f' must be at most n_clusters {n_clusters}')

        # TODO: perhaps use hierarchical clustering to support other distance
        # functions
        if dist_name not in {'euclidean', 'sqeuclidean'}:
            raise ValueError('Currently IVF only supports euclidean and'
                             ' sqeuclidean distance')
        
        self.n_clusters = n_clusters
        self.n_probe = n_probe

        self.centers = None
        self.index_base = None
        self.inverted_lists = [None] * n_clusters

        self.km = KMeans(n_clusters=n_clusters)

    # How do we store the index? We do not want to store vectors twice as that
    # would be very wasteful.
    #
    # We choose to store the vectors as they already are, in self.vectors.
    # Inverted lists are stored as arrays in self.inverted_lists, all of which
    # are actually slices of a single array in self.vectors. Since that means
    # indexes are changed, we use the keys mechanism to reflect the original
    # index/key when querying or iterating. self.index_base helps with that.
    #
    # Benefits:   
    # * Fast queries as it allows dist_func.one2many 
    # * Vectors are stored as a 2D array in self.vectors as in most other
    #   indexes, making them easy to work with.
    #
    # Downsides:
    # * Implementing something like insert_vector would be slower due as we
    #   might be inserting to the middle of the self.vectors array.
    # * Derived classes must be careful and not ignore the keys.

    def query(self, x: ArrayLike, k: int = 1, n_probe: Optional[int] = None, 
              *args, **kwargs) -> list[Any] | list[int]:
        """
        Return k approximate nearest neighbours to x.

        Parameters
        ----------
        x : ArrayLike
            Vector to search for, dimension d.
        k : int, optional
            How many neighbours to return, by default 1
        n_probe : int, optional
            How many cluster to search. If not provided, uses `self.n_probe`.

        Returns
        -------
        out :
            list of keys or indexes of k nearest neighbours to x.
        """                
        super().query(x, k)

        if n_probe is None:
            n_probe = self.n_probe        
        else:
            if n_probe > self.n_clusters:
                raise ValueError(f'n_probe {n_probe} cannot be greater than' 
                                 f' n_clusters {self.n_clusters}')            

        if self.index_base == None:
            raise RuntimeError('call build() before query()')
        
        x = np.asarray(x)
        # Compute distances from all centroids        
        distances = self.km.transform([x])[0]

        # Find bottom n_probe distances, in no particular order
        # Faster than distances.argsort()[:n_probe]
        probe = np.argpartition(distances, n_probe)[:n_probe]
        # Probe one cluster at a time
        vector_indexes = []
        distances = []
        for c in probe:
            # Get distance from x to cluster vectors
            dists = self.dist_func.one2many(x, self.inverted_lists[c])            
            # Get index of top k (or less) in cluster (in any order)
            idx = arg_ksmallest(k, dists, skip_sorting=True)
            # Correct the index so it applies to the vectors array
            vector_indexes.extend(idx + self.index_base[c])
            distances.extend(dists[idx])
            
        # distances array now holds one-to-many distances from x to
        # self.vectors[vector_indexes]
        vector_indexes = np.array(vector_indexes)
        distances = np.array(distances)
        # Get index into self.vectors of the k vectors with lowest distance
        knns = vector_indexes[arg_ksmallest(k, distances)]
        
        return self._idxs_to_keys(knns)

    def build(self) -> None:  
        """
        Build the index from stored vector data.
        """                
        super().build()

        # Find nearest cluster for each vector
        cluster_assignment = self.km.fit_predict(self.vectors)
        order = cluster_assignment.argsort()
        _, num_per_cluster = np.unique(cluster_assignment, return_counts=True)
        # Rearrange vector array
        self.vectors = self.vectors[order]
        # Rearrange keys (or create keys from old indexes so that externally
        # indexes appear to be the same)
        if self.keys is None:
            self.keys = order 
        else:
            self.keys = [ self.keys[i] for i in order ]
        # Update mapping of index
        self.key_index = { k:i for i,k in enumerate(self.keys) }        
        # Build slices for inverted lists, and record the base index for each
        ends = num_per_cluster.cumsum().tolist()
        starts = [0] + ends[:-1]
        self.inverted_lists = [ self.vectors[s:e] for s,e in zip(starts, ends) ]
        self.index_base = starts
