import numpy as np
from typing import Any
from numpy.typing import ArrayLike

from .base import BaseIndex
from .utils import arg_ksmallest

class Flat(BaseIndex):
    """
    Implements "flat" index i.e., straightforward brute-force search.

    Parameters
    ----------
    d : int
        Vector dimension
    dist_name : str
        Distance function to use. By default, `euclidean`.
    """    
    def __init__(self, d, dist_name = 'sqeuclidean'):
        super().__init__(d, dist_name)

    def query(self, x: ArrayLike, k: int = 1, *args, **kwargs) -> list[Any] | list[int]:           
        # (Docstring inherited from parent)        
        super().query(x, k)        
        x = np.asarray(x)
        # Compute distances from all vecotrs
        distances = self.dist_func.one2many(x, self.vectors)
        # Find k smallest, sorted
        knearest_indices = arg_ksmallest(k, distances)        
        return self._idxs_to_keys(knearest_indices)

    def build(self) -> None:  
        # (Docstring inherited from parent)
        super().build()
        # Nothing to do!