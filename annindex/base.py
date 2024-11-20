
from typing import Callable, Sequence, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray
from . import distance

# Type variable for progress bars. Creates a progress bar from a sequence, with a string title.
ProgressWrapper = Callable[[Sequence, str], Sequence]

class BaseIndex(ABC):
    """
    Base type for indexes.

    Parameters
    ----------
    d : int
        Vector dimension
    dist_func : str
        Distance function to use: `euclidean`, `cosine`, or `inner` (for inner product). By default, `euclidean'.
    progress_wrapper : Sequence, str -> Sequence, optional
            None, or a function that accepets a sequence S and description str, and yields the same sequence S. Useful for progress bar (try `tqdm`).
    """            
    def __init__(self, d: int, dist_func: str = 'euclidean', progress_wrapper: Optional[ProgressWrapper] = None) -> None:

        if d <= 0:
            raise ValueError('Cannot create index with negative or zero d')
        if dist_func not in distance.dist_funcs:
            raise ValueError(f'Unkown distance function {dist_func}. Supported: {list(distance.dist_funcs.keys())}')

        self.d = d
        self.dist_func_name = dist_func
        self.dist_func = distance.dist_funcs[dist_func]
        self.progress_wrapper = progress_wrapper if progress_wrapper is not None else lambda S, d: S 

        self.npts = 0
        self.vectors = None
        self.keys = None
        self.key_index = None
    
    def get(self, idx_or_key: int | Any) -> NDArray:
        """
        Return vector by index (or key if keys where passed)

        Parameters
        ----------
        idx_or_key : int or Any
            Index or key of vector to retrieve.

        Returns
        -------
        out : NDArray
            Vector of length d.
        """
        idx = self.key_index[idx_or_key] if self.keys is None else idx_or_key
        return self.vectors[idx]
    
    @abstractmethod
    def query(self, x: ArrayLike, k:int = 1, *args, **kwargs) -> list[Any] | list[int]:
        """
        Return k approximate nearest neighbours to x.

        Parameters
        ----------
        x : ArrayLike
            Vector to search for, dimension d.
        k : int, optional
            How many neighbours to return, by default 1

        Returns
        -------
        out :
            list of keys or indexes of k nearest neighbours to x.
        """        
        if len(x) != self.d:
            raise ValueError(f'Dimension of x {len(x)} does not match index dimension {self.d}')
        if k < 1:
            raise ValueError(f'Must return at least 1 neighbour')
                      
    @abstractmethod
    def build(self, data: Sequence[ArrayLike], dtype: np.dtype = np.float64, keys: Optional[Sequence[Any]] = None, *args, **kwargs) -> None:
        """
        Build the index from vector data.

        Parameters
        ----------
        data : sequence of vectors
            N vectors. Length must match index dimension `d`.
        dtype : np.dtype, optional
            Datatype for stored vector data, by default np.float64.
        keys : sequence of keys, optional
            Use supplied keys as index instead of integer index.
        """        
        # Sanity checks
        if len(data) == 0:
            raise ValueError('Cannot build index from empty array')        
        if keys is not None and len(data) != len(keys):
            raise ValueError(f"List of keys supplied, but it's length {len(keys)} does not match length of data {len}")        
        if len(data[0]) != self.d:
            raise ValueError(f'Data dimention {len(data[0])} does not match expected dimension {self.d}')    

        # Copy keys and vectors
        self.npts = len(data)    
        if keys:
            self.keys = list(keys)
            self.key_index = { k:i for i,k in enumerate(self.keys) }
        # Efficient way to load data from sequence ( https://perfpy.com/871 )
        self.vectors = np.fromiter(data, dtype=(dtype, self.d), count=len(data))        

    def _idxs_to_keys(self, indexes: list[int]) -> list[Any] | list[int]:
        """
        Convert sequence of indexes to list of keys (if provided) or leave as is.

        Parameters
        ----------
        indexes : list[int]
            List of indexes

        Returns
        -------
        out : list[Any] | list[int]
            List of keys if keys are used, or same list of indexes.
        """        
        return indexes if self.keys is None else [ self.keys[idx] for idx in indexes ] 
