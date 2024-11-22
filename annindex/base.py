
from typing import Callable, Sequence, Optional, Any, Iterable, Iterator
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .distance import get_distance_func
from .utils import peek_iterator

# Type variable for progress bars. Creates a progress bar from a sequence, with a string title.
ProgressWrapper = Callable[[Sequence, str], Sequence]

class BaseStore(ABC):
    """
    Base class for anything that loads and stores vectors in memory. Allows access by key.

    Do not instantiate directly -- use derived classes.
   
    Parameters
    ----------
    d : int
        Vector dimension
    """            
    def __init__(self, d: int) -> None:

        if d <= 0:
            raise ValueError(f'Dimension d must be at least 1, not {d}')

        self.d = d

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
    
    def iterate(self) -> Iterator[tuple[Any, NDArray]]:
        """
        Iterate over stored vectors, returning pairs of (key, vector).
        If no keys are given, the key is the index.

        Yields
        ------
        key : Any
            Vector key or index if no key given./
        vec : NDArray
            Vector as ndarray.        
        """        
        key_iter = self.keys if self.keys else range(self.npts)
        yield from zip(key_iter.keys, self.vectors)
                      
    def load(self, data: Iterable[ArrayLike], data_len: int, dtype: np.dtype = np.float64, keys: Optional[Sequence[Any]] = None) -> None:
        """
        Load vectors and optionally keys into memory.

        Parameters
        ----------
        data : sequence of vectors
            N vectors. Length must match index dimension `d`.
        data_len : int
            Length of data N. Must be at least 1.
        dtype : np.dtype, optional
            Datatype for stored vector data, by default np.float64.
        keys : sequence of keys, optional
            Use supplied keys to identify vectors instead of integer index.
        """                
        # Sanity checks
        if data_len == 0:
            raise ValueError('Cannot build index from empty array')        
        if keys is not None and data_len != len(keys):
            raise ValueError(f"List of keys supplied, but it's length {len(keys)} does not match length of data {data_len}")                
        # Check that vector dimension matches loader
        first_vec, data = peek_iterator(data)
        if len(first_vec) != self.d:
            raise ValueError(f'Data dimention {len(first_vec)} does not match expected dimension {self.d}')            

        # Copy keys
        self.npts = data_len    
        if keys:
            self.keys = list(keys)
            self.key_index = { k:i for i,k in enumerate(self.keys) }
        # Copy vectors efficiently ( https://perfpy.com/871 )
        self.vectors = np.fromiter(data, dtype=(dtype, self.d), count=data_len)
        
  
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


class BaseIndex(BaseStore):
    """
    Base type for indexes. 

    Do not instantiate directly -- use derived classes.
     
    Parameters
    ----------
    d : int
        Vector dimension
    dist_name : str
        Distance function to use: ``euclidean``, ``sqeuclidean`` (squared Euclidean) ``cosine``, or ``inner`` (for inner product). 
        By default, ``sqeuclidean``.
    progress_wrapper : Sequence, str -> Sequence, optional
            None, or a function that accepets a sequence S and description str, and yields the same sequence S. Useful for progress bar (try `tqdm`).
    """            
    def __init__(self, d: int, dist_name: str = 'sqeuclidean', progress_wrapper: Optional[ProgressWrapper] = None) -> None:

        if d <= 0:
            raise ValueError('Cannot create index with negative or zero d')

        self.d = d
        self.external_dist_name = dist_name
        self.dist_func = get_distance_func(dist_name, internal_use=True)
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
    def build(self) -> None:
        """
        Build the index from stored vector data.
        """        
        if self.npts <= 0:
            raise RuntimeError('No vectors to build index from -- call load() first.')
