
from typing import Callable, Sequence, Optional, Any, Iterable, Iterator
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray
from functools import wraps

from .distance import get_distance_func, DistanceFunction
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
        idx = self.key_index[idx_or_key] if self.keys is not None else idx_or_key
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

        super().__init__(d)

        self.external_dist_name = dist_name
        self.dist_func = get_distance_func(dist_name, internal_use=True)
        self.progress_wrapper = progress_wrapper if progress_wrapper is not None else lambda S, d: S 

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


class BaseCompressor(ABC):
    """
    Base type for compressors like PQ, OPQ, SQ, and so on. 

    Do not instantiate directly -- use derived classes.
     
    Parameters
    ----------
    d : int
        Vector dimension
    progress_wrapper : Sequence, str -> Sequence, optional
            None, or a function that accepets a sequence S and description str, and yields the same sequence S. Useful for progress bar (try `tqdm`).
    """    
    def __init__(self, d: int, progress_wrapper: Optional[ProgressWrapper] = None) -> None:        
        if d <= 0:
            raise ValueError(f'Dimension d must be at least 1, not {d}')
        self.d = d
        self.progress_wrapper = progress_wrapper if progress_wrapper is not None else lambda S, d: S 

    @abstractmethod
    def load_and_compress(self, data: Iterable[ArrayLike], data_len: int, *args, **kwargs) -> Sequence[Any]:
        """
        Initializes compression and return compressed vectors.

        The Compressor object only stores information needed for compression and decompression.
        It does not store the compressed vectors internally.
        
        Parameters
        ----------
        data : sequence of vectors
            N vectors. Length must match index dimension `d`.
        data_len : int
            Length of data N. Must be at least 1.
        
        Returns
        -------
        out : Sequence[Any]
            Compressed vectors.
            
        """
        pass

    @abstractmethod
    def decompress(self, compressed: Any) -> np.ndarray[float]:
        """
        Decompress vector(s)

        Parameters
        ----------
        compressed :
            Single compressed vector or a sequence (array/list) of vectors.

        Returns
        -------
        out : np.ndarray[float]
            Decompressed vector or vectorsd.
        """                
        pass

    @abstractmethod
    def compress(self, x: np.ndarray) -> Any | Sequence[Any]:
        """
        Compress a single or a sequence of vectors.

        Parameters
        ----------
        x : np.ndarray
            Vector to compress, or 2D array for compressing multiple vectors.

        Returns
        -------
        out : 
            Compressed vector or vectors.
        """                        
        return self.vec_to_code(x)    
    
    def get_distance_function(self, dist_name: str = 'sqeuclidean', specialized: bool = True) -> DistanceFunction:
        """
        Return a potentially specialized distance function object that works directly on compressed vectors
        and return the distance betrween the decompressed vectors.

        Requesting a specialized implementation does not guarantee getting one.
        The result may simply decompresses the vectors and calls the original distance function.                

        Parameters
        ----------
        dist_name : str
            Distance function to use, by default, ``sqeuclidean``.

        Returns
        -------
        out : DistanceFunction
            Distance function for compressed data.
        is_specialized : bool
            True if out is an optimized implementation for compressed data. False if not.

        Notes
        -----        
        Do not call this function in performance-critical paths -- it may require significant precomputation. 
        Instead, get the resulting DistanceObject object in advance and store it.

        """                
        # Default implementation simply wraps the original distance function 
        # to decompress parameters.
        original = get_distance_func(dist_name)
        def decompressed(f):
            @wraps(f)
            def decompressed_distance(*args, **kwargs):                
                return f(*(self.decompress(argument) for argument in args), **kwargs)
            return decompressed_distance
        
        new_dist_func = DistanceFunction(original.name + ' compressed',
                                         allpairs=decompressed(original.allpairs),
                                         pairwise=decompressed(original.pairwise),
                                         one2many=decompressed(original.one2many),
                                         distance=decompressed(original.distance),
                                         allpairs_nonsquare=decompressed(original.allpairs_nonsquare),
                                         paired=decompressed(original.paired),
                                         )
        return new_dist_func, False
