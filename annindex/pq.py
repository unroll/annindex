from typing import Optional, Sequence, Iterable, Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans

from .distance import get_distance_func, DistanceFunction
from .base import ProgressWrapper, BaseCompressor
from .utils import load_from_iterator

# Compressed form of a vector is a list/array of integers
CodeWord = Sequence[np.unsignedinteger]

class ProductQuantization(BaseCompressor):    
    """
    Compresses vectors by partitioning the space to chunk and running vector quantization there.

    Dense vectors are compressed to *code words* -- sequence of numbers -- using a *code book* derived from the data when building the compression.
    The class also provides a distance function between code words.
    Current implementation may not reduce memory by as much as it could, see notes.

    Parameters
    ----------
    d : int
        Dimension of vectors.
    chunk_dim : int, optional
        How many dimensions to include in each chunk, by default 4. For example, if d is 64 and chunk_dim is 4, there will be 16 chunks.
    chunk_bits : int, optional
        How many bits to allocate for each chunk, by default 8. (In practice, more bits are allocated; see notes.)
    allow_precalc_distances : bool, optional
        For certain distance functions and chunk_bits, distance computation can be optimzied with precalculated
        tables at the cost of more memory (see notes). Set to False to disable this optimization. By default, True.
    progress_wrapper : Sequence, str -> Sequence, optional
            None, or a function that accepets a sequence S and description str, and yields the same sequence S. Useful for progress bar (try `tqdm`).
    
    Notes
    -----
    1. Current implementation allocates the next sufficient native datatype (uint8, uint16, uint32...) per chunk, so not all memory savings are realized.
    2. For certain distance functions (currently `euclidean` and `inner`) when chunk_bits is not too high (10 or less),
       distance computation can be accelerated by computing it as sum of pre-calculated codebook distances.
       This incurs extra memory and can be disabled.    
    """                
    def __init__(self, 
                 d: int,
                 chunk_dim: int = 4, chunk_bits: int = 8,
                 progress_wrapper: Optional[ProgressWrapper] = None) -> None:        
        super().__init__(d, progress_wrapper)
        
        if chunk_dim > d or (d % chunk_dim) != 0:
            raise ValueError(f'Dimension {d} must be dividisble by chunk dimension {chunk_dim}')
        if chunk_bits < 2 or chunk_bits > 16:
            raise ValueError(f'Can only support between 2 and 16 bits for each chunk, not {chunk_bits}')

        self.chunk_dim = chunk_dim
        self.chunk_bits = chunk_bits
        self.chunk_clusters = 2**chunk_bits
        self.n_chunks = d // chunk_dim
        self.kms = [ KMeans(n_clusters=self.chunk_clusters) for i in range(self.n_chunks) ]        
        self.clusters_initialized = False

        self.center_distances = dict() # set later by _init_precalc_distances
        

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
        X = load_from_iterator(data, data_len, self.d)

        # Build array to hold centroid ids, using the nearest dtype size to chunk_bits 
        # TODO: replace with something like bitarray
        potential_dtypes = [np.uint8, np.uint16, np.uint32]
        self.chunk_dtype = next( dt for dt in potential_dtypes if np.dtype(dt).itemsize*8 >= self.chunk_bits )        
        codes = np.zeros((len(X), self.n_chunks), dtype=self.chunk_dtype) 

        for i in self.progress_wrapper(range(self.n_chunks), 'clustering chunks'):
            X_chunk = X[:, i*self.chunk_dim:(i+1)*self.chunk_dim]
            ids = self.kms[i].fit_predict(X_chunk)
            codes[:,i] = ids
        
        self.clusters_initialized = True
        
        return codes
        
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
        if not self.clusters_initialized:
            raise RuntimeError('Must load and compress data before precalculating distances.')

        # Some configuration cannot or should not be specialized                
        unoptimized_dist_func, _ = super().get_distance_function(dist_name, False)
        is_specialized = specialized and dist_name in {'sqeuclidean', 'inner'} and self.chunk_bits <= 10
        if not specialized:
            return unoptimized_dist_func, False
            
        # Precompute distances between cluster centers               
        dist_func = get_distance_func(dist_name, internal_use=True)
        center_distances = np.array([ dist_func.allpairs_nonsquare(km.cluster_centers_) for km in self.progress_wrapper(self.kms, 'precomputing distances') ])        
        # Compute dist (x,x) for pseudo-distance functions that are not reflexive (e.g., inner prod)
        if dist_name in {'inner'}:
            self_distances = np.array([ dist_func.paired(km.cluster_centers_, km.cluster_centers_) for km in self.progress_wrapper(self.kms, 'precomputing distances') ])
        else:
            self_distances = None
        # Caching arange, used by for all precalced distance functions regardless of dist_name
        self._arange_n = np.arange(self.n_chunks) 
        
        # Build new distance function that uses preco
        new_dist_func = DistanceFunction(dist_name + ' pq compressed',
                                         allpairs=unoptimized_dist_func.allpairs,
                                         pairwise=unoptimized_dist_func.pairwise,
                                         # One-to-many is implemented by repeating x and computing paired distances between X[i] and Y[i]
                                         one2many=lambda x,Y: self._precalced_distance_paired(np.tile(x, (len(Y), 1)), Y, center_distances, self_distances),
                                         distance=lambda x,y: self._precalced_distance_one_pair(x, y, center_distances, self_distances),
                                         allpairs_nonsquare=unoptimized_dist_func.allpairs_nonsquare,
                                         paired=lambda X,Y: self._precalced_distance_paired(X, Y, center_distances, self_distances),
                                         )
        # Recorded precalculated distances as members
        new_dist_func.center_distances = center_distances 
        new_dist_func.self_distances = self_distances

        return new_dist_func, True
                
    def code_to_vec(self, code: CodeWord) -> np.ndarray[float]:
        """
        Convert code word to its representing vector.

        Parameters
        ----------
        code : CodeWord
            Code word to convert

        Returns
        -------
        out : np.ndarray[float]
            Representing vector.
        """
        if len(code.shape) == 1:
            x = np.zeros(self.d, dtype=float)
            for i in range(self.n_chunks):
                x[i*self.chunk_dim:(i+1)*self.chunk_dim] = self.kms[i].cluster_centers_[code[i]]            
            return x
        else:
            X = np.zeros((len(code), self.d), dtype=float)        
            for i in range(self.n_chunks):
                X[:, i*self.chunk_dim:(i+1)*self.chunk_dim] = self.kms[i].cluster_centers_[code[:, i]]
            return X

    def vec_to_code(self, x: np.ndarray[float]) -> CodeWord:
        """
        Convert vector to compressed code word.

        Parameters
        ----------
        x : CodeWord
            Vector to compress.

        Returns
        -------
        out : CodeWord
            Code word (sequence of integers) for the vector.
        """         
        if len(x.shape) == 1:       
            code = np.zeros((self.n_chunks), dtype=self.chunk_dtype) 
            for i in range(self.n_chunks):
                code[i] = self.kms[i].predict(x[None, i*self.chunk_dim:(i+1)*self.chunk_dim])[0]
            return code
        else:
            codes = np.zeros((len(x), self.n_chunks), dtype=self.chunk_dtype) 
            for i in range(self.n_chunks):
                codes[:, i] = self.kms[i].predict(x[:, i*self.chunk_dim:(i+1)*self.chunk_dim])
            return codes
    
    def decompress(self, compressed: Any) -> np.ndarray[float]:
        """
        Decompress vector

        Parameters
        ----------
        compressed :
            Compressed vector

        Returns
        -------
        out : np.ndarray[float]
            Decompressed vector.
        """                
        return self.code_to_vec(compressed)

    def compress(self, x: np.ndarray) -> Any:
        """
        Compress vector

        Parameters
        ----------
        x : np.ndarray
            Vector to compress.

        Returns
        -------
        out :
            Compressed vector.
        """                        
        return self.vec_to_code(x)
    
    def _precalced_distance_one_pair(self, x_code: CodeWord, y_code: CodeWord, precalced_center_distances: np.ndarray, precalced_self_distances: np.ndarray) -> float:
        """
        Implement optimized precalculated version
        """
        # This works by precalculating distances between all cluster centers ("pivots") of each chunk.
        # For many distance functions the distance is a sum of scalars computed from the vector scalars. 
        # In such cases so we can compute the distance d(x,y) as a sum of partial distances d(x[..], y[..]) from each chunk.
        assert self.center_distances is not None
        
        m = self.chunk_clusters
        # Make a (2, m) array of ints
        xy = np.vstack([x_code, y_code]).astype(int)
        # Sort such that xy[0] < xy[1] 
        xy.sort(axis=0)
        i, j = xy[0], xy[1]
        # Compute array of indexes into center_distances[0], center_distances[1], ...
        idx = m * i + j - ((i + 2) * (i + 1)) // 2
        # idx[i] indexs center_distances[i], so use 2D array index using [0,1,2,3...] for first dim
        partials = precalced_center_distances[self._arange_n, idx]    
        # The non-square form does not include the cases where i==j, it is designed for dist(x,x)=0
        mask = i == j        
        # Condensed form does not include the cases where i==j as it assumes distance is reflexive
        if precalced_self_distances is not None:
            # Restore dist(x,x) for cases where dist(x,x) != 0
            # (non-square form does not include the cases where i==j as it assumes distance is reflexive)     
            partials[mask] = precalced_self_distances[self._arange_n, i][mask]
            return partials.sum()
        else:
            return precalced_center_distances[self._arange_n, idx][i != j].sum()      

        # The preceding code is equivalent to the following plain Python code for when dist(x,x)=0:
        # 
        # d = 0.0
        # m = self.chunk_clusters
        # for x, y, precalced in zip(x_code, y_code, precalced_center_distances):
        #     # We are only storing the i<j cases
        #     if x == y:
        #         continue            
        #     i, j = (int(x), int(y)) if x < y else (int(y), int(x))
        #     #  Compressed form formula: dist(i,j) where i<j is stored in: 
        #     idx = m * i + j - ((i + 2) * (i + 1)) // 2
        #     d += precalced[idx]
        # return d

    def _precalced_distance_paired(self, x_codes: Sequence[CodeWord], y_codes: Sequence[CodeWord], precalced_center_distances: np.ndarray, precalced_self_distances: np.ndarray) -> float:
        """
        Implement optimized precalculated version compariing x[i] and y[i]
        """
        # This works by precalculating distances between all cluster centers ("pivots") of each chunk.
        # For many distance functions the distance is a sum of scalars computed from the vector scalars. 
        # In such cases so we can compute the distance d(x,y) as a sum of partial distances d(x[..], y[..]) from each chunk.
        #        
        n = len(x_codes)
        assert len(y_codes) == n
        m = self.chunk_clusters
        # Make a (2, n, m) array of ints
        xy = np.stack([x_codes, y_codes]).astype(int)
        # Sort such that xy[0] < xy[1] 
        xy.sort(axis=0)
        i, j = xy[0], xy[1]
        # Compute array of indexes into center_distances[0], center_distances[1], ...
        idx = m * i + j - ((i + 2) * (i + 1)) // 2
        # idx[i] indexs center_distances[i], so use 2D array index using [0,1,2,3...] for first dim
        partials = precalced_center_distances[self._arange_n, idx]
        
        # Condensed form does not include the cases where i==j as it assumes distance is reflexive
        if precalced_self_distances is not None:
            # Restore dist(x,x) for cases where dist(x,x) != 0
            mask = i == j
            partials[mask] = precalced_self_distances[self._arange_n, i][mask]                  
        else:
            # Do not sum where i==j
            partials[i == j] = 0
        return partials.sum(axis=1) 

        

