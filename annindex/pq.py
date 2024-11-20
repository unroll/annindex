from typing import Optional, Sequence
import numpy as np
from scipy.spatial.distance import squareform
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from . import distance
from .base import ProgressWrapper

# Compressed form of a vector is a list/array of integers
CodeWord = Sequence[np.unsignedinteger]

class ProductQuantization():    
    """
    Compresses incoming vectors by partitioning the space to chunk and running vector quantization there.

    Dense vectors are compressed to *code words* -- sequence of numbers -- using a *code book* derived from the data when building the compression.
    The class also provides a distance function between code words.
    Current implementation may not reduce memory by as much as it could, see notes.

    Parameters
    ----------
    data : sequence of vectors
        N vectors of length d.
    chunk_dim : int, optional
        How many dimensions to include in each chunk, by default 4. For example if d is 64 and chunk_dim is 4, there will be 16 chunks.
    chunk_bits : int, optional
        How many bits to allocate for each chunk, by default 8. (In practice, more bits are allocated; see notes.)
    dist_func : str, optional
        Distance function to use: `euclidean`, `cosine`, or `inner` (for inner product). By default, `euclidean'.
    allow_precalc_distances : bool, optional
        For certain distance functions and chunk_bits, distance computation can be optimzied with precalculated tables at the cost of more memory (see notes). Set to False to disable this optimization. By default, True.
    progress_wrapper : Sequence, str -> Sequence, optional
            None, or a function that accepets a sequence S and description str, and yields the same sequence S. Useful for progress bar (try `tqdm`).
    
    Notes
    -----
    1. Current implementation allocates the next sufficient native datatype (uint8, uint16, uint32...) per chunk, so not all memory savings are realized.
    2. For certain distance functions (currently `euclidean` and `inner`) when chunk_bits is not too high (10 or less),
       distance computation can be accelerated by computing it as sum of pre-calculated codebook distances. 
       This incurs extra memory and can be disabled.
    """                
    def __init__(self, data: Sequence[ArrayLike], 
                 chunk_dim: int = 4, chunk_bits: int = 8,
                 dist_func: str = 'euclidean', # not used for clustering
                 allow_precalc_distances: bool = True,
                 progress_wrapper: Optional[ProgressWrapper] = None) -> None:        
        
        if len(data) == 0:
            raise ValueError('Cannot work on no data')
        d = len(data[0])
        if d <= 0:
            raise ValueError(f'Dimension {d} must be at least 1')
        if chunk_dim > d or (d % chunk_dim) != 0:
            raise ValueError(f'Dimension {d} must be dividisble by chunk dimension {chunk_dim}')
        if chunk_bits < 2 or chunk_bits > 16:
            raise ValueError(f'Can only support between 2 and 16 bits for each chunk, not {chunk_bits}')
        if dist_func not in distance.dist_funcs:
            raise ValueError(f'Unkown distance function {dist_func}. Supported: {list(distance.dist_funcs.keys())}')        

        
        self.d = d
        self.chunk_dim = chunk_dim
        self.chunk_bits = chunk_bits
        self.chunk_clusters = 2**chunk_bits
        self.n_chunks = d // chunk_dim
        self.kms = [ KMeans(n_clusters=self.chunk_clusters) for i in range(self.n_chunks) ]        
        self.dist_func_name = dist_func
        self.dist_func = distance.dist_funcs[dist_func]
        self.precalc_distances = (allow_precalc_distances) and dist_func in {'euclidean', 'inner'} and self.chunk_bits <= 10

                        
        X = np.fromiter(data, dtype=(float, self.d), count=len(data))
        n = len(X)
        
        # Build array to hold centroid ids, using the nearest dtype size to chunk_bits 
        # (yes, this is actually fudging things)
        potential_dtypes = [np.uint8, np.uint16, np.uint32]
        self.chunk_dtype = next( dt for dt in potential_dtypes if np.dtype(dt).itemsize*8 >= chunk_bits )        
        self.codes = np.zeros((n, self.n_chunks), dtype=self.chunk_dtype) 

        for i in progress_wrapper(range(self.n_chunks), 'clustering chunks'):
            X_chunk = X[:, i*self.chunk_dim:(i+1)*self.chunk_dim]
            ids = self.kms[i].fit_predict(X_chunk)
            self.codes[:,i] = ids
        
        def get_compressed_distances(vecs):
            # Get symmetric distance matrix
            d = distance.symmetric_distance_matrix(vecs, dist_func)
            return squareform(d, checks=False)

        if self.precalc_distances:
            self.center_distances = [ get_compressed_distances(km.cluster_centers_) for km in progress_wrapper(self.kms, 'precomputing distances') ]
        else:
            self.center_distances = None

    def get_code(self, idx: int) -> CodeWord:
        """
        Return the code word (compressed from) of stored vector idx.

        Parameters
        ----------
        idx : int
            Index of stored vector

        Returns
        -------
        out : CodeWord
            The stored code word for the vector.
        """        
        return self.codes[idx]

    def distance(self, x_code: CodeWord, y_code: CodeWord, allow_precalc: bool = True) -> float:
        """
        Distance between two compressed vectors (two code words).

        Parameters
        ----------
        x_code : CodeWord
            Code for vector x
        y_code : CodeWord
            Code for vector y
        allow_precalc : bool, optional
            If true and precalculated distance computation is enabled, use it. By default True

        Returns
        -------
        float
            Distance between vectors x and y.
        """        
        if self.precalc_distances and allow_precalc:
            return self._precalced_distance(x_code, y_code)
        else:
            return self.dist_func(self.code_to_vec(x_code), self.code_to_vec(y_code))

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
        x = np.zeros(self.d, dtype=float)
        for i in range(self.n_chunks):
            x[i*self.chunk_dim:(i+1)*self.chunk_dim] = self.kms[i].cluster_centers_[code[i]]
        return x

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
        code = np.zeros((self.n_chunks), dtype=self.chunk_dtype) 
        for i in range(self.n_chunks):
            code[i] = self.kms[i].predict(x[i*self.chunk_dim:(i+1)*self.chunk_dim])
        return code
    
    def _precalced_distance(self, x_code: CodeWord, y_code: CodeWord) -> float:
        """
        Implement optimized precalculated version
        """
        # This works by precalculating distances between all cluster centers ("pivots") of each chunk.
        # For many distance functions the distance is a sum of scalars computed from the vector scalars. 
        # In such cases so we can compute the distance d(x,y) as a sum of partial distances d(x[..], y[..]) from each chunk.
        assert self.center_distances
        d = 0.0
        m = self.chunk_clusters
        for x, y, precalced in zip(x_code, y_code, self.center_distances):
            # We are only storing the i<j cases
            if x == y:
                continue            
            i, j = (int(x), int(y)) if x < y else (int(y), int(x))
            #  Compressed form formula: dist(i,j) where i<j is stored in: 
            idx = m * i + j - ((i + 2) * (i + 1)) // 2
            d += precalced[idx]
        return d


    
