from typing import Iterable, Optional, Any, Sequence
import numpy as np
from numpy.typing import ArrayLike

from .base import BaseIndex, BaseCompressor

class ComressionAdapter(BaseIndex):    
    def __init__(self, index: BaseIndex, compressor: BaseCompressor, build_from_compressed=False):
        super().__init__(index.d, index.external_dist_name)
        self.compressed_d = compressor.get_compressed_d() if build_from_compressed else self.d
        self.index = index
        self.compressor = compressor
        self.build_from_compressed = build_from_compressed
        self.use_assymetric_distance = not build_from_compressed
           
    def load(self, data: Iterable[ArrayLike], data_len: int, dtype: np.dtype = np.float64, keys: Optional[Sequence[Any]] = None):
        if self.build_from_compressed:
            self.index.d = self.compressed_d
            X = self.compressor.load_and_compress(data, data_len)
            self._update_distance_function()
            self.index.load(X, len(X), dtype=dtype, keys=keys)
        else:
            self.index.load(data, data_len, dtype, keys=keys)
    
    def query(self, x: ArrayLike, k:int = 1, *args, **kwargs) -> list[Any] | list[int]:
        if not self.use_assymetric_distance:
            x = self.compressor.compress(x)
        return self.index.query(x, k, *args, **kwargs)

    def build(self):
        self.index.build()
        if not self.build_from_compressed:
            X = self.compressor.load_and_compress(self.index.vectors, len(self.index.vectors))
            self.index.vectors = X
            self.index.d = self.compressed_d
            self._update_distance_function()

        
    def _update_distance_function(self):        
        self.original_dist_func = self.index.dist_func
        if self.use_assymetric_distance:
            self.index.dist_func, _ = self.compressor.get_assymetric_distance_function(self.original_dist_func.name)
        else:            
            self.index.dist_func, _ = self.compressor.get_distance_function(self.original_dist_func.name) 

            
