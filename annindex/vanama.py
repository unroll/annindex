import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Sequence, Iterable, Optional, Any
from . import distance



def randomly_wired_edges(npts: int, nedges: int) -> NDArray[np.int32]:
    """
    Generate randomly wired outgoing edges for all nodes. 
    Avoids self-loops and double edges, but does not guarantee back-edges: 
    if u is a neighbour of v, the reverse is not guaranteed.

    Parameters
    ----------
    npts : int
        Number of nodes in the graph
    nedges : int
        Number of edges for each node

    Returns
    -------
    G : NDArray[np.int32]
        An integer array of size (npts, nedges) with edges of each node.
    """    
    if (nedges <= npts):
        raise RuntimeError('Must have more points than edges per point')

    edges = np.zeros((npts, nedges), dtype=np.int32)    
    # Set random edges row by row
    for i in range(npts):
        # To avoid self loop, generate nedges+1 neighbours, then replace i with the last neighbour if needed
        nbrs = np.random.choice(npts, size=nedges+1, replace=False)
        for j in range(nedges):
            if nbrs[j] == i:
                nbrs[j] = nbrs[-1]
                break
        edges[i] = nbrs
    
    return edges

class VanamaIndex():
    """
    Vanama nearest neighbour search index.

    Parameters
    ----------
    d : int
        Vector dimension
    dist_func : str
        Distance function to use: `euclidean`, `cosine`, or `inner` (for inner product). By default, `euclidean'.
    R : int
        Maximum out-degree. By default, 64.
    L : int
        Maximum size of candidate list during graph construction/traversal. By default, 100.
    alpha : float
        Growth factor for steps towards goal. By default, 1.2.
    """            
    def __init__(self, d: int, dist_func: str = 'euclidean', R: int = 64, L: int = 100, alpha: float = 1.2) -> None:
        if R > L:
            raise ValueError(f'Cannot create index where R ({R}) is larger than L ({L})')
        if d <= 0:
            raise ValueError('Cannot create index with negative or zero d')
        if alpha < 1:
            raise ValueError('alpha must be at least 1.0')
        if dist_func not in distance.dist_funcs:
            raise ValueError(f'Unkown distance function {dist_func}. Supported: {list(distance.dist_funcs.keys())}')

        self.d = d
        self.R = R
        self.L = L
        self.alpha = alpha
        
        self.dist_func_name = dist_func
        self.dist_func = distance.dist_funcs[dist_func]
        
        self.npts = 0
        self.edges = None
        self.vectors = None
        self.keys = []
              
    def build(self, data: Sequence[ArrayLike], keys: Optional[Sequence[Any]] = None) -> None:
        # Sanity checks
        if len(data) == 0:
            raise ValueError('Cannot build index from empty array')        
        if len(data) < self.R:
            raise ValueError(f'Cannot build index for fewer vectors than R={self.R}')
        if keys is not None and len(data) != len(keys):
            raise ValueError(f"List of keys supplied, but it's length {len(keys)} does not match length of data {len}")        
        if len(data[0]) != self.d:
            raise ValueError(f'Data dimention {len(data[0])} does not match expected dimension {self.d}')
        
        # Copy keys
        self.npts = len(data)        
        self.keys = list(keys) if keys else np.arange(self.npts)
                        
        # Initialize to randomly chosen edges
        self.edges = randomly_wired_edges(self.npts, self.R)
        self.vectors = np.array(data)

        # Set entry point to medoid.
        self.entry_point = distance.medoid(self.vectors, self.dist_func_name)



    
    




        







                              

        



    