import numpy as np
from numpy.typing import ArrayLike, NDArray
from heapq import heappush, heappop, nsmallest
from typing import Sequence, Iterable, Optional, Any
from . import distance
from .utils import VisitPriorityQueue



def randomly_wired_edges(npts: int, nedges: int) -> list[set[int]]:
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
    G : list of sets of integers
        A list of length npts with a sets of size nedges edges per node.
    """    
    if (nedges >= npts):
        raise RuntimeError('Must have more points than edges per point')

    edges = [None] * npts
    # Set random edges row by row
    for i in range(npts):
        # Generate nedges+1 neighbours without replacement
        nbrs = np.random.choice(npts, size=nedges+1, replace=False)
        for j in range(nedges):
            # To avoid self loop, replace i with the last generated neighbour if needed
            if nbrs[j] == i:
                nbrs[j] = nbrs[-1] # guaranteed not be i since we sample without replacement
                break
        edges[i] = set(nbrs[:-1])
    
    return edges

class VamanaIndex():
    """
    Vamana nearest neighbour search index, as described in Section 2 of the [DiskANN paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf).

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
        if R > L and L != 0:
            raise ValueError(f'Cannot create index where R ({R}) is larger than non-zero L ({L})')
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
        self.edges = []
        self.vectors = None
        self.keys = []
        self.key_index = {}

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
        idx = self.key_index[idx_or_key]
        return self.vectors[idx]
    
    def query(self, x: ArrayLike, k:int = 1) -> Any:

        if len(x) != self.d:
            raise ValueError(f'Dimension of x {len(x)} does not match index dimension {self.d}')
        if k < 1:
            raise ValueError(f'Must return at least 1 neighbour')

        x = np.asarray(x)
        knns, _ = self._greedy_search(x, k)
        return knns
                      

    def build(self, data: Sequence[ArrayLike], keys: Optional[Sequence[Any]] = None) -> None:
        """
        Build the index from vector data.

        Parameters
        ----------
        data : sequence of vectors
            N vectors. Length must match index dimension `d`.
        keys : sequence of keys, optional
            Use supplied keys as index instead of integer index.
        """        
        # Sanity checks
        if len(data) == 0:
            raise ValueError('Cannot build index from empty array')        
        if len(data) < self.R:
            raise ValueError(f'Cannot build index for fewer vectors than R={self.R}')
        if keys is not None and len(data) != len(keys):
            raise ValueError(f"List of keys supplied, but it's length {len(keys)} does not match length of data {len}")        
        if len(data[0]) != self.d:
            raise ValueError(f'Data dimention {len(data[0])} does not match expected dimension {self.d}')
        
        # Copy keys and vectors
        self.npts = len(data)        
        self.keys = list(keys) if keys else np.arange(self.npts)
        self.key_index = { k:i for i,k in enumerate(self.keys) }
        self.vectors = np.array(data)

        # Build graph
        self._vamana_indexing()

    def _vamana_indexing(self) -> None:
        """
        Construct a Vamana graph.
        Implements Algorithm 3 in the [paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf).=
        """        
        # Initialize to randomly chosen edges
        self.edges = randomly_wired_edges(self.npts, self.R)
        # Set entry point to medoid.
        self.entry_point = distance.medoid(self.vectors, self.dist_func_name)
        # Update paths
        for p in np.random.permutation(self.npts):
            # Find path to p
            nearest, visited = self._greedy_search(self.vectors[p], 1)
            # Update neighbour of p based on visited path
            self._robust_prune(p, visited)
            # Update p's neigbhbours to point back to p
            for nbr in self.edges[p]:
                if len(self.edges[nbr]) < self.R:
                    self.edges[nbr].add(p)
                else:
                    self._robust_prune(nbr, self.edges[nbr].union([p]))

    def _greedy_search(self, x: ArrayLike, k: int = 1, start: Optional[int] = None, L: Optional[int] = None) -> tuple[list[int], set[int]]:
        """
        Greedily explore to nearest point, and return set of visited nodes.
        Implements Algorithm 1 in the [paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf).

        Parameters
        ----------
        x : ArrayLike
            Vector to search for, dimension d.
        k : int, optional
            Number of nearest neighbours to find, by default 1
        start : int, optional
            If provided, index of start node instead of `self.entry_point`.
        L : Optional[int], optional
            Length of neighbour candidate list. If not provided, uses `self.L`.

        Returns
        -------
        C : list[int]
            List of k approximate nearest neighbours
        
        V : set[int]
            Set of indexes of visited nodes
        """        
        # Normalize L and start, make sure everything is valid        
        if L is None:
            L = self.L
        assert L >= k or L == 0
        assert len(x) == self.d
        if start is None:
            start = self.entry_point
        else:
            assert start >= 0 and start < self.npts

        # Distance to query                
        x = np.asarray(x)
        distance = lambda idx: self.dist_func(self.vectors[idx], x)
        
        # Begin at start point
        search_list = VisitPriorityQueue(maxlen=L)
        search_list.insert(distance(start), start) # push distance first to allow sorting
        visited = set()
        # Expand until no unvisited candidate is left
        # (VisitPriorityQueue takes care of iterating over unvisited nodes in order of distance)
        for p in search_list.visit():            
            # Mark p as visited
            visited.add(p)
            # Add all unvisited neighbours to candidate list
            # (VisitPriorityQueue takes care of truncating to L top neighbours)
            for nbr in self.edges[p]:
                if nbr not in visited: 
                    search_list.insert(distance(nbr), nbr)
            # TODO: checking if nbr is in visited list may not be worth it, since search_list eliminates duplicates
        # Return k nearest nodes and visited nodes
        return search_list.ksmallest(k), visited

    def _robust_prune(self, p: int, visited: set[int], alpha: Optional[float] = None) -> None:
        """
        Use the visited path from the entry point during greedy search to prune out edges of a point. 
        Implements Algorithm 2 in the [paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf).
        
        Note `visited` is modified during the run.

        Parameters
        ----------
        p : int
            Index of point whose neighbours we are pruning.
        visited : set[int]
            Candidates for edges of p (the set of points visited during `_greedy_search`). Modified during operation.
        alpha : Optional[float], optional
            Distance growth factor. Omit to use `self.alpha`.
        """
        assert p >= 0 and p < self.npts
        assert len(visited) > 0
        if alpha is None:
            alpha = self.alpha

        # Add current neighbours of p to list of candidates
        visited.update(self.edges[p])
        # Avoid self loops if any
        visited.discard(p)

        # Precompute distances from p
        dist_from_p = { v: self.dist_func(self.vectors[p], self.vectors[v]) for v in visited }
        # TODO: Since visited never grows, can optimize using array and setting inf distance to "remove".

        # Start without neighbours
        out_edges = set()
        # Iterate until no more candidates
        while visited:
            # Find neareast candidate 
            d, p_star = min( (dist_from_p[v], v) for v in visited )
            # Add it to list of neigbhours
            out_edges.add(p_star)
            # If we have R edges, we're done
            if len(out_edges) == self.R:
                break
            # Remove from candidate list points that are too close            
            visited = set( v for v in visited if dist_from_p[v] < alpha*self.dist_func(self.vectors[p_star], self.vectors[v]) )
        
        # Update the out neighbours of p        
        self.edges[p] = out_edges


