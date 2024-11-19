import numpy as np
from numpy.typing import ArrayLike, NDArray
from heapq import heappush, heappop, nsmallest
from typing import Sequence, Iterable, Optional, Any, Callable
from . import distance
from .utils import VisitPriorityQueue
from dataclasses import dataclass

ProgressWrapper = Callable[[Sequence, str], Sequence]

def randomly_wired_edges(npts: int, nedges: int, progress_wrapper: Optional[ProgressWrapper] = None) -> list[set[int]]:
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
    progress_wrapper : Sequence, str -> Sequence, optional
            None, or a function that accepets a sequence S and description str, and yields the same sequence S. Useful for progress bar (try `tqdm`).

    Returns
    -------
    G : list of sets of integers
        A list of length npts with a sets of size nedges edges per node.
    """    
    if (nedges >= npts):
        raise RuntimeError('Must have more points than edges per point')
    
    if progress_wrapper is None:
        progress_wrapper = lambda S, d: S

    edges = [None] * npts
    # Set random edges row by row
    for i in progress_wrapper(range(npts), 'init edges'):
        # Generate nedges+1 neighbours without replacement
        nbrs = np.random.choice(npts, size=nedges+1, replace=False)
        for j in range(nedges):
            # To avoid self loop, replace i with the last generated neighbour if needed
            if nbrs[j] == i:
                nbrs[j] = nbrs[-1] # guaranteed not be i since we sample without replacement
                break
        edges[i] = set(nbrs[:-1])
    
    return edges

@dataclass
class QueryStats():
    """Statistics for Vamana queries."""    
    nhops: int = 0

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
    progress_wrapper : Sequence, str -> Sequence, optional
            None, or a function that accepets a sequence S and description str, and yields the same sequence S. Useful for progress bar (try `tqdm`).
    """            
    def __init__(self, d: int, dist_func: str = 'euclidean', R: int = 64, L: int = 100, alpha: float = 1.2,
                 progress_wrapper: Optional[ProgressWrapper] = None) -> None:
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

        self.progress_wrapper = progress_wrapper if progress_wrapper is not None else lambda S, d: S
        
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
    
    def query(self, x: ArrayLike, k:int = 1, L: Optional[int] = None, out_stats: Optional[QueryStats] = None) -> list[Any] | list[int]:
        """
        Return k approximate nearest neighbours to x.

        Parameters
        ----------
        x : ArrayLike
            Vector to search for, dimension d.
        k : int, optional
            How many neighbours to return, by default 1
        L : int, optional
            Length of neighbour candidate list. If not provided, uses `self.L`.
        out_stats : QueryStats, option
            Statistics about the query.            

        Returns
        -------
        out :
            list of keys or indexes of k nearest neighbours to x.
        """        
        if len(x) != self.d:
            raise ValueError(f'Dimension of x {len(x)} does not match index dimension {self.d}')
        if k < 1:
            raise ValueError(f'Must return at least 1 neighbour')
        if L is None:
            L = self.L        
        if L < k:
            raise ValueError(f'L ({L}) must be at least k ({k})')
        if out_stats is None:
            out_stats = QueryStats()
        
        x = np.asarray(x)
        knns, _ = self._greedy_search(x, k, L=L, out_stats=out_stats)
        return knns
                      

    def build(self, data: Sequence[ArrayLike], dtype: np.dtype = np.float64, keys: Optional[Sequence[Any]] = None) -> None:
        """
        Build the index from vector data.

        Parameters
        ----------
        data : sequence of vectors
            N vectors. Length must match index dimension `d`.
        dtype : np.dtype, optional
            Datatype for vector data, by default np.float64.
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
        # Efficient way to load data from sequence ( https://perfpy.com/871 )
        self.vectors = np.fromiter(data, dtype=(dtype, self.d), count=len(data))

        # Build graph
        self._vamana_indexing()

    def _vamana_indexing(self) -> None:
        """
        Construct a Vamana graph.
        Implements Algorithm 3 in the [paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf).=
        """        
        # Initialize to randomly chosen edges
        self.edges = randomly_wired_edges(self.npts, self.R, self.progress_wrapper)
        # Set entry point to medoid.
        self.entry_point = distance.medoid(self.vectors, self.dist_func_name)
        # Update paths
        for p in self.progress_wrapper(np.random.permutation(self.npts), 'indexing'):
            # Find path to p
            nearest, visited = self._greedy_search(self.vectors[p], 1)
            # Update neighbour of p based on visited path
            p_edges = self._robust_prune(p, visited)
            self.edges[p] = p_edges
            # Update p's neigbhbours to point back to p
            for nbr in p_edges:
                if len(self.edges[nbr]) < self.R:
                    self.edges[nbr].add(p)
                else:
                    self.edges[nbr] = self._robust_prune(nbr, self.edges[nbr].union([p]))

    def _greedy_search(self, x: ArrayLike, k: int = 1, start: Optional[int] = None, L: Optional[int] = None, out_stats: Optional[QueryStats] = None) -> tuple[list[int], set[int]]:
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
        out_stats : QueryStats, optional
            Statistics about the query.

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
        if out_stats is None:
            out_stats = QueryStats()

        # Distance to query
        x = np.asarray(x)
        distance = lambda idx: self.dist_func(self.vectors[idx], x)
        
        # Begin at start point
        search_list = VisitPriorityQueue(maxlen=L)
        search_list.insert(distance(start), start) # push distance first to allow sorting
        inserted = set([start])
        visited = set()
        # Expand until no unvisited candidate is left
        # (VisitPriorityQueue takes care of iterating over unvisited nodes in order of distance)
        for p in search_list.visit():            
            out_stats.nhops += 1
            # Mark p as visited
            visited.add(p)
            # Add all unvisited neighbours to candidate list (unless already there)
            for nbr in self.edges[p]:
                # Avoid reinserting same point twice to search list (also avoids recomputing distance)
                if nbr not in inserted: 
                    search_list.insert(distance(nbr), nbr, trim=True)
                    inserted.add(nbr)
            # Truncate to L top neighbours
            search_list.trim()            
        # Return k nearest nodes and visited nodes
        return search_list.ksmallest(k), visited

    def _robust_prune(self, p: int, visited: set[int], alpha: Optional[float] = None) -> set[int]:
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

        Returns
        -------
        out : set[int]
            New set of neighbours for p.
        """
        assert p >= 0 and p < self.npts
        assert len(visited) > 0
        if alpha is None:
            alpha = self.alpha

        # Add current neighbours of p to list of candidates
        visited.update(self.edges[p])
        # Avoid self loops if any
        visited.discard(p)

        # This implementation looks quite from paper the description.
        # It takes advantage of several facts:
        # * In every iteration iteration we select neighbour using distance from p to candidates
        # * The set of candidates ("visited") can only shrink: it never grows or changes otherwise.
        # * Hence the order of candidate evaluation is fixed, as it's based on min distance, which cannot change.
        # * Once a candidate is removed, we never need to consider it in any way again.
        # This allows some optimizations:
        # 1) Precompute distances to p
        # 2) Use arrays to avoid set construction, searches, and iteration: 
        #    instead of creating and managing the visited, we store indexes and and distances from p in array.
        #    To remove a node, set its distance to a sentinel (-1).
        # 3) Rather than executing argmin at each turn,  we can pre-sort based on distance from p.
        #    When iterating over nodes, simply skip those that were removed in previous iteration.
        # 4) Once we process or skip a candidate, we no longer need to compute or check its distance from p_star.
        # 5) Use numpy vector operation with masks (boolean indexing) to avoid iteration when checking distances from p and p_star

        n = len(visited) 
        # Convert to array
        visited = np.array(list(visited)) 
        # Precompute distances from p         
        # (this creates an extra copy of vectors, but it's worth it so we can use one_to_many)
        dist_from_p = distance.one_to_many(self.vectors[p], self.vectors[visited])        
        # Get the order of looking at candidates
        order = dist_from_p.argsort()
        # Update order of visited and dist arrays
        dist_from_p = dist_from_p[order]
        visited = visited[order]
        # Create contiguous array of the vectors of visited nodes, allowing for fast one-to-many computation (worth it)
        visited_vectors = self.vectors[visited]
        # Set dist_from_p[i] to mark_removed to mark as removed from candidate set
        mark_removed = -1
                       
        # Start without neighbours
        out_edges = set()
        # Iterate until no more candidates
        for idx in range(n):
            # If this candidate was already removed, move to next
            if dist_from_p[idx] == mark_removed:
                continue
            # Find neareast candidate (no need for argmin since we visit in sorted order)
            p_star = visited[idx]
            # Add it to list of neigbhours
            out_edges.add(p_star)
            # If we have R edges, we're done
            if len(out_edges) == self.R:
                break
            # Precompute distances from p_star to remaining vectors, times factor alpha 
            # (in theory wasteful, as some of these may have been removed from visited set; in practice it's faster)
            threshold_dist_from_p_star = distance.one_to_many(self.vectors[p_star], visited_vectors[idx:]) * alpha
            # Remove from candidate list points that are too close (by setting their distance to mark_removed).
            # We only need to update nodes not already seen (whose index >= idx)
            mask = dist_from_p[idx:] >= threshold_dist_from_p_star
            # Set items to remove. Since mask is shorter than dist array, use slicing and boolean indexing.
            dist_from_p[idx:][mask] = mark_removed
        
        # Return the out neighbours of p        
        return out_edges


