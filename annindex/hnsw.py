import numpy as np
from typing import Any, Optional
from numpy.typing import ArrayLike, NDArray
from heapq import heapify, heappush, heappop, heapreplace

from .base import BaseIndex, ProgressWrapper
from .distance import medoid

class HNSW(BaseIndex):
    """
    HNSW nearest neighbour search index, as described in the relevant
    [paper](https://dl.acm.org/doi/10.1109/tpami.2018.2889473).

    Despite the many parameters of the index, in general only the first three
    (`n_neighbours`, `ef_construction`, `ef_search`) must be tuned. See the
    notes below for further explanations.
     
    Parameters
    ----------
    d : int
        Vector dimension
    dist_name : str
        Distance function to use, by default, ``sqeuclidean``.
    n_neighbours : int, optional
        Number of neighbours for each node (M in paper), by default 16.
    ef_construction : int, optional
        Expansion factor during graph construction (efConstruction in paper).
        By default, 40.
    ef_search : int, optional
        Expansion factor during search (ef in paper).
        By default, 16.
    max_neighbours : int, optional
        Maximum neighbours for each node (M_max in paper), by default 
        `n_neighbours`.
    max_zero : int, optional
        Maximum neigbhours in layer 0 (M_max0 in paper). If not specified,
        set to 2 * `n_neighbours`.
    height_factor : float, optional
        Factor controlling level heigh selection (m_L in paper). If not
        specified, set to 1/ln(n_neighbours).
    select_method : str, optional
        How to select neighbours for a node. If ``simple``, a node is simply
        connected to its nearest neighbours. If ``heuristic``, the index makes
        sure neighbours of a node are not more close to each other than the
        node itself. By default, ``heuristic``.
    heuristic_extend : bool, optional
        When selecting neighbours using the ``heuristic`` method, should
        neigbhours of original candidate set also be added to the candidate set.
        By default, False.
    heuristic_keep : bool, optional
        When selecting neighbours using the ``heuristic`` method, should
        candidate neighbours pruned by the heuristic be added back to the 
        the neigbhour list until it is exactly ``n_neighbours`` long. 
        By default, False.    
    progress_wrapper : Sequence, str -> Sequence, optional
        None, or a function that accepets a sequence S and description str, 
        and yields the same sequence S. Useful for progress bar (try `tqdm`).

    Notes
    -----
    1. Generally only three parameters need to be tuned: `n_neighbours` (M),
       `ef_construction`, and `ef_search`. Their default values are based on
       the HNSW paper choices for SIFT (Table 3, Fig 15) and match default
       FAISS values.
    2. The `max_zero` (M_max0) and `height_factor` (m_L) build parameters use 
       heuristics from the paper, and normally need not be specified or tuned.
    3. The paper does not provide a heuristic for `max_neighbours` (M_max).
       We set it to `n_neighbours` (M) by default, matching FAISS behaviour
       in [set_default_probas](https://github.com/facebookresearch/faiss/blob/90e4c4dee00f4393c16cc8886f63cd7deb785fb1/faiss/impl/HNSW.h#L151).
    4. The default neighbour selection method `select_method` is set to 
       ``heuristic`` following the paper discussion (Sec. 4.1, Figs 3--5).
       The heuristic parameters `heuristic_extend` and `heuristic_keep` are
       by default ``False`` based on Sec 4. The advice in the paper is that
       extension should only be used for "extremely clustered data".       

    """             
    def __init__(self, d, dist_name = 'sqeuclidean',
                 n_neighbours: int = 16,
                 ef_contruction: int = 40,
                 ef_search: int = 16,
                 max_neighbours: Optional[int] = None,
                 max_zero: Optional[int] = None,
                 height_factor: Optional[float] = None,
                 select_method: str = 'heuristic',
                 heuristic_extend: bool = False,
                 heuristic_keep: bool = False,                 
                 progress_wrapper: Optional[ProgressWrapper] = None):
        super().__init__(d, dist_name, progress_wrapper)
        
        # Verify parameters
        if n_neighbours < 1:
            raise ValueError(f'n_neighbours should be at least 1,'
                             f' not {n_neighbours}')
        if ef_contruction < 1:
            raise ValueError(f'ef_contruction should be at least 1,'
                             f' not {ef_contruction}')
        if ef_search < 1:
            raise ValueError(f'ef_search shoul d be at least 1,'
                             f' not {ef_search}')
        if max_neighbours is None:
            max_neighbours = n_neighbours
        if max_zero is None:
            max_zero = 2 * n_neighbours
        if height_factor is None:
            height_factor = 1.0/np.log(n_neighbours)
        if max_neighbours < n_neighbours:
            raise ValueError(f'max_neighbours {max_neighbours} should be at'
                             f' least n_neighbours {n_neighbours}')
        if max_zero < n_neighbours:
            raise ValueError(f'max_zero {max_zero} should be at'
                             f' least n_neighbours {n_neighbours}')
        if height_factor < 0:
            raise ValueError(f'height_factor {height_factor} must be at least 0')      
        
        selection_methods = { 'simple': self._select_neighbours_simple,
                              'heuristic': self._select_neighbours_heuristic }
        if select_method not in selection_methods:
            raise ValueError(f'select_method {select_method}')

        self.n_neighbours = n_neighbours
        self.ef_contruction = ef_contruction
        self.ef_search = ef_search
        self.max_neighbours = max_neighbours
        self.max_zero = max_zero
        self.height_factor = height_factor
        self.select_method = select_method
        self.heuristic_extend = heuristic_extend
        self.heuristic_keep = heuristic_keep
        self._select_neighbours = selection_methods[select_method]

        self.graphs: list[dict[int, set]] = []

    def query(self, x: ArrayLike, k: int = 1, ef_search: Optional[int] = None,
              *args, **kwargs):
        """
        Return k approximate nearest neighbours to x.

        Parameters
        ----------
        x : ArrayLike
            Vector to search for, dimension d.
        k : int, optional
            How many neighbours to return, by default 1
        ef_search : int, optional
            Expansion factor during search (ef in paper), which should be 
            larger than `k`. If not provided, uses `self.ef_search`.

        Returns
        -------
        out :
            list of keys or indexes of k nearest neighbours to x.
        """        
        # Verify arguments
        super().query(x, k)

        if ef_search is None:
            ef_search = self.ef_search        
        if ef_search < k:
            raise ValueError(f'ef_search ({ef_search}) must be at least k ({k})')
        
        x = np.asarray(x)
        knns = self._knn_search(x, k, ef_search)
        return self._idxs_to_keys(knns)


    def build(self) -> None:
        # (Docstring inherited from parent)
        
        super().build()
        if len(self.graphs) > 0:
            raise RuntimeError('Index already built')

        # Insert first vector as entry point: graph with one node and no edges.
        self.entry_point = 0;
        self.graphs = [ {self.entry_point: set() }]     
        # It may be better to force medoid (inserted last) to be in all layers
        # and be the entry point. In that vein, it may be better to insert the
        # vectors in reverse distance from medoid so that medoid and nearby
        # are inserted last, getting a chance to be included in all layers.

        # Insert othjer vectors one at a time
        for i in self.progress_wrapper(range(1, self.npts), 'indexing'):
            self._insert_vector(i)

    def _knn_search(self, x: NDArray, k: int, ef: int) -> list[int]:  
        """
        Called by `query` to find the nearest neighbours of a vector. 
        Implements Algorithm 5 in the HNSW paper.

        Parameters
        ----------
        x : NDArray
            Query vector.
        k : int
            How many neighbours to return.
        ef : int
            Expansion factor during search.

        Returns
        -------
        out : list[int]
            Indexes of k nearest neighbours to x sorted by distance.

        See Also
        --------
        _search_layer: called to search within each layer
        
        """        
        pts = [ self.entry_point ]
        L = len(self.graphs)-1
        # From layer L to layer 1 use one point (ef=1)
        for layer in range(L, 0, -1):
            pts = self._search_layer(x, pts, 1, layer)[:1]
        # For layer 0, use ef points
        return self._search_layer(x, pts, ef, 0)[:k]
    
    def _search_layer(self, qvec: NDArray, 
                      entry_points: list[int],
                      ef: int,
                      layer) -> list[int]:
        """
        Searches for nearest elements in layer to a vector. Implements
        Algorithm 2 in the HNSW paper.

        Parameters
        ----------
        qvec : NDArray
            Query vector.
        entry_points : list[int]
            Initial list of nodes from which search will continue.
        ef : int
            How many neighbours to return.
        layer : _type_
            Which layer are we searching in.

        Returns
        -------
        out : list[int]
            `ef` nearest neighbours to qvec sorted by distance.

        See Also
        --------
        _insert_vector: calls `_search_layer` to select candidate neighbours
        _knn_search: calls `_search_layer` to find nearest neighbours of query
        """        
        
        # While the search here may be equivalent to Vamana greedy search, 
        # one of our goals is to adhere closely to the paper hence we base this
        # implementation on Algorithm 2.

        # Create heap for nearest neigbhour candidates so we can find nearest
        # in O(1) (Alg 2 line 5)
        distances = self.dist_func.one2many(qvec, self.vectors[entry_points])
        candidates = list(zip(distances, entry_points))
        heapify(candidates)

        # Create heap in reverse order (furthest is at top) so we can remove
        # most distant element (Alg 2 line 6)
        output = list(zip(-distances, entry_points))
        heapify(output)
        
        visited = set(entry_points)
      
        while len(candidates) > 0:
            # Get nearest candidate
            dist_cq, c = heappop(candidates)
            dist_fq = -(output[0][0])
            # Stop if nearest candidate is more distant than furthest in output
            if dist_cq > dist_fq:
                break
            # Explore neighbourhood of C
            for e in self.graphs[layer][c]:
                if e in visited:
                    continue
                visited.add(e)                
                # Add e to candidate and output heaps if needed
                dist_eq = self.dist_func.distance(qvec, self.vectors[e])
                dist_fq = -(output[0][0])
                if len(output) < ef:
                    # Add e to candidates and current output if there is space
                    heappush(output, (-dist_eq, e))
                    heappush(candidates, (dist_eq, e))
                elif dist_fq > dist_eq:
                    # Replace furthest item in output if e is closer to q
                    heapreplace(output, (-dist_eq, e)) # Alg 2 line 15--17
                    heappush(candidates, (dist_eq, e))
        
        # Pop the nearest ef neigbhours in reverse order from the output heap,
        # and return to user in the sorted order
        res = []
        while output:
            res.append(heappop(output)[1])
        res.reverse()
        return res
            
    def _select_neighbours_heuristic(self, q: int, 
                                     candidates: list[int],
                                     n_neighbours: int,
                                     layer: int) -> set[int]:
        """
        Heuristic version of select nearest neigbhour for a node that avoids
        having neighbours too close to each other. Implements Algorithm 4 in 
        the HNSW paper.

        Called via `self._select_neighbhours` which is set during `__init__`.

        Parameters
        ----------
        q : int
            Index of vector whose neighbours we want to select.
        candidates : list[int]
            Index of neighbour candidates (C in Alg 4)
        n_neighbours : int
            How many neighbours to select (M in Alg 4)
        layer : int
            Which layer are we searching.

        Returns
        -------
        out : set[int]
            Set of neighbours for q.
        
        See Also
        --------
        _select_neighbours_simple: the alternative simple approach
        _insert_vector: invokes neighbour selection as part of adding a vector
        _search_layer: produces the candidate list of neighbours
        __init__: selects between this and the heuristic approach

        """                
        # Extend working set of candidates if needed
        W = set(candidates)
        if self.heuristic_extend:
            for c in candidates:
                c_neigborhood = self.graphs[layer][c]
                for nbr in c_neigborhood:
                    W.add(nbr)
        candidates = list(W)
        
        distances = lambda x_idx, Y_idx: self.dist_func.one2many(self.vectors[x_idx], self.vectors[Y_idx])          
        dist_to_q = distances(q, candidates)
        # TODO: one can cache distance computaiton from search layer and use it
        # here in select neighbours, avoiding some recomputation 

        # TODO may be faster to precompute O(n^2) distances between
        # all candidates instead of distances(idx, output) each iteration as
        # ef is generally small

        output = list()      # R in Alg 4 of the paper
        discarded = list()   # W_d in Alg 4 of the paper         
        # Use argsort to go through candidates in order of distance from q
        order = np.argsort(dist_to_q)
        for idx in order:
            # idx is the nearest candidate to q
           
            # Add candidates that are closer to q than to any element in the
            # output list (Alg 4 Line 11)
            if len(output) == 0 or dist_to_q[idx] < distances(idx, output).max():                
                output.append(candidates[idx])
            else:
                discarded.append(candidates[idx])        
            # Stop if we have enough neighbours
            if len(output) == n_neighbours:
                break
        
        # Add back pruned connections until we have n_neighbours
        if self.heuristic_keep and len(output) < n_neighbours:
            # Indexes were added to discarded in order of distance from q so it            
            # is easy to just take from beginning of discarded
            n = min(len(discarded), n_neighbours - len(output))
            output.extend(discarded[:n])
            
        # Convert to set
        return set(output)

    def _select_neighbours_simple(self, q: int, 
                                  candidates: list[int],
                                  n_neighbours: int,
                                  layer: int) -> set[int]:
        """
        Simple version of select nearest neigbhour for a node. Implements 
        Algorithm 3 in the HNSW paper.

        Called via `self._select_neighbhours` which is set during `__init__`.

        Parameters
        ----------
        q : int
            Index of vector whose neighbours we want to select.
        candidates : list[int]
            Index of neighbour candidates (C in Alg 3)
        n_neighbours : int
            How many neighbours to select (M in Alg 3)
        layer : int
            Ignored. Included for compatibility with 
            `_select_neighbours_heuristic`.

        Returns
        -------
        out : set[int]
            Set of neighbours for q.
        
        See Also
        --------
        _select_neighbours_heuristic: the alternative heuristic approach
        _insert_vector: invokes neighbour selection as part of adding a vector
        _search_layer: produces the candidate list of neighbours
        __init__: selects between this and the heuristic approach

        """        
        # layer is ignored in the simple version

        # TODO: one can cache distance computaiton from search layer and use it
        # here in select neighbours, avoiding some recomputation 
        d = self.dist_func.one2many(self.vectors[q], self.vectors[candidates])
        return np.argpartition(d, n_neighbours)[:n_neighbours]
        

    def _insert_vector(self, q: int) -> None:        
        """
        Called from `build` to add the stored vector with index q to the index.
        Implements Algorithm 1 from the HNSW paper.

        Parameters
        ----------
        q : int
            Index of vectors to add from self.vectors.

        See Also
        --------
        _select_neighbours_simple: called via ``_select_neighbours``
        _select_neighbours_heuristic: called via ``_select_neighbours``
        _search_layer: called to produces the candidate list of neighbours
        """        
        M = self.n_neighbours
        qvec = self.vectors[q]

        # Current top level
        L = len(self.graphs)-1
        # Select level for inserted point
        q_layer = int(np.floor(-np.log(np.random.rand()) * self.height_factor))
        # Search layers L down to new_layer+1 (inclusive)
        cur_ep = self.entry_point
        for layer in range(L, q_layer, -1):
            # Find ef_construction closest nodes in layer l
            one_nearest = self._search_layer(qvec, [cur_ep], 1, layer)
            assert len(one_nearest) == 1
            # Continue with nearest point
            distances = self.dist_func.one2many(qvec, self.vectors[one_nearest])
            cur_ep = one_nearest[distances.argmin()]
        # Adding from new_layer down, this time with ef_construction candidates
        cur_ep = [cur_ep]
        for layer in range(min(L, q_layer), -1, -1):            
            # Find ef_construction candidates as neigbhours in this layer
            nearest = self._search_layer(qvec, cur_ep, self.ef_contruction, layer)
            # Find M neighbours
            neighbours = self._select_neighbours(q, nearest, M, layer)
            # Add q to layer and add its neighours
            self.graphs[layer][q] = neighbours
            # Add back-endges from edges to q
            M_max = self.max_neighbours if layer > 0 else self.max_zero
            for nbr in neighbours:
                nbr_edges = self.graphs[layer][nbr]
                nbr_edges.add(q)
                # Prune if too many out-edges for nbr
                # (Here we merge lines 11 and 12--16 of Algorithm 1. This is OK
                # since adding back edge to q from neighbour i does not affect 
                # pruning of neighbour j)                
                if len(nbr_edges) > M_max:
                    new_edges = self._select_neighbours(nbr, nbr_edges, M_max, layer)
                    self.graphs[layer][nbr] = new_edges           
            # Move to next layer with nearest neighbours as current candidates
            ep = nearest
            # If new_layer > L, we set q as new entry point, but also need to
            # actually create add those new layers
            if q_layer > L:
                self.entry_point = q
                for i in range(q_layer-L):
                    self.graphs.append({self.entry_point:set()})

