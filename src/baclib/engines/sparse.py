"""
Engines that rely on scipy.sparse data and algorithms.
"""
from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Union, Callable, Optional, Literal

import numpy as np
from scipy.sparse import csgraph, csr_matrix

from baclib.lib.resources import jit, RESOURCES
from baclib.containers.cluster import ClusterBatch
from baclib.containers.graph import Graph, Path, PathBatch

if RESOURCES.has_module('numba'):
    from numba import prange
else:
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Aggregator(IntEnum):
    """Enumeration of aggregation modes for node attributes."""
    TO = 0
    FROM = 1
    SUM = 2
    MEAN = 3
    MIN = 4
    MAX = 5


class AttributeSource(IntEnum):
    """Whether a graph weight is sourced from an edge or a node attribute."""
    EDGE = 0
    NODE = 1


@dataclass(frozen=True, slots=True)
class CsGraphPolicy:
    """
    Defines HOW a matrix should be constructed.
    Frozen = Immutable and Hashable (can be used as a cache key).

    Attributes:
        attr: The attribute name to use for weighting.
        source: Whether to pull the attribute from the 'edge' or the 'node'.
        aggregator: How to combine node attributes if source='node'.
        invert: If True, uses 1/value as the weight (useful for converting similarity to distance).
        default: Default value if the attribute is missing.
    """
    attr: bytes
    source: Union[AttributeSource, str] = AttributeSource.EDGE
    aggregator: Union[Aggregator, str, int] = Aggregator.TO
    invert: bool = False
    default: float = 1.0

    def __post_init__(self):  # Robust Aggregator Normalization
        # Source Normalization
        if not isinstance(self.source, AttributeSource):
            try:
                val = AttributeSource[self.source.upper()]
                object.__setattr__(self, 'source', val)
            except KeyError:
                raise ValueError(f"Invalid source: {self.source}")

        if isinstance(self.aggregator, Aggregator):
            return

        val = None
        if isinstance(self.aggregator, str):
            try:
                val = Aggregator[self.aggregator.upper()]
            except KeyError:
                raise ValueError(f"Invalid aggregator name: {self.aggregator}")
        else:
            try:
                val = Aggregator(self.aggregator)
            except ValueError:
                raise ValueError(f"Invalid aggregator value: {self.aggregator}")

        object.__setattr__(self, 'aggregator', val)


class CsGraphBuilder:
    """Constructs scipy CSR matrices from a ``Graph`` according to a ``CsGraphPolicy``.

    Results are cached on the ``Graph`` for repeated queries.
    """
    _MAX_PENALTY = 1e12

    @classmethod
    def _build_matrix(cls, graph: Graph, policy: CsGraphPolicy) -> 'csr_matrix':
        """
        Optimized matrix builder using Coordinate Format (COO) -> CSR.
        Uses pre-allocation and vectorized operations where possible.
        """
        # 1. Check Cache
        if (cached := graph.matrix_cache.get(policy)) is not None: return cached

        n = len(graph.nodes)
        if n == 0: return csr_matrix((0, 0))

        graph.ensure_topology()
        u_idx, v_idx, edge_list = graph.topology_cache
        if len(u_idx) == 0: return csr_matrix((n, n))

        # Determine reduction mode for multigraphs
        # Map TO(0)/FROM(1) to MIN(4) as 'direction' doesn't imply summation for parallel edges
        reduce_mode = policy.aggregator.value
        if reduce_mode <= 1: reduce_mode = 4

        if policy.source == AttributeSource.NODE:
            node_vals = np.full(n, policy.default, dtype=np.float64)
            for n_id, attrs in graph.node_attributes.items():
                if (idx := graph.node_to_idx.get(n_id)) is not None:
                    val = attrs.get(policy.attr)
                    if val is not None: node_vals[idx] = float(val)

            agg_mode = policy.aggregator.value

            rows, cols, data = _build_node_graph_kernel(
                u_idx, v_idx, node_vals, agg_mode, graph.directed, policy.invert, cls._MAX_PENALTY
            )
        else:
            # Optimization: Pre-allocate array to avoid list overhead
            n_edges = len(edge_list)
            vals_arr = np.full(n_edges, policy.default, dtype=np.float64)
            attr = policy.attr
            for i, e in enumerate(edge_list):
                if (val := e.attributes.get(attr)) is not None: vals_arr[i] = float(val)

            rows, cols, data = _build_edge_graph_kernel(
                u_idx, v_idx, vals_arr, graph.directed, policy.invert, cls._MAX_PENALTY
            )

        # Multigraph Reduction: Sort and Reduce
        if len(rows) > 0:
            order = np.lexsort((cols, rows))
            rows, cols, data = rows[order], cols[order], data[order]
            rows, cols, data = _reduce_duplicates_kernel(rows, cols, data, reduce_mode)

        csr = csr_matrix((data, (rows, cols)), shape=(n, n))
        graph.matrix_cache[policy] = csr
        return csr

    @classmethod
    def get_matrix(cls, graph: Graph, policy: CsGraphPolicy) -> csr_matrix:
        """
        Fetches the matrix corresponding to the policy from the cache,
        or builds and caches it if it does not exist.

        Args:
            policy: The WeightingPolicy defining how to build the matrix.

        Returns:
            A scipy.sparse.csr_matrix representing the graph weights.
        """
        if (cached := graph.matrix_cache.get(policy)) is None:
            graph.matrix_cache[policy] = (cached := cls._build_matrix(graph, policy))
        return cached


class PathAlgorithm(str, Enum):
    """Shortest-path algorithm selector wrapping scipy.sparse.csgraph routines."""
    DIJKSTRA = 'D'
    BELLMAN_FORD = 'BF'
    JOHNSON = 'J'
    FLOYD_WARSHALL = 'FW'
    
    @classmethod
    def normalize(cls, arg: Union[str, 'PathAlgorithm']) -> 'PathAlgorithm':
        """Normalizes a string or enum member to a PathAlgorithm."""
        if isinstance(arg, cls): return arg
        s = str(arg).upper()
        try: return cls[s]  # Try name (e.g. DIJKSTRA)
        except KeyError: pass
        try: return cls(s)  # Try value (e.g. D)
        except ValueError: raise ValueError(f"Unknown algorithm: {arg}")


class PathFinder(CsGraphBuilder):
    """Shortest-path and traversal queries over a weighted ``Graph``.

    Inherits ``CsGraphBuilder`` to transparently build and cache CSR matrices.
    """
    _REGISTRY: dict[PathAlgorithm, Callable] = {
        PathAlgorithm.DIJKSTRA: csgraph.dijkstra,
        PathAlgorithm.FLOYD_WARSHALL: csgraph.floyd_warshall,
        PathAlgorithm.JOHNSON: csgraph.johnson,
        PathAlgorithm.BELLMAN_FORD: csgraph.bellman_ford
    }

    @classmethod
    def _get_pathfinder(cls, algo: PathAlgorithm) -> Callable:
        if finder := cls._REGISTRY.get(algo): return finder
        raise ValueError(f"Algorithm {algo} is not available.")

    @staticmethod
    def _reconstruct_path(graph: Graph, start_idx: int, end_idx: int, predecessors: np.ndarray, 
                          cost: float) -> Optional[Path]:
        """Helper to reconstruct path from predecessor array."""
        # Use Numba kernel for fast array traversal
        path_indices = _reconstruct_path_kernel_single(predecessors, start_idx, end_idx)
        if len(path_indices) == 0: return None
        path_nodes = [graph.nodes[i] for i in path_indices]  # Kernel returns Start->End
        return Path(path_nodes, cost)

    @classmethod
    def shortest_path(cls, graph: Graph, policy: CsGraphPolicy, start: bytes, end: bytes,
                      algorithm: Union[str, PathAlgorithm] = PathAlgorithm.DIJKSTRA,
                      max_cost: float = np.inf) -> Optional[Path]:
        """
        Finds the shortest path between two nodes using the specified algorithm.
        """
        if (start_idx := graph.node_to_idx.get(start)) is None or (end_idx := graph.node_to_idx.get(end)) is None:
            return None
        csr =cls.get_matrix(graph, policy)
        finder = cls._get_pathfinder(algo := PathAlgorithm.normalize(algorithm))
        try:
            if algo == PathAlgorithm.DIJKSTRA:
                dist, preds = finder(csr, directed=graph.directed, indices=start_idx, return_predecessors=True,
                                     limit=max_cost)
            elif algo == PathAlgorithm.FLOYD_WARSHALL:
                dist_mat, pred_mat = finder(csr, directed=graph.directed, return_predecessors=True)
                dist = dist_mat[start_idx, end_idx]
                preds = pred_mat[start_idx]
            else:  # BF, J
                dist, preds = finder(csr, directed=graph.directed, indices=start_idx, return_predecessors=True)
        except ValueError:
            return None  # Negative cycle
        # Handle result extraction
        d = dist if np.isscalar(dist) else dist[end_idx]
        if np.isinf(d) or d >= (cls._MAX_PENALTY * 0.1): return None
        return cls._reconstruct_path(graph, start_idx, end_idx, preds, float(d))

    @classmethod
    def traverse(cls, graph: Graph, policy: CsGraphPolicy, start: bytes, mode: Literal['DFS', 'BFS'] = 'DFS',
                 return_predecessors: bool = True) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Returns a breadth-first or depth-first traversal of the graph.
        Wraps scipy.sparse.csgraph.breadth_first_order or scipy.sparse.csgraph.depth_first_order.

        Args:
            graph: The graph.
            policy: Matrix construction policy.
            start: Start node ID.
            mode: Traverse mode.
            return_predecessors: If True, returns (node_indices, predecessors).

        Returns:
            node_indices: Array of visited node indices in order.
            predecessors: Array of predecessor indices (if requested).
        """
        if (start_idx := graph.node_to_idx.get(start)) is None:
            raise ValueError(f"Start node {start} not found")
        csr = cls.get_matrix(graph, policy)
        func = csgraph.depth_first_order if mode == 'DFS' else csgraph.breadth_first_order
        return func(csr, start_idx, directed=graph.directed, return_predecessors=return_predecessors)


    @classmethod
    def find_paths(cls, graph: Graph, policy: CsGraphPolicy, start: bytes, end: bytes, max_hops: int = 10) -> PathBatch:
        """
        Finds all simple paths between start and end nodes within a hop limit.
        Useful for exploring local graph topology (e.g., bubbles) between anchors.
        """
        s_idx = graph.node_to_idx.get(start)
        e_idx = graph.node_to_idx.get(end)
        if s_idx is None or e_idx is None: return PathBatch.from_paths([])

        # Access CSR internals directly for speed
        csr = cls.get_matrix(graph, policy)
        indices = csr.indices
        indptr = csr.indptr
        data = csr.data

        results = []

        # Optimization: Recursive Backtracking to avoid list copying overhead
        # Using a single mutable list 'path' is much faster than 'path + [neighbor]'
        def _dfs(curr, path, cost):
            if curr == e_idx:
                nodes = [graph.nodes[i] for i in path]
                results.append(Path(nodes, cost))
                return

            if len(path) > max_hops: return

            for i in range(indptr[curr], indptr[curr + 1]):
                neighbor = indices[i]
                if neighbor not in path:  # O(Depth) check, acceptable for small max_hops
                    weight = data[i]
                    path.append(neighbor)
                    _dfs(neighbor, path, cost + weight)
                    path.pop()

        _dfs(s_idx, [s_idx], 0.0)
        return PathBatch.from_paths(results)


class ClusterFinder(CsGraphBuilder):
    """
    Algorithms for clustering and covering a Graph.
    """
    @staticmethod
    def otsu(similarity_matrix) -> float:
        """
        Calculates Otsu's threshold for a similarity matrix.

        Args:
            similarity_matrix (np.ndarray): Array of similarity scores.

        Returns:
            float: The calculated threshold.
        """
        if len(similarity_matrix) == 0: return 0.5
        hist, bin_edges = np.histogram(similarity_matrix, bins=256, range=(0.0, 1.0))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight = hist.cumsum()
        mean = (hist * bin_centers).cumsum()
        total_mean = mean[-1]
        total_weight = weight[-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_bg = mean / weight
            mean_fg = (total_mean - mean) / (total_weight - weight)
        mean_bg[np.isnan(mean_bg)] = 0.0
        mean_fg[np.isnan(mean_fg)] = 0.0
        w0 = weight
        w1 = total_weight - weight
        between_class_variance = w0 * w1 * (mean_bg - mean_fg) ** 2
        idx = np.argmax(between_class_variance)
        return bin_centers[idx]

    @classmethod
    def connected_components(cls, graph: Graph, policy: CsGraphPolicy) -> ClusterBatch:
        """
        Returns a ClusterBatch of connected components.
        """
        # Use a default PathFinder to get connectivity (weights don't matter)
        csr = cls.get_matrix(graph, policy)
        
        n, labels = csgraph.connected_components(csr, connection='weak', directed=graph.directed)

        # Sort nodes by component label to group them
        order = np.argsort(labels)
        sorted_labels = labels[order]
        
        # Find boundaries where labels change
        _, start_indices = np.unique(sorted_labels, return_index=True)
        
        offsets = np.zeros(len(start_indices) + 1, dtype=np.int32)
        offsets[:-1] = start_indices
        offsets[-1] = len(labels)
        
        # Map indices to node IDs
        nodes_arr = np.array(graph.nodes, dtype=object)
        flat_nodes = nodes_arr[order]
        
        ids = np.array([b'cc_%d' % i for i in range(len(start_indices))], dtype=object)
        
        return ClusterBatch(flat_nodes, offsets, ids=ids)

    @classmethod
    def greedy_set_cover(cls, graph: Graph, policy: CsGraphPolicy) -> ClusterBatch:
        """
        Solves Set Cover and returns a ClusterBatch where each cluster is a selected set.
        Rows are treated as Sets, Columns as Elements.
        """
        csr = cls.get_matrix(graph, policy)
        
        # Use Scipy to build the inverted index (CSC) efficiently
        csc = csr.tocsc()
        
        # Calculate max score (max set size) for bucket initialization
        # We can infer this from the CSR index pointers
        max_score = 0
        if csr.shape[0] > 0:
            max_score = np.max(np.diff(csr.indptr))

        indices = _greedy_set_cover_kernel(
            csr.shape[0], int(max_score), 
            csr.indptr, csr.indices, 
            csc.indptr, csc.indices
        )
        
        # Gather members
        sizes = np.diff(csr.indptr)[indices]
        offsets = np.zeros(len(indices) + 1, dtype=np.int32)
        np.cumsum(sizes, out=offsets[1:])
        
        total_size = offsets[-1]
        flat_indices = np.empty(total_size, dtype=np.int32)
        
        _fill_cluster_members(indices, csr.indptr, csr.indices, offsets, flat_indices)
        
        nodes_arr = np.array(graph.nodes, dtype=object)
        flat_nodes = nodes_arr[flat_indices]
        representatives = nodes_arr[indices]
        
        return ClusterBatch(flat_nodes, offsets, representatives=representatives)


# Kernels -------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _greedy_set_cover_kernel(n_sets, max_score, csr_indptr, csr_indices, csc_indptr, csc_indices):
    set_scores = np.zeros(n_sets, dtype=np.int32)
    is_covered = np.zeros(len(csc_indptr) - 1, dtype=np.bool_)

    buckets = np.full(max_score + 1, -1, dtype=np.int32)
    next_set = np.full(n_sets, -1, dtype=np.int32)
    prev_set = np.full(n_sets, -1, dtype=np.int32)

    for i in range(n_sets):
        score = csr_indptr[i + 1] - csr_indptr[i]
        set_scores[i] = score
        if score > 0:
            head = buckets[score]
            next_set[i] = head
            if head != -1:
                prev_set[head] = i
            buckets[score] = i

    result_sets = []
    current_max_score = max_score
    while current_max_score > 0:
        while current_max_score > 0 and buckets[current_max_score] == -1: current_max_score -= 1
        if current_max_score == 0: break
        best_set = buckets[current_max_score]
        next_node = next_set[best_set]
        buckets[current_max_score] = next_node
        if next_node != -1: prev_set[next_node] = -1
        result_sets.append(best_set)
        set_scores[best_set] = 0
        start_ptr = csr_indptr[best_set]
        end_ptr = csr_indptr[best_set + 1]

        for idx in range(start_ptr, end_ptr):
            elem = csr_indices[idx]
            if is_covered[elem]: continue
            is_covered[elem] = True
            col_start = csc_indptr[elem]
            col_end = csc_indptr[elem + 1]
            for c_idx in range(col_start, col_end):
                other_set = csc_indices[c_idx]
                old_score = set_scores[other_set]
                if old_score == 0: continue
                p = prev_set[other_set]
                n = next_set[other_set]
                if p != -1: next_set[p] = n
                else: buckets[old_score] = n
                if n != -1: prev_set[n] = p
                new_score = old_score - 1
                set_scores[other_set] = new_score
                if new_score > 0:
                    head = buckets[new_score]
                    next_set[other_set] = head
                    prev_set[other_set] = -1
                    if head != -1: prev_set[head] = other_set
                    buckets[new_score] = other_set

    return result_sets


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _fill_cluster_members(selected_indices, indptr, indices, offsets, out_flat):
    n = len(selected_indices)
    for i in prange(n):
        set_idx = selected_indices[i]
        start = indptr[set_idx]
        end = indptr[set_idx+1]
        length = end - start
        
        out_start = offsets[i]
        out_flat[out_start:out_start+length] = indices[start:end]


@jit(nopython=True, parallel=True, cache=True)
def _build_edge_graph_kernel(u_indices, v_indices, values, directed, invert, max_penalty):
    n_edges = len(u_indices)
    out_len = n_edges if directed else n_edges * 2

    rows = np.empty(out_len, dtype=np.int32)
    cols = np.empty(out_len, dtype=np.int32)
    data = np.empty(out_len, dtype=np.float64)

    for i in prange(n_edges):
        u = u_indices[i]
        v = v_indices[i]
        val = values[i]

        if val <= 1e-9:
            val = max_penalty
        elif invert:
            val = 1.0 / val

        rows[i] = u
        cols[i] = v
        data[i] = val

        if not directed:
            idx_rev = n_edges + i
            rows[idx_rev] = v
            cols[idx_rev] = u
            data[idx_rev] = val

    return rows, cols, data


@jit(nopython=True, parallel=True, cache=True)
def _build_node_graph_kernel(u_indices, v_indices, node_vals, agg_mode, directed, invert, max_penalty):
    n_edges = len(u_indices)
    out_len = n_edges if directed else n_edges * 2

    rows = np.empty(out_len, dtype=np.int32)
    cols = np.empty(out_len, dtype=np.int32)
    data = np.empty(out_len, dtype=np.float64)

    for i in prange(n_edges):
        u = u_indices[i]
        v = v_indices[i]
        val_u = node_vals[u]
        val_v = node_vals[v]

        val_raw = 1.0
        if agg_mode == Aggregator.TO:
            val_raw = val_v
        elif agg_mode == Aggregator.FROM:
            val_raw = val_u
        elif agg_mode == Aggregator.SUM:
            val_raw = val_u + val_v
        elif agg_mode == Aggregator.MEAN:
            val_raw = (val_u + val_v) * 0.5
        elif agg_mode == Aggregator.MIN:
            val_raw = min(val_u, val_v)
        elif agg_mode == Aggregator.MAX:
            val_raw = max(val_u, val_v)

        val = val_raw
        if val <= 1e-9:
            val = max_penalty
        elif invert:
            val = 1.0 / val

        rows[i] = u
        cols[i] = v
        data[i] = val

        if not directed:
            idx_rev = n_edges + i
            rows[idx_rev] = v
            cols[idx_rev] = u

            # Handle asymmetric aggregators for reverse edge
            if agg_mode == Aggregator.TO:
                val_rev_raw = val_u  # to (now u)
            elif agg_mode == Aggregator.FROM:
                val_rev_raw = val_v  # from (now v)
            else:
                val_rev_raw = val_raw  # Use raw value to avoid double inversion

            val_rev = val_rev_raw
            if val_rev <= 1e-9:
                val_rev = max_penalty
            elif invert:
                val_rev = 1.0 / val_rev

            data[idx_rev] = val_rev

    return rows, cols, data


@jit(nopython=True, cache=True, nogil=True)
def _reduce_duplicates_kernel(rows, cols, data, agg_mode):
    """
    Reduces duplicate (row, col) entries in sorted arrays using the specified aggregation mode.
    Assumes rows and cols are lexsorted.
    """
    n = len(rows)
    if n == 0: return rows, cols, data

    # 1. Count unique entries
    count = 1
    for i in range(1, n):
        if rows[i] != rows[i - 1] or cols[i] != cols[i - 1]:
            count += 1

    new_rows = np.empty(count, dtype=rows.dtype)
    new_cols = np.empty(count, dtype=cols.dtype)
    new_data = np.empty(count, dtype=data.dtype)

    # 2. Aggregate
    idx = 0
    new_rows[0] = rows[0]
    new_cols[0] = cols[0]
    curr_val = data[0]

    for i in range(1, n):
        if rows[i] != rows[i - 1] or cols[i] != cols[i - 1]:
            new_data[idx] = curr_val
            idx += 1
            new_rows[idx] = rows[i]
            new_cols[idx] = cols[i]
            curr_val = data[i]
        else:
            v = data[i]
            if agg_mode == 2:
                curr_val += v  # SUM
            elif agg_mode == 5:  # MAX
                if v > curr_val: curr_val = v
            else:  # MIN (Default for TO/FROM/MIN)
                if v < curr_val: curr_val = v

    new_data[idx] = curr_val
    return new_rows, new_cols, new_data


@jit(nopython=True, cache=True, nogil=True)
def _reconstruct_path_kernel_single(predecessors, start_idx, end_idx):
    # predecessors is 1D array
    curr = end_idx
    length = 0
    max_iter = len(predecessors)

    # First pass: calculate length
    while curr != -9999 and length <= max_iter:
        length += 1
        if curr == start_idx: break
        curr = predecessors[curr]

    if curr != start_idx and curr != -9999: return np.empty(0, dtype=np.int32)  # Cycle or broken
    if length == 0: return np.empty(0, dtype=np.int32)

    # Second pass: fill
    path = np.empty(length, dtype=np.int32)
    curr = end_idx
    idx = length - 1
    while idx >= 0:
        path[idx] = curr
        curr = predecessors[curr]
        idx -= 1

    return path