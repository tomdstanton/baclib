"""
Graph system using `scipy.sparse.csgraph` with separated Weighting Policy and intelligent caching.
"""
from collections import defaultdict
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Literal, Any, Optional, Union, List, Dict, Set, Iterable, Generator

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _cc, bellman_ford as _bf, dijkstra as _d, \
    johnson as _j, floyd_warshall as _fw

from baclib.utils.resources import RESOURCES, jit
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


class GraphAlgorithm(str, Enum):
    DIJKSTRA = 'D'
    BELLMAN_FORD = 'BF'
    JOHNSON = 'J'
    FLOYD_WARSHALL = 'FW'


@dataclass(frozen=True, slots=True)
class WeightingPolicy:
    """
    Defines HOW a matrix should be constructed.
    Frozen = Immutable and Hashable (can be used as a cache key).

    Attributes:
        attr: The attribute name to use for weighting.
        source: Whether to pull the attribute from the 'edge' or the 'node'.
        aggregator: How to combine node attributes if source='node'.
        invert: If True, uses 1/value as the weight (useful for converting similarity to distance).
        default: Default value if the attribute is missing.
        name: Optional name for the policy.
    """
    attr: bytes
    source: Literal['edge', 'node'] = 'edge'
    aggregator: Aggregator = Aggregator.TO
    invert: bool = False
    default: float = 1.0
    name: Optional[str] = None


class Edge:
    """
    Represents a directed edge between two nodes with optional attributes.

    Attributes:
        u (bytes): Source node ID.
        v (bytes): Target node ID.
        attributes (dict): Edge attributes.

    Examples:
        >>> e = Edge(b'node1', b'node2', {b'weight': 10})
        >>> e.u
        b'node1'
    """
    __slots__ = ('_u', '_v', 'attributes')

    def __init__(self, u: Any, v: Any, attributes: dict[bytes, Any] = None):
        """
        Initializes an Edge.

        Args:
            u: Source node (or object with .id).
            v: Target node (or object with .id).
            attributes: Dictionary of edge attributes.
        """
        self._u: bytes = self._coerce_node(u)  # Extract the pointer
        self._v: bytes = self._coerce_node(v)  # Extract the pointer
        self.attributes = attributes or {}

    @property
    def u(self) -> bytes: return self._u
    @property
    def v(self) -> bytes: return self._v

    @staticmethod
    def _coerce_node(obj: Any) -> bytes:
        if isinstance(obj, bytes): return obj
        if hasattr(obj, 'id'): return obj.id
        if isinstance(obj, str): return obj.encode('ascii')
        return str(obj).encode('ascii')

    def reverse(self):
        """Returns a reversed copy of the edge."""
        return Edge(self.v, self.u, self.attributes.copy())

    def __eq__(self, other):
        if not isinstance(other, Edge): return NotImplemented
        return self._u == other._u and self._v == other._v and self.attributes == other.attributes

    def __hash__(self):
        return hash((self._u, self._v, tuple(sorted(self.attributes.items()))))

    def __repr__(self): return f"Edge({self.u}->{self.v})"
    def __getitem__(self, item): return self.attributes.get(item, None)
    def __iter__(self): return iter((self.u, self.v, self.attributes))


class Path:
    """
    Represents a path through the graph.

    Attributes:
        nodes (list[bytes]): List of node IDs in the path.
        total_cost (float): Total cost of the path.
        policy (WeightingPolicy): The policy used to calculate cost.
    """
    __slots__ = ('nodes', 'total_cost', 'policy')

    def __init__(self, nodes: list[bytes], cost: float, policy: WeightingPolicy = None):
        """
        Initializes a Path.

        Args:
            nodes: List of node IDs in the path.
            cost: Total cost of the path.
            policy: The weighting policy used to calculate the cost.
        """
        self.nodes = nodes
        self.total_cost = cost
        self.policy = policy

    def __len__(self): return len(self.nodes)
    def __iter__(self): return iter(self.nodes)
    def __getitem__(self, item): return self.nodes[item]
    def __add__(self, other: 'Path') -> 'Path':
        if not isinstance(other, Path): return NotImplemented
        if not self.nodes or not other.nodes: return self if self.nodes else other
        if self.nodes[-1] != other.nodes[0]:
            raise ValueError(f"Paths are not continuous: {self.nodes[-1]} != {other.nodes[0]}")
        if self.policy != other.policy:
            raise ValueError("Cannot concatenate paths with different weighting policies")

        new_nodes = self.nodes + other.nodes[1:]
        new_cost = self.total_cost + other.total_cost
        return Path(new_nodes, new_cost, self.policy)

    def __repr__(self):
        p_str = self.policy.name or self.policy.attr if self.policy else "None"
        return f"Path(steps={len(self.nodes)}, cost={self.total_cost:.4f}, policy={p_str})"


class EdgeBatch:
    """
    A columnar container for edges, optimized for batch processing and IO.
    Stores source/target nodes and attributes as numpy arrays.
    """
    __slots__ = ('_u', '_v', '_attributes')

    def __init__(self, u: np.ndarray, v: np.ndarray, attributes: Dict[bytes, np.ndarray] = None):
        self._u = np.array(u, dtype=object, copy=False)
        self._v = np.array(v, dtype=object, copy=False)
        self._attributes = attributes or {}
        
        if len(self._u) != len(self._v):
            raise ValueError("u and v arrays must have the same length")

    def __len__(self): return len(self._u)

    def __iter__(self):
        for i in range(len(self)):
            attrs = {k: v[i] for k, v in self._attributes.items()}
            yield Edge(self._u[i], self._v[i], attrs)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            attrs = {k: v[item] for k, v in self._attributes.items()}
            return Edge(self._u[item], self._v[item], attrs)
        elif isinstance(item, (slice, np.ndarray, list)):
            new_attrs = {k: v[item] for k, v in self._attributes.items()}
            return EdgeBatch(self._u[item], self._v[item], new_attrs)
        raise TypeError(f"Invalid index type: {type(item)}")

    @property
    def u(self) -> np.ndarray: return self._u
    @property
    def v(self) -> np.ndarray: return self._v
    @property
    def attributes(self) -> Dict[bytes, np.ndarray]: return self._attributes


class Graph:
    """
    A simple graph object which act as an abstraction layer on top on scipy sparce matrices; nodes are string
    references to objects in memory.

    Examples:
        >>> g = Graph()
        >>> g.add_node(b'A')
        >>> g.add_node(b'B')
        >>> g.add_edges([Edge(b'A', b'B', {b'weight': 5})])
        >>> p = WeightingPolicy(b'weight')
        >>> path = g.shortest_path(b'A', b'B', p)
        >>> path.total_cost
        5.0
    """
    _MAX_PENALTY = 1e12
    _DEFAULT_POLICY = WeightingPolicy(b'weight', default=1.0)
    _PATHFINDERS = {
        GraphAlgorithm.DIJKSTRA: _d,
        GraphAlgorithm.FLOYD_WARSHALL: _fw,
        GraphAlgorithm.BELLMAN_FORD: _bf,
        GraphAlgorithm.JOHNSON: _j
    }
    _ALGORITHMS = Union[str, GraphAlgorithm]
    __slots__ = ('_directed', '_multi', 'edges', '_nodes', '_node_to_idx', '_node_attributes', '_matrix_cache',
                 '_csc_cache', '_components_cache', '_topology_cache', '_topo_lists')

    def __init__(self, edges: Iterable[Union[Edge, tuple]] = None, directed: bool = True, multi: bool = True):
        """
        Initializes the Graph.

        Args:
            edges: Initial edges to add.
            directed: Whether the graph is directed.
            multi: Whether to allow multiple edges between the same nodes (Multigraph).
        """
        self._directed = directed
        self._multi = multi
        self.edges: Set[Edge] = set()
        self._nodes: List[bytes] = []
        self._node_to_idx: Dict[bytes, int] = {}
        self._node_attributes: Dict[bytes, dict] = {}
        # Cache: Policy -> CSR Matrix
        self._matrix_cache: Dict[WeightingPolicy, csr_matrix] = {}
        self._csc_cache: Dict[WeightingPolicy, csr_matrix] = {}
        self._components_cache = None
        self._topology_cache = None
        self._topo_lists = ([], [], [])
        if edges: self.add_edges(edges)

    def __repr__(self):
        # Note: len(self.edges) counts only the *unique* edge objects added,
        # not the total number of traversable connections in the undirected case.
        type_str = "Directed" if self._directed else "Undirected"
        kind_str = "Multigraph" if self._multi else "Graph"
        return (f"{type_str} {kind_str} with "
                f"{len(self._nodes)} nodes and {len(self.edges)} defined edges")

    def __iter__(self): return iter(self.edges)
    def __len__(self): return len(self.edges)
    def __getitem__(self, item): return self._nodes[item]
    def _invalidate_cache(self):
        self._matrix_cache.clear()
        self._csc_cache.clear()
        self._components_cache = None

    def add_node(self, node: Any, attributes: dict = None):
        """
        Adds a node to the graph.

        Args:
            node: The node ID.
            attributes: Optional dictionary of node attributes.
        """
        node = Edge._coerce_node(node)
        if node not in self._node_to_idx:
            self._node_to_idx[node] = len(self._nodes)
            self._nodes.append(node)
            if attributes: self._node_attributes[node] = attributes
            self._topology_cache = None
            self._invalidate_cache()
        elif attributes:
            if node not in self._node_attributes: self._node_attributes[node] = {}
            self._node_attributes[node].update(attributes)
            self._invalidate_cache()

    def add_edges(self, edges: Iterable[Union[Edge, tuple]]):
        """
        Batch optimization for adding multiple edges.

        Args:
            edges: An iterable of Edge objects or tuples.
        """
        # Optimization: Collect unique nodes first to reduce dict lookups from O(E) to O(V)
        if isinstance(edges, EdgeBatch):
            # Fast path for batch: access arrays directly
            potential_nodes = set(edges.u)
            potential_nodes.update(edges.v)
            edge_objs = edges
        else:
            potential_nodes = set()
            edge_objs = []
            for e in edges:
                if not isinstance(e, Edge): e = Edge(*e)
                edge_objs.append(e)
                potential_nodes.add(e.u)
                potential_nodes.add(e.v)
        
        # Batch add nodes
        for n in potential_nodes:
            if n not in self._node_to_idx: # Double check
                self._node_to_idx[n] = len(self._nodes)
                self._nodes.append(n)

        # Batch add edges
        u_list, v_list, e_list = self._topo_lists
        node_map = self._node_to_idx
        added = False
        for e in edge_objs:
            if e not in self.edges:
                self.edges.add(e)
                u_list.append(node_map[e.u])
                v_list.append(node_map[e.v])
                e_list.append(e)
                added = True

        if added:
            self._topology_cache = None
            self._invalidate_cache()

    def get_matrix(self, policy: WeightingPolicy) -> csr_matrix:
        """
        Fetches the matrix corresponding to the policy from the cache,
        or builds and caches it if it does not exist.

        Args:
            policy: The WeightingPolicy defining how to build the matrix.

        Returns:
            A scipy.sparse.csr_matrix representing the graph weights.
        """
        if (cached := self._matrix_cache.get(policy)) is None:
            self._matrix_cache[policy] = (cached := self._build_matrix(policy))
        return cached

    def _ensure_topology(self):
        if self._topology_cache is not None: return

        u_list, v_list, e_list = self._topo_lists
        self._topology_cache = (
            np.array(u_list, dtype=np.int32),
            np.array(v_list, dtype=np.int32),
            e_list
        )

    def _build_matrix(self, p: WeightingPolicy) -> csr_matrix:
        """
        Optimized matrix builder using Coordinate Format (COO) -> CSR.
        Uses pre-allocation and vectorized operations where possible.

        Args:
            p: The WeightingPolicy.

        Returns:
            The constructed CSR matrix.
        """
        n = len(self._nodes)
        if n == 0: return csr_matrix((0, 0))
        
        self._ensure_topology()
        u_idx, v_idx, edge_list = self._topology_cache
        if len(u_idx) == 0: return csr_matrix((n, n))

        default = float(p.default)
        
        # Determine reduction mode for multigraphs
        # Map TO(0)/FROM(1) to MIN(4) as 'direction' doesn't imply summation for parallel edges
        reduce_mode = p.aggregator.value
        if reduce_mode <= 1: reduce_mode = 4
        
        if p.source == 'node':
            node_vals = np.full(n, default, dtype=np.float64)
            for n_id, attrs in self._node_attributes.items():
                if (idx := self._node_to_idx.get(n_id)) is not None:
                    val = attrs.get(p.attr)
                    if val is not None: node_vals[idx] = float(val)
            
            agg_mode = p.aggregator.value
            
            rows, cols, data = _build_node_graph_kernel(
                u_idx, v_idx, node_vals, agg_mode, self._directed, p.invert, self._MAX_PENALTY
            )
        else:
            # Optimization: Pre-allocate array to avoid list overhead
            n_edges = len(edge_list)
            vals_arr = np.full(n_edges, default, dtype=np.float64)
            attr = p.attr
            for i, e in enumerate(edge_list):
                if (val := e.attributes.get(attr)) is not None: vals_arr[i] = float(val)
            
            rows, cols, data = _build_edge_graph_kernel(
                u_idx, v_idx, vals_arr, self._directed, p.invert, self._MAX_PENALTY
            )

        # Multigraph Reduction: Sort and Reduce
        if len(rows) > 0:
            order = np.lexsort((cols, rows))
            rows, cols, data = rows[order], cols[order], data[order]
            rows, cols, data = _reduce_duplicates_kernel(rows, cols, data, reduce_mode)

        matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
        return matrix

    def _reconstruct_path(self, start_idx: int, end_idx: int, predecessors: np.ndarray, cost: float, policy: WeightingPolicy) -> Optional[Path]:
        """Helper to reconstruct path from predecessor array."""
        # Use Numba kernel for fast array traversal
        path_indices = _reconstruct_path_kernel_single(predecessors, start_idx, end_idx)
        if len(path_indices) == 0: return None
        path_nodes = [self._nodes[i] for i in path_indices] # Kernel returns Start->End
        return Path(path_nodes, cost, policy)

    def shortest_path(self, start: bytes, end: bytes, policy: WeightingPolicy,
                      algorithm: _ALGORITHMS = 'D', max_cost: float = np.inf) -> Optional[Path]:
        """
        Finds the shortest path between two nodes using the specified algorithm.

        Args:
            start: The start node ID.
            end: The end node ID.
            policy: The WeightingPolicy.
            algorithm: Algorithm to use ('D', 'BF', 'J', 'FW'). Defaults to 'D' (Dijkstra).
            max_cost: Maximum cost (Dijkstra only).

        Returns:
            A Path object if a path exists, otherwise None.
        """
        if (start_idx := self._node_to_idx.get(start)) is None or (end_idx := self._node_to_idx.get(end)) is None: return None
        matrix = self.get_matrix(policy)
        if not (finder := self._PATHFINDERS.get(algorithm if isinstance(algorithm, GraphAlgorithm) else str(algorithm))): raise ValueError(f"Unknown algorithm: {algorithm}")
        try:
            if algorithm == GraphAlgorithm.DIJKSTRA:
                dist, preds = finder(matrix, directed=self._directed, indices=start_idx, return_predecessors=True, limit=max_cost)
            elif algorithm == GraphAlgorithm.FLOYD_WARSHALL:
                dist_mat, pred_mat = finder(matrix, directed=self._directed, return_predecessors=True)
                dist = dist_mat[start_idx, end_idx]
                preds = pred_mat[start_idx]
            else: # BF, J
                dist, preds = finder(matrix, directed=self._directed, indices=start_idx, return_predecessors=True)
        except ValueError: return None  # Negative cycle
        # Handle result extraction
        d = dist if np.isscalar(dist) else dist[end_idx]
        if np.isinf(d) or d >= 1e11: return None
        return self._reconstruct_path(start_idx, end_idx, preds, float(d), policy)

    def find_paths(self, start: bytes, end: bytes, max_hops: int = 10,
                   policy: WeightingPolicy = None) -> List[Path]:
        """
        Finds all simple paths between start and end nodes within a hop limit.
        Useful for exploring local graph topology (e.g., bubbles) between anchors.

        Args:
            start: Start node ID.
            end: End node ID.
            max_hops: Maximum number of edges to traverse.
            policy: Optional policy to calculate cost for the paths.

        Returns:
            List of Path objects.
        """
        s_idx = self._node_to_idx.get(start)
        e_idx = self._node_to_idx.get(end)
        if s_idx is None or e_idx is None: return []

        # Access CSR internals directly for speed
        if policy is None: policy = self._DEFAULT_POLICY
        matrix = self.get_matrix(policy)
        indices = matrix.indices
        indptr = matrix.indptr
        data = matrix.data

        results = []

        # Optimization: Recursive Backtracking to avoid list copying overhead
        # Using a single mutable list 'path' is much faster than 'path + [neighbor]'
        def _dfs(curr, path, cost):
            if curr == e_idx:
                nodes = [self._nodes[i] for i in path]
                results.append(Path(nodes, cost, policy))
                return

            if len(path) > max_hops: return

            for i in range(indptr[curr], indptr[curr+1]):
                neighbor = indices[i]
                if neighbor not in path: # O(Depth) check, acceptable for small max_hops
                    weight = data[i]
                    path.append(neighbor)
                    _dfs(neighbor, path, cost + weight)
                    path.pop()

        _dfs(s_idx, [s_idx], 0.0)
        return results

    def resolve_bridges(self, bridges: Iterable[tuple[bytes, bytes]], policy: WeightingPolicy,
                        algorithm: _ALGORITHMS = 'D', max_cost: float = np.inf,
                        create_edges: bool = False, pool: Executor = None) -> Generator[Path, None, None]:
        """
        Identifies paths between bridged nodes. Automatically switches to dense
        Floyd-Warshall for small graphs to improve performance.

        Args:
            bridges: Iterable of (start, end) tuples.
            policy: WeightingPolicy.
            algorithm: Algorithm to use.
            max_cost: Max path cost.
            create_edges: If True, creates bridge edges for missing paths.
            pool: Executor for parallel processing.

        Yields:
            Path objects.
        """
        # 1. Group bridges
        bridges_map = defaultdict(set)
        for u, v in bridges:
            bridges_map[u].add(v)
        if not bridges_map: return

        # --- HYBRID OPTIMIZATION START ---
        # If the graph is tiny (e.g. a small plasmid or contig cluster),
        # Dijkstra's overhead is wasteful. Solve ALL pairs at once using Dense FW.
        if len(self._nodes) < 200:
            yield from self._resolve_bridges_dense(bridges_map, policy, max_cost, create_edges)
            return
        # --- HYBRID OPTIMIZATION END ---
        if not (finder := self._PATHFINDERS.get(algorithm if isinstance(algorithm, GraphAlgorithm) else str(algorithm))): raise ValueError(f"Unknown algorithm: {algorithm}")
        # Existing Sparse Logic (Chunked Dijkstra) for large graphs
        if algorithm == GraphAlgorithm.FLOYD_WARSHALL:
            raise ValueError("Floyd-Warshall cannot be used with batched sparse resolution.")

        # ... (Rest of your existing Dijkstra logic) ...
        source_ids = list(bridges_map.keys())
        if policy is None: policy = self._DEFAULT_POLICY
        matrix = self.get_matrix(policy)

        chunk_size = 100
        tasks = []
        for i in range(0, len(source_ids), chunk_size):
            chunk = source_ids[i:i + chunk_size]
            tasks.append((chunk, bridges_map, matrix, finder, max_cost, policy, create_edges))

        new_edges_to_add = []
        if pool is None: pool = RESOURCES.pool

        # Note: We disable Numba parallelism inside the workers to prevent oversubscription
        # Assuming you have a flag or config for this in RESOURCES
        for paths, edges in pool.map(self._resolve_chunk, tasks):
            yield from paths
            if edges: new_edges_to_add.extend(edges)

        if new_edges_to_add: self.add_edges(new_edges_to_add)

    def resolve_scaffolds(self, scaffolds: Iterable[List[bytes]], policy: WeightingPolicy,
                          algorithm: _ALGORITHMS = 'D', max_cost: float = np.inf) -> Generator[Path, None, None]:
        """
        Resolves paths for multi-node skeletons (e.g., from AlignmentBatch.find_scaffolds).
        Stitches adjacent nodes in the skeleton using the shortest path.

        Args:
            scaffolds: Iterable of lists of node IDs (e.g. [A, B, C]).
            policy: WeightingPolicy.
            algorithm: Algorithm to use.
            max_cost: Max cost per hop.

        Yields:
            Path objects representing the fully resolved scaffold.
        """
        # 1. Collect all unique segments needed across all scaffolds
        # This allows us to batch-solve them using parallel Dijkstra
        scaffold_list = list(scaffolds)
        needed_segments = set()
        for skel in scaffold_list:
            for i in range(len(skel) - 1):
                needed_segments.add((skel[i], skel[i+1]))
        
        # 2. Batch resolve using the optimized resolve_bridges
        solved_segments = {}
        for path in self.resolve_bridges(needed_segments, policy, algorithm, max_cost):
            if not path.nodes: continue
            # Map (Start, End) -> Path
            key = (path.nodes[0], path.nodes[-1])
            solved_segments[key] = path
            
        # 3. Stitch scaffolds
        for skeleton in scaffold_list:
            if len(skeleton) < 2: continue
            
            full_path = None
            valid = True
            
            for i in range(len(skeleton) - 1):
                u, v = skeleton[i], skeleton[i+1]
                segment = solved_segments.get((u, v))
                if segment is None:
                    valid = False
                    break
                
                if full_path is None: full_path = segment
                else: full_path = full_path + segment
            
            if valid and full_path:
                yield full_path

    def _resolve_bridges_dense(self, bridges_map: Dict[bytes, Set[bytes]],
                               policy: WeightingPolicy, max_cost: float,
                               create_edges: bool) -> Generator[Path, None, None]:
        """
        Specialized solver for small graphs using Dense matrices + Floyd-Warshall.
        """
        # 1. Get Sparse Matrix and Densify
        # .toarray() is very fast for N < 200
        csr = self.get_matrix(policy)
        dense_matrix = csr.toarray()

        # 2. Run Floyd-Warshall (Compute ALL paths)
        # We use the scipy csgraph version but on a dense array.
        # It handles the dense format efficiently.
        dist_matrix, predecessors = _fw(dense_matrix, directed=self._directed, return_predecessors=True)

        found_edges = []

        # 3. Iterate requests and reconstruct
        for u, targets in bridges_map.items():
            u_idx = self._node_to_idx.get(u)

            # Source missing?
            if u_idx is None:
                if create_edges:
                    for v in targets:
                        found_edges.append(Edge(u, v, {b'type': b'bridge'}))
                        yield Path([u, v], cost=0.0, policy=policy)
                continue

            paths_for_u = []
            for v in targets:
                v_idx = self._node_to_idx.get(v)

                # Target missing?
                if v_idx is None:
                    if create_edges:
                        found_edges.append(Edge(u, v, {b'type': b'bridge'}))
                        yield Path([u, v], cost=0.0, policy=policy)
                    continue

                # Check path existence
                cost = dist_matrix[u_idx, v_idx]
                if cost > max_cost:  # FW doesn't support 'limit', so we check here
                    if create_edges:
                        found_edges.append(Edge(u, v, {b'type': b'bridge'}))
                        yield Path([u, v], cost=0.0, policy=policy)
                    continue

                # Reconstruct
                path_obj = self._reconstruct_path(u_idx, v_idx, predecessors[u_idx], float(cost), policy)
                if path_obj:
                    paths_for_u.append((v, path_obj))

            # Filter subsets
            endpoints = {v for v, p in paths_for_u}
            redundant = set()
            for v, p in paths_for_u:
                for node in p.nodes[:-1]:
                    if node in endpoints:
                        redundant.add(node)

            for v, p in paths_for_u:
                if v not in redundant:
                    yield p

        if found_edges:
            self.add_edges(found_edges)

    def _resolve_chunk(self, args: tuple) -> tuple[List[Path], List[Edge]]:
        """Worker method for resolve_bridges."""
        chunk_sources, bridges_map, matrix, finder, max_cost, policy, create_edges = args
        
        found_paths = []
        found_edges = []
        
        valid_sources = []
        valid_source_indices = []
        missing_sources = []
        
        for u in chunk_sources:
            if u in self._node_to_idx:
                valid_sources.append(u)
                valid_source_indices.append(self._node_to_idx[u])
            else:
                missing_sources.append(u)

        if valid_sources:
            kwargs = {'directed': self._directed, 'indices': valid_source_indices, 'return_predecessors': True, 'limit': max_cost}
            try:
                dists, preds = finder(matrix, **kwargs)
                
                # Batch preparation
                req_local = []
                req_global = []
                req_meta = [] # (u_id, v_id)

                for local_idx, u_id in enumerate(valid_sources):
                    targets = bridges_map[u_id]
                    for v_id in targets:
                        if v_id in self._node_to_idx:
                            req_local.append(local_idx)
                            req_global.append(self._node_to_idx[v_id])
                            req_meta.append((u_id, v_id))
                        elif create_edges:
                            found_edges.append(Edge(u_id, v_id, {b'type': b'bridge'}))
                            found_paths.append(Path([u_id, v_id], cost=0.0, policy=policy))

                if req_local:
                    flat, offs, valid = _reconstruct_paths_kernel(
                        preds, np.array(req_local, dtype=np.int32), np.array(req_global, dtype=np.int32),
                        dists, float(max_cost)
                    )
                    
                    paths_by_source = defaultdict(list)
                    for i in range(len(req_local)):
                        u_id, v_id = req_meta[i]
                        if valid[i]:
                            nodes = [self._nodes[n] for n in flat[offs[i]:offs[i+1]]]
                            cost = dists[req_local[i], req_global[i]]
                            p = Path(nodes, float(cost), policy)
                            paths_by_source[u_id].append((v_id, p))
                        elif create_edges:
                            found_edges.append(Edge(u_id, v_id, {b'type': b'bridge'}))
                            found_paths.append(Path([u_id, v_id], cost=0.0, policy=policy))

                    # Filter subsets
                    for u_id, items in paths_by_source.items():
                        endpoints = {v for v, p in items}
                        redundant = set()
                        for v, p in items:
                            for node in p.nodes[:-1]:
                                if node in endpoints:
                                    redundant.add(node)

                        for v, p in items:
                            if v not in redundant:
                                found_paths.append(p)
            except ValueError: pass

        if create_edges:
            for u_id in missing_sources:
                for v_id in bridges_map[u_id]:
                    found_edges.append(Edge(u_id, v_id, {b'type': b'bridge'}))
                    found_paths.append(Path([u_id, v_id], cost=0.0, policy=policy))

        return found_paths, found_edges

    def connected_components(self) -> Generator[List[bytes], None, None]:
        """
        Yields lists of node IDs for each connected component.

        Yields:
            Lists of node IDs.
        """
        if self._components_cache is None:
            # Use a dummy policy just to get connectivity
            n, labels = _cc(self.get_matrix(self._DEFAULT_POLICY), connection='weak', directed=self._directed)
            components = defaultdict(list)
            for idx, label in enumerate(labels):
                components[label].append(self._nodes[idx])
            self._components_cache = list(components.values())

        yield from self._components_cache

    def greedy_set_cover(self, policy: WeightingPolicy, return_ids: bool = False) -> Union[List[int], List[bytes]]:
        """
        Solves the Set Cover problem using a greedy approach on the graph matrix defined by the policy.
        Rows are treated as Sets, Columns as Elements.

        Args:
            policy: The WeightingPolicy to define the matrix.
            return_ids: If True, returns node identifiers (bytes) instead of indices.

        Returns:
            A list of indices representing the selected sets.
        """
        matrix = self.get_matrix(policy)
        # Ensure CSC for column access (Element -> Sets)
        # This is O(N) memory but critical for speed. Cached to avoid re-transposing.
        if (csc := self._csc_cache.get(policy)) is None:
            self._csc_cache[policy] = (csc := matrix.tocsc())

        # (We can optimize this, but getting max row len is cheap)
        if matrix.shape[0] == 0: return []
        max_score = matrix.getnnz(axis=1).max()
        # Run Numba optimized core
        indices = _greedy_set_cover_kernel(matrix.shape[0], int(max_score), matrix.indptr, matrix.indices, csc.indptr,
                                        csc.indices)
        if return_ids:
            return [self._nodes[i] for i in indices]
        return indices


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _greedy_set_cover_kernel(n_sets, max_score, csr_indptr, csr_indices, csc_indptr, csc_indices):
    """
    Numba-accelerated greedy set cover using a bucket queue system
    similar to MMSeqs2's C++ implementation.

    Args:
        n_sets: Number of sets.
        max_score: Maximum possible score (initial bucket size).
        csr_indptr: CSR index pointer array.
        csr_indices: CSR indices array.
        csc_indptr: CSC index pointer array.
        csc_indices: CSC indices array.

    Returns:
        A list of selected set indices.
    """

    # --- 1. Initialize Data Structures ---
    # Current score of each set (initially just the number of elements)
    # We map sets to their scores to know which bucket they are in
    set_scores = np.zeros(n_sets, dtype=np.int32)
    # Track which elements have been covered
    is_covered = np.zeros(len(csc_indptr) - 1, dtype=np.bool_)

    # --- 2. Build the Bucket Queue (Doubly Linked List in Arrays) ---
    # buckets[s] holds the 'head' set_id for score 's'
    # next_set / prev_set hold the links. -1 indicates null/none.

    buckets = np.full(max_score + 1, -1, dtype=np.int32)
    next_set = np.full(n_sets, -1, dtype=np.int32)
    prev_set = np.full(n_sets, -1, dtype=np.int32)

    # Calculate initial scores and populate buckets
    for i in range(n_sets):
        # Initial score is just the number of elements in the set (row length)
        # Note: If you have custom weights, pass them in here instead
        score = csr_indptr[i + 1] - csr_indptr[i]
        set_scores[i] = score
        if score > 0:
            # Insert into the head of the bucket for this score
            head = buckets[score]
            next_set[i] = head
            if head != -1:
                prev_set[head] = i
            buckets[score] = i

    # --- 3. The Greedy Loop ---
    result_sets = []
    # Start checking from the highest possible score
    current_max_score = max_score
    while current_max_score > 0:
        # 3a. Find the next non-empty bucket
        while current_max_score > 0 and buckets[current_max_score] == -1: current_max_score -= 1
        if current_max_score == 0: break
        # 3b. Pop the first set from this bucket
        best_set = buckets[current_max_score]
        # Remove best_set from the bucket list
        next_node = next_set[best_set]
        buckets[current_max_score] = next_node
        if next_node != -1: prev_set[next_node] = -1
        # Add to results
        result_sets.append(best_set)
        # We implicitly "remove" this set by setting its score to 0 so it's ignored later
        set_scores[best_set] = 0
        # 3c. "Cover" the elements and update neighbors
        start_ptr = csr_indptr[best_set]
        end_ptr = csr_indptr[best_set + 1]

        for idx in range(start_ptr, end_ptr):
            elem = csr_indices[idx]
            if is_covered[elem]: continue
            is_covered[elem] = True
            # CRITICAL STEP: Downgrade all OTHER sets that contain this element
            # This is the "unplug_set" equivalent from C++
            col_start = csc_indptr[elem]
            col_end = csc_indptr[elem + 1]
            for c_idx in range(col_start, col_end):
                other_set = csc_indices[c_idx]
                # Skip if it's the set we just picked, or if it's already "dead" (score 0)
                old_score = set_scores[other_set]
                if old_score == 0: continue
                # Remove 'other_set' from its CURRENT bucket (doubly linked list surgery)
                p = prev_set[other_set]
                n = next_set[other_set]
                if p != -1:
                    next_set[p] = n
                else:
                    buckets[old_score] = n  # It was the head of the list
                if n != -1: prev_set[n] = p
                # Update Score
                new_score = old_score - 1
                set_scores[other_set] = new_score
                # Insert 'other_set' into the NEW bucket (if score > 0)
                if new_score > 0:
                    head = buckets[new_score]
                    next_set[other_set] = head
                    prev_set[other_set] = -1  # It becomes new head
                    if head != -1: prev_set[head] = other_set
                    buckets[new_score] = other_set

    return result_sets

@jit(nopython=True, parallel=True, cache=True)
def _reconstruct_paths_kernel(predecessors, source_indices, target_indices, dists, max_cost):
    n_req = len(source_indices)
    path_lens = np.zeros(n_req, dtype=np.int32)
    valid = np.zeros(n_req, dtype=np.bool_)

    for i in prange(n_req):
        s_local = source_indices[i]
        t_global = target_indices[i]
        d = dists[s_local, t_global]

        if np.isinf(d) or d > max_cost:
            valid[i] = False
            continue

        curr = t_global
        length = 0
        # -9999 is SciPy's NULL index
        while curr != -9999:
            length += 1
            curr = predecessors[s_local, curr]

        if length > 0:
            path_lens[i] = length
            valid[i] = True

    offsets = np.zeros(n_req + 1, dtype=np.int32)
    total = 0
    for i in range(n_req):
        offsets[i] = total
        if valid[i]: total += path_lens[i]
    offsets[n_req] = total

    flat_nodes = np.empty(total, dtype=np.int32)

    for i in prange(n_req):
        if not valid[i]: continue
        s_local = source_indices[i]
        t_global = target_indices[i]
        curr = t_global
        idx = offsets[i+1] - 1
        while curr != -9999:
            flat_nodes[idx] = curr
            curr = predecessors[s_local, curr]
            idx -= 1
    return flat_nodes, offsets, valid


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
        
        if val <= 1e-9: val = max_penalty
        elif invert: val = 1.0 / val
            
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
        if agg_mode == Aggregator.TO: val_raw = val_v
        elif agg_mode == Aggregator.FROM: val_raw = val_u
        elif agg_mode == Aggregator.SUM: val_raw = val_u + val_v
        elif agg_mode == Aggregator.MEAN: val_raw = (val_u + val_v) * 0.5
        elif agg_mode == Aggregator.MIN: val_raw = min(val_u, val_v)
        elif agg_mode == Aggregator.MAX: val_raw = max(val_u, val_v)
        
        val = val_raw
        if val <= 1e-9: val = max_penalty
        elif invert: val = 1.0 / val
            
        rows[i] = u
        cols[i] = v
        data[i] = val
        
        if not directed:
            idx_rev = n_edges + i
            rows[idx_rev] = v
            cols[idx_rev] = u
            
            # Handle asymmetric aggregators for reverse edge
            if agg_mode == Aggregator.TO: val_rev_raw = val_u # to (now u)
            elif agg_mode == Aggregator.FROM: val_rev_raw = val_v # from (now v)
            else: val_rev_raw = val_raw # Use raw value to avoid double inversion
            
            val_rev = val_rev_raw
            if val_rev <= 1e-9: val_rev = max_penalty
            elif invert: val_rev = 1.0 / val_rev
            
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
        if rows[i] != rows[i-1] or cols[i] != cols[i-1]:
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
        if rows[i] != rows[i-1] or cols[i] != cols[i-1]:
            new_data[idx] = curr_val
            idx += 1
            new_rows[idx] = rows[i]
            new_cols[idx] = cols[i]
            curr_val = data[i]
        else:
            v = data[i]
            if agg_mode == 2: curr_val += v # SUM
            elif agg_mode == 5: # MAX
                if v > curr_val: curr_val = v
            else: # MIN (Default for TO/FROM/MIN)
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

    if curr != start_idx and curr != -9999: return np.empty(0, dtype=np.int32) # Cycle or broken
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
