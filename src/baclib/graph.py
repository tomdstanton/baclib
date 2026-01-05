"""
Graph system using `scipy.sparse.csgraph` with separated Weighting Policy and intelligent caching.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Any, Optional, Union, List, Dict, Set, Iterable, Generator

import numpy as np
import scipy.sparse as sp

from . import jit


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
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
    attr: str
    source: Literal['edge', 'node'] = 'edge'
    aggregator: Literal['to', 'from', 'mean', 'sum', 'min', 'max'] = 'to'
    invert: bool = False
    default: float = 1.0
    name: Optional[str] = None


class Edge:
    """
    Represents a directed edge between two nodes with optional attributes.
    """
    __slots__ = ('u', 'v', 'attributes')

    def __init__(self, u: Any, v: Any, attributes: dict[str, Any] = None):
        """
        Initializes an Edge.

        Args:
            u: Source node (or object with .id).
            v: Target node (or object with .id).
            attributes: Dictionary of edge attributes.
        """
        self.u: str = _coerce_node(u)  # Extract the pointer
        self.v: str = _coerce_node(v)  # Extract the pointer
        self.attributes = attributes or {}

    def reverse(self):
        """Returns a reversed copy of the edge."""
        return Edge(self.v, self.u, self.attributes.copy())

    def __repr__(self): return f"Edge({self.u}->{self.v})"
    def __getitem__(self, item): return self.attributes.get(item, None)
    def __iter__(self): return iter((self.u, self.v, self.attributes))
    # Critical for set operations:
    def __hash__(self): return hash((self.u, self.v))
    def __eq__(self, other): return (self.u, self.v) == (other.u, other.v)


class Path:
    """
    Represents a path through the graph.
    """
    __slots__ = ('nodes', 'total_cost')

    def __init__(self, nodes: list[str], cost: float):
        """
        Initializes a Path.

        Args:
            nodes: List of node IDs in the path.
            cost: Total cost of the path.
        """
        self.nodes = nodes
        self.total_cost = cost

    def __repr__(self): return f"Path(steps={len(self.nodes)}, cost={self.total_cost:.4f})"


class Graph:
    """
    A simple graph object which act as an abstraction layer on top on scipy sparce matrices; nodes are string
    references to objects in memory.
    """
    _MAX_PENALTY = 1e12
    __slots__ = ('directed', 'edges', '_nodes', '_node_to_idx', '_node_attributes', '_matrix_cache')

    def __init__(self, *edges: Union[Edge, tuple], directed: bool = True):
        """
        Initializes the Graph.

        Args:
            *edges: Initial edges to add.
            directed: Whether the graph is directed.
        """
        self.directed = directed
        self.edges: Set[Edge] = set()
        self._nodes: List[str] = []
        self._node_to_idx: Dict[str, int] = {}
        self._node_attributes: Dict[str, dict] = {}
        # Cache: Policy -> CSR Matrix
        self._matrix_cache: Dict[WeightingPolicy, sp.csr_matrix] = {}
        if edges: self.add_edges_from(edges)

    def __repr__(self):
        # Note: len(self.edges) counts only the *unique* edge objects added,
        # not the total number of traversable connections in the undirected case.
        return (f"{'Directed' if self.directed else 'Undirected'} Graph with "
                f"{len(self._nodes)} nodes and {len(self.edges)} defined edges")

    def __iter__(self): return iter(self.edges)
    def __len__(self): return len(self.edges)
    def __getitem__(self, item): return self._nodes[item]
    def _invalidate_cache(self): self._matrix_cache.clear()

    def add_node(self, node: str, attributes: dict = None):
        """
        Adds a node to the graph.

        Args:
            node: The node ID.
            attributes: Optional dictionary of node attributes.
        """
        if node not in self._node_to_idx:
            self._node_to_idx[node] = len(self._nodes)
            self._nodes.append(node)
            if attributes: self._node_attributes[node] = attributes
            self._invalidate_cache()
        elif attributes:
            if node not in self._node_attributes: self._node_attributes[node] = {}
            self._node_attributes[node].update(attributes)
            self._invalidate_cache()

    def add_edge(self, edge: Union[Edge, tuple]):
        """
        Adds a single edge to the graph.

        Args:
            edge: An Edge object or a tuple representing an edge.
        """
        if not isinstance(edge, Edge):
            edge = Edge(*edge)
        self.add_node(edge.v)
        self.add_node(edge.u)
        if edge not in self.edges:
            self.edges.add(edge)
            self._invalidate_cache()

    def add_edges_from(self, edges: Iterable[Union[Edge, tuple]]):
        """
        Batch optimization for adding multiple edges.

        Args:
            edges: An iterable of Edge objects or tuples.
        """
        # Pre-process nodes to avoid checking dictionary repeatedly
        new_nodes = set()
        edge_objs = []

        for e in edges:
            if not isinstance(e, Edge): e = Edge(*e)
            edge_objs.append(e)
            if e.u not in self._node_to_idx: new_nodes.add(e.u)
            if e.v not in self._node_to_idx: new_nodes.add(e.v)

        # Batch add nodes
        for n in new_nodes:
            if n not in self._node_to_idx: # Double check
                self._node_to_idx[n] = len(self._nodes)
                self._nodes.append(n)

        # Batch add edges
        len_before = len(self.edges)
        self.edges.update(edge_objs)
        if len(self.edges) > len_before:
            self._invalidate_cache()

    def get_matrix(self, policy: WeightingPolicy) -> sp.csr_matrix:
        """
        Fetches the matrix corresponding to the policy from the cache,
        or builds and caches it if it does not exist.

        Args:
            policy: The WeightingPolicy defining how to build the matrix.

        Returns:
            A scipy.sparse.csr_matrix representing the graph weights.
        """
        if policy in self._matrix_cache:
            return self._matrix_cache[policy]
        matrix = self._build_matrix(policy)
        self._matrix_cache[policy] = matrix
        return matrix

    def _build_matrix(self, p: WeightingPolicy) -> sp.csr_matrix:
        """
        Optimized matrix builder using Coordinate Format (COO) -> CSR.

        Args:
            p: The WeightingPolicy.

        Returns:
            The constructed CSR matrix.
        """
        n = len(self._nodes)
        if n == 0: return sp.csr_matrix((0, 0))

        # 1. Prepare fast lookups
        node_map = self._node_to_idx
        node_attrs = self._node_attributes
        default = p.default
        attr_key = p.attr

        # 2. Accumulate coordinates and data
        # Using simple lists is faster than appending to arrays iteratively
        rows = []
        cols = []
        data = []

        # Optimization: Pre-fetch boolean flags
        is_edge_src = (p.source == 'edge')
        is_inverted = p.invert

        for edge in self.edges:
            try:
                u = node_map[edge.u]
                v = node_map[edge.v]
            except KeyError: continue # Should not happen if graph consistent

            # Determine Raw Value
            if is_edge_src:
                val = edge.attributes.get(attr_key, default)
            else:
                # Node Attribute Logic
                val_u = node_attrs.get(edge.u, {}).get(attr_key, default)
                val_v = node_attrs.get(edge.v, {}).get(attr_key, default)

                if p.aggregator == 'to': val = val_v
                elif p.aggregator == 'from': val = val_u
                elif p.aggregator == 'sum': val = val_u + val_v
                elif p.aggregator == 'mean': val = (val_u + val_v) * 0.5
                elif p.aggregator == 'min': val = min(val_u, val_v)
                elif p.aggregator == 'max': val = max(val_u, val_v)
                else: val = default

            # Apply Inversion
            weight = self._MAX_PENALTY if val <= 1e-9 else (1.0 / val) if is_inverted else val

            # Add forward edge
            rows.append(u)
            cols.append(v)
            data.append(weight)

            # Handle Undirected (Symmetry)
            if not self.directed:
                rows.append(v)
                cols.append(u)

                # If source is node, direction matters (u->v vs v->u attributes)
                if not is_edge_src and p.aggregator in ('to', 'from'):
                    # Recalculate for reverse
                    r_val = val_u if p.aggregator == 'to' else val_v

                    if is_inverted: data.append(self._MAX_PENALTY if r_val <= 1e-9 else 1.0/r_val)
                    else: data.append(r_val)
                # Symmetric edge attributes or symmetric aggregators (sum/mean)
                else: data.append(weight)

        # 3. Construct CSR Matrix directly
        # Duplicate entries (multigraphs) are summed by default in sparse constructor,
        # which is usually acceptable or desired for weight accumulation.
        matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        return matrix

    def dijkstra(self, start: str, end: str, policy: WeightingPolicy) -> Optional[Path]:
        """
        Finds the shortest path between two nodes using Dijkstra's algorithm.

        Args:
            start: The start node ID.
            end: The end node ID.
            policy: The WeightingPolicy to determine edge weights.

        Returns:
            A Path object if a path exists, otherwise None.
        """
        if start not in self._node_to_idx or end not in self._node_to_idx: return None

        start_idx = self._node_to_idx[start]
        end_idx = self._node_to_idx[end]
        dist, preds = sp.csgraph.dijkstra(
            self.get_matrix(policy), directed=self.directed, return_predecessors=True, indices=start_idx)

        if np.isinf(dist[end_idx]) or dist[end_idx] >= 1e11: return None

        # Reconstruct
        path_indices = []
        curr = end_idx
        while curr != -9999:  # scipy uses -9999 for "no predecessor"
            path_indices.append(curr)
            if curr == start_idx: break
            curr = preds[curr]
            # Safety break for disjoint graphs if -9999 check fails
            if curr == -9999 and path_indices[-1] != start_idx: return None

        path_nodes = [self._nodes[i] for i in reversed(path_indices)]

        # Double check reconstruction integrity
        if path_nodes[0] != start: return None

        return Path(path_nodes, float(dist[end_idx]))

    def connected_components(self) -> Generator[List[str], None, None]:
        """
        Yields lists of node IDs for each connected component.

        Yields:
            Lists of node IDs.
        """
        # Use a dummy policy just to get connectivity
        policy = WeightingPolicy(attr='_connectivity', default=1.0)
        n, labels = sp.csgraph.connected_components(self.get_matrix(policy), connection='weak', directed=self.directed)
        components = defaultdict(list)
        for idx, label in enumerate(labels):
            components[label].append(self._nodes[idx])

        yield from components.values()

    def greedy_set_cover(self, policy: WeightingPolicy) -> List[int]:
        """
        Solves the Set Cover problem using a greedy approach on the graph matrix defined by the policy.
        Rows are treated as Sets, Columns as Elements.

        Args:
            policy: The WeightingPolicy to define the matrix.

        Returns:
            A list of indices representing the selected sets.
        """
        matrix = self.get_matrix(policy)
        # Ensure CSC for column access (Element -> Sets)
        # This is O(N) memory but critical for speed
        csc = matrix.tocsc()
        # Max possible score is the max number of elements in a single set
        # (We can optimize this, but getting max row len is cheap)
        if matrix.shape[0] == 0: return []
        max_score = matrix.getnnz(axis=1).max()
        # Run Numba optimized core
        return _greedy_set_cover_kernel(matrix.shape[0], int(max_score), matrix.indptr, matrix.indices, csc.indptr,
                                        csc.indices)


# Functions ------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def _greedy_set_cover_kernel(n_sets,max_score, csr_indptr, csr_indices, csc_indptr, csc_indices):
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
    is_covered = np.zeros(len(csc_indptr) - 1, dtype=np.bool)  # This used to be numba.boolean, but numba is optional

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
                if p != -1: next_set[p] = n
                else: buckets[old_score] = n  # It was the head of the list
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


def _coerce_node(obj: Any) -> str:
    """
    Coerces an object into a node ID string.

    Args:
        obj: The object to coerce.

    Returns:
        The string ID of the object.
    """
    # Check for 'id' first, then fall back to string representation
    return str(obj.id) if hasattr(obj, 'id') else str(obj)
