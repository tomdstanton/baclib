from typing import Any, Union, List, Dict, Set, Iterable

import numpy as np

from baclib.core.interval import Strand
from baclib.utils import Batch, RaggedBatch


# Classes --------------------------------------------------------------------------------------------------------------
class Edge:
    """
    Represents a directed edge between two nodes with optional attributes.

    Attributes:
        u (bytes): Source node ID.
        v (bytes): Target node ID.
        u_strand (Strand): Orientation of the source node.
        v_strand (Strand): Orientation of the target node.
        attributes (dict): Edge attributes.
    """
    __slots__ = ('_u', '_v', '_u_strand', '_v_strand', 'attributes')

    def __init__(self, u: Any, v: Any, u_strand: Union[Strand, int] = Strand.FORWARD, 
                 v_strand: Union[Strand, int] = Strand.FORWARD, attributes: dict[bytes, Any] = None):
        """
        Initializes an Edge.

        Args:
            u: Source node (or object with .id).
            v: Target node (or object with .id).
            attributes: Dictionary of edge attributes.
        """
        self._u: bytes = self._coerce_node(u)  # Extract the pointer
        self._v: bytes = self._coerce_node(v)  # Extract the pointer
        self._u_strand = Strand(u_strand)
        self._v_strand = Strand(v_strand)
        self.attributes = attributes or {}

    @property
    def u(self) -> bytes: return self._u
    @property
    def v(self) -> bytes: return self._v
    @property
    def u_strand(self) -> Strand: return self._u_strand
    @property
    def v_strand(self) -> Strand: return self._v_strand

    @staticmethod
    def _coerce_node(obj: Any) -> bytes:
        if isinstance(obj, bytes): return obj
        if hasattr(obj, 'id'): return obj.id
        if isinstance(obj, str): return obj.encode('ascii')
        return str(obj).encode('ascii')

    def reverse(self):
        """Returns a reversed copy of the edge."""
        return Edge(self.v, self.u, self.v_strand, self.u_strand, self.attributes.copy())

    def __eq__(self, other):
        if not isinstance(other, Edge): return NotImplemented
        return (self._u == other._u and self._v == other._v and 
                self._u_strand == other._u_strand and self._v_strand == other._v_strand and
                self.attributes == other.attributes)

    def __hash__(self):
        return hash((self._u, self._v, self._u_strand, self._v_strand, tuple(sorted(self.attributes.items()))))

    def __repr__(self): 
        u_s = b'+' if self._u_strand == 1 else b'-'
        v_s = b'+' if self._v_strand == 1 else b'-'
        return f"Edge({self.u.decode()}({u_s.decode()})->{self.v.decode()}({v_s.decode()}))"
        
    def __getitem__(self, item): return self.attributes.get(item, None)
    def __iter__(self): return iter((self.u, self.v, self.u_strand, self.v_strand, self.attributes))


class EdgeBatch(Batch):
    """
    A columnar container for edges, optimized for batch processing and IO.
    Stores source/target nodes and attributes as numpy arrays.
    """
    __slots__ = ('_u', '_v', '_u_strands', '_v_strands', '_attributes')

    def __init__(self, u: np.ndarray, v: np.ndarray, 
                 u_strands: np.ndarray = None, v_strands: np.ndarray = None,
                 attributes: Dict[bytes, np.ndarray] = None):
        self._u = np.array(u, dtype=object, copy=False)
        self._v = np.array(v, dtype=object, copy=False)
        n = len(self._u)
        self._u_strands = u_strands.astype(np.int8, copy=False) if u_strands is not None else np.full(n, 1, dtype=np.int8)
        self._v_strands = v_strands.astype(np.int8, copy=False) if v_strands is not None else np.full(n, 1, dtype=np.int8)
        self._attributes = attributes or {}
        
        if len(self._u) != len(self._v):
            raise ValueError("u and v arrays must have the same length")

    def __len__(self): return len(self._u)

    def __iter__(self):
        for i in range(len(self)):
            attrs = {k: v[i] for k, v in self._attributes.items()}
            yield Edge(self._u[i], self._v[i], self._u_strands[i], self._v_strands[i], attrs)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            attrs = {k: v[item] for k, v in self._attributes.items()}
            return Edge(self._u[item], self._v[item], self._u_strands[item], self._v_strands[item], attrs)
        elif isinstance(item, (slice, np.ndarray, list)):
            new_attrs = {k: v[item] for k, v in self._attributes.items()}
            return EdgeBatch(self._u[item], self._v[item], self._u_strands[item], self._v_strands[item], new_attrs)
        raise TypeError(f"Invalid index type: {type(item)}")

    def empty(self) -> 'EdgeBatch':
        return EdgeBatch(np.empty(0, dtype=object), np.empty(0, dtype=object), 
                         np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int8), {})

    def __repr__(self):
        return f"<EdgeBatch: {len(self)} edges>"

    @property
    def u(self) -> np.ndarray: return self._u
    @property
    def v(self) -> np.ndarray: return self._v
    @property
    def u_strands(self) -> np.ndarray: return self._u_strands
    @property
    def v_strands(self) -> np.ndarray: return self._v_strands
    @property
    def attributes(self) -> Dict[bytes, np.ndarray]: return self._attributes


class Graph:
    __slots__ = ('_directed', '_multi', '_edges', '_nodes', '_node_to_idx', '_node_attributes',
                 '_topology_cache', '_topo_lists', '_matrix_cache')

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
        self._edges: Set[Edge] = set()
        self._nodes: List[bytes] = []
        self._node_to_idx: Dict[bytes, int] = {}
        self._node_attributes: Dict[bytes, dict] = {}
        self._topology_cache = None
        self._topo_lists = ([], [], [])
        # Cache: (attr, source, aggregator, invert, default) -> CSR Matrix
        self._matrix_cache: Dict['MatrixPolicy', 'csr_matrix'] = {}
        if edges: self.add_edges(edges)

    @property
    def directed(self): return self._directed
    @property
    def multi(self): return self._multi
    @property
    def nodes(self): return self._nodes
    @property
    def edges(self): return self._edges
    @property
    def node_to_idx(self): return self._node_to_idx
    @property
    def node_attributes(self): return self._node_attributes
    @property
    def topology_cache(self): return self._topology_cache
    @property
    def matrix_cache(self): return self._matrix_cache

    def __repr__(self):
        type_str = "Directed" if self._directed else "Undirected"
        kind_str = "Multigraph" if self._multi else "Graph"
        return (f"{type_str} {kind_str} with "
                f"{len(self._nodes)} nodes and {len(self._edges)} defined edges")

    def __iter__(self): return iter(self._edges)
    def __len__(self): return len(self._edges)
    def __getitem__(self, item): return self._nodes[item]

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
            self._matrix_cache.clear()
        elif attributes:
            if node not in self._node_attributes: self._node_attributes[node] = {}
            self._node_attributes[node].update(attributes)
            self._matrix_cache.clear()

    def add_edges(self, edges: Iterable[Union[Edge, tuple]]):
        """
        Batch optimization for adding multiple edges.

        Args:
            edges: An iterable of Edge objects or tuples.
        """
        # Optimization: Collect unique nodes first to reduce dict lookups from O(E) to O(V)
        if isinstance(edges, EdgeBatch):
            # Fast path for batch: access arrays directly
            # np.unique is faster than set() for large numpy arrays of objects/bytes
            potential_nodes = np.unique(np.concatenate((edges.u, edges.v)))
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
            if e not in self._edges:
                self._edges.add(e)
                u_list.append(node_map[e.u])
                v_list.append(node_map[e.v])
                e_list.append(e)
                added = True

        if added: self._topology_cache = None
        if added: self._matrix_cache.clear()

    def ensure_topology(self):
        if self._topology_cache is not None: return

        u_list, v_list, e_list = self._topo_lists
        self._topology_cache = (
            np.array(u_list, dtype=np.int32),
            np.array(v_list, dtype=np.int32),
            e_list
        )

    def subgraph(self, nodes: Iterable[Any]) -> 'Graph':
        """
        Returns a new Graph containing only the specified nodes and the edges between them (induced subgraph).

        Args:
            nodes: Iterable of node IDs to keep.

        Returns:
            A new Graph instance.
        """
        keep_set = {Edge._coerce_node(n) for n in nodes}
        # Filter to only existing nodes
        keep_set.intersection_update(self._node_to_idx)
        
        if not keep_set:
            return Graph(directed=self._directed, multi=self._multi)

        valid_edges = []
        
        # Use topology cache for vectorized filtering if available
        if self._topology_cache is not None:
            u_idx, v_idx, e_list = self._topology_cache
            
            # Build boolean mask of nodes to keep
            n_nodes = len(self._nodes)
            mask = np.zeros(n_nodes, dtype=bool)
            
            indices = [self._node_to_idx[n] for n in keep_set]
            mask[indices] = True
            
            # Find edges where both u and v are in mask
            edge_mask = mask[u_idx] & mask[v_idx]
            
            valid_indices = np.flatnonzero(edge_mask)
            for i in valid_indices:
                valid_edges.append(e_list[i])
        else:
            for e in self._edges:
                if e.u in keep_set and e.v in keep_set:
                    valid_edges.append(e)

        sub = Graph(valid_edges, directed=self._directed, multi=self._multi)
        for n in keep_set:
            attrs = self._node_attributes.get(n)
            sub.add_node(n, attributes=attrs.copy() if attrs else None)
            
        return sub


class Path:
    __slots__ = ('nodes', 'total_cost')

    def __init__(self, nodes: list[bytes], cost: float = 0.0):
        """
        Initializes a Path.

        Args:
            nodes: List of node IDs in the path.
            cost: Total cost of the path.
        """
        self.nodes = nodes
        self.total_cost = cost

    def __len__(self): return len(self.nodes)
    def __iter__(self): return iter(self.nodes)
    def __getitem__(self, item): return self.nodes[item]
    def __add__(self, other: 'Path') -> 'Path':
        if not isinstance(other, Path): return NotImplemented
        if not self.nodes or not other.nodes: return self if self.nodes else other
        if self.nodes[-1] != other.nodes[0]:
            raise ValueError(f"Paths are not continuous: {self.nodes[-1]} != {other.nodes[0]}")
        return Path(self.nodes + other.nodes[1:], self.total_cost + other.total_cost)

    def __repr__(self): return f"Path(steps={len(self.nodes)}, cost={self.total_cost:.4f})"


class PathBatch(RaggedBatch):
    """
    Efficient storage for many paths (Structure of Arrays).
    """
    __slots__ = ('_flat_nodes', '_costs')

    def __init__(self, flat_nodes: np.ndarray, offsets: np.ndarray, costs: np.ndarray = None):
        super().__init__(offsets)
        self._flat_nodes = flat_nodes
        n = len(offsets) - 1
        self._costs = costs if costs is not None else np.zeros(n, dtype=np.float32)

    def empty(self) -> 'PathBatch':
        return PathBatch(
            np.empty(0, dtype=object),
            np.array([0], dtype=np.int32),
            np.empty(0, dtype=np.float32)
        )

    def __repr__(self): return f"<PathBatch: {len(self)} paths>"

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            start = self._offsets[item]
            end = self._offsets[item+1]
            nodes = self._flat_nodes[start:end]
            # Convert numpy array of bytes to list of bytes for Path compatibility
            return Path(list(nodes), cost=self._costs[item])

        if isinstance(item, slice):
            new_offsets, val_start, val_end = self._get_slice_info(item)
            return PathBatch(
                self._flat_nodes[val_start:val_end],
                new_offsets,
                self._costs[item]
            )
        raise TypeError(f"Invalid index type: {type(item)}")

    @classmethod
    def from_paths(cls, paths: Iterable[Path]):
        paths = list(paths)
        if not paths:
            return cls(np.empty(0, dtype=object), np.array([0], dtype=np.int32))
        
        flat_nodes = []
        offsets = [0]
        curr = 0
        for p in paths:
            flat_nodes.extend(p.nodes)
            curr += len(p.nodes)
            offsets.append(curr)
            
        return cls(
            np.array(flat_nodes, dtype=object),
            np.array(offsets, dtype=np.int32),
            np.array([p.total_cost for p in paths], dtype=np.float32)
        )
