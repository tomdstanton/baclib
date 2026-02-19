"""Graph data structures for representing directed and undirected networks of genomic segments."""
from typing import Any, Union, List, Dict, Set, Iterable

import numpy as np

from baclib.containers import Batch, RaggedBatch
from baclib.core.interval import Strand


# Classes --------------------------------------------------------------------------------------------------------------
class Edge:
    """
    A directed edge between two nodes with strand orientation and attributes.

    Nodes can be specified as ``bytes`` IDs, strings, or objects with an
    ``.id`` attribute — they are coerced to ``bytes`` internally.

    Args:
        u: Source node ID (or object with ``.id``).
        v: Target node ID (or object with ``.id``).
        u_strand: Orientation of the source node (default ``FORWARD``).
        v_strand: Orientation of the target node (default ``FORWARD``).
        attributes: Optional dictionary of edge attributes.

    Examples:
        >>> e = Edge(b'contig_1', b'contig_2')
        >>> e.reverse()
        Edge(contig_2(+)->contig_1(+))
    """
    __slots__ = ('_u', '_v', '_u_strand', '_v_strand', 'attributes')

    def __init__(self, u: Any, v: Any, u_strand: Union[Strand, int] = Strand.FORWARD, 
                 v_strand: Union[Strand, int] = Strand.FORWARD, attributes: dict[bytes, Any] = None):
        self._u: bytes = self._coerce_node(u)  # Extract the pointer
        self._v: bytes = self._coerce_node(v)  # Extract the pointer
        self._u_strand = Strand(u_strand)
        self._v_strand = Strand(v_strand)
        self.attributes = attributes or {}

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``EdgeBatch`` class.
        """
        return EdgeBatch

    @property
    def u(self) -> bytes:
        """Returns the source node ID.

        Returns:
            Source node as ``bytes``.
        """
        return self._u

    @property
    def v(self) -> bytes:
        """Returns the target node ID.

        Returns:
            Target node as ``bytes``.
        """
        return self._v

    @property
    def u_strand(self) -> Strand:
        """Returns the source node strand orientation.

        Returns:
            A ``Strand`` value.
        """
        return self._u_strand

    @property
    def v_strand(self) -> Strand:
        """Returns the target node strand orientation.

        Returns:
            A ``Strand`` value.
        """
        return self._v_strand

    @staticmethod
    def _coerce_node(obj: Any) -> bytes:
        if isinstance(obj, bytes): return obj
        if hasattr(obj, 'id'): return obj.id
        if isinstance(obj, str): return obj.encode('ascii')
        return str(obj).encode('ascii')

    def reverse(self) -> 'Edge':
        """Returns a new edge with swapped source and target.

        Returns:
            A reversed ``Edge`` with copied attributes.

        Examples:
            >>> Edge(b'A', b'B').reverse()
            Edge(B(+)->A(+))
        """
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
    Columnar container for edges, optimized for batch I/O and graph construction.

    Stores source/target node IDs, strand orientations, and attributes as
    parallel numpy arrays.

    Args:
        u: Object array of source node IDs.
        v: Object array of target node IDs.
        u_strands: ``int8`` array of source strand orientations.
        v_strands: ``int8`` array of target strand orientations.
        attributes: Optional dict mapping attribute names to per-edge arrays.

    Examples:
        >>> batch = EdgeBatch.build([edge1, edge2])
        >>> len(batch)
        2
    """
    __slots__ = ('_u', '_v', '_u_strands', '_v_strands', '_attributes')

    def __init__(self, u: np.ndarray, v: np.ndarray, 
                 u_strands: np.ndarray = None, v_strands: np.ndarray = None,
                 attributes: Dict[bytes, np.ndarray] = None):
        self._u = np.array(u, copy=False)
        self._v = np.array(v, copy=False)
        n = len(self._u)
        self._u_strands = u_strands.astype(np.int8, copy=False) if u_strands is not None else np.full(n, 1, dtype=np.int8)
        self._v_strands = v_strands.astype(np.int8, copy=False) if v_strands is not None else np.full(n, 1, dtype=np.int8)
        self._attributes = attributes or {}
        
        if len(self._u) != len(self._v):
            raise ValueError("u and v arrays must have the same length")

    @classmethod
    def build(cls, components: Iterable[Edge]) -> 'EdgeBatch':
        """Constructs an EdgeBatch from an iterable of Edge objects.

        Args:
            components: An iterable of ``Edge`` objects.

        Returns:
            A new ``EdgeBatch``.

        Examples:
            >>> batch = EdgeBatch.build([edge1, edge2])
        """
        edges = list(components)
        if not edges: return cls.empty()
        u = np.array([e.u for e in edges])
        v = np.array([e.v for e in edges])
        us = np.array([e.u_strand for e in edges], dtype=np.int8)
        vs = np.array([e.v_strand for e in edges], dtype=np.int8)
        return cls(u, v, us, vs)

    @classmethod
    def concat(cls, batches: Iterable['EdgeBatch']) -> 'EdgeBatch':
        """Concatenates multiple EdgeBatch objects into one.

        Args:
            batches: An iterable of ``EdgeBatch`` objects.

        Returns:
            A single concatenated ``EdgeBatch``.
        """
        batches = list(batches)
        if not batches: return cls.empty()
        u = np.concatenate([b.u for b in batches])
        v = np.concatenate([b.v for b in batches])
        us = np.concatenate([b.u_strands for b in batches])
        vs = np.concatenate([b.v_strands for b in batches])
        return cls(u, v, us, vs)

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by all internal arrays.
        """
        return self._u.nbytes + self._v.nbytes + self._u_strands.nbytes + self._v_strands.nbytes

    @classmethod
    def empty(cls) -> 'EdgeBatch':
        """Creates an empty EdgeBatch with zero edges.

        Returns:
            An empty ``EdgeBatch``.
        """
        return cls.zeros(0)

    @classmethod
    def zeros(cls, n: int) -> 'EdgeBatch':
        """Creates an EdgeBatch with *n* placeholder edges.

        Args:
            n: Number of placeholder edge slots.

        Returns:
            An ``EdgeBatch`` with empty IDs and forward strands.
        """
        return cls(
            np.full(n, b'', dtype='S1'),
            np.full(n, b'', dtype='S1'),
            np.ones(n, dtype=np.int8),
            np.ones(n, dtype=np.int8),
            {}
        )

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``Edge`` class.
        """
        return Edge

    def __repr__(self): return f"<EdgeBatch: {len(self)} edges>"

    @property
    def u(self) -> np.ndarray:
        """Returns the source node IDs.

        Returns:
            An object array of ``bytes``.
        """
        return self._u

    @property
    def v(self) -> np.ndarray:
        """Returns the target node IDs.

        Returns:
            An object array of ``bytes``.
        """
        return self._v

    @property
    def u_strands(self) -> np.ndarray:
        """Returns the source strand orientations.

        Returns:
            An ``int8`` numpy array.
        """
        return self._u_strands

    @property
    def v_strands(self) -> np.ndarray:
        """Returns the target strand orientations.

        Returns:
            An ``int8`` numpy array.
        """
        return self._v_strands

    @property
    def attributes(self) -> Dict[bytes, np.ndarray]:
        """Returns the per-edge attribute arrays.

        Returns:
            A dict mapping attribute names to numpy arrays.
        """
        return self._attributes

    def copy(self) -> 'EdgeBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``EdgeBatch`` with copied arrays.
        """
        # Note: attributes dict copy is shallow for arrays inside
        return self.__class__(self._u.copy(), self._v.copy(), self._u_strands.copy(), self._v_strands.copy(), self._attributes.copy())

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


class Graph:
    """A general-purpose graph supporting directed/undirected and simple/multi-edge topologies.

    Nodes are identified by ``bytes`` IDs and may carry arbitrary attribute dicts.
    Edges are stored as a set of ``Edge`` objects. An internal topology cache
    (COO arrays) is maintained lazily for efficient matrix construction.

    Args:
        edges: Optional initial edges to add (``Edge`` objects or tuples).
        directed: Whether the graph is directed (default ``True``).
        multi: Whether to allow parallel edges (default ``True``).

    Examples:
        >>> g = Graph(directed=True)
        >>> g.add_node(b'A')
        >>> g.add_edges([Edge(b'A', b'B')])
        >>> len(g)
        1
    """
    __slots__ = ('_directed', '_multi', '_edges', '_nodes', '_node_to_idx', '_node_attributes',
                 '_topology_cache', '_topo_lists', '_matrix_cache')

    def __init__(self, edges: Iterable[Union[Edge, tuple]] = None, directed: bool = True, multi: bool = True):
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
    def directed(self) -> bool:
        """Returns ``True`` if the graph is directed.

        Returns:
            Direction flag.
        """
        return self._directed

    @property
    def multi(self) -> bool:
        """Returns ``True`` if the graph allows parallel edges.

        Returns:
            Multi-edge flag.
        """
        return self._multi

    @property
    def nodes(self) -> List[bytes]:
        """Returns the ordered list of node IDs.

        Returns:
            A list of ``bytes`` node IDs.
        """
        return self._nodes

    @property
    def edges(self) -> Set[Edge]:
        """Returns the set of all edges in the graph.

        Returns:
            A set of ``Edge`` objects.
        """
        return self._edges

    @property
    def node_to_idx(self) -> Dict[bytes, int]:
        """Returns the node-to-index mapping.

        Returns:
            A dict mapping ``bytes`` node IDs to integer indices.
        """
        return self._node_to_idx

    @property
    def node_attributes(self) -> Dict[bytes, dict]:
        """Returns the per-node attribute dictionaries.

        Returns:
            A dict mapping node IDs to attribute dicts.
        """
        return self._node_attributes

    @property
    def topology_cache(self):
        """Returns the COO topology cache, or ``None`` if not yet built.

        Returns:
            A tuple of ``(u_indices, v_indices, edge_list)`` or ``None``.
        """
        return self._topology_cache

    @property
    def matrix_cache(self) -> Dict:
        """Returns the sparse matrix cache.

        Returns:
            A dict mapping ``MatrixPolicy`` keys to CSR matrices.
        """
        return self._matrix_cache

    def __repr__(self):
        type_str = "Directed" if self._directed else "Undirected"
        kind_str = "Multigraph" if self._multi else "Graph"
        return (f"{type_str} {kind_str} with "
                f"{len(self._nodes)} nodes and {len(self._edges)} defined edges")

    def __iter__(self): return iter(self._edges)
    def __len__(self): return len(self._edges)
    def __getitem__(self, item): return self._nodes[item]

    def add_node(self, node: Any, attributes: dict = None):
        """Adds a node to the graph, optionally with attributes.

        If the node already exists, its attributes are merged (updated).

        Args:
            node: The node ID (coerced to ``bytes``).
            attributes: Optional dictionary of node attributes.

        Examples:
            >>> g.add_node(b'contig_1', attributes={'length': 5000})
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
        """Adds multiple edges to the graph, auto-creating nodes as needed.

        Accepts ``Edge`` objects, ``EdgeBatch`` instances, or tuples that
        can be unpacked into an ``Edge`` constructor. Invalidates the
        topology and matrix caches.

        Args:
            edges: An iterable of ``Edge`` objects, tuples, or an ``EdgeBatch``.

        Examples:
            >>> g.add_edges([Edge(b'A', b'B'), Edge(b'B', b'C')])
        """
        # Optimization: Collect unique nodes first to reduce dict lookups from O(E) to O(V)
        if isinstance(edges, EdgeBatch):
            # Fast path for batch: access arrays directly
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
        """Builds or refreshes the COO topology cache if stale.

        After calling this, ``self.topology_cache`` is guaranteed to be
        a valid ``(u_indices, v_indices, edge_list)`` tuple.
        """
        if self._topology_cache is not None: return

        u_list, v_list, e_list = self._topo_lists
        self._topology_cache = (
            np.array(u_list, dtype=np.int32),
            np.array(v_list, dtype=np.int32),
            e_list
        )

    def subgraph(self, nodes: Iterable[Any]) -> 'Graph':
        """Returns an induced subgraph containing only the specified nodes.

        Edges are kept only if both endpoints are in the node set.

        Args:
            nodes: An iterable of node IDs to keep.

        Returns:
            A new ``Graph`` with the selected nodes and their inter-edges.

        Examples:
            >>> sub = g.subgraph([b'A', b'B'])
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
    """An ordered sequence of node IDs representing a walk through a graph.

    Args:
        nodes: List of ``bytes`` node IDs in traversal order.
        cost: Cumulative edge weight along the path (default ``0.0``).

    Examples:
        >>> p = Path([b'A', b'B', b'C'], cost=2.5)
        >>> len(p)
        3
    """
    __slots__ = ('nodes', 'total_cost')

    def __init__(self, nodes: list[bytes], cost: float = 0.0):
        self.nodes = nodes
        self.total_cost = cost
    
    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``PathBatch`` class.
        """
        return PathBatch

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
    Columnar batch of paths using ragged array storage.

    Stores all node IDs in a flat array with per-path offsets and costs.

    Args:
        flat_nodes: Flat object array of all node IDs concatenated.
        offsets: ``int32`` offset array (length = ``n_paths + 1``).
        costs: Optional ``float32`` array of per-path costs.

    Examples:
        >>> batch = PathBatch.build([path1, path2])
        >>> len(batch)
        2
    """
    __slots__ = ('_flat_nodes', '_costs')

    def __init__(self, flat_nodes: np.ndarray, offsets: np.ndarray, costs: np.ndarray = None):
        super().__init__(offsets)
        self._flat_nodes = flat_nodes
        n = len(offsets) - 1
        self._costs = costs if costs is not None else np.zeros(n, dtype=np.float32)

    @classmethod
    def empty(cls) -> 'PathBatch':
        """Creates an empty PathBatch with zero paths.

        Returns:
            An empty ``PathBatch``.
        """
        return cls.zeros(0)

    @classmethod
    def zeros(cls, n: int) -> 'PathBatch':
        """Creates a PathBatch with *n* empty placeholder paths.

        Args:
            n: Number of placeholder path slots.

        Returns:
            A ``PathBatch`` with zero-cost, zero-length paths.
        """
        return cls(
            np.empty(0, dtype='S1'),
            np.zeros(n + 1, dtype=np.int32),
            np.zeros(n, dtype=np.float32)
        )

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``Path`` class.
        """
        return Path

    @classmethod
    def concat(cls, batches: Iterable['PathBatch']) -> 'PathBatch':
        """Concatenates multiple PathBatch objects into one.

        Args:
            batches: An iterable of ``PathBatch`` objects.

        Returns:
            A single concatenated ``PathBatch``.
        """
        batches = list(batches)
        if not batches: return cls.empty()
        flat_nodes = np.concatenate([b._flat_nodes for b in batches])
        costs = np.concatenate([b._costs for b in batches])
        offsets = cls._stack_offsets(batches)
        return cls(flat_nodes, offsets, costs)

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by all internal arrays.
        """
        return super().nbytes + self._flat_nodes.nbytes + self._costs.nbytes

    def copy(self) -> 'PathBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``PathBatch`` with copied arrays.
        """
        return self.__class__(self._flat_nodes.copy(), self._offsets.copy(), self._costs.copy())

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
    def build(cls, components: Iterable[Path]) -> 'PathBatch':
        """Constructs a PathBatch from an iterable of Path objects.

        Args:
            components: An iterable of ``Path`` objects.

        Returns:
            A new ``PathBatch``.

        Examples:
            >>> batch = PathBatch.build([path1, path2])
        """
        paths = list(components)
        if not paths:
            return cls(np.empty(0, dtype='S1'), np.array([0], dtype=np.int32))
        
        flat_nodes = []
        offsets = [0]
        curr = 0
        for p in paths:
            flat_nodes.extend(p.nodes)
            curr += len(p.nodes)
            offsets.append(curr)
            
        return cls(
            np.array(flat_nodes),
            np.array(offsets, dtype=np.int32),
            np.array([p.total_cost for p in paths], dtype=np.float32)
        )
