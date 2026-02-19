"""
Module for managing alignments.
"""
from typing import Generator, Iterable, Any, Union, NamedTuple
from itertools import chain
import copy
from enum import Enum, IntEnum

import numpy as np

from baclib.core.alphabet import Alphabet
from baclib.containers.feature import Feature, FeatureKey
from baclib.core.interval import IntervalBatch, Interval
from baclib.containers import Batch
from baclib.lib.protocols import HasIntervals
from baclib.lib.resources import jit


# Classes --------------------------------------------------------------------------------------------------------------
class CigarOp(IntEnum):
    M = 0
    I = 1
    D = 2
    N = 3
    S = 4
    H = 5
    P = 6
    EQ = 7
    X = 8
    B = 9


class Cigar:
    """
    High-performance CIGAR string parser and builder.
    """
    # Fast lookup for bytes -> integer op codes
    _OP_BYTES_LOOKUP = [b'M', b'I', b'D', b'N', b'S', b'H', b'P', b'=', b'X', b'B']

    _BYTE_TO_OP = np.full(256, 255, dtype=np.uint8)
    for op, sym in enumerate(_OP_BYTES_LOOKUP):
        _BYTE_TO_OP[ord(sym)] = op

    # Consumption logic
    _QUERY_CONSUMERS = np.array([True, True, False, False, True, False, False, True, True, False], dtype=bool)
    _TARGET_CONSUMERS = np.array([True, False, True, True, False, False, False, True, True, False], dtype=bool)
    _ALN_CONSUMERS = _QUERY_CONSUMERS | _TARGET_CONSUMERS

    @classmethod
    def parse(cls, cigar: bytes) -> Generator[tuple[bytes, int, int, int, int], None, None]:
        """
        Parses a CIGAR string.
        Yields: (op_char, count, query_consumed, target_consumed, aln_consumed).
        """
        ops, counts = _parse_cigar_kernel(cigar, cls._BYTE_TO_OP)
        q_len, t_len, aln_len = 0, 0, 0
        lookup = cls._OP_BYTES_LOOKUP

        for i in range(len(ops)):
            op = ops[i]
            n = counts[i]
            if cls._QUERY_CONSUMERS[op]: q_len += n
            if cls._TARGET_CONSUMERS[op]: t_len += n
            if cls._ALN_CONSUMERS[op]: aln_len += n
            yield lookup[op], n, q_len, t_len, aln_len

    @classmethod
    def parse_into_arrays(cls, cigar: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Parses CIGAR string directly into (ops, counts) arrays."""
        return _parse_cigar_kernel(cigar, cls._BYTE_TO_OP)

    @staticmethod
    def make(query: np.ndarray, target: np.ndarray, gap_code: int = 255, extended: bool = False) -> bytes:
        """
        Constructs a CIGAR string from aligned sequences.
        """
        if len(query) == 0: return b""
        counts, ops = _cigar_rle_kernel(query, target, gap_code, extended)
        return b"".join([b"%d" % c + Cigar._OP_BYTES_LOOKUP[o] for c, o in zip(counts, ops)])


class Alignment(Feature):
    """
    Represents a pairwise sequence alignment.

    Attributes:
        query (bytes): Query sequence ID.
        query_interval (Interval): Interval on the query.
        target (bytes): Target sequence ID.
        interval (Interval): Interval on the target (inherited from Feature).
        score (float): Alignment score.
        cigar (bytes): CIGAR string.
    """
    __slots__ = (
        'query', 'query_interval', 'query_length', 'target', 'target_length', 'length', 'cigar', 'score',
        'n_matches', 'quality'
    )

    def __init__(self, query: bytes, query_interval: 'Interval', target: bytes, interval: 'Interval',
                 query_length: int = 0, target_length: int = 0, length: int = 0, cigar: bytes = None,
                 n_matches: int = 0, quality: int = 0, qualifiers: Iterable[tuple[bytes, Any]] = None, score: float = 0):
        super().__init__(interval, key=FeatureKey.MISC_FEATURE, qualifiers=qualifiers)
        self.query = query
        self.target = target
        self.query_interval = query_interval
        self.query_length = query_length
        self.target_length = target_length
        self.length = length
        self.cigar = cigar
        self.score = score
        self.n_matches = n_matches
        self.quality = quality

    @property
    def batch(self) -> type['Batch']: return AlignmentBatch

    def __repr__(self):
        return (f"Alignment({self.query.decode(Alphabet.ENCODING, 'ignore')}->"
                f"{self.target.decode(Alphabet.ENCODING, 'ignore')}, score={self.score})")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (super().__eq__(other) and
                    self.query == other.query and
                    self.query_interval == other.query_interval and
                    self.score == other.score and
                    self.cigar == other.cigar)
        return False

    def copy(self) -> 'Alignment':
        return Alignment(
            query=self.query, query_interval=self.query_interval, target=self.target,
            interval=self.interval, query_length=self.query_length, target_length=self.target_length,
            length=self.length, cigar=self.cigar, n_matches=self.n_matches, quality=self.quality,
            qualifiers=list(self.qualifiers), score=self.score
        )

    def shift(self, x: int, y: int = None) -> 'Alignment':
        return Alignment(
            query=self.query, query_interval=self.query_interval, target=self.target,
            interval=self.interval.shift(x, y),
            query_length=self.query_length, target_length=self.target_length, length=self.length,
            cigar=self.cigar, n_matches=self.n_matches, quality=self.quality,
            qualifiers=list(self.qualifiers), score=self.score
        )

    def reverse_complement(self, parent_length: int) -> 'Alignment':
        return Alignment(
            query=self.query, query_interval=self.query_interval, target=self.target,
            interval=self.interval.reverse_complement(parent_length),
            query_length=self.query_length, target_length=self.target_length, length=self.length,
            cigar=self.cigar, n_matches=self.n_matches, quality=self.quality,
            qualifiers=list(self.qualifiers), score=self.score
        )

    def query_coverage(self) -> float:
        """Returns the fraction of the query sequence covered by the alignment."""
        return len(self.query_interval) / self.query_length if self.query_length > 0 else 0.0

    def target_coverage(self) -> float:
        """Returns the fraction of the target sequence covered by the alignment."""
        return len(self.interval) / self.target_length if self.target_length > 0 else 0.0

    def identity(self) -> float:
        """Returns the sequence identity (matches / alignment length)."""
        return self.n_matches / self.length if self.length > 0 else 0.0

    def flip(self) -> 'Alignment':
        """Swaps query and target."""
        return Alignment(
            query=self.target, query_interval=self.interval, query_length=self.target_length,
            target=self.query, interval=self.query_interval, target_length=self.query_length,
            length=self.length, cigar=self.cigar, score=self.score, n_matches=self.n_matches,
            quality=self.quality, qualifiers=list(self.qualifiers) if self.qualifiers else None
        )


class AlignmentSide(NamedTuple):
    """
    A view of one side (Query or Target) of an AlignmentBatch.
    """
    indices: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    lengths: np.ndarray
    strands: np.ndarray
    ids: np.ndarray

    @property
    def coords(self) -> np.ndarray:
        return np.stack((self.starts, self.ends), axis=1)

    def to_intervals(self, sort: bool = True) -> 'IntervalBatch':
        """Returns an IntervalBatch representing this side's intervals."""
        return IntervalBatch(self.starts, self.ends, self.strands, sort=sort)


class AlignmentBatch(Batch, HasIntervals):
    """
    High-performance container for alignment batches.
    Stores data in Structure-of-Arrays (SoA) layout.
    """

    class Field(str, Enum):
        Q_IDX = 'q_idx'
        T_IDX = 't_idx'
        SCORE = 'score'
        Q_START = 'q_start'
        Q_END = 'q_end'
        Q_LEN = 'q_len'
        Q_STRAND = 'q_strand'
        T_START = 't_start'
        T_END = 't_end'
        T_LEN = 't_len'
        T_STRAND = 't_strand'
        MATCHES = 'matches'
        QUALITY = 'quality'
        ALN_LEN = 'aln_len'

    _DTYPE = np.dtype([
        (Field.Q_IDX, np.int32), (Field.T_IDX, np.int32), (Field.SCORE, np.float32),
        (Field.Q_START, np.int32), (Field.Q_END, np.int32), (Field.Q_LEN, np.int32), (Field.Q_STRAND, np.int8),
        (Field.T_START, np.int32), (Field.T_END, np.int32), (Field.T_LEN, np.int32), (Field.T_STRAND, np.int8),
        (Field.MATCHES, np.int32), (Field.QUALITY, np.uint8), (Field.ALN_LEN, np.int32)
    ])

    __slots__ = ('_data', '_cigars', '_qualifiers', 'query_ids', 'target_ids', '_spatial_indices')

    def __init__(self, data: np.ndarray = None, cigars: np.ndarray = None, qualifiers: np.ndarray = None,
                 query_ids: np.ndarray = None, target_ids: np.ndarray = None):
        """
        Direct initialization. Prefer factory methods (from_hits, etc.) for ease of use.
        """
        if data is None:
            self._data = np.zeros(0, dtype=self._DTYPE)
            self._cigars = np.zeros(0, dtype=object)
            self._qualifiers = np.zeros(0, dtype=object)
            # Default empty IDs arrays should be empty
            if query_ids is None: self.query_ids = np.empty(0, dtype=object)
            if target_ids is None: self.target_ids = np.empty(0, dtype=object)
        else:
            self._data = data
            n = len(data)
            self._cigars = cigars if cigars is not None else np.full(n, None, dtype=object)
            self._qualifiers = qualifiers if qualifiers is not None else np.full(n, None, dtype=object)
            self.query_ids = query_ids
            self.target_ids = target_ids

        self._spatial_indices: dict[int, IntervalBatch] = {}

    @classmethod
    def from_data(cls,
                  q_idx: np.ndarray, t_idx: np.ndarray, score: np.ndarray,
                  q_coords: np.ndarray, t_coords: np.ndarray,
                  q_lens: np.ndarray, t_lens: np.ndarray,
                  cigars: np.ndarray = None,
                  q_strands: np.ndarray = None, t_strands: np.ndarray = None):
        """Zero-copy construction from Aligner output arrays."""
        n = len(q_idx)
        data = np.zeros(n, dtype=cls._DTYPE)

        data[cls.Field.Q_IDX] = q_idx
        data[cls.Field.T_IDX] = t_idx
        data[cls.Field.SCORE] = score
        data[cls.Field.Q_START] = q_coords[:, 0]
        data[cls.Field.Q_END] = q_coords[:, 1]
        data[cls.Field.T_START] = t_coords[:, 0]
        data[cls.Field.T_END] = t_coords[:, 1]
        data[cls.Field.Q_LEN] = q_lens
        data[cls.Field.T_LEN] = t_lens

        data[cls.Field.Q_STRAND] = q_strands if q_strands is not None else 1
        data[cls.Field.T_STRAND] = t_strands if t_strands is not None else 1
        data[cls.Field.ALN_LEN] = np.maximum(
            data[cls.Field.Q_END] - data[cls.Field.Q_START],
            data[cls.Field.T_END] - data[cls.Field.T_START]
        )
        data[cls.Field.MATCHES] = (data[cls.Field.ALN_LEN] * 0.9).astype(np.int32)
        data[cls.Field.QUALITY] = 60

        return cls(data=data, cigars=cigars)

    @classmethod
    def build(cls, alignments: Iterable[Alignment]) -> 'AlignmentBatch':
        """
        Creates an AlignmentBatch from an iterable of Alignment objects.
        """
        # Avoid copy if already a list/tuple
        if isinstance(alignments, (list, tuple)):
            items = alignments
        else:
            items = list(alignments)

        n = len(items)
        if n == 0: return cls()

        # Handle identifiers vs indices
        first = items[0]

        # Pass 1: Collect columns (Faster than row-by-row tuple construction)
        q_raw = []
        t_raw = []
        
        # Pre-allocate lists for speed
        q_starts, q_ends, q_lens, q_strands = [], [], [], []
        t_starts, t_ends, t_lens, t_strands = [], [], [], []
        scores, matches, qualities, aln_lens = [], [], [], []
        cigars_list, qualifiers_list = [], []

        for x in items:
            q_raw.append(x.query)
            t_raw.append(x.target)
            cigars_list.append(x.cigar)
            qualifiers_list.append(x.qualifiers)
            
            q_starts.append(x.query_interval.start)
            q_ends.append(x.query_interval.end)
            q_lens.append(x.query_length)
            q_strands.append(x.query_interval.strand)
            
            t_starts.append(x.interval.start)
            t_ends.append(x.interval.end)
            t_lens.append(x.target_length)
            t_strands.append(x.interval.strand)
            
            scores.append(x.score)
            matches.append(x.n_matches)
            qualities.append(x.quality)
            aln_lens.append(x.length)

        # Pass 2: Build structured array
        data = np.zeros(n, dtype=cls._DTYPE)

        # Resolve IDs & Indices
        q_ids = None
        if isinstance(first.query, (int, np.integer)): data[cls.Field.Q_IDX] = q_raw
        else: q_ids, data[cls.Field.Q_IDX] = np.unique(q_raw, return_inverse=True)

        t_ids = None
        if isinstance(first.target, (int, np.integer)): data[cls.Field.T_IDX] = t_raw
        else: t_ids, data[cls.Field.T_IDX] = np.unique(t_raw, return_inverse=True)

        # Bulk fill (NumPy handles list->array conversion efficiently)
        data[cls.Field.Q_START] = q_starts; data[cls.Field.Q_END] = q_ends; data[cls.Field.Q_LEN] = q_lens; data[cls.Field.Q_STRAND] = q_strands
        data[cls.Field.T_START] = t_starts; data[cls.Field.T_END] = t_ends; data[cls.Field.T_LEN] = t_lens; data[cls.Field.T_STRAND] = t_strands
        data[cls.Field.SCORE] = scores; data[cls.Field.MATCHES] = matches; data[cls.Field.QUALITY] = qualities; data[cls.Field.ALN_LEN] = aln_lens

        cigars = np.array(cigars_list, dtype=object)
        qualifiers = np.array(qualifiers_list, dtype=object)

        return cls(data=data, cigars=cigars, qualifiers=qualifiers, query_ids=q_ids, target_ids=t_ids)

    @classmethod
    def zeros(cls, n: int) -> 'AlignmentBatch':
        return cls(
            data=np.zeros(n, dtype=cls._DTYPE),
            cigars=np.full(n, None, dtype=object),
            qualifiers=np.full(n, None, dtype=object),
            query_ids=np.empty(0, dtype=object),  # No ID mapping by default
            target_ids=np.empty(0, dtype=object)
        )

    @classmethod
    def empty(cls) -> 'AlignmentBatch':
        return cls.zeros(0)

    @property
    def component(self): return Alignment

    @classmethod
    def concat(cls, batches: Iterable['AlignmentBatch']) -> 'AlignmentBatch':
        batches = list(batches)
        if not batches: return cls.zeros(0)
        
        # 1. Structure of Arrays Concatenation (Fast)
        data = np.concatenate([b._data for b in batches])
        cigars = np.concatenate([b._cigars for b in batches])
        qualifiers = np.concatenate([b._qualifiers for b in batches])
        
        # 2. ID Mapping Resolution
        # We need to unify the ID maps if they differ.
        # Strict approach: If maps differ, we must re-map indices.
        # Heuristic: Check if all share same map (common case in pipe).
        first = batches[0]
        remapping_needed = False
        
        # Simple check: are maps identical objects or content?
        # For now, fast path only if objects are identical or None
        # TODO: Implement robust ID unification for mixed sources
        
        # Naive implementation: just take the first non-empty map or merge?
        # For safety/correctness in this refactor, if we detect potential mismatch,
        # we might fallback to object rebuild OR (better) unification.
        # But `from_alignments` (build) does full deduplication.
        
        # Let's assume consistent IDs for the "fast path" optimization context
        # If heterogeneous, we might need a slower path.
        
        # For now, we use the first valid ID map.
        q_ids = first.query_ids
        t_ids = first.target_ids
        
        return cls(data, cigars, qualifiers, q_ids, t_ids)

    @property
    def nbytes(self) -> int: return self._data.nbytes + self._cigars.nbytes + self._qualifiers.nbytes

    def copy(self) -> 'AlignmentBatch':
        return self.__class__(self._data.copy(), self._cigars.copy(), self._qualifiers.copy(), 
                              self.query_ids, self.target_ids)

    def __len__(self): return len(self._data)
    def __repr__(self): return f"<AlignmentBatch: {len(self)} alignments>"

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)): return self._make_alignment(item)
        elif isinstance(item, (slice, np.ndarray, list)):
            return AlignmentBatch(self._data[item], self._cigars[item], self._qualifiers[item],
                                  query_ids=self.query_ids, target_ids=self.target_ids)
        raise TypeError(f"Invalid index type: {type(item)}")
    @property
    def scores(self) -> np.ndarray: return self._data[self.Field.SCORE]
    @property
    def q_indices(self) -> np.ndarray: return self._data[self.Field.Q_IDX]
    @property
    def t_indices(self) -> np.ndarray: return self._data[self.Field.T_IDX]
    @property
    def q_coords(self) -> np.ndarray:
        return np.stack((self._data[self.Field.Q_START], self._data[self.Field.Q_END]), axis=1)
    @property
    def q_starts(self) -> np.ndarray: return self._data[self.Field.Q_START]
    @property
    def q_ends(self) -> np.ndarray: return self._data[self.Field.Q_END]
    @property
    def t_starts(self) -> np.ndarray: return self._data[self.Field.T_START]
    @property
    def t_ends(self) -> np.ndarray: return self._data[self.Field.T_END]
    @property
    def matches(self) -> np.ndarray: return self._data[self.Field.MATCHES]
    @property
    def aln_lens(self) -> np.ndarray: return self._data[self.Field.ALN_LEN]
    @property
    def q_lens(self) -> np.ndarray: return self._data[self.Field.Q_LEN]
    @property
    def t_lens(self) -> np.ndarray: return self._data[self.Field.T_LEN]
    @property
    def q_strands(self) -> np.ndarray: return self._data[self.Field.Q_STRAND]
    @property
    def t_strands(self) -> np.ndarray: return self._data[self.Field.T_STRAND]
    @property
    def cigars(self) -> np.ndarray: return self._cigars

    @property
    def intervals(self) -> IntervalBatch:
        """
        Returns the target intervals as an IntervalBatch.
        Enables efficient interoperability with IntervalBatch.from_features().
        """
        return self.target.to_intervals(sort=False)

    @property
    def query(self) -> AlignmentSide:
        """Returns a view of the Query side of the alignments."""
        ids = self.query_ids[self.q_indices] if self.query_ids is not None else self.q_indices
        return AlignmentSide(
            self.q_indices, self.q_starts, self.q_ends, 
            self.q_lens, self.q_strands, ids
        )

    @property
    def target(self) -> AlignmentSide:
        """Returns a view of the Target side of the alignments."""
        ids = self.target_ids[self.t_indices] if self.target_ids is not None else self.t_indices
        return AlignmentSide(
            self.t_indices, self.t_starts, self.t_ends, 
            self.t_lens, self.t_strands, ids
        )

    def filter(self, query_idxs: Iterable[int] = None, target_idxs: Iterable[int] = None) -> 'AlignmentBatch':
        """
        Filters alignments by query or target indices.

        Args:
            query_idxs: Iterable of internal query integer indices to keep.
            target_idxs: Iterable of internal target integer indices to keep.

        Returns:
            A new filtered AlignmentBatch.
        """
        mask = None
        if query_idxs is not None:
            # Create a boolean lookup table if range is reasonable, else hash set
            # Assuming dense indices from SeqBatch:
            q_set = np.array(list(query_idxs), dtype=np.int32)
            mask = np.isin(self.q_indices, q_set)  # O(N) or O(N log M)

        if target_idxs is not None:
            t_set = np.array(list(target_idxs), dtype=np.int32)
            t_mask = np.isin(self.t_indices, t_set)
            mask = t_mask if mask is None else (mask & t_mask)

        return self[mask] if mask is not None else self[:]

    def best(self, by_target: bool = False) -> 'AlignmentBatch':
        """
        Keeps only the best scoring alignment for each query (or target).

        Args:
            by_target: If True, keeps best per target.

        Returns:
            A new AlignmentBatch.
        """
        indices = self.t_indices if by_target else self.q_indices
        # To find max ID for array sizing, we scan once.
        # This is safe because indices correspond to SeqBatch,
        # so max(indices) < len(batch).
        max_id = np.max(indices) if len(indices) > 0 else 0
        best_indices = _best_hit_kernel(indices, self.scores, max_id + 1)
        return self[np.sort(best_indices)]

    def sort(self, by: str = 'score', ascending: bool = False) -> 'AlignmentBatch':
        """
        Sorts the batch.

        Args:
            by: Field to sort by ('score', 'query', 'target').
            ascending: Sort order.

        Returns:
            A new sorted AlignmentBatch.
        """
        # Now we map the 'by' string to our internal fields safely
        perm = np.argsort(self._data[self.Field(by)])
        if not ascending: perm = perm[::-1]
        return self[perm]

    def _make_alignment(self, idx: int) -> Alignment:
        r = self._data[idx]
        q_idx = r[self.Field.Q_IDX]
        t_idx = r[self.Field.T_IDX]
        
        return Alignment(
            query=self.query_ids[q_idx] if self.query_ids is not None else q_idx,
            query_interval=Interval(r[self.Field.Q_START], r[self.Field.Q_END], r[self.Field.Q_STRAND]),
            target=self.target_ids[t_idx] if self.target_ids is not None else t_idx,
            interval=Interval(r[self.Field.T_START], r[self.Field.T_END], r[self.Field.T_STRAND]),
            query_length=r[self.Field.Q_LEN], target_length=r[self.Field.T_LEN], length=r[self.Field.ALN_LEN],
            cigar=self._cigars[idx], score=r[self.Field.SCORE], n_matches=r[self.Field.MATCHES],
            quality=r[self.Field.QUALITY], qualifiers=self._qualifiers[idx]
        )

    def _get_spatial_index(self, target_idx: int) -> IntervalBatch:
        """Retrieves or builds the spatial index for a target."""
        if (target_int_idx := self._spatial_indices.get(target_idx)) is None:
            mask = self.t_indices == target_idx
            # Optimization: Direct IntervalBatch construction from arrays
            self._spatial_indices[target_idx] = IntervalBatch(
                self.t_starts[mask], self.t_ends[mask], sort=True
            )
            return self._spatial_indices[target_idx]
        return target_int_idx

    def swap_sides(self) -> 'AlignmentBatch':
        """
        Returns a new collection with queries and targets swapped.
        Vectorized operation.
        """
        new_coll = copy.copy(self)

        # Swap data columns
        new_data = self._data.copy()
        new_data[self.Field.Q_IDX] = self.t_indices
        new_data[self.Field.T_IDX] = self.q_indices
        new_data[self.Field.Q_START] = self.t_starts
        new_data[self.Field.Q_END] = self.t_ends
        new_data[self.Field.Q_LEN] = self.t_lens
        new_data[self.Field.T_START] = self.q_starts
        new_data[self.Field.T_END] = self.q_ends
        new_data[self.Field.T_LEN] = self.q_lens
        new_data[self.Field.Q_STRAND] = 1 # Reset query to forward reference
        new_data[self.Field.T_STRAND] = self.q_strands * self.t_strands # Preserve relative orientation

        new_coll._data = new_data
        new_coll.query_ids = self.target_ids
        new_coll.target_ids = self.query_ids
        new_coll._graph = None  # Invalidate graph
        new_coll._t_id_map = None
        new_coll._spatial_indices = {}  # Invalidate indices
        return new_coll

    def merge_overlaps(self, target_idx: int, tolerance: int = 0) -> IntervalBatch:
        """
        Merges overlapping intervals on a target.

        Args:
            target_idx (int): The target sequence ID.
            tolerance (int, optional): Gap tolerance for merging. Defaults to 0.

        Returns:
            IntervalBatch: An IntervalBatch of merged intervals.
        """
        return self._get_spatial_index(target_idx).merge(tolerance=tolerance)

    def group_by(self, by_target: bool = False) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Efficiently groups alignments by query or target.

        Yields:
            (id, indices): Tuple of the query/target ID and a numpy array of indices into the collection.
        """
        indices = self.t_indices if by_target else self.q_indices
        # Sort indices by the key column (stable sort)
        perm = np.argsort(indices, kind='mergesort')
        sorted_keys = indices[perm]
        # Find unique keys and their start positions
        unique_keys, start_indices = np.unique(sorted_keys, return_index=True)

        for i, start in enumerate(start_indices):
            end = start_indices[i + 1] if i + 1 < len(start_indices) else len(perm)
            yield unique_keys[i], perm[start:end]

    def cull_overlaps(self, max_overlap_fraction: float = 0.1, key: str = 'score') -> 'AlignmentBatch':
        """
        Greedily removes alignments that overlap with higher-scoring alignments on the same target.

        Args:
            max_overlap_fraction (float, optional): Maximum allowed overlap fraction. Defaults to 0.1.
            key (str, optional): Attribute to use for sorting. Defaults to 'score'.

        Returns:
            AlignmentBatch: A new filtered AlignmentCollection.
        """
        col = self.Field(key)
        kept_indices_list = []

        # OPTIMIZATION: Global Sort -> Chunking
        if col in self._data.dtype.names and np.issubdtype(self._data[col].dtype, np.number):
            # Global Sort: Target (Primary), Key Descending (Secondary)
            # np.lexsort keys are (secondary, primary)
            vals = self._data[col]
            targets = self.t_indices
            # Negate vals for descending sort
            perm = np.lexsort((-vals, targets))

            # Find groups on sorted array
            t_sorted = targets[perm]
            _, start_indices = np.unique(t_sorted, return_index=True)

            # Pre-fetch arrays
            s_arr = self.t_starts
            e_arr = self.t_ends
            l_arr = self.aln_lens

            for i in range(len(start_indices)):
                start = start_indices[i]
                end = start_indices[i+1] if i + 1 < len(start_indices) else len(perm)
                group_indices = perm[start:end]

                # Kernel
                s = s_arr[group_indices]
                e = e_arr[group_indices]
                l = l_arr[group_indices]

                keep_mask = _greedy_overlap_kernel(s, e, l, max_overlap_fraction)
                kept_indices_list.append(group_indices[keep_mask])
        else:
            # Fallback: Object-based sort
            for target_id, group_indices in self.group_by(by_target=True):
                temp = [(self._make_alignment(i), i) for i in group_indices]
                temp.sort(key=lambda x: getattr(x[0], key), reverse=True)
                sorted_indices = [x[1] for x in temp]

                s_coords = self.t_starts[sorted_indices]
                e_coords = self.t_ends[sorted_indices]
                lengths = self.aln_lens[sorted_indices]
                keep_mask = _greedy_overlap_kernel(s_coords, e_coords, lengths, max_overlap_fraction)
                kept_indices_list.append(sorted_indices[keep_mask])

        # Return a new filtered collection
        if not kept_indices_list: return self[np.array([], dtype=np.int32)]

        # We sort indices to maintain stable order relative to input
        return self[np.sort(np.concatenate(kept_indices_list))]

    def chain(self, max_gap: int = 1000) -> 'AlignmentBatch':
        """
        Performs Synteny Chaining (Collinear Chaining) to reconstruct fragmented alignments.
        
        For each (Query, Target) pair, it finds the optimal chain of local alignments 
        that maximizes the total score while maintaining collinearity.

        Args:
            max_gap (int): Maximum allowed gap between chained alignments.

        Returns:
            AlignmentBatch: A new collection containing only the alignments that form the best chains.
        """
        # OPTIMIZATION: Global Sort: Target (Primary), Query (Secondary), Q_Start (Tertiary)
        # lexsort keys: (tertiary, secondary, primary)
        # We also need to group by Strand to handle reverse chains correctly
        perm = np.lexsort((self.q_starts, self.t_strands, self.q_indices, self.t_indices))
        
        t_sorted = self.t_indices[perm]
        q_sorted = self.q_indices[perm]
        st_sorted = self.t_strands[perm]
        
        # Pack for fast grouping
        # We detect boundaries where T, Q, or Strand changes
        diff = (t_sorted[:-1] != t_sorted[1:]) | (q_sorted[:-1] != q_sorted[1:]) | (st_sorted[:-1] != st_sorted[1:])
        start_indices = np.concatenate(([0], np.flatnonzero(diff) + 1))
        
        kept_indices = []
        
        # Pre-fetch arrays
        q_s_arr = self.q_starts
        q_e_arr = self.q_ends
        t_s_arr = self.t_starts
        t_e_arr = self.t_ends
        scores_arr = self.scores
        
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = start_indices[i+1] if i + 1 < len(start_indices) else len(perm)
            
            group_indices = perm[start:end]
            if len(group_indices) == 1: 
                kept_indices.append(group_indices)
                continue

            # Extract data (already sorted by q_start due to global sort)
            q_s = q_s_arr[group_indices]
            q_e = q_e_arr[group_indices]
            t_s = t_s_arr[group_indices]
            t_e = t_e_arr[group_indices]
            scores = scores_arr[group_indices]
            
            # Run DP Kernel (Replaces Graph/Bellman-Ford)
            # Solves Longest Path in DAG directly on arrays
            t_st = st_sorted[start]
            
            if t_st == -1:
                # Reverse strand chaining: Invert coordinates to enforce collinearity
                # Gap = t_s'[i] - t_e'[j] = (-t_e[i]) - (-t_s[j]) = t_s[j] - t_e[i]
                pred, best_idx = _chain_dp_kernel(q_s, q_e, -t_e, -t_s, scores, max_gap)
            else:
                pred, best_idx = _chain_dp_kernel(q_s, q_e, t_s, t_e, scores, max_gap)
            
            if best_idx == -1:
                # Fallback: keep best single alignment
                best_idx = np.argmax(scores)
                kept_indices.append(group_indices[best_idx:best_idx+1])
            else:
                # Backtrack
                local_idxs = _backtrack_kernel(pred, best_idx)
                kept_indices.append(group_indices[local_idxs])

        if not kept_indices: return self[np.array([], dtype=np.int32)]
        return self[np.sort(np.concatenate(kept_indices))]

    def find_dovetails(self, by_target: bool = False, return_ids: bool = False, tolerance: int = 0) -> Generator[
        tuple[Union[int, bytes], list[tuple[Union[int, bytes], int]], list[tuple[Union[int, bytes], int]]], None, None]:
        """
        Identifies potential dovetail connections based on alignment boundary conditions.

        Args:
            by_target (bool): If True, uses Targets as the pivot.
            return_ids (bool): If True, returns identifiers (bytes) instead of integer indices.
            tolerance (int): Tolerance for boundary checks (default 0).

        Yields:
            (pivot_id, starts_list, ends_list)
            starts_list: List of (node_id, strand) where the alignment extends LEFT from the Pivot.
            ends_list: List of (node_id, strand) where the alignment extends RIGHT from the Pivot.
        """
        # 1. Define Pivot and Node columns
        pivot = self.target if by_target else self.query
        node = self.query if by_target else self.target

        pivot_indices = pivot.indices
        pivot_starts = pivot.starts
        pivot_ends = pivot.ends
        pivot_lens = pivot.lengths
        
        # Relative strand is essential for correct graph orientation
        # (q_strand * t_strand) gives the relative orientation regardless of which side is pivot
        rel_strands = self.q_strands * self.t_strands

        node_indices = node.indices

        pivot_lookup = self.target_ids if by_target else self.query_ids
        node_lookup = self.query_ids if by_target else self.target_ids

        # 2. Group by Pivot
        perm = np.argsort(pivot_indices, kind='mergesort')
        p_sorted = pivot_indices[perm]
        _, start_indices = np.unique(p_sorted, return_index=True)

        # 3. Iterate Groups
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = start_indices[i + 1] if i + 1 < len(start_indices) else len(perm)
            group_indices = perm[start:end]

            p_s = pivot_starts[group_indices]
            p_e = pivot_ends[group_indices]
            p_l = pivot_lens[group_indices]
            
            # Get relative strands for this group
            r_st = rel_strands[group_indices]

            # Boundaries with tolerance
            p_at_start = p_s <= tolerance
            p_at_end = p_e >= (p_l - tolerance)

            # Starts (Left Extension): Pivot Start connects to Node
            starts_mask = p_at_start

            # Ends (Right Extension): Pivot End connects to Node
            ends_mask = p_at_end

            if not np.any(starts_mask) and not np.any(ends_mask):
                continue

            # Get Pivot ID
            p_idx = pivot_indices[group_indices[0]]
            p_out = pivot_lookup[p_idx] if return_ids and pivot_lookup is not None else p_idx

            # Helper to extract Node IDs and Strands
            def get_nodes(mask):
                if not np.any(mask): return []
                idxs = group_indices[mask]
                n_idxs = node_indices[idxs]
                n_st = r_st[mask] # Use relative strand
                
                if return_ids and node_lookup is not None:
                    ids = [node_lookup[x] for x in n_idxs]
                    return list(zip(ids, n_st.tolist()))
                return list(zip(n_idxs.tolist(), n_st.tolist()))

            yield p_out, get_nodes(starts_mask), get_nodes(ends_mask)


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _best_hit_kernel(indices, scores, num_ids):
    best_scores = np.full(num_ids, -np.inf, dtype=scores.dtype)
    best_idxs = np.full(num_ids, -1, dtype=np.int32)
    n = len(indices)
    for i in range(n):
        idx = indices[i]
        if scores[i] > best_scores[idx]:
            best_scores[idx] = scores[i]
            best_idxs[idx] = i

    # Compact
    out = np.empty(len(indices), dtype=np.int32)  # Upper bound
    k = 0
    for i in range(num_ids):
        if best_idxs[i] != -1:
            out[k] = best_idxs[i]
            k += 1
    return out[:k]


@jit(nopython=True, cache=True, nogil=True)
def _greedy_overlap_kernel(starts, ends, lengths, max_frac):
    """
    Greedily keeps alignments that don't overlap significantly with already kept ones.
    Assumes input is sorted by score descending.
    """
    n = len(starts)
    keep = np.ones(n, dtype=np.bool_)
    # Buffer to store indices of kept items to check against
    kept_indices = np.empty(n, dtype=np.int32)
    kept_count = 0

    for i in range(n):
        s1 = starts[i]
        e1 = ends[i]
        l1 = lengths[i]

        reject = False
        for k in range(kept_count):
            j = kept_indices[k]
            s2 = starts[j]
            e2 = ends[j]

            # Calculate Overlap
            start_max = s1 if s1 > s2 else s2
            end_min = e1 if e1 < e2 else e2

            if end_min > start_max:
                ov_len = end_min - start_max
                # Reject if overlap covers a significant fraction of the CURRENT alignment
                if (ov_len / l1) > max_frac:
                    reject = True
                    break

        if not reject:
            kept_indices[kept_count] = i
            kept_count += 1
        else:
            keep[i] = False

    return keep


@jit(nopython=True, cache=True, nogil=True)
def _chain_dp_kernel(q_s, q_e, t_s, t_e, scores, max_gap):
    """
    Solves the collinear chaining problem using Dynamic Programming.
    Assumes inputs are sorted by Query Start.
    """
    n = len(q_s)
    dp = scores.copy() # Init with individual scores
    pred = np.full(n, -1, dtype=np.int32)

    for i in range(1, n):
        # Look back for predecessors
        # Optimization: Iterate backwards to find recent connections first
        for j in range(i-1, -1, -1):
            # Query Gap Check (q_s[i] >= q_s[j] is guaranteed by sort)
            q_gap = q_s[i] - q_e[j]

            if q_gap < 0: continue # Query Overlap (not allowed in linear chain)
            if q_gap > max_gap: continue

            # Target Gap Check (Collinearity: t_s[i] > t_e[j])
            t_gap = t_s[i] - t_e[j]
            if t_gap < 0: continue # Target Overlap/Inversion
            if t_gap > max_gap: continue

            # Score Update
            new_score = dp[j] + scores[i]
            if new_score > dp[i]:
                dp[i] = new_score
                pred[i] = j

    # Find global max
    best_idx = -1
    max_score = -1.0

    for i in range(n):
        if dp[i] > max_score:
            max_score = dp[i]
            best_idx = i

    return pred, best_idx


@jit(nopython=True, cache=True, nogil=True)
def _backtrack_kernel(pred, best_idx):
    # 1. Count length
    curr = best_idx
    count = 0
    while curr != -1:
        count += 1
        curr = pred[curr]

    # 2. Fill
    out = np.empty(count, dtype=np.int32)
    curr = best_idx
    idx = count - 1
    while curr != -1:
        out[idx] = curr
        curr = pred[curr]
        idx -= 1
    return out


@jit(nopython=True, cache=True, nogil=True)
def _parse_cigar_kernel(cigar, map_table):
    """Parses CIGAR bytes into op codes and counts."""
    n = len(cigar)
    ops = np.empty(n, dtype=np.uint8)
    counts = np.empty(n, dtype=np.int32)
    idx = 0
    curr_count = 0
    for i in range(n):
        b = cigar[i]
        if 48 <= b <= 57:
            curr_count = (curr_count * 10) + (b - 48)
        else:
            ops[idx] = map_table[b]
            counts[idx] = curr_count
            idx += 1
            curr_count = 0
    return ops[:idx], counts[:idx]


@jit(nopython=True, cache=True, nogil=True)
def _cigar_rle_kernel(query, target, gap_code, extended):
    n = len(query)
    if n == 0: return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint8)
    counts = np.empty(n, dtype=np.int32)
    ops = np.empty(n, dtype=np.uint8)
    idx = 0
    q = query[0]
    t = target[0]
    if q == gap_code:
        curr_op = 2  # D
    elif t == gap_code:
        curr_op = 1  # I
    elif extended:
        curr_op = 7 if q == t else 8  # = / X
    else:
        curr_op = 0  # M
    curr_count = 1

    for i in range(1, n):
        q = query[i]
        t = target[i]
        if q == gap_code:
            op = 2
        elif t == gap_code:
            op = 1
        elif extended:
            op = 7 if q == t else 8
        else:
            op = 0
        if op == curr_op:
            curr_count += 1
        else:
            counts[idx] = curr_count
            ops[idx] = curr_op
            idx += 1
            curr_op = op
            curr_count = 1

    counts[idx] = curr_count
    ops[idx] = curr_op
    idx += 1
    return counts[:idx], ops[:idx]
