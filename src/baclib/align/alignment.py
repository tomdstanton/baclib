"""
Module for managing alignments.
"""
from typing import Generator, Iterable, Final, Any, Union
import copy

import numpy as np

from baclib.containers.record import Feature
from baclib.core.interval import Interval, IntervalIndex
from baclib.utils.resources import jit


# Classes --------------------------------------------------------------------------------------------------------------
class CigarParser:
    """Parses CIGAR strings into operations and lengths"""
    _SYM_TO_OP = {b'M': 0, b'I': 1, b'D': 2, b'N': 3, b'S': 4, b'H': 5, b'P': 6, b'=': 7, b'X': 8, b'B': 9}
    _OP_BYTES_LOOKUP = [b'?'] * 10
    for op, sym in {v: k for k, v in _SYM_TO_OP.items()}.items(): _OP_BYTES_LOOKUP[op] = sym

    # Fast lookup for bytes -> integer op codes
    _BYTE_TO_OP = np.full(256, 255, dtype=np.uint8)
    for s, op in _SYM_TO_OP.items(): _BYTE_TO_OP[ord(s)] = op

    # Consumption logic
    _QUERY_CONSUMERS = np.array([True, True, False, False, True, False, False, True, True, False], dtype=bool)
    _TARGET_CONSUMERS = np.array([True, False, True, True, False, False, False, True, True, False], dtype=bool)
    _ALN_CONSUMERS = _QUERY_CONSUMERS | _TARGET_CONSUMERS

    @classmethod
    def parse(cls, cigar: bytes) -> Generator[tuple[bytes, int, int, int, int], None, None]:
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

    @staticmethod
    def make(query: np.ndarray, target: np.ndarray, gap_code: int = 255, extended: bool = False) -> bytes:
        if len(query) == 0: return b""
        counts, ops = _cigar_rle_kernel(query, target, gap_code, extended)
        return b"".join([b"%d" % c + CigarParser._OP_BYTES_LOOKUP[o] for c, o in zip(counts, ops)])


class Alignment(Feature):
    """Represents a pairwise sequence alignment."""
    __slots__ = (
        'query', 'query_interval', 'query_length', 'target', 'target_length', 'length', 'cigar', 'score',
        'n_matches', 'quality'
    )

    def __init__(self, query: bytes, query_interval: Interval, target: bytes, interval: Interval,
                 query_length: int = 0, target_length: int = 0, length: int = 0, cigar: bytes = None,
                 n_matches: int = 0, quality: int = 0, qualifiers: Iterable[tuple[bytes, Any]] = None, score: float = 0):
        super().__init__(interval, kind=b'Alignment', qualifiers=qualifiers)
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

    def __repr__(self):
        return f"Alignment({self.query.decode('ascii', 'ignore')}->{self.target.decode('ascii', 'ignore')}, score={self.score})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (super().__eq__(other) and
                    self.query == other.query and
                    self.query_interval == other.query_interval and
                    self.score == other.score and
                    self.cigar == other.cigar)
        return False

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
        return len(self.query_interval) / self.query_length if self.query_length > 0 else 0.0

    def target_coverage(self) -> float:
        return len(self.interval) / self.target_length if self.target_length > 0 else 0.0

    def identity(self) -> float:
        return self.n_matches / self.length if self.length > 0 else 0.0

    def flip(self) -> 'Alignment':
        return Alignment(
            query=self.target, query_interval=self.interval, query_length=self.target_length,
            target=self.query, interval=self.query_interval, target_length=self.query_length,
            length=self.length, cigar=self.cigar, score=self.score, n_matches=self.n_matches,
            quality=self.quality, qualifiers=list(self.qualifiers) if self.qualifiers else None
        )


class AlignmentBatch:
    """
    High-performance container for alignment batches.
    Stores data in Structure-of-Arrays (SoA) layout.
    """

    class _Fields:
        Q_IDX: Final[str] = 'q_idx'; T_IDX: Final[str] = 't_idx'; SCORE: Final[str] = 'score'
        Q_START: Final[str] = 'q_start'; Q_END: Final[str] = 'q_end'; Q_LEN: Final[str] = 'q_len'; Q_STRAND: Final[str] = 'q_strand'
        T_START: Final[str] = 't_start'; T_END: Final[str] = 't_end'; T_LEN: Final[str] = 't_len'; T_STRAND: Final[str] = 't_strand'
        MATCHES: Final[str] = 'matches'; QUALITY: Final[str] = 'quality'; ALN_LEN: Final[str] = 'aln_len'

    _DTYPE = np.dtype([
        (_Fields.Q_IDX, np.int32), (_Fields.T_IDX, np.int32), (_Fields.SCORE, np.float32),
        (_Fields.Q_START, np.int32), (_Fields.Q_END, np.int32), (_Fields.Q_LEN, np.int32), (_Fields.Q_STRAND, np.int8),
        (_Fields.T_START, np.int32), (_Fields.T_END, np.int32), (_Fields.T_LEN, np.int32), (_Fields.T_STRAND, np.int8),
        (_Fields.MATCHES, np.int32), (_Fields.QUALITY, np.uint8), (_Fields.ALN_LEN, np.int32)
    ])

    def __init__(self, data: np.ndarray = None, cigars: np.ndarray = None, qualifiers: np.ndarray = None,
                 query_ids: np.ndarray = None, target_ids: np.ndarray = None):
        """
        Direct initialization. Prefer factory methods (from_hits, etc.) for ease of use.
        """
        if data is None:
            self._data = np.zeros(0, dtype=self._DTYPE)
            self._cigars = np.zeros(0, dtype=object)
            self._qualifiers = np.zeros(0, dtype=object)
        else:
            self._data = data
            n = len(data)
            self._cigars = cigars if cigars is not None else np.full(n, None, dtype=object)
            self._qualifiers = qualifiers if qualifiers is not None else np.full(n, None, dtype=object)

        self.query_ids = query_ids
        self.target_ids = target_ids
        self._spatial_indices: dict[int, IntervalIndex] = {}

    @classmethod
    def from_data(cls,
                  q_idx: np.ndarray, t_idx: np.ndarray, score: np.ndarray,
                  q_coords: np.ndarray, t_coords: np.ndarray,
                  q_lens: np.ndarray, t_lens: np.ndarray,
                  cigars: np.ndarray = None):
        """Zero-copy construction from Aligner output arrays."""
        n = len(q_idx)
        data = np.zeros(n, dtype=cls._DTYPE)

        data[cls._Fields.Q_IDX] = q_idx
        data[cls._Fields.T_IDX] = t_idx
        data[cls._Fields.SCORE] = score
        data[cls._Fields.Q_START] = q_coords[:, 0]
        data[cls._Fields.Q_END] = q_coords[:, 1]
        data[cls._Fields.T_START] = t_coords[:, 0]
        data[cls._Fields.T_END] = t_coords[:, 1]
        data[cls._Fields.Q_LEN] = q_lens
        data[cls._Fields.T_LEN] = t_lens

        data[cls._Fields.Q_STRAND] = 1
        data[cls._Fields.T_STRAND] = 1
        data[cls._Fields.ALN_LEN] = np.maximum(
            data[cls._Fields.Q_END] - data[cls._Fields.Q_START],
            data[cls._Fields.T_END] - data[cls._Fields.T_START]
        )
        data[cls._Fields.MATCHES] = (data[cls._Fields.ALN_LEN] * 0.9).astype(np.int32)
        data[cls._Fields.QUALITY] = 60

        return cls(data=data, cigars=cigars)

    @classmethod
    def from_alignments(cls, alignments: Iterable[Alignment]) -> 'AlignmentBatch':
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
        
        q_ids = None
        if isinstance(first.query, (int, np.integer)):
            q_indices = [x.query for x in items]
        else:
            q_ids, q_indices = np.unique([x.query for x in items], return_inverse=True)
            
        t_ids = None
        if isinstance(first.target, (int, np.integer)):
            t_indices = [x.target for x in items]
        else:
            t_ids, t_indices = np.unique([x.target for x in items], return_inverse=True)

        data = np.zeros(n, dtype=cls._DTYPE)

        # Column-wise assignment is faster than row-wise loop in Python
        data[cls._Fields.Q_IDX] = q_indices
        data[cls._Fields.T_IDX] = t_indices
        data[cls._Fields.SCORE] = [x.score for x in items]
        data[cls._Fields.Q_START] = [x.query_interval.start for x in items]
        data[cls._Fields.Q_END] = [x.query_interval.end for x in items]
        data[cls._Fields.Q_LEN] = [x.query_length for x in items]
        data[cls._Fields.Q_STRAND] = [x.query_interval.strand for x in items]
        data[cls._Fields.T_START] = [x.interval.start for x in items]
        data[cls._Fields.T_END] = [x.interval.end for x in items]
        data[cls._Fields.T_LEN] = [x.target_length for x in items]
        data[cls._Fields.T_STRAND] = [x.interval.strand for x in items]
        data[cls._Fields.MATCHES] = [x.n_matches for x in items]
        data[cls._Fields.QUALITY] = [x.quality for x in items]
        data[cls._Fields.ALN_LEN] = [x.length for x in items]

        cigars = np.array([x.cigar for x in items], dtype=object)
        qualifiers = np.array([x.qualifiers for x in items], dtype=object)

        return cls(data=data, cigars=cigars, qualifiers=qualifiers, query_ids=q_ids, target_ids=t_ids)

    def __len__(self): return len(self._data)
    def __iter__(self):
        for i in range(len(self)): yield self._make_alignment(i)
    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)): return self._make_alignment(item)
        elif isinstance(item, (slice, np.ndarray, list)):
            return AlignmentBatch(self._data[item], self._cigars[item], self._qualifiers[item],
                                  query_ids=self.query_ids, target_ids=self.target_ids)
        raise TypeError(f"Invalid index type: {type(item)}")
    @property
    def scores(self) -> np.ndarray: return self._data[self._Fields.SCORE]
    @property
    def q_indices(self) -> np.ndarray: return self._data[self._Fields.Q_IDX]
    @property
    def t_indices(self) -> np.ndarray: return self._data[self._Fields.T_IDX]
    @property
    def q_coords(self) -> np.ndarray:
        return np.stack((self._data[self._Fields.Q_START], self._data[self._Fields.Q_END]), axis=1)
    @property
    def q_starts(self) -> np.ndarray: return self._data[self._Fields.Q_START]
    @property
    def q_ends(self) -> np.ndarray: return self._data[self._Fields.Q_END]
    @property
    def t_starts(self) -> np.ndarray: return self._data[self._Fields.T_START]
    @property
    def t_ends(self) -> np.ndarray: return self._data[self._Fields.T_END]
    @property
    def matches(self) -> np.ndarray: return self._data[self._Fields.MATCHES]
    @property
    def aln_lens(self) -> np.ndarray: return self._data[self._Fields.ALN_LEN]
    @property
    def q_lens(self) -> np.ndarray: return self._data[self._Fields.Q_LEN]
    @property
    def t_lens(self) -> np.ndarray: return self._data[self._Fields.T_LEN]
    @property
    def q_strands(self) -> np.ndarray: return self._data[self._Fields.Q_STRAND]
    @property
    def t_strands(self) -> np.ndarray: return self._data[self._Fields.T_STRAND]

    def filter(self, query_idxs: Iterable[int] = None, target_idxs: Iterable[int] = None) -> 'AlignmentBatch':
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
        indices = self.t_indices if by_target else self.q_indices
        # To find max ID for array sizing, we scan once.
        # This is safe because indices correspond to SeqBatch,
        # so max(indices) < len(batch).
        max_id = np.max(indices) if len(indices) > 0 else 0
        best_indices = _best_hit_kernel(indices, self.scores, max_id + 1)
        return self[np.sort(best_indices)]

    def sort(self, by: str = 'score', ascending: bool = False) -> 'AlignmentBatch':
        # Now we map the 'by' string to our internal fields safely
        lookup = {
            'score': self._Fields.SCORE,
            'query': self._Fields.Q_IDX,
            'target': self._Fields.T_IDX
        }
        col = lookup.get(by, by)  # Fallback allows direct field name usage

        perm = np.argsort(self._data[col])
        if not ascending: perm = perm[::-1]
        return self[perm]

    def _make_alignment(self, idx: int) -> Alignment:
        r = self._data[idx]
        q_idx = r[self._Fields.Q_IDX]
        t_idx = r[self._Fields.T_IDX]
        
        return Alignment(
            query=self.query_ids[q_idx] if self.query_ids is not None else q_idx,
            query_interval=Interval(r[self._Fields.Q_START], r[self._Fields.Q_END], r[self._Fields.Q_STRAND]),
            target=self.target_ids[t_idx] if self.target_ids is not None else t_idx,
            interval=Interval(r[self._Fields.T_START], r[self._Fields.T_END], r[self._Fields.T_STRAND]),
            query_length=r[self._Fields.Q_LEN], target_length=r[self._Fields.T_LEN], length=r[self._Fields.ALN_LEN],
            cigar=self._cigars[idx], score=r[self._Fields.SCORE], n_matches=r[self._Fields.MATCHES],
            quality=r[self._Fields.QUALITY], qualifiers=self._qualifiers[idx]
        )

    def _get_spatial_index(self, target_idx: int) -> IntervalIndex:
        """Retrieves or builds the spatial index for a target."""
        if (target_int_idx := self._spatial_indices.get(target_idx)) is None:
            mask = self.t_indices == target_idx
            starts = self.t_starts[mask]
            ends = self.t_ends[mask]
            intervals = [Interval(s, e) for s, e in zip(starts, ends)]
            self._spatial_indices[target_idx] = IntervalIndex.from_intervals(*intervals)
        return target_int_idx

    def flip(self) -> 'AlignmentBatch':
        """
        Returns a new collection with queries and targets swapped.
        Vectorized operation.
        """
        new_coll = copy.copy(self)

        # Swap data columns
        new_data = self._data.copy()
        new_data[self._Fields.Q_IDX] = self.t_indices
        new_data[self._Fields.T_IDX] = self.q_indices
        new_data[self._Fields.Q_START] = self.t_starts
        new_data[self._Fields.Q_END] = self.t_ends
        new_data[self._Fields.Q_LEN] = self.t_lens
        new_data[self._Fields.Q_STRAND] = self.t_strands
        new_data[self._Fields.T_START] = self.q_starts
        new_data[self._Fields.T_END] = self.q_ends
        new_data[self._Fields.T_LEN] = self.q_lens
        new_data[self._Fields.T_STRAND] = self.q_strands

        new_coll._data = new_data
        new_coll.query_ids = self.target_ids
        new_coll.target_ids = self.query_ids
        new_coll._graph = None  # Invalidate graph
        new_coll._t_id_map = None
        new_coll._spatial_indices = {}  # Invalidate indices
        return new_coll

    def pileup(self, target_idx: int, length: int = None) -> np.ndarray:
        """
        Calculates coverage depth across the target sequence.

        Args:
            target_idx (int): The target sequence index.
            length (int, optional): Length of the target sequence. Defaults to max interval end.

        Returns:
            np.ndarray: A numpy array of depths.
        """
        idx = self._get_spatial_index(target_idx)
        if len(idx) == 0: return np.zeros(length or 0, dtype=np.int32)
        size = length or (idx.ends.max() + 1)
        
        # Optimization: Use Numba kernel instead of unbuffered np.add.at
        diffs = _pileup_kernel(idx.starts, idx.ends, size)
        
        return np.add.accumulate(diffs)[:-1]

    def merge_overlaps(self, target_idx: int, tolerance: int = 0) -> IntervalIndex:
        """
        Merges overlapping intervals on a target.

        Args:
            target_idx (int): The target sequence ID.
            tolerance (int, optional): Gap tolerance for merging. Defaults to 0.

        Returns:
            IntervalIndex: An IntervalIndex of merged intervals.
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
        kept_indices_list = []

        # Use the new group_by method
        for target_id, group_indices in self.group_by(by_target=True):

            # Sort within group by key (usually score)
            lookup = {
                'score': self._Fields.SCORE,
                'query': self._Fields.Q_IDX,
                'target': self._Fields.T_IDX
            }
            col = lookup.get(key, key)

            if col in self._data.dtype.names:
                # Vectorized sort
                vals = self._data[col][group_indices]
                local_perm = np.argsort(vals)[::-1]
                sorted_indices = group_indices[local_perm]
            else:
                # Object-based sort (fallback)
                temp = [(self._make_alignment(i), i) for i in group_indices]
                temp.sort(key=lambda x: getattr(x[0], key), reverse=True)
                sorted_indices = [x[1] for x in temp]

            # Extract relevant columns for the sorted group (Views if possible)
            s_coords = self.t_starts[sorted_indices]
            e_coords = self.t_ends[sorted_indices]
            lengths = self.aln_lens[sorted_indices]
            
            # Run Numba Kernel
            keep_mask = _greedy_overlap_kernel(s_coords, e_coords, lengths, max_overlap_fraction)
            
            # Append the indices that passed the filter
            kept_indices_list.append(sorted_indices[keep_mask])

        # Return a new filtered collection
        if not kept_indices_list: return self[np.array([], dtype=np.int32)]

        # We sort indices to maintain stable order relative to input
        return self[np.sort(np.concatenate(kept_indices_list))]

    def find_bridges(self, tolerance: int = 0, min_length: int = 0, by_target: bool = False,
                     return_ids: bool = False) -> Generator[tuple[Union[int, bytes], Union[int, bytes]], None, None]:
        """
        Yields pairs of alignments that bridge two sequences via a common pivot.

        A bridge is defined as two alignments on the same pivot sequence (Query or Target):
        1. A 'Left' alignment starting at the beginning of the pivot (start <= tolerance).
        2. A 'Right' alignment ending at the end of the pivot (end >= length - tolerance).

        Args:
            tolerance (int): Maximum distance from the ends to be considered an edge alignment.
            min_length (int): Minimum alignment length to consider.
            by_target (bool): If True, uses Targets as the pivot. Defaults to False (Queries).
            return_ids (bool): If True, returns identifiers (bytes) instead of integer indices.

        Yields:
            tuple[int, int]:
                (index_Left, index_Right)
                index_Left/Right are the indices of the sequences connected to the pivot.
        """
        # Define columns based on pivot
        if by_target:
            starts_arr, ends_arr, lens_arr = self.t_starts, self.t_ends, self.t_lens
            other_indices = self.q_indices
            lookup = self.query_ids if return_ids else None
        else:
            starts_arr, ends_arr, lens_arr = self.q_starts, self.q_ends, self.q_lens
            other_indices = self.t_indices
            lookup = self.target_ids if return_ids else None

        for pivot_id, group_indices in self.group_by(by_target=by_target):
            # Filter by length first
            if min_length > 0:
                lens = self._data[self._Fields.ALN_LEN][group_indices]
                group_indices = group_indices[lens >= min_length]

            if len(group_indices) < 2: continue

            # Extract coordinates
            starts = starts_arr[group_indices]
            ends = ends_arr[group_indices]
            lengths = lens_arr[group_indices]

            # Identify Left and Right candidates
            left_mask = starts <= tolerance
            right_mask = ends >= (lengths - tolerance)

            left_indices = group_indices[left_mask]
            right_indices = group_indices[right_mask]

            if len(left_indices) == 0 or len(right_indices) == 0: continue

            # Yield pairs
            for l_idx in left_indices:
                for r_idx in right_indices:
                    if l_idx == r_idx: continue
                    u, v = other_indices[l_idx], other_indices[r_idx]
                    if lookup is not None: yield lookup[u], lookup[v]
                    else: yield u, v

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
        kept_indices = []
        
        # Process each pair independently
        for q_id, t_id, group_indices in self.group_by_pair():
            if len(group_indices) == 0: continue
            if len(group_indices) == 1: 
                kept_indices.append(group_indices)
                continue

            # Extract data for kernel
            # Sort by Query Start to simplify DAG construction
            q_starts = self.q_starts[group_indices]
            perm = np.argsort(q_starts)
            sorted_indices = group_indices[perm]
            
            q_s = self.q_starts[sorted_indices]
            q_e = self.q_ends[sorted_indices]
            t_s = self.t_starts[sorted_indices]
            t_e = self.t_ends[sorted_indices]
            scores = self.scores[sorted_indices]
            
            # Run DP Kernel (Replaces Graph/Bellman-Ford)
            # Solves Longest Path in DAG directly on arrays
            pred, best_idx = _chain_dp_kernel(q_s, q_e, t_s, t_e, scores, max_gap)
            
            if best_idx == -1:
                # Fallback: keep best single alignment
                best_idx = np.argmax(scores)
                kept_indices.append(sorted_indices[best_idx:best_idx+1])
                continue
                
            # Backtrack
            chain_idxs = []
            curr = best_idx
            while curr != -1:
                chain_idxs.append(curr)
                curr = pred[curr]
                
            # Restore chronological order
            kept_indices.append(sorted_indices[chain_idxs[::-1]])

        if not kept_indices: return self[np.array([], dtype=np.int32)]
        return self[np.sort(np.concatenate(kept_indices))]

    def group_by_pair(self) -> Generator[tuple[int, int, np.ndarray], None, None]:
        """Groups alignments by (Query, Target) pair."""
        # Create a composite key for sorting
        # We can't easily lexsort strings in numpy, so we use the indices
        q_idx = self.q_indices
        t_idx = self.t_indices
        
        # Lexsort by Target then Query
        perm = np.lexsort((t_idx, q_idx))
        
        q_sorted = q_idx[perm]
        t_sorted = t_idx[perm]
        
        # Find unique runs
        # Combine into complex number or struct for uniqueness check if needed, 
        # or just iterate since we are sorted.
        n = len(perm)
        if n == 0: return
        
        start = 0
        curr_q = q_sorted[0]
        curr_t = t_sorted[0]
        
        for i in range(1, n):
            nq = q_sorted[i]
            nt = t_sorted[i]
            if nq != curr_q or nt != curr_t:
                yield curr_q, curr_t, perm[start:i]
                start = i
                curr_q = nq
                curr_t = nt
        yield curr_q, curr_t, perm[start:]


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _parse_cigar_kernel(cigar, map_table):
    """Parses CIGAR bytes into op codes and counts."""
    n = len(cigar)
    ops = np.empty(n, dtype=np.uint8)
    counts = np.empty(n, dtype=np.int32)
    idx = 0; curr_count = 0
    for i in range(n):
        b = cigar[i]
        if 48 <= b <= 57:
            curr_count = (curr_count * 10) + (b - 48)
        else:
            ops[idx] = map_table[b]; counts[idx] = curr_count
            idx += 1; curr_count = 0
    return ops[:idx], counts[:idx]


@jit(nopython=True, cache=True, nogil=True)
def _cigar_rle_kernel(query, target, gap_code, extended):
    n = len(query)
    if n == 0: return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint8)
    counts = np.empty(n, dtype=np.int32); ops = np.empty(n, dtype=np.uint8); idx = 0; q = query[0]; t = target[0]
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
        q = query[i]; t = target[i]
        if q == gap_code: op = 2
        elif t == gap_code: op = 1
        elif extended: op = 7 if q == t else 8
        else: op = 0
        if op == curr_op:
            curr_count += 1
        else:
            counts[idx] = curr_count; ops[idx] = curr_op; idx += 1
            curr_op = op; curr_count = 1

    counts[idx] = curr_count; ops[idx] = curr_op; idx += 1
    return counts[:idx], ops[:idx]


@jit(nopython=True, cache=True, nogil=True)
def _best_hit_kernel(indices, scores, num_ids):
    best_scores = np.full(num_ids, -np.inf, dtype=scores.dtype)
    best_idxs = np.full(num_ids, -1, dtype=np.int32)
    n = len(indices)
    for i in range(n):
        idx = indices[i]
        if scores[i] > best_scores[idx]:
            best_scores[idx] = scores[i]; best_idxs[idx] = i

    # Compact
    out = np.empty(len(indices), dtype=np.int32)  # Upper bound
    k = 0
    for i in range(num_ids):
        if best_idxs[i] != -1: out[k] = best_idxs[i]; k += 1
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
def _pileup_kernel(starts: np.ndarray, ends: np.ndarray, size: int) -> np.ndarray:
    """
    Fast histogramming for coverage calculation.
    """
    diffs = np.zeros(size + 1, dtype=np.int32)
    n = len(starts)
    for i in range(n):
        s = starts[i]
        e = ends[i]
        if s < size: diffs[s] += 1
        if e < size: diffs[e] -= 1
    return diffs
