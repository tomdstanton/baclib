from typing import Union, Any, Match, MutableSequence, Iterable
from enum import IntEnum

import numpy as np

from baclib.utils.resources import RESOURCES, jit

if 'numba' in RESOURCES.optional_packages: from numba import prange
else: prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Strand(IntEnum):
    """
    Enumeration for genomic strands.
    """
    FORWARD = 1
    REVERSE = -1
    UNSTRANDED = 0

    @property
    def token(self) -> bytes:
        """Returns the byte representation of the strand (+, -, .)."""
        if self.value == 1: return b'+'
        if self.value == -1: return b'-'
        return b'.'

    @classmethod
    def from_symbol(cls, symbol: Any) -> 'Strand':
        """
        Creates a Strand from various symbols.

        Args:
            symbol: The symbol to convert (e.g., '+', '-', 1, -1, b'+').

        Returns:
            A Strand enum member.
        """
        if symbol is None: return cls.UNSTRANDED
        # Fast path for integers (including numpy scalars)
        if symbol in (1, -1, 0): return cls(int(symbol))

        # Handle bytes specifically to avoid str(b'+') -> "b'+'"
        if isinstance(symbol, bytes):
            if symbol in (b'+', b'1'): return cls.FORWARD
            if symbol in (b'-', b'-1'): return cls.REVERSE
            return cls.UNSTRANDED

        # String parsing for everything else
        s = str(symbol)
        if s in ('+', '1'): return cls.FORWARD
        if s in ('-', '-1'): return cls.REVERSE
        return cls.UNSTRANDED


class Interval:
    """
    Immutable genomic interval. Safe for hashing and use in sets/dicts.

    Attributes:
        start: The start position (0-based, inclusive).
        end: The end position (0-based, exclusive).
        strand: The strand (FORWARD, REVERSE, or UNSTRANDED).

    Examples:
        >>> i = Interval(10, 20, '+')
        >>> len(i)
        10
        >>> i.strand
        <Strand.FORWARD: 1>
    """
    # 1. Use private slots
    __slots__ = ('_start', '_end', '_strand')

    def __init__(self, start: int, end: int, strand: Any = None):
        """
        Initializes an Interval.

        Args:
            start: Start position.
            end: End position.
            strand: Strand symbol or enum.
        """
        self._start: int = int(start)
        self._end: int = int(end)
        self._strand: Strand = Strand.from_symbol(strand)

    @property
    def start(self): return self._start
    @property
    def end(self): return self._end
    @property
    def strand(self) -> Strand: return self._strand
    def __hash__(self): return hash((self._start, self._end, self._strand))
    def __repr__(self): return f"{self._start}:{self._end}({self._strand.token.decode('ascii')})"
    def __len__(self): return max(0, self._end - self._start)
    def __iter__(self): return iter((self._start, self._end, self._strand))

    def __array__(self, dtype=None):
        """Allows the Interval to be treated as a numpy array (e.g. np.array(interval))."""
        return np.array([self._start, self._end, self._strand], dtype=dtype or np.int32)

    def __eq__(self, other):
        if not isinstance(other, Interval): return False
        # Direct slot access is fastest
        return (self._start == other._start and
                self._end == other._end and
                self._strand == other._strand)

    def __contains__(self, item: Union[slice, int, 'Interval']):
        if isinstance(item, int): return self._start <= item < self._end
        item = Interval.from_item(item)
        return self._start <= item.start and self._end >= item.end

    def overlap(self, other: Union[slice, int, 'Interval']) -> int:
        """
        Calculates the overlap length with another interval.

        Args:
            other: The other interval.

        Returns:
            The number of overlapping bases.
        """
        other = Interval.from_item(other)
        return max(0, min(self._end, other.end) - max(self._start, other.start))

    # --- Hull Arithmetic ---
    def __add__(self, other: Union[slice, int, 'Interval']) -> 'Interval':
        """Returns the convex hull (union extent) as a NEW Interval."""
        other = Interval.from_item(other)
        new_strand = self._strand if self._strand == other.strand else 0
        return Interval(min(self._start, other.start), max(self._end, other.end), new_strand)

    def __radd__(self, other: Union[slice, int, 'Interval']) -> 'Interval':
        return self.__add__(other)

    def intersection(self, other: Union[slice, int, 'Interval']) -> 'Interval':
        """Returns the intersection of this interval and another as a NEW Interval."""
        other = Interval.from_item(other)
        new_start = max(self._start, other.start)
        new_end = min(self._end, other.end)
        # Result inherits self's strand? Or 0? Usually self's strand for intersection.
        if new_start >= new_end: return Interval(new_start, new_start, self._strand)
        return Interval(new_start, new_end, self._strand)

    def shift(self, x: int, y: int = None) -> 'Interval':
        """
        Shifts the interval coordinates.

        Args:
            x: Amount to shift start (and end if y is None).
            y: Amount to shift end (optional).

        Returns:
            A new shifted Interval.
        """
        return Interval(self._start + x, self._end + (y if y is not None else x), self._strand)

    def reverse_complement(self, parent_length: int) -> 'Interval':
        """
        Returns the interval coordinates on the reverse complement strand.

        Args:
            parent_length: The length of the parent sequence.

        Returns:
            A new Interval on the opposite strand.
        """
        return Interval(parent_length - self._end, parent_length - self._start, self._strand * -1)

    @classmethod
    def random(cls, rng: np.random.Generator = None, length: int = None, min_len: int = 1, max_len: int = 10_000,
               min_start: int = 0, max_start: int = 1_000_000):
        """
        Generates a random Interval.

        Args:
            rng: Random number generator.
            length: Fixed length (optional).
            min_len: Minimum length.
            max_len: Maximum length.
            min_start: Minimum start position.
            max_start: Maximum start position.

        Returns:
            A random Interval.
        """
        if rng is None: rng = RESOURCES.rng
        if not length: length = rng.integers(min_len, max_len)
        # Ensure we don't go negative on the bounds
        safe_max_start = max(min_start + 1, max_start - length)
        start = rng.integers(min_start, safe_max_start)
        return cls(start, start + length, rng.choice([1, -1]))

    @classmethod
    def from_item(cls, item: Union[slice, int, 'Interval', Match], strand: int = 0, length: int = None) -> 'Interval':
        """
        Coerces various types into an Interval.

        Args:
            item: The item to coerce (slice, int, Interval, Match).
            strand: Default strand if not present in item.
            length: Length of the sequence (needed for slice with None stop).

        Returns:
            An Interval object.

        Raises:
            TypeError: If coercion is not possible.
        """
        if isinstance(item, Interval): return item
        if interval := getattr(item, 'interval', None): return interval
        if isinstance(item, Match): return Interval(item.start(), item.end(), strand or 1)
        if isinstance(item, int):
            if item < 0 and length is not None: item += length
            return cls(item, item + 1, strand or 1)
        if isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            if start is None: start = 0
            if stop is None and length is not None: stop = length
            if stop is None: raise ValueError("Cannot create Interval from slice with None stop without 'length'")

            if step == -1: return cls(stop + 1, start + 1, strand or -1)
            return cls(start, stop, strand or 1)
        raise TypeError(f"Cannot coerce {type(item)} to Interval")


class IntervalIndex:
    """
    High-performance index for genomic intervals, powered by NumPy.

    This class provides efficient querying, intersection, and manipulation of
    large sets of genomic intervals.

    Examples:
        >>> i1 = Interval(0, 100, '+')
        >>> i2 = Interval(50, 150, '-')
        >>> idx = IntervalIndex.from_intervals(i1, i2)
        >>> len(idx)
        2
        >>> idx.coverage()
        150
    """
    __slots__ = ('_starts', '_ends', '_strands', '_original_indices', '_max_len')
    _DTYPE = np.int32  # int32 allows up to 2.14 Billion bp (fits all bacterial genomes)

    def __init__(self, starts: np.ndarray = None, ends: np.ndarray = None, strands: np.ndarray = None,
                 original_indices: np.ndarray = None):
        """
        Initializes an IntervalIndex.

        Args:
            starts: Array of start positions.
            ends: Array of end positions.
            strands: Array of strands.
            original_indices: Array of original indices (for tracking after sort).
        """
        if starts is None:
            self._starts = np.empty(0, dtype=self._DTYPE)
            self._ends = np.empty(0, dtype=self._DTYPE)
            self._strands = np.empty(0, dtype=self._DTYPE)
            self._max_len = 0
        else:
            # Ensure contiguous arrays for Numba performance
            self._starts = np.ascontiguousarray(starts, dtype=self._DTYPE)
            self._ends = np.ascontiguousarray(ends, dtype=self._DTYPE)
            self._strands = np.ascontiguousarray(strands, dtype=self._DTYPE) if strands is not None else np.zeros(len(starts), dtype=self._DTYPE)
            # Calculate max length for query optimization
            self._max_len = np.max(self._ends - self._starts) if len(self._starts) > 0 else 0

        if original_indices is None and starts is not None:
            self._original_indices = np.arange(len(self._starts), dtype=np.int32)
        else:
            self._original_indices = original_indices
        self.sort()

    def sort(self):
        """Sorts the intervals by start position, then end position."""
        # Sort both data and the index tracker
        if len(self._starts) > 0:
            # Lexsort: Primary key is last in the tuple (starts), Secondary is ends
            order = np.lexsort((self._ends, self._starts))
            self._starts = self._starts[order]
            self._ends = self._ends[order]
            self._strands = self._strands[order]
            self._original_indices = self._original_indices[order]

    @classmethod
    def from_intervals(cls, *intervals: Interval):
        """
        Creates an IntervalIndex from a list of Interval objects.

        Args:
            *intervals: Variable number of Interval objects.

        Returns:
            An IntervalIndex.
        """
        if not intervals: return cls()
        # Convert to SoA
        arr = np.array([tuple(i) for i in intervals], dtype=cls._DTYPE)
        return cls(arr[:, 0], arr[:, 1], arr[:, 2])

    @classmethod
    def from_features(cls, *features):
        """
        Creates an IntervalIndex from a list of Feature objects.

        Args:
            *features: Variable number of Feature objects (must have .interval attribute).

        Returns:
            An IntervalIndex.
        """
        if not features: return cls()
        arr = np.array([tuple(i.interval) for i in features], dtype=cls._DTYPE)
        return cls(arr[:, 0], arr[:, 1], arr[:, 2])

    @classmethod
    def from_items(cls, *items: Union[slice, int, 'Interval', Match]):
        """
        Creates an IntervalIndex from various items.

        Args:
            *items: Items to convert to intervals.

        Returns:
            An IntervalIndex.
        """
        if not items: return cls()
        arr = np.array([tuple(Interval.from_item(i)) for i in items], dtype=cls._DTYPE)
        return cls(arr[:, 0], arr[:, 1], arr[:, 2])

    def __len__(self): return len(self._starts)
    def __iter__(self): return iter(zip(self.starts, self.ends, self.strands))
    def copy(self):
        """Returns a deep copy of the IntervalIndex."""
        return IntervalIndex(
            self._starts.copy(), self._ends.copy(), self._strands.copy(),
            self._original_indices.copy() if self._original_indices is not None else None
        )

    @property
    def starts(self): return self._starts
    @property
    def ends(self): return self._ends
    @property
    def strands(self): return self._strands

    @staticmethod
    def _generate_intervals(s, e, st): yield from (Interval(*i) for i in zip(s, e, st))

    def query(self, start: int, end: int) -> np.ndarray:
        """Returns indices of intervals overlapping [start, end)."""
        if len(self) == 0: return np.empty(0, dtype=np.int32)
        # Use Numba kernel for fast search
        return _query_kernel(self.starts, self.ends, self._original_indices,
                             start, end, self._max_len)

    def intersect(self, other: 'IntervalIndex', stranded: bool = False) -> 'IntervalIndex':
        """
        Computes the intersection with another IntervalIndex.

        Args:
            other: The other IntervalIndex.
            stranded: If True, only intersects intervals on the same strand.

        Returns:
            A new IntervalIndex representing the intersection.
        """
        if len(self) == 0 or len(other) == 0: return IntervalIndex()
        # Call Numba Kernel
        out = _intersect_kernel(self.starts, self.ends, self.strands,
                                other.starts, other.ends, other.strands, stranded)
        # Kernel returns tuple of arrays
        if len(out[0]) == 0: return IntervalIndex()
        return IntervalIndex(out[0], out[1], out[2])

    def subtract(self, other: 'IntervalIndex', stranded: bool = False) -> 'IntervalIndex':
        """
        Subtracts regions in 'other' from this index.

        Args:
            other: The IntervalIndex to subtract.
            stranded: If True, only subtracts intervals on the same strand.

        Returns:
            A new IntervalIndex.
        """
        if len(other) == 0: return self.copy()
        # Merge B to simplify subtraction
        b_merged = other.merge()

        out = _subtract_kernel(self.starts, self.ends, self.strands,
                               b_merged.starts, b_merged.ends, b_merged.strands, stranded)
        if len(out[0]) == 0: return IntervalIndex()
        return IntervalIndex(out[0], out[1], out[2])

    def promoters(self, upstream: int = 100, downstream: int = 0) -> 'IntervalIndex':
        """
        Extracts promoter regions relative to strand.
        Forward: [Start - Up, Start + Down]
        Reverse: [End - Down, End + Up]

        Args:
            upstream: Bases upstream of the start.
            downstream: Bases downstream of the start.

        Returns:
            A new IntervalIndex of promoters.
        """
        return self.flank(upstream, downstream, direction='upstream')

    def flank(self, left: int, right: int = None, direction: str = 'both') -> 'IntervalIndex':
        """
        Generates flanking regions.

        Args:
            left: Bases to the left (relative to direction).
            right: Bases to the right (relative to direction).
            direction: 'both', 'upstream', or 'downstream'.

        Returns:
            A new IntervalIndex.
        """
        if right is None: right = left
        if len(self) == 0: return IntervalIndex()

        # Map string direction to int for Numba
        d_code = 0  # both
        if direction == 'upstream':
            d_code = 1
        elif direction == 'downstream':
            d_code = 2

        # Deterministic output size calculation
        factor = 2 if d_code == 0 else 1
        n = len(self)
        out_s = np.empty(n * factor, dtype=self._DTYPE)
        out_e = np.empty(n * factor, dtype=self._DTYPE)
        out_st = np.empty(n * factor, dtype=self._DTYPE)

        _flank_kernel(self.starts, self.ends, self.strands, left, right, d_code, out_s, out_e, out_st)

        # Filter empty intervals (length <= 0) resulting from boundary clipping
        mask = out_e > out_s
        if not np.all(mask):
            return IntervalIndex(out_s[mask], out_e[mask], out_st[mask])

        return IntervalIndex(out_s, out_e, out_st)

    def jaccard(self, other: 'IntervalIndex') -> float:
        """
        Calculates Jaccard Index (Intersection / Union) in bp.
        Useful for comparing gene predictions or annotations.

        Args:
            other: The other IntervalIndex.

        Returns:
            The Jaccard index (0.0 to 1.0).
        """
        union_len = self.merge().coverage() + other.merge().coverage()
        if union_len == 0: return 0.0

        intersect_len = self.intersect(other).coverage()
        # Union = A + B - Intersection
        real_union = union_len - intersect_len

        if real_union == 0: return 0.0
        return intersect_len / real_union

    def merge(self, tolerance: int = 0) -> 'IntervalIndex':
        """
        Merges overlapping or adjacent intervals.

        Args:
            tolerance: Maximum distance between intervals to merge.

        Returns:
            A new merged IntervalIndex.
        """
        if len(self) == 0: return self
        # Note: self.sort() must be guaranteed before calling this kernel
        # Since we sort on init, we are usually safe, but you might want to ensure it.

        out = _merge_kernel(self.starts, self.ends, self.strands, tolerance)

        return IntervalIndex(out[0], out[1], out[2])

    def tile(self, width: int, step: int = None) -> 'IntervalIndex':
        """
        Splits intervals into sliding windows of 'width'.
        Keeps windows strictly INSIDE the original intervals.

        Args:
            width: Window width.
            step: Step size (defaults to width).

        Returns:
            A new IntervalIndex of tiles.
        """
        if step is None: step = width
        if len(self) == 0: return IntervalIndex()

        # Pass 1: Count tiles per interval (Parallel)
        counts = _tile_count_kernel(self.starts, self.ends, width, step)
        total_tiles = counts.sum()
        if total_tiles == 0: return IntervalIndex()

        # Calculate Offsets
        offsets = np.zeros(len(counts), dtype=np.int32)
        np.cumsum(counts[:-1], out=offsets[1:])

        # Pass 2: Fill (Parallel)
        out_s = np.empty(total_tiles, dtype=self._DTYPE)
        out_e = np.empty(total_tiles, dtype=self._DTYPE)
        out_st = np.empty(total_tiles, dtype=self._DTYPE)

        _tile_fill_kernel(self.starts, self.ends, self.strands, width, step, offsets, out_s, out_e, out_st)

        return IntervalIndex(out_s, out_e, out_st)

    def coverage(self) -> int:
        """Returns total unique bases covered."""
        # Optimization: coverage is just sum(lengths) of the merged set
        merged = self.merge()
        return np.sum(merged.ends - merged.starts)

    def pad(self, upstream: int, downstream: int = None) -> 'IntervalIndex':
        """
        Expands intervals by 'upstream' and 'downstream' bp.
        Respects strand (upstream is 5', downstream is 3').

        Args:
            upstream: Bases to add upstream.
            downstream: Bases to add downstream (defaults to upstream).

        Returns:
            A new padded IntervalIndex.
        """
        if downstream is None: downstream = upstream
        if len(self) == 0: return self

        # Copy data to avoid mutating the current index
        s, e, st = self._starts.copy(), self._ends.copy(), self._strands.copy()

        # 1. Forward/Unstranded (+ or 0): Start - Up, End + Down
        mask_fwd = st >= 0
        s[mask_fwd] -= upstream
        e[mask_fwd] += downstream

        # 2. Reverse (-): Start - Down, End + Up
        mask_rev = st < 0
        s[mask_rev] -= downstream
        e[mask_rev] += upstream

        # 3. Clip negative values to 0
        np.maximum(s, 0, out=s)

        # 4. Re-sort required?
        # Padding can cause re-ordering (e.g. a small interval expanding past a large one)
        # It's safest to create a new index which enforces sort/validation
        return IntervalIndex(s, e, st)

    def complement(self, length: int) -> 'IntervalIndex':
        """
        Returns the 'gaps' in the index up to 'length'.
        Essential for finding intergenic regions.

        Args:
            length: The total length of the region (e.g., genome size).

        Returns:
            A new IntervalIndex representing the gaps.
        """
        if len(self) == 0:
            return IntervalIndex.from_intervals(Interval(0, length))

        # Merge first to remove overlaps
        merged = self.merge()
        s, e = merged.starts, merged.ends

        gap_starts = []
        gap_ends = []

        # Gap before first interval
        if s[0] > 0:
            gap_starts.append(0)
            gap_ends.append(s[0])

        # Gaps between intervals (Start[i+1] > End[i])
        # Since merged, we know Start[i+1] >= End[i], strictly > means gap.
        # Vectorized check:
        gap_mask = s[1:] > e[:-1]
        if np.any(gap_mask):
            gap_starts.extend(e[:-1][gap_mask])
            gap_ends.extend(s[1:][gap_mask])

        # Gap after last interval
        if e[-1] < length:
            gap_starts.append(e[-1])
            gap_ends.append(length)

        if not gap_starts:
            return IntervalIndex()

        # Gaps are unstranded (0)
        gap_starts = np.array(gap_starts, dtype=self._DTYPE)
        gap_ends = np.array(gap_ends, dtype=self._DTYPE)
        gap_strands = np.zeros(len(gap_starts), dtype=self._DTYPE)

        return IntervalIndex(gap_starts, gap_ends, gap_strands)

    def span(self) -> int:
        """Returns the distance from the very start to the very end."""
        if len(self) == 0: return 0
        # Since we are sorted, this is easy O(1)
        # Note: If self.sort() isn't guaranteed, use .min()/.max()
        return self._ends[-1] - self._starts[0]


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _query_kernel(starts, ends, original_indices, q_start, q_end, max_len):
    # 1. Binary search for the rightmost possible start
    # Any interval starting >= q_end cannot overlap [q_start, q_end)
    limit = np.searchsorted(starts, q_end, side='left')

    # 2. Backward scan with early stopping
    # We collect hits in a list (or pre-count). Since N is usually small for overlaps, list is fine.
    # But Numba lists can be slow. Let's do two-pass count-then-fill.
    
    count = 0
    # Optimization: We only need to look back as far as the longest interval
    min_start_check = q_start - max_len
    
    for i in range(limit - 1, -1, -1):
        if starts[i] < min_start_check: break
        if ends[i] > q_start: count += 1
            
    out = np.empty(count, dtype=np.int32)
    idx = 0
    for i in range(limit - 1, -1, -1):
        if starts[i] < min_start_check: break
        if ends[i] > q_start:
            out[idx] = original_indices[i]
            idx += 1
            
    # The backward scan produces indices in reverse sorted order. 
    # Usually we want sorted order.
    return out[::-1]


@jit(nopython=True, cache=True, nogil=True)
def _intersect_kernel(s_a, e_a, st_a, s_b, e_b, st_b, stranded):
    n_a = len(s_a)
    n_b = len(s_b)

    # Pre-allocate worst-case size.
    # Logic: In the worst case (perfect tiling), output count is n_a + n_b.
    max_size = n_a + n_b
    out_s = np.empty(max_size, dtype=s_a.dtype)
    out_e = np.empty(max_size, dtype=e_a.dtype)
    out_st = np.empty(max_size, dtype=st_a.dtype)

    out_idx = 0
    b_idx = 0

    for i in range(n_a):
        curr_s_a = s_a[i]
        curr_e_a = e_a[i]
        curr_st_a = st_a[i]

        # Optimization: Advance B pointer based on A's start
        while b_idx < n_b and e_b[b_idx] <= curr_s_a:
            b_idx += 1

        temp_b = b_idx
        while temp_b < n_b:
            curr_s_b = s_b[temp_b]

            # If B starts after A ends, we are done with this A
            if curr_s_b >= curr_e_a:
                break

            curr_e_b = e_b[temp_b]
            curr_st_b = st_b[temp_b]

            # Strand Check
            if not stranded or (curr_st_a == curr_st_b):
                new_s = max(curr_s_a, curr_s_b)
                new_e = min(curr_e_a, curr_e_b)

                # Write to array
                out_s[out_idx] = new_s
                out_e[out_idx] = new_e
                out_st[out_idx] = curr_st_a
                out_idx += 1

            temp_b += 1

    return out_s[:out_idx], out_e[:out_idx], out_st[:out_idx]


@jit(nopython=True, cache=True, nogil=True)
def _subtract_kernel(s_a, e_a, st_a, s_b, e_b, st_b, stranded):
    n_a, n_b = len(s_a), len(s_b)

    # Worst case: Every A is split into 2 pieces by a B sitting in the middle
    # Safe upper bound is 2 * n_a + n_b (though usually just n_a)
    max_size = (n_a * 2) + n_b
    out_s = np.empty(max_size, dtype=s_a.dtype)
    out_e = np.empty(max_size, dtype=e_a.dtype)
    out_st = np.empty(max_size, dtype=st_a.dtype)

    out_idx = 0
    b_idx = 0

    for i in range(n_a):
        curr_s = s_a[i]
        end_a = e_a[i]
        strand_a = st_a[i]

        # Advance B
        while b_idx < n_b and e_b[b_idx] <= curr_s:
            b_idx += 1

        temp_b = b_idx
        while temp_b < n_b and s_b[temp_b] < end_a:
            b_start = s_b[temp_b]
            b_end = e_b[temp_b]
            b_strand = st_b[temp_b]

            if stranded and (b_strand != strand_a):
                temp_b += 1
                continue

            if b_start > curr_s:
                # Keep the chunk before the overlap
                out_s[out_idx] = curr_s
                out_e[out_idx] = b_start
                out_st[out_idx] = strand_a
                out_idx += 1

            curr_s = max(curr_s, b_end)
            if curr_s >= end_a:
                break
            temp_b += 1

        if curr_s < end_a:
            out_s[out_idx] = curr_s
            out_e[out_idx] = end_a
            out_st[out_idx] = strand_a
            out_idx += 1

    return out_s[:out_idx], out_e[:out_idx], out_st[:out_idx]


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _tile_count_kernel(starts, ends, width, step):
    n = len(starts)
    counts = np.empty(n, dtype=np.int32)
    for i in prange(n):
        s, e = starts[i], ends[i]
        if e - s >= width:
            counts[i] = (e - s - width) // step + 1
        else:
            counts[i] = 0
    return counts

@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _tile_fill_kernel(starts, ends, strands, width, step, offsets, out_s, out_e, out_st):
    n = len(starts)
    for i in prange(n):
        s, e, st = starts[i], ends[i], strands[i]
        count = (e - s - width) // step + 1 if (e - s >= width) else 0
        
        start_idx = offsets[i]
        for j in range(count):
            tile_start = s + (j * step)
            out_s[start_idx + j] = tile_start
            out_e[start_idx + j] = tile_start + width
            out_st[start_idx + j] = st


@jit(nopython=True, cache=True, nogil=True)
def _merge_kernel(starts, ends, strands, tolerance):
    """
    Merges overlapping intervals in a single O(N) pass.
    Assumes inputs are sorted by start.
    """
    n = len(starts)
    if n == 0:
        return (np.empty(0, dtype=starts.dtype),
                np.empty(0, dtype=ends.dtype),
                np.empty(0, dtype=strands.dtype))

    # Worst case: No merges, output size = input size
    # We allocate max size and slice at the end (standard Numba pattern)
    temp_s = np.empty(n, dtype=starts.dtype)
    temp_e = np.empty(n, dtype=ends.dtype)
    temp_st = np.empty(n, dtype=strands.dtype)

    # Initialize with first interval
    curr_s = starts[0]
    curr_e = ends[0]
    curr_st = strands[0]
    out_idx = 0

    for i in range(1, n):
        s = starts[i]
        e = ends[i]
        st = strands[i]

        if s <= curr_e + tolerance:
            # MERGE
            curr_e = max(curr_e, e)
            # Strand Logic: If strands conflict, set to 0 (ambiguous)
            if curr_st != st:
                curr_st = 0
        else:
            # PUSH & RESET
            temp_s[out_idx] = curr_s
            temp_e[out_idx] = curr_e
            temp_st[out_idx] = curr_st
            out_idx += 1

            curr_s = s
            curr_e = e
            curr_st = st

    # Push final interval
    temp_s[out_idx] = curr_s
    temp_e[out_idx] = curr_e
    temp_st[out_idx] = curr_st
    out_idx += 1

    return temp_s[:out_idx], temp_e[:out_idx], temp_st[:out_idx]


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _flank_kernel(starts, ends, strands, left, right, direction, out_s, out_e, out_st):
    """
    Generates flanking regions in parallel.
    direction: 0=both, 1=upstream, 2=downstream
    """
    n = len(starts)
    for i in prange(n):
        s, e, st = starts[i], ends[i], strands[i]

        # UPSTREAM FLANK
        if direction == 0 or direction == 1:
            idx = 2 * i if direction == 0 else i
            if st >= 0:  # Forward (+): [Start - Left, Start]
                new_s = max(0, s - left)
                new_e = s
            else:  # Reverse (-): [End, End + Left]
                new_s = e
                new_e = e + left
            
            out_s[idx], out_e[idx], out_st[idx] = new_s, new_e, st

        # DOWNSTREAM FLANK
        if direction == 0 or direction == 2:
            idx = (2 * i + 1) if direction == 0 else i
            if st >= 0:  # Forward (+): [End, End + Right]
                new_s = e
                new_e = e + right
            else:  # Reverse (-): [Start - Right, Start]
                new_s = max(0, s - right)
                new_e = s
            
            out_s[idx], out_e[idx], out_st[idx] = new_s, new_e, st
