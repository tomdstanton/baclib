"""Genomic interval representation with strand and context, plus batched interval operations."""
from typing import Union, Any, Match, Iterable, ClassVar
from enum import IntEnum, auto

import numpy as np

from baclib.lib.resources import RESOURCES, jit
from baclib.containers import Batch

if RESOURCES.has_module('numba'):
    from numba import prange
else:
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Context(IntEnum):
    """Spatial relationship between two genomic intervals."""
    UPSTREAM = auto()
    DOWNSTREAM = auto()
    INSIDE = auto()
    OVERLAPPING = auto()
    OVERLAPPING_START = auto()
    OVERLAPPING_END = auto()


class Strand(IntEnum):
    """
    Enumeration for genomic strands.
    """
    FORWARD = 1
    REVERSE = -1
    UNSTRANDED = 0
    _STR_CACHE: ClassVar[dict]
    _BYTES_CACHE: ClassVar[dict]
    _FROM_BYTES_CACHE: ClassVar[dict]

    def __str__(self): return self._STR_CACHE[self]
    @property
    def bytes(self) -> bytes: return self._BYTES_CACHE[self]

    @classmethod
    def from_bytes(cls, b: bytes) -> 'Strand':
        return cls._FROM_BYTES_CACHE.get(b, cls.UNSTRANDED)

    @classmethod
    def from_symbol(cls, s: Any) -> 'Strand':
        if s is None: return cls.UNSTRANDED
        if isinstance(s, cls): return s
        if isinstance(s, int):
            try: return cls(s)
            except ValueError: return cls.UNSTRANDED
        if isinstance(s, bytes): return cls.from_bytes(s)
        if isinstance(s, str): return cls.from_bytes(s.encode('ascii'))
        return cls.UNSTRANDED

    @classmethod
    def _init_caches(cls):
        cls._STR_CACHE = {cls.FORWARD: '+', cls.REVERSE: '-', cls.UNSTRANDED: '.'}
        cls._BYTES_CACHE = {cls.FORWARD: b'+', cls.REVERSE: b'-', cls.UNSTRANDED: b'.'}
        cls._FROM_BYTES_CACHE = { b'+': cls.FORWARD, b'-': cls.REVERSE, b'.': cls.UNSTRANDED}


class Interval:
    """
    Immutable genomic interval. Safe for hashing and use in sets/dicts.

    Attributes:
        start: The start position (0-based, inclusive).
        end: The end position (0-based, exclusive).
        strand: The strand (FORWARD, REVERSE, or UNSTRANDED).
    """
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
    def __repr__(self): return f"{self._start}:{self._end}({self._strand})"
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
        """Returns the convex hull (union extent) as a NEW Interval.

        Args:
            other: The other interval.

        Returns:
            A new ``Interval`` spanning from the minimum start to maximum end.
        """
        other = Interval.from_item(other)
        new_strand = self._strand if self._strand == other.strand else 0
        return Interval(min(self._start, other.start), max(self._end, other.end), new_strand)

    def __radd__(self, other: Union[slice, int, 'Interval']) -> 'Interval':
        return self.__add__(other)

    def intersection(self, other: Union[slice, int, 'Interval']) -> 'Interval':
        """Returns the intersection of this interval and another as a NEW Interval.

        Args:
            other: The other interval.

        Returns:
            A new ``Interval`` representing the overlapping region.
        """
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

    def reverse_complement(self, length: int = None) -> 'Interval':
        """
        Returns the interval coordinates on the reverse complement strand.

        Args:
            length: The length of the parent sequence.

        Returns:
            A new Interval on the opposite strand.
        """
        if length is None: length = self._end  # - self._start
        return Interval(length - self._end, length - self._start, self._strand * -1)

    def relate(self, other: Union[slice, int, 'Interval']) -> Context:
        """
        Determines the spatial relationship of another interval relative to this one,
        respecting the strand of this interval.

        Args:
            other: The other interval to compare.

        Returns:
            A Context enum member.
        """
        other = Interval.from_item(other)
        
        # Determine absolute orientation
        if other.end <= self._start:
            return Context.UPSTREAM if self._strand >= 0 else Context.DOWNSTREAM
        if other.start >= self._end:
            return Context.DOWNSTREAM if self._strand >= 0 else Context.UPSTREAM
        
        # Overlap cases
        if other.start >= self._start and other.end <= self._end: return Context.INSIDE
        
        if other.start < self._start:
            if other.end > self._end: return Context.OVERLAPPING
            return Context.OVERLAPPING_START if self._strand >= 0 else Context.OVERLAPPING_END
        
        return Context.OVERLAPPING_END if self._strand >= 0 else Context.OVERLAPPING_START

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
    def from_match(cls, item: Match, strand: int = Strand.UNSTRANDED) -> 'Interval': 
        return Interval(item.start(), item.end(), strand)
    
    @classmethod
    def from_int(cls, item: int, strand: int = Strand.UNSTRANDED, length: int = None) -> 'Interval':
        """Creates an interval from a single integer index.

        Args:
            item: The integer index (can be negative).
            strand: The strand.
            length: Sequence length (required for negative indices).

        Returns:
            A new ``Interval`` of length 1.
        """
        if item < 0 and length is not None: item += length
        return cls(item, item + 1, strand)

    @classmethod
    def from_slice(cls, item: slice, strand: int = Strand.UNSTRANDED, length: int = None) -> 'Interval':
        """Creates an interval from a slice object.

        Args:
            item: The slice object.
            strand: The strand.
            length: Sequence length (required for slices with ``None`` stop).

        Returns:
            A new ``Interval``.
        """
        start, stop, step = item.start, item.stop, item.step
        if start is None: start = 0
        if stop is None and length is not None: stop = length
        if stop is None: raise ValueError("Cannot create Interval from slice with None stop without 'length'")
        if step == -1: return cls(stop + 1, start + 1, strand)
        return cls(start, stop, strand)

    @classmethod
    def from_item(cls, item: Union[slice, int, 'Interval', Match], strand: int = Strand.UNSTRANDED, length: int = None) -> 'Interval':
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
        if isinstance(item, cls): return item
        if interval := getattr(item, 'interval', None): return interval
        if isinstance(item, Match): return cls.from_match(item, strand)
        if isinstance(item, int): return cls.from_int(item, strand, length)
        if isinstance(item, slice): return cls.from_slice(item, strand, length)
        raise TypeError(f"Cannot coerce {type(item)} to Interval")


class IntervalBatch(Batch):
    """
    High-performance batch of genomic intervals, powered by NumPy.

    This class provides efficient querying, intersection, and manipulation of
    large sets of genomic intervals.
    """
    __slots__ = ('_starts', '_ends', '_strands', '_original_indices', '_max_len')
    _DTYPE = np.int32  # int32 allows up to 2.14 Billion bp (fits all bacterial genomes)

    def __init__(self, starts: np.ndarray = None, ends: np.ndarray = None, strands: np.ndarray = None,
                 original_indices: np.ndarray = None, sort: bool = True):
        """
        Initializes an IntervalBatch.

        Args:
            starts: Array of start positions.
            ends: Array of end positions.
            strands: Array of strands.
            original_indices: Array of original indices (for tracking after sort).
            sort: Whether to sort the intervals (default: True).
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

        self._original_indices = original_indices
        if sort: self.sort()

    def sort(self):
        """Sorts the intervals by start position, then end position."""
        # Sort both data and the index tracker
        if len(self._starts) > 1:
            # Optimization: Check if already sorted to avoid expensive lexsort
            if _is_sorted_kernel(self._starts, self._ends): return

            # Lexsort: Primary key is last in the tuple (starts), Secondary is ends
            order = np.lexsort((self._ends, self._starts))

            self._starts = self._starts[order]
            self._ends = self._ends[order]
            self._strands = self._strands[order]
            
            if self._original_indices is not None:
                self._original_indices = self._original_indices[order]
            else:
                # Create the mapping only now that we know we are scrambling the order
                self._original_indices = order.astype(np.int32)


    @classmethod
    def zeros(cls, n: int) -> 'IntervalBatch':
        """Creates a batch of *n* 0-length intervals at position 0.

        Args:
            n: Number of intervals.

        Returns:
            A new ``IntervalBatch``.
        """
        return cls(
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            np.ones(n, dtype=np.int32) # Default strand? Or 0? Or 1 (FORWARD)? Strand(1) is standard.
        )

    @classmethod
    def random(cls, n: int, rng: np.random.Generator = None, length: int = None, min_len: int = 1, max_len: int = 1000,
               min_start: int = 0, max_start: int = 1_000_000) -> 'IntervalBatch':
        """
        Creates a batch of n random intervals.

        Args:
            n: Number of intervals to generate.
            rng: Random number generator (optional).
            length: Fixed length for all intervals (optional).
            min_len: Minimum length (default: 1).
            max_len: Maximum length (default: 1000).
            min_start: Minimum start position (default: 0).
            max_start: Maximum start position (default: 1,000,000).

        Returns:
            An IntervalBatch.
        """
        if rng is None: rng = RESOURCES.rng
        if n <= 0: return cls.empty()

        # 1. Generate Lengths
        if length is not None:
             lengths = np.full(n, length, dtype=np.int32)
        else:
             lengths = rng.integers(min_len, max_len, size=n, dtype=np.int32)

        # 2. Generate Starts
        # We treat max_start as the upper bound for the start coordinate (exclusive).
        if max_start <= min_start:
             raise ValueError(f"max_start ({max_start}) must be > min_start ({min_start})")
             
        starts = rng.integers(min_start, max_start, size=n, dtype=np.int32)
        
        # 3. Generate Ends
        ends = starts + lengths
        
        # 4. Generate Strands (-1, 0, 1)
        strands = rng.choice([-1, 0, 1], size=n).astype(np.int32)
        
        return cls(starts, ends, strands, sort=True)

    @classmethod
    def empty(cls) -> 'IntervalBatch':
        """Creates an empty IntervalBatch.

        Returns:
            An empty ``IntervalBatch``.
        """
        return cls.zeros(0)

    @classmethod
    def build(cls, *intervals: Union[Interval, Iterable[Interval]]) -> 'IntervalBatch':
        """Creates an IntervalBatch from an iterable of Interval objects (or varargs).

        Args:
            *intervals: Iterable of intervals or individual ``Interval`` arguments.

        Returns:
            A new ``IntervalBatch``.
        """
        if not intervals: return cls.empty()
        # Handle single iterable argument
        if len(intervals) == 1 and isinstance(intervals[0], Iterable) and not isinstance(intervals[0], Interval):
            intervals = intervals[0]
            
        # Vectorized construction using list comprehension is generally faster than explicit loops
        data = [(x._start, x._end, x._strand) for x in intervals]
        arr = np.array(data, dtype=cls._DTYPE)
        if len(arr) == 0: return cls.empty()
        return cls(arr[:, 0], arr[:, 1], arr[:, 2])

    @classmethod
    def from_features(cls, *features) -> 'IntervalBatch':
        """
        Creates an IntervalBatch from a list of Feature objects.

        Args:
            *features: Variable number of Feature objects (must have .interval attribute).

        Returns:
            An IntervalBatch.
        """
        if not features: return cls.empty()
        
        # Handle single argument (list, batch, iterator)
        if len(features) == 1:
            obj = features[0]
            
            # Fast Path: Container with intervals property (e.g. FeatureList, Record, FeatureBatch)
            if batch := getattr(obj, 'intervals', None):
                if isinstance(batch, cls): return batch
            
            # 3. Unwrap Iterable if it's not a single Feature (Features are iterable)
            if isinstance(obj, Iterable) and not hasattr(obj, 'interval'):
                features = list(obj)

        # Fallback: Iteration (Optimized for list of objects)
        # We assume objects have .interval attribute (duck typing) for speed
        n = len(features)
        starts = np.empty(n, dtype=cls._DTYPE)
        ends = np.empty(n, dtype=cls._DTYPE)
        strands = np.empty(n, dtype=cls._DTYPE)
        for i, f in enumerate(features):
            iv = f.interval
            starts[i] = iv._start
            ends[i] = iv._end
            strands[i] = iv._strand
        return cls(starts, ends, strands)

    @classmethod
    def from_items(cls, *items: Union[slice, int, 'Interval', Match]) -> 'IntervalBatch':
        """
        Creates an IntervalBatch from various items.

        Args:
            *items: Items to convert to intervals.

        Returns:
            An IntervalBatch.
        """
        if not items: return cls.empty()
        arr = np.array([tuple(Interval.from_item(i)) for i in items], dtype=cls._DTYPE)
        return cls(arr[:, 0], arr[:, 1], arr[:, 2])

    @classmethod
    def concat(cls, batches: Iterable['IntervalBatch']) -> 'IntervalBatch':
        """Concatenates multiple IntervalBatches.

        Args:
            batches: Iterable of ``IntervalBatch`` objects.

        Returns:
            A new concatenated ``IntervalBatch``.
        """
        batches = list(batches)
        if not batches: return cls.empty()
        
        starts = np.concatenate([b._starts for b in batches])
        ends = np.concatenate([b._ends for b in batches])
        strands = np.concatenate([b._strands for b in batches])
        
        return cls(starts, ends, strands, sort=True)



    def __repr__(self): return f"<IntervalBatch: {len(self)} intervals>"

    def __len__(self): return len(self._starts)
    
    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            return Interval(self._starts[item], self._ends[item], self._strands[item])
        elif isinstance(item, (slice, np.ndarray, list)):
            # Slicing or boolean masking a sorted batch preserves sort order
            is_slice = isinstance(item, slice)
            is_mask = isinstance(item, np.ndarray) and item.dtype == bool
            return IntervalBatch(self._starts[item], self._ends[item], self._strands[item], 
                                 sort=not (is_slice or is_mask))
        raise TypeError(f"Invalid index type: {type(item)}")

    def filter(self, mask: Union[np.ndarray, Iterable, slice]) -> 'IntervalBatch':
        """
        Filters the batch using a boolean mask, integer indices, or slice.
        Always returns an IntervalBatch.

        Args:
            mask: Boolean array, integer indices, slice, or iterable.

        Returns:
            A new IntervalBatch.
        """
        if isinstance(mask, (slice, int, np.integer)):
            if isinstance(mask, (int, np.integer)):
                mask = [mask]
            return self[mask]

        # Ensure mask is a numpy array (handles lists of bools correctly as masks)
        return self[np.asarray(mask)]

    def __iter__(self):
        for i in range(len(self)):
            yield Interval(self._starts[i], self._ends[i], self._strands[i])

    def copy(self) -> 'IntervalBatch':
        """Returns a deep copy of the IntervalBatch."""
        return IntervalBatch(
            self._starts.copy(), self._ends.copy(), self._strands.copy(),
            self._original_indices.copy() if self._original_indices is not None else None,
            sort=False
        )

    @property
    def starts(self): return self._starts
    @property
    def ends(self): return self._ends
    @property
    def strands(self): return self._strands
    
    @property
    def component(self): return Interval

    
    @property
    def nbytes(self) -> int:
        return self._starts.nbytes + self._ends.nbytes + self._strands.nbytes + (self._original_indices.nbytes if self._original_indices is not None else 0)

    @property
    def centers(self) -> np.ndarray:
        """Returns the center points of the intervals (float array)."""
        return (self._starts + self._ends) / 2

    @property
    def lengths(self) -> np.ndarray:
        """Returns the lengths of the intervals (int array)."""
        return self._ends - self._starts

    def query(self, start: int, end: int) -> np.ndarray:
        """Returns indices of intervals overlapping [start, end)."""
        if len(self) == 0: return np.empty(0, dtype=np.int32)
        
        if self._original_indices is not None:
            return _query_kernel(self.starts, self.ends, self._original_indices, start, end, self._max_len)
        
        return _query_kernel_identity(self.starts, self.ends,
                             start, end, self._max_len)

    def intersect(self, other: 'IntervalBatch', stranded: bool = False) -> 'IntervalBatch':
        """Computes the intersection with another IntervalBatch.

        Args:
            other: The other IntervalBatch.
            stranded: If ``True``, only intersects intervals on the same strand.

        Returns:
            A new ``IntervalBatch`` representing the intersection.
        """
        if len(self) == 0 or len(other) == 0: return IntervalBatch.empty()
        # Call Numba Kernel
        out = _intersect_kernel(self.starts, self.ends, self.strands,
                                other.starts, other.ends, other.strands, stranded)
        # Kernel returns tuple of arrays
        if len(out[0]) == 0: return IntervalBatch.empty()
        return IntervalBatch(out[0], out[1], out[2], sort=False)

    def subtract(self, other: 'IntervalBatch', stranded: bool = False) -> 'IntervalBatch':
        """Subtracts regions in ``other`` from this index.

        Args:
            other: The IntervalBatch to subtract.
            stranded: If ``True``, only subtracts intervals on the same strand.

        Returns:
            A new ``IntervalBatch`` with regions removed.
        """
        if len(other) == 0: return self.copy()
        # Merge B to simplify subtraction
        b_merged = other.merge()

        out = _subtract_kernel(self.starts, self.ends, self.strands,
                               b_merged.starts, b_merged.ends, b_merged.strands, stranded)
        if len(out[0]) == 0: return IntervalBatch.empty()
        return IntervalBatch(out[0], out[1], out[2], sort=False)

    def promoters(self, upstream: int = 100, downstream: int = 0) -> 'IntervalBatch':
        """
        Extracts promoter regions relative to strand.
        Forward: [Start - Up, Start + Down]
        Reverse: [End - Down, End + Up]

        Args:
            upstream: Bases upstream of the start.
            downstream: Bases downstream of the start.

        Returns:
            A new IntervalBatch of promoters.
        """
        if len(self) == 0: return IntervalBatch.empty()

        s = self._starts
        e = self._ends
        st = self._strands

        new_s = np.empty_like(s)
        new_e = np.empty_like(e)

        # Forward (+ or 0)
        mask_fwd = st >= 0
        if np.any(mask_fwd):
            fwd_s = s[mask_fwd]
            new_s[mask_fwd] = fwd_s - upstream
            new_e[mask_fwd] = fwd_s + downstream

        # Reverse (-)
        mask_rev = ~mask_fwd
        if np.any(mask_rev):
            rev_e = e[mask_rev]
            new_s[mask_rev] = rev_e - downstream
            new_e[mask_rev] = rev_e + upstream

        np.maximum(new_s, 0, out=new_s)
        return IntervalBatch(new_s, new_e, st.copy(), sort=True)

    def flank(self, left: int, right: int = None, direction: str = 'both') -> 'IntervalBatch':
        """
        Generates flanking regions.

        Args:
            left: Bases to the left (relative to direction).
            right: Bases to the right (relative to direction).
            direction: 'both', 'upstream', or 'downstream'.

        Returns:
            A new IntervalBatch.
        """
        if right is None: right = left
        if len(self) == 0: return IntervalBatch.empty()

        # Map string direction to int for Numba
        d_code = 0  # both
        if direction == 'upstream': d_code = 1
        elif direction == 'downstream': d_code = 2

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
            return IntervalBatch(out_s[mask], out_e[mask], out_st[mask], sort=True)

        return IntervalBatch(out_s, out_e, out_st, sort=True)

    def jaccard(self, other: 'IntervalBatch') -> float:
        """
        Calculates Jaccard Index (Intersection / Union) in bp.
        Useful for comparing gene predictions or annotations.

        Args:
            other: The other IntervalBatch.

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

    def relate(self, other: Union['Interval', 'IntervalBatch']) -> np.ndarray:
        """
        Vectorized determination of spatial relationships.

        Args:
            other: An Interval or IntervalBatch.

        Returns:
            np.ndarray: Array of Context values (int32).
        """
        if isinstance(other, Interval):
            s2 = np.full(len(self), other.start, dtype=self._DTYPE)
            e2 = np.full(len(self), other.end, dtype=self._DTYPE)
        elif isinstance(other, IntervalBatch):
            if len(self) != len(other):
                raise ValueError(f"Batch length mismatch: {len(self)} vs {len(other)}")
            s2 = other.starts
            e2 = other.ends
        else:
            try:
                iv = Interval.from_item(other)
                s2 = np.full(len(self), iv.start, dtype=self._DTYPE)
                e2 = np.full(len(self), iv.end, dtype=self._DTYPE)
            except TypeError:
                raise TypeError(f"Cannot relate IntervalBatch with {type(other)}")

        return _relate_kernel(self.starts, self.ends, self.strands, s2, e2)

    def merge(self, tolerance: int = 0) -> 'IntervalBatch':
        """
        Merges overlapping or adjacent intervals.

        Args:
            tolerance: Maximum distance between intervals to merge.

        Returns:
            A new merged IntervalBatch.
        """
        if len(self) == 0: return self
        # Note: self.sort() must be guaranteed before calling this kernel
        # Since we sort on init, we are usually safe, but you might want to ensure it.

        out = _merge_kernel(self.starts, self.ends, self.strands, tolerance)

        return IntervalBatch(out[0], out[1], out[2], sort=False)

    def tile(self, width: int, step: int = None) -> 'IntervalBatch':
        """
        Splits intervals into sliding windows of 'width'.
        Keeps windows strictly INSIDE the original intervals.

        Args:
            width: Window width.
            step: Step size (defaults to width).

        Returns:
            A new IntervalBatch of tiles.
        """
        if step is None: step = width
        if len(self) == 0: return IntervalBatch.empty()

        # Pass 1: Count tiles per interval (Parallel)
        counts = _tile_count_kernel(self.starts, self.ends, width, step)
        total_tiles = counts.sum()
        if total_tiles == 0: return IntervalBatch.empty()

        # Calculate Offsets
        offsets = np.zeros(len(counts), dtype=np.int32)
        np.cumsum(counts[:-1], out=offsets[1:])

        # Pass 2: Fill (Parallel)
        out_s = np.empty(total_tiles, dtype=self._DTYPE)
        out_e = np.empty(total_tiles, dtype=self._DTYPE)
        out_st = np.empty(total_tiles, dtype=self._DTYPE)

        _tile_fill_kernel(self.starts, self.ends, self.strands, width, step, offsets, out_s, out_e, out_st)

        return IntervalBatch(out_s, out_e, out_st, sort=False)

    def coverage(self) -> int:
        """Returns total unique bases covered."""
        return _coverage_kernel(self.starts, self.ends)

    def pad(self, upstream: int, downstream: int = None) -> 'IntervalBatch':
        """
        Expands intervals by 'upstream' and 'downstream' bp.
        Respects strand (upstream is 5', downstream is 3').

        Args:
            upstream: Bases to add upstream.
            downstream: Bases to add downstream (defaults to upstream).

        Returns:
            A new padded IntervalBatch.
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
        return IntervalBatch(s, e, st, sort=True)

    def reverse_complement(self, length: Union[int, np.ndarray]) -> 'IntervalBatch':
        """
        Returns a new IntervalBatch on the reverse complement strand.

        Args:
            length: Length of the parent sequence(s). Can be a scalar or an array.

        Returns:
            A new IntervalBatch.
        """
        if len(self) == 0: return IntervalBatch.empty()
        return IntervalBatch(length - self._ends, length - self._starts, self._strands * -1, sort=True)

    def complement(self, length: int = None) -> 'IntervalBatch':
        """
        Returns the 'gaps' in the index up to 'length'.
        Essential for finding intergenic regions.

        Args:
            length: The total length of the region (e.g., genome size).
                    If None, defaults to the maximum end position in the batch.

        Returns:
            A new IntervalBatch representing the gaps.
        """
        if len(self) == 0:
            if length is None: return IntervalBatch.empty()
            return IntervalBatch.build(Interval(0, length))

        # Merge first to remove overlaps and sort
        merged = self.merge()
        
        if length is None:
            length = merged.ends[-1]

        out = _complement_kernel(merged.starts, merged.ends, length)
        return IntervalBatch(out[0], out[1], out[2], sort=False)

    def span(self) -> int:
        """Returns the distance from the very start to the very end."""
        if len(self) == 0: return 0
        # Starts are sorted, but ends are not necessarily sorted by the last element
        return np.max(self._ends) - self._starts[0]

    def cover_linear(self, length: int = None, min_coverage: float = 0.0, max_overlap: float = 0.1) -> np.ndarray:
        """
        Identifies a linear chain of intervals that covers the range [0, length] with minimal overlap.
        Useful for scaffolding or filtering overlapping features.

        Args:
            length: The total length of the region (e.g. contig length). Required if min_coverage > 0.
            min_coverage: Minimum fraction of 'length' covered by the chain.
            max_overlap: Maximum allowed fractional overlap between adjacent intervals.

        Returns:
            Indices of the selected intervals (relative to the original unsorted input).
        """
        kept_sorted_indices = _cover_linear_kernel(self.starts, self.ends, length or 0, min_coverage, max_overlap)
        if self._original_indices is not None: return self._original_indices[kept_sorted_indices]
        return kept_sorted_indices


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
def _query_kernel_identity(starts, ends, q_start, q_end, max_len):
    """Optimized query kernel for when original_indices is identity (0..N)."""
    limit = np.searchsorted(starts, q_end, side='left')
    count = 0
    min_start_check = q_start - max_len
    
    for i in range(limit - 1, -1, -1):
        if starts[i] < min_start_check: break
        if ends[i] > q_start: count += 1
            
    out = np.empty(count, dtype=np.int32)
    idx = 0
    for i in range(limit - 1, -1, -1):
        if starts[i] < min_start_check: break
        if ends[i] > q_start:
            out[idx] = i  # Direct index
            idx += 1
    return out[::-1]


@jit(nopython=True, cache=True, nogil=True)
def _intersect_kernel(s_a, e_a, st_a, s_b, e_b, st_b, stranded):
    n_a = len(s_a)
    n_b = len(s_b)

    # Pass 1: Count intersections
    # We cannot safely predict max_size (worst case is N*M), so we count first.
    count = 0
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
                count += 1

            temp_b += 1

    # Pass 2: Fill
    out_s = np.empty(count, dtype=s_a.dtype)
    out_e = np.empty(count, dtype=e_a.dtype)
    out_st = np.empty(count, dtype=st_a.dtype)
    
    out_idx = 0
    b_idx = 0
    for i in range(n_a):
        curr_s_a = s_a[i]
        curr_e_a = e_a[i]
        curr_st_a = st_a[i]

        while b_idx < n_b and e_b[b_idx] <= curr_s_a:
            b_idx += 1

        temp_b = b_idx
        while temp_b < n_b:
            curr_s_b = s_b[temp_b]
            if curr_s_b >= curr_e_a: break
            curr_e_b = e_b[temp_b]
            curr_st_b = st_b[temp_b]

            if not stranded or (curr_st_a == curr_st_b):
                out_s[out_idx] = max(curr_s_a, curr_s_b)
                out_e[out_idx] = min(curr_e_a, curr_e_b)
                out_st[out_idx] = curr_st_a
                out_idx += 1
            temp_b += 1

    return out_s, out_e, out_st


@jit(nopython=True, cache=True, nogil=True)
def _subtract_kernel(s_a, e_a, st_a, s_b, e_b, st_b, stranded):
    n_a, n_b = len(s_a), len(s_b)

    # Pass 1: Count
    count = 0
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
                count += 1

            curr_s = max(curr_s, b_end)
            if curr_s >= end_a:
                break
            temp_b += 1

        if curr_s < end_a:
            count += 1

    # Pass 2: Fill
    out_s = np.empty(count, dtype=s_a.dtype)
    out_e = np.empty(count, dtype=e_a.dtype)
    out_st = np.empty(count, dtype=st_a.dtype)
    
    out_idx = 0
    b_idx = 0
    for i in range(n_a):
        curr_s = s_a[i]
        end_a = e_a[i]
        strand_a = st_a[i]

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
            
    return out_s, out_e, out_st


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


@jit(nopython=True, cache=True, nogil=True)
def _coverage_kernel(starts, ends):
    """Calculates total coverage of sorted intervals without allocating merged arrays."""
    n = len(starts)
    if n == 0: return 0
    
    total_len = 0
    curr_start = starts[0]
    curr_end = ends[0]
    
    for i in range(1, n):
        s, e = starts[i], ends[i]
        if s < curr_end:
            if e > curr_end: curr_end = e
        else:
            total_len += (curr_end - curr_start)
            curr_start = s
            curr_end = e
            
    total_len += (curr_end - curr_start)
    return total_len


@jit(nopython=True, cache=True, nogil=True)
def _cover_linear_kernel(starts, ends, length, min_cov, max_overlap):
    """
    Filters a sorted list of intervals to form a consistent linear chain.
    Returns indices into the sorted arrays.
    """
    n = len(starts)
    if n == 0: return np.empty(0, dtype=np.int32)

    # 1. Greedy Filter (Keep first, then append if non-overlapping)
    kept = np.empty(n, dtype=np.int32)
    k = 0

    last_end = -1
    total_cov = 0

    for i in range(n):
        s, e = starts[i], ends[i]
        l = e - s

        # Check overlap with previous
        if k > 0:
            overlap = last_end - s
            if overlap > 0:
                # If overlap is too large relative to the current alignment, skip
                if (overlap / l) > max_overlap: continue
                # Adjust coverage calculation for overlap
                total_cov -= overlap

        kept[k] = i
        k += 1
        last_end = e
        total_cov += l

    # 2. Check Total Coverage
    if length > 0:
        frac = total_cov / length
        if frac < min_cov: return np.empty(0, dtype=np.int32)

    return kept[:k]


@jit(nopython=True, cache=True, nogil=True)
def _relate_kernel(s1, e1, st1, s2, e2):
    n = len(s1)
    out = np.empty(n, dtype=np.int32)
    
    # Context Enum Values (auto starts at 1)
    UPSTREAM = 1
    DOWNSTREAM = 2
    INSIDE = 3
    OVERLAPPING = 4
    OVERLAPPING_START = 5
    OVERLAPPING_END = 6
    
    for i in range(n):
        _s1 = s1[i]
        _e1 = e1[i]
        _st1 = st1[i]
        _s2 = s2[i]
        _e2 = e2[i]
        
        if _e2 <= _s1:
            out[i] = UPSTREAM if _st1 >= 0 else DOWNSTREAM
        elif _s2 >= _e1:
            out[i] = DOWNSTREAM if _st1 >= 0 else UPSTREAM
        else:
            if _s2 >= _s1 and _e2 <= _e1:
                out[i] = INSIDE
            elif _s2 < _s1:
                if _e2 > _e1: out[i] = OVERLAPPING
                else: out[i] = OVERLAPPING_START if _st1 >= 0 else OVERLAPPING_END
            else:
                out[i] = OVERLAPPING_END if _st1 >= 0 else OVERLAPPING_START
                
    return out


@jit(nopython=True, cache=True, nogil=True)
def _complement_kernel(starts, ends, length):
    n = len(starts)
    
    # Pass 1: Count
    count = 0
    curr = 0
    for i in range(n):
        s = starts[i]
        e = ends[i]
        if s > curr:
            if curr >= length: break
            count += 1
        curr = max(curr, e)
        if curr >= length: break
        
    if curr < length:
        count += 1
        
    # Pass 2: Fill
    out_s = np.empty(count, dtype=starts.dtype)
    out_e = np.empty(count, dtype=ends.dtype)
    out_st = np.zeros(count, dtype=starts.dtype)
    
    idx = 0
    curr = 0
    for i in range(n):
        s = starts[i]
        e = ends[i]
        if s > curr:
            if curr >= length: break
            out_s[idx] = curr
            out_e[idx] = min(s, length)
            idx += 1
        curr = max(curr, e)
        if curr >= length: break
        
    if curr < length:
        out_s[idx] = curr
        out_e[idx] = length
        idx += 1
        
    return out_s[:idx], out_e[:idx], out_st[:idx]


@jit(nopython=True, cache=True, nogil=True)
def _is_sorted_kernel(starts, ends):
    """Checks if the arrays are already sorted by start, then end."""
    n = len(starts)
    for i in range(n - 1):
        if starts[i] > starts[i+1]: return False
        if starts[i] == starts[i+1]:
            if ends[i] > ends[i+1]: return False
    return True


# Cache initialisations ------------------------------------------------------------------------------------------------
Strand._init_caches()
