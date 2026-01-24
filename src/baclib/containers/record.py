from copy import deepcopy
from operator import attrgetter
from typing import Generator, Union, Iterator, MutableSequence, Any, Iterable
from uuid import uuid4

import numpy as np

from baclib.core.seq import Seq, SeqBatch
from baclib.core.interval import Interval, IntervalIndex
from baclib.utils.resources import RESOURCES


# Classes --------------------------------------------------------------------------------------------------------------
class QualifierList(MutableSequence):
    """
    A list-like container for (key, value) tuples that also supports dictionary-style access.
    Maintains insertion order and allows duplicate keys.
    """
    __slots__ = ('_data',)

    def __init__(self, items: Iterable[tuple[bytes, Any]] = None):
        self._data = list(items) if items else []

    def __getitem__(self, item):
        # List-style access
        if isinstance(item, (int, slice)): return self._data[item]
        # Dict-style access (First match)
        for k, v in self._data:
            if k == item: return v
        return None

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            self._data[key] = value
            return

        # Dict-style set: Replace first occurrence or append
        for i, (k, v) in enumerate(self._data):
            if k == key:
                self._data[i] = (key, value)
                return
        self._data.append((key, value))

    def __delitem__(self, index): del self._data[index]
    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)
    def __repr__(self): return repr(self._data)
    def insert(self, index, value): self._data.insert(index, value)

    def __eq__(self, other):
        if isinstance(other, QualifierList): return self._data == other._data
        if isinstance(other, list): return self._data == other
        return False

    def get(self, key, default=None):
        for k, v in self._data:
            if k == key: return v
        return default

    def get_all(self, key: bytes) -> list[Any]:
        return [v for k, v in self._data if k == key]

    def add(self, key: bytes, value: Any):
        self._data.append((key, value))

    def to_dict(self) -> dict[bytes, Any]:
        """Converts the list to a standard dictionary (lossy for duplicates)."""
        return {k: v for k, v in self._data}

class Feature:
    """
    Represents a genomic feature, such as a gene or CDS.

    Attributes:
        interval (Interval): The genomic interval of the feature.
        kind (bytes): The type of feature (e.g., 'CDS', 'gene', 'misc_feature').
        qualifiers (list[Qualifier]): A list of qualifiers annotating the feature.
    """
    __slots__ = ('interval', 'kind', 'qualifiers')
    def __init__(self, interval: 'Interval', kind: bytes = b'misc_feature', qualifiers: Iterable[tuple[bytes, Any]] = None):
        self.interval = interval
        self.kind = kind
        self.qualifiers = QualifierList(qualifiers)

    def __len__(self) -> int: return len(self.interval)
    def __iter__(self): return self.interval.__iter__()
    def __contains__(self, item) -> bool: return self.interval.__contains__(item)
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.interval == other.interval and self.kind == other.kind and
                    self.qualifiers == other.qualifiers)
        return False
    
    def __repr__(self):
        return f"Feature({self.kind.decode('ascii', 'ignore')}, {self.interval})"

    # Delegate dict-like access to qualifiers, but ensure we use .get() for lookups
    # to avoid ambiguity with integer indices if Feature were to support them.
    def __getitem__(self, item): return self.qualifiers.get(item)
    def __setitem__(self, key: bytes, value: Any): self.qualifiers[key] = value

    def overlap(self, other) -> int: return self.interval.overlap(other)
    def get(self, item, default = None): return self.qualifiers.get(item, default)
    def get_all(self, key: bytes) -> list[Any]: return self.qualifiers.get_all(key)
    def add_qualifier(self, key: bytes, value: Any = True): self.qualifiers.add(key, value)
    def shift(self, x: int, y: int = None) -> 'Feature':
        return Feature(self.interval.shift(x, y), self.kind, list(self.qualifiers))
    def reverse_complement(self, parent_length: int) -> 'Feature':
        return Feature(self.interval.reverse_complement(parent_length), self.kind, list(self.qualifiers))
    def extract(self, parent_seq: Seq) -> Seq:
        """Extracts the feature's sequence from the parent."""
        sub = parent_seq[self.interval.start:self.interval.end]
        if self.interval.strand == -1: return parent_seq.alphabet.reverse_complement(sub)
        return sub


class FeatureList(MutableSequence):
    """
    A list-like object that manages genomic features and their spatial index.
    
    It automatically invalidates the spatial index when features are modified.
    """
    __slots__ = ('_data', '_interval_index')

    def __init__(self, features: Iterable['Feature'] = None):
        self._data: list[Feature] = list(features) if features else []
        self._interval_index = None

    def __getitem__(self, index): return self._data[index]
    def __repr__(self): return repr(self._data)
    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)
    def _flag_dirty(self): self._interval_index = None
    @property
    def interval_index(self) -> 'IntervalIndex':
        if self._interval_index is None:
            self._interval_index = IntervalIndex.from_features(*self._data)
        return self._interval_index

    def __setitem__(self, index, value):
        self._data[index] = value
        self._flag_dirty()

    def __delitem__(self, index):
        del self._data[index]
        self._flag_dirty()

    def insert(self, index, value):
        self._data.insert(index, value)
        self._flag_dirty()

    def extend(self, values: Iterable['Feature']):
        self._data.extend(values)
        self._flag_dirty()

    def reverse(self):
        self._data.reverse()
        self._flag_dirty()

    def sort(self, key=None, reverse=False):
        self._data.sort(key=key, reverse=reverse)
        self._flag_dirty()

    def get_overlapping(self, start: int, end: int) -> Generator['Feature', None, None]:
        """
        Yields features overlapping [start, end).
        Delegates to the IntervalIndex for robust and fast querying.
        """
        # self.interval_index ensures the index is built and up-to-date
        indices = self.interval_index.query(start, end)
        for i in indices: yield self._data[i]


class Record:
    """
    Represents a sequence record, such as a contig or chromosome.
    """
    __slots__ = ('seq', 'id', 'description', 'qualifiers', '_features')
    def __init__(self, seq: Seq, id_: bytes = None, desc: bytes = None, qualifiers: Iterable[tuple] = None,
                 features: list['Feature'] = None):
        self.seq: Seq = seq
        self.id: bytes = id_ or uuid4().hex.encode('ascii')
        self.description: bytes = desc or b''
        self.qualifiers = QualifierList(qualifiers)
        # FeatureList is a list subclass that likely handles parent linkage
        self._features = FeatureList(features)
    def __str__(self): return self.id.decode()
    def __repr__(self) -> str: return f'{self.id} {self.seq.__repr__()}'
    def __len__(self) -> int: return len(self.seq)
    def __hash__(self) -> int: return hash(self.id)
    def __iter__(self) -> Iterator['Feature']: return iter(self.features)
    def __eq__(self, other) -> bool: return self.id == other.id if isinstance(other, Record) else False

    @classmethod
    def from_seq(cls, seq: Seq, id_: bytes = None) -> 'Record': return cls(seq, id_=id_)

    @classmethod
    def from_batch(cls, batch: SeqBatch, ids: list[bytes] = None) -> Generator['Record', None, None]:
        if ids is None: ids = [b'Seq_%d' % i for i in range(len(batch))]
        for id_, seq in zip(ids, batch): yield cls(seq, id_=id_)

    @property
    def features(self) -> 'FeatureList': return self._features

    @features.setter
    def features(self, value): self._features = FeatureList(value)

    @property
    def interval_index(self) -> 'IntervalIndex': return self.features.interval_index

    def add_features(self, *features: 'Feature'):
        self.features.extend(features)
        self.features.sort(key=attrgetter('interval.start'))

    def extract_feature(self, feature: Feature) -> Seq:
        """Extracts the sequence of a specific feature."""
        return feature.extract(self.seq)

    def reverse_complement(self) -> 'Record':
        """Returns the reverse complement of the record and all its features."""
        new_seq = self.seq.alphabet.reverse_complement(self.seq)
        parent_len = len(self)
        # Transform features: coordinates flip, strand flips
        new_feats = [f.reverse_complement(parent_len) for f in self.features]
        new_feats.sort(key=attrgetter('interval.start'))
        return Record(new_seq, id_=self.id, desc=self.description,
                      qualifiers=list(self.qualifiers), features=new_feats)

    # --- Slicing & Access ---
    def __getitem__(self, item) -> 'Record':
        """
        Slices the record and correctly truncates overlapping features.
        """
        item = Interval.from_item(item, length=len(self.seq))
        new_record = Record(self.seq[item])

        # Use the robust overlap logic instead of naive bisect
        # We collect ANY feature overlapping the slice region
        overlapping = sorted(self.features.get_overlapping(item.start, item.end), key=attrgetter('interval.start'))

        new_features = []
        for feature in overlapping:
            # Shift (-start) moves the feature into the new coordinate system
            # The Feature.shift method or Feature constructor usually handles truncation
            # if the interval goes negative or exceeds length.
            # If NOT, you must truncate manually here:

            # Logic: We want the intersection of Feature and Slice
            # In the NEW coordinates (0 to slice_len)

            # 1. Shift
            new_f = feature.shift(-item.start)

            # 2. Truncate (Clamp to 0 and new_len)
            # Assuming Interval is mutable or has a method for intersection
            start_in_new = max(0, new_f.interval.start)
            end_in_new = min(len(new_record), new_f.interval.end)

            if start_in_new < end_in_new:
                # Update interval if truncated
                if start_in_new != new_f.interval.start or end_in_new != new_f.interval.end:
                    new_f.interval = Interval(start_in_new, end_in_new, new_f.interval.strand)
                new_features.append(new_f)

        new_record.features.extend(new_features)
        return new_record

    # --- Modification ---
    def __add__(self, other: 'Record') -> 'Record':
        if not isinstance(other, Record): raise TypeError(other)
        # 1. Merge Sequence
        new_seq = self.seq + other.seq
        # 2. Handle Features
        features = [deepcopy(f) for f in self.features]
        offset = len(self)

        # Merge Boundary Features Logic
        # (Only if adjacent and same kind)
        if (self.features and other.features and
                self.features[-1].interval.end == len(self) and
                other.features[0].interval.start == 0 and
                self.features[-1].kind == other.features[0].kind):

            last = features.pop()  # Remove last from self
            first = other.features[0]  # Get first from other

            # Extend the last feature
            # FIX: Interval is immutable, must create new instance
            new_end = last.interval.end + (first.interval.end - first.interval.start)
            last.interval = Interval(last.interval.start, new_end, last.interval.strand)
            # Qualifiers merge logic? (For now, kept self's qualifiers)
            features.append(last)

            # Add remaining other features
            for f in other.features[1:]:
                features.append(f.shift(offset))
        else:
            # Standard append
            for f in other.features:
                features.append(f.shift(offset))

        return Record(new_seq, features=features)

    def __delitem__(self, key: Union[slice, int]):
        """
        Deletes a slice. Handles overlapping features by TRUNCATING them.
        """
        if not isinstance(key, slice):
            raise TypeError("Deletion supported for slices only.")

        start, stop, step = key.indices(len(self))
        if step != 1: raise ValueError("Deletion step not supported.")

        slice_len = stop - start
        if slice_len <= 0: return

        new_features = []

        # We iterate existing features and decide: Keep, Drop, or Modify
        # Since deletion breaks the coordinate system, binary search is tricky for *modifying*
        # but useful for finding start points.

        for feature in self.features:
            f_start, f_end = feature.interval.start, feature.interval.end

            # Case 1: Feature is entirely BEFORE the cut (Keep)
            if f_end <= start:
                new_features.append(feature)
                continue

            # Case 2: Feature is entirely AFTER the cut (Shift Left)
            if f_start >= stop:
                new_features.append(feature.shift(-slice_len))
                continue

            # Case 3: Feature is INSIDE or OVERLAPPING the cut (Modify)
            # This handles:
            #   - Enclosed: Cut [100:200], Feature [120:150] -> Deleted
            #   - Left Overlap: Cut [100:200], Feature [50:150] -> [50:100]
            #   - Right Overlap: Cut [100:200], Feature [150:250] -> [150-len:250-len] -> [100:150] (shifted)
            #   - Spanning: Cut [100:200], Feature [50:250] -> [50:150] (Gap closed)

            # Calculate new duration (remove the part that was cut)
            # Overlap start
            ov_start = max(start, f_start)
            # Overlap end
            ov_end = min(stop, f_end)
            overlap_len = max(0, ov_end - ov_start)

            if overlap_len == len(feature):
                # Entirely deleted
                continue

            # If we survive, we are effectively just "collapsing" the cut region
            # It's easier to think about the new coordinates:
            # - Start: min(f_start, start) if spanning left, else f_start - slice_len
            # Actually, standard logic:

            new_f = deepcopy(feature)

            if f_start < start:
                # Starts before cut.
                # New end = old_end - overlap_len
                new_f.interval = Interval(new_f.interval.start, new_f.interval.end - overlap_len, new_f.interval.strand)
            else:
                # Starts inside or after cut.
                # Start must be shifted
                # Since we are iterating all features, we know f_start < stop here (Case 2 handled)
                # So f_start is inside. It becomes `start`.
                new_start = start
                new_end = start + (len(feature) - overlap_len)
                new_f.interval = Interval(new_start, new_end, new_f.interval.strand)

            new_features.append(new_f)

        self.features = new_features
        self.seq = self.seq[:start] + self.seq[stop:]  # This delegates to Seq slicing

    # --- Utilities ---
    def insert(self, other: 'Record', at: int, replace: bool = False) -> 'Record':
        """
        Optimized insertion.
        """
        if not 0 <= at <= len(self): raise IndexError(f"Insert index {at} out of range.")
        # Optimization: Don't slice everything if we don't have to.
        # But Record is immutable-ish (returns new), so slicing is inevitable.
        suffix_start = at + (len(other) if replace else 0)
        # If replace=True, we skip 'len(other)' from self.
        # Wait, usually replace=True implies overwriting sequence of length len(other).
        # Your previous logic: self[at if not replace else at + len(other):]
        # This implies if replace=True, we delete `len(other)` from self.
        return self[:at] + other + self[suffix_start:]

    def shred(self, rng: np.random.Generator = None, n_breaks: int = None, break_points: list[int] = None
              ) -> Generator['Record', None, None]:
        if rng is None: rng = RESOURCES.rng
        if not n_breaks and not break_points:
            n_breaks = rng.integers(1, max(2, len(self) // 1000))  # improved default

        if not break_points:
            break_points = sorted(rng.choice(len(self), size=n_breaks, replace=False))

        previous_end = 0
        for break_point in break_points:
            # Yield slice
            yield self[previous_end:break_point]
            previous_end = break_point
        yield self[previous_end:]


class RecordBatch:
    """
    A high-performance, read-only container for a collection of Records.
    
    Uses Structure-of-Arrays (SoA) layout to store sequences and features contiguously,
    enabling vectorized operations and Numba compatibility.
    """
    __slots__ = ('_seqs', '_ids', '_feature_intervals', '_feature_kinds', '_feature_rec_indices', '_feature_offsets')

    def __init__(self, records: Iterable[Record]):
        # 1. Materialize list to avoid multiple passes if iterator
        recs = list(records)
        n_recs = len(recs)

        # 2. Batch Sequences
        self._seqs = SeqBatch((r.seq for r in recs))

        # 3. Batch IDs (Fixed length numpy array for speed, or object array)
        # We use object array of bytes to handle variable length IDs safely
        self._ids = np.array([r.id for r in recs], dtype=object)

        # 4. Flatten Features
        # We need to build a CSR-like structure (Compressed Sparse Row)
        # offsets[i] -> start index of features for record i
        # offsets[i+1] -> end index
        
        total_features = sum(len(r.features) for r in recs)
        self._feature_offsets = np.zeros(n_recs + 1, dtype=np.int32)
        
        # Pre-allocate feature arrays
        # We use IntervalIndex internals directly for speed
        all_starts = np.empty(total_features, dtype=np.int32)
        all_ends = np.empty(total_features, dtype=np.int32)
        all_strands = np.empty(total_features, dtype=np.int32)
        
        # Kind is usually short ASCII, S16 is usually enough, or object
        self._feature_kinds = np.empty(total_features, dtype='S16')
        
        # Mapping back to record index (useful for filtering)
        self._feature_rec_indices = np.empty(total_features, dtype=np.int32)

        curr_idx = 0
        for i, r in enumerate(recs):
            self._feature_offsets[i] = curr_idx
            
            # Bulk copy if possible, otherwise loop
            # Assuming FeatureList is iterable
            for f in r.features:
                all_starts[curr_idx] = f.interval.start
                all_ends[curr_idx] = f.interval.end
                all_strands[curr_idx] = f.interval.strand
                self._feature_kinds[curr_idx] = f.kind
                self._feature_rec_indices[curr_idx] = i
                curr_idx += 1
        
        self._feature_offsets[n_recs] = curr_idx
        
        # Build the master IntervalIndex
        self._feature_intervals = IntervalIndex(all_starts, all_ends, all_strands)

    def __len__(self): return len(self._seqs)
    @property
    def ids(self) -> np.ndarray: return self._ids
    @property
    def seqs(self) -> SeqBatch: return self._seqs
    @property
    def n_features(self) -> int: return len(self._feature_intervals)

    def features_for(self, record_idx: int) -> 'IntervalIndex':
        """Returns the IntervalIndex for a specific record."""
        start = self._feature_offsets[record_idx]
        end = self._feature_offsets[record_idx+1]
        # Slicing IntervalIndex is not yet implemented in your code, 
        # but assuming standard numpy slicing on internal arrays:
        return IntervalIndex(
            self._feature_intervals.starts[start:end],
            self._feature_intervals.ends[start:end],
            self._feature_intervals.strands[start:end]
        )

    def extract_features(self, kind: bytes = None) -> 'SeqBatch':
        """
        Extracts sequences for ALL features in the batch, optionally filtered by kind.
        Returns a new BatchSeq of the feature sequences.
        """
        if kind is not None:
            mask = self._feature_kinds == kind
            starts = self._feature_intervals.starts[mask]
            ends = self._feature_intervals.ends[mask]
            strands = self._feature_intervals.strands[mask]
            rec_idxs = self._feature_rec_indices[mask]
        else:
            starts, ends, strands = self._feature_intervals.starts, self._feature_intervals.ends, self._feature_intervals.strands
            rec_idxs = self._feature_rec_indices

        # We need a kernel to extract slices from BatchSeq based on these coordinates
        # This requires a new method in BatchSeq or a helper kernel.
        # For now, we can rely on the fact that BatchSeq holds a flat array.
        
        # Construct new BatchSeq manually
        # We need to handle Reverse Complements for negative strands
        # This is complex to do zero-copy. 
        # A simple approach is to yield Seqs, but we want BatchSeq.
        
        # Let's assume a helper exists or implement a basic list comprehension fallback for safety
        # until a specific kernel is written.
        extracted = []
        for i in range(len(starts)):
            # Reconstruct Seq object (view)
            r_idx = rec_idxs[i]
            s = starts[i]
            e = ends[i]
            st = strands[i]
            
            # Get parent seq
            parent = self.seqs[r_idx]
            sub = parent[s:e]
            if st == -1:
                sub = sub.alphabet.reverse_complement(sub)
            extracted.append(sub)
            
        return SeqBatch(extracted, self.seqs.alphabet)


# class MutationEffect(IntEnum):
#     """
#     Enum for efficient integer storage of mutation effects.
#     Using IntEnum allows storage in numpy uint8 arrays.
#     """
#     UNKNOWN = 0
#     SYNONYMOUS = 1
#     MISSENSE = 2
#     NONSENSE = 3
#     FRAMESHIFT = 4
#     INTERGENIC = 5
#     NON_CODING = 6


# class Mutation(Feature):
#     """
#     Represents a discrete change: SNP, Insertion, or Deletion.
#
#     Attributes:
#         interval (Interval): The location on the REFERENCE sequence.
#         ref_seq (Seq): The reference sequence content.
#         alt_seq (Seq): The alternative sequence content.
#         effect (MutationEffect): The predicted functional impact.
#     """
#     __slots__ = ('ref_seq', 'alt_seq', 'effect', '_aa_change')
#
#     def __init__(self, interval: Interval, ref_seq: Seq, alt_seq: Seq,
#                  effect: MutationEffect = MutationEffect.UNKNOWN,
#                  aa_change: bytes = None):
#         super().__init__(interval, kind=b'mutation')
#         self.ref_seq = ref_seq
#         self.alt_seq = alt_seq
#         self.effect = effect
#         self._aa_change = aa_change
#
#     def __repr__(self):
#         # VCF-style notation: Pos Ref>Alt (1-based for display)
#         return f"{self.interval.start + 1}:{self.ref_seq}>{self.alt_seq}"
#
#     @property
#     def is_snp(self): return len(self.ref_seq) == 1 and len(self.alt_seq) == 1
#     @property
#     def is_indel(self): return len(self.ref_seq) != len(self.alt_seq)
#     @property
#     def diff(self) -> int:
#         """Returns the net change in sequence length (Alt - Ref)."""
#         return len(self.alt_seq) - len(self.ref_seq)