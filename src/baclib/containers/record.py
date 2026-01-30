from copy import deepcopy
from operator import attrgetter
from typing import Generator, Union, Iterator, MutableSequence, Any, Iterable
from uuid import uuid4

import numpy as np

from baclib.core.seq import Seq, SeqBatch
from baclib.core.interval import Interval, IntervalIndex
from baclib.utils.resources import RESOURCES, jit

if RESOURCES.has_module('numba'): 
    from numba import prange
else: 
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class QualifierList(MutableSequence):
    """
    A list-like container for (key, value) tuples that also supports dictionary-style access.
    Maintains insertion order and allows duplicate keys.

    Examples:
        >>> q = QualifierList([(b'gene', 'gyrA')])
        >>> q[b'gene']
        'gyrA'
        >>> q.add(b'note', 'important')
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
        """Returns the first value for a key, or default."""
        for k, v in self._data:
            if k == key: return v
        return default

    def get_all(self, key: bytes) -> list[Any]:
        """Returns all values for a key."""
        return [v for k, v in self._data if k == key]

    def add(self, key: bytes, value: Any):
        """Adds a new key-value pair."""
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

    Examples:
        >>> i = Interval(100, 200, '+')
        >>> f = Feature(i, kind=b'gene', qualifiers=[(b'gene', 'gyrA')])
        >>> f[b'gene']
        'gyrA'
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
    def copy(self) -> 'Feature':
        """Creates a shallow copy of the feature (qualifiers are copied)."""
        return Feature(self.interval, self.kind, list(self.qualifiers))


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
            self._interval_index = IntervalIndex.from_features(self._data)
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

    Attributes:
        seq (Seq): The sequence data.
        id (bytes): The record identifier.
        description (bytes): Description text.
        qualifiers (QualifierList): Record-level annotations.
        features (FeatureList): List of features on the record.

    Examples:
        >>> from baclib.core.seq import Alphabet
        >>> dna = Alphabet.dna()
        >>> s = dna.seq("ACGT")
        >>> r = Record(s, id_=b'seq1')
        >>> len(r)
        4
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
    def from_seq(cls, seq: Seq, id_: bytes = None) -> 'Record':
        """Creates a Record from a Seq object."""
        return cls(seq, id_=id_)

    @classmethod
    def from_batch(cls, batch: SeqBatch, ids: list[bytes] = None) -> Generator['Record', None, None]:
        """
        Generates Records from a SeqBatch.

        Args:
            batch: The SeqBatch.
            ids: Optional list of IDs.

        Yields:
            Record objects.
        """
        if ids is None: ids = [b'Seq_%d' % i for i in range(len(batch))]
        for id_, seq in zip(ids, batch): yield cls(seq, id_=id_)

    @property
    def features(self) -> 'FeatureList': return self._features

    @features.setter
    def features(self, value): self._features = FeatureList(value)

    @property
    def interval_index(self) -> 'IntervalIndex': return self.features.interval_index

    def add_features(self, *features: 'Feature'):
        """Adds features and sorts them by start position."""
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
        features = [f.copy() for f in self.features]
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
        """
        Shreds the record into smaller pieces.

        Args:
            rng: Random number generator.
            n_breaks: Number of random breaks.
            break_points: Specific break points.

        Yields:
            Record fragments.
        """
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


class QualifierBatch:
    """
    A batch of qualifier lists, stored in a columnar format (Structure-of-Arrays).
    """
    __slots__ = ('_key_vocab', '_key_ids', '_values', '_offsets')

    def __init__(self, key_vocab, key_ids, values, offsets):
        self._key_vocab = key_vocab
        self._key_ids = key_ids
        self._values = values
        self._offsets = offsets

    @classmethod
    def from_qualifiers(cls, qualifiers_list: Iterable[Iterable[tuple[bytes, Any]]]) -> 'QualifierBatch':
        key_to_id = {}
        key_vocab = []
        flat_key_ids = []
        flat_values = []
        
        offsets = [0]
        curr_idx = 0
        
        for quals in qualifiers_list:
            for k, v in quals:
                if k not in key_to_id:
                    key_to_id[k] = len(key_vocab)
                    key_vocab.append(k)
                flat_key_ids.append(key_to_id[k])
                flat_values.append(v)
                curr_idx += 1
            offsets.append(curr_idx)
            
        return cls(
            np.array(key_vocab, dtype=object),
            np.array(flat_key_ids, dtype=np.int32),
            np.array(flat_values, dtype=object),
            np.array(offsets, dtype=np.int32)
        )

    def get(self, idx: int) -> list[tuple]:
        start = self._offsets[idx]
        end = self._offsets[idx+1]
        if start == end: return []
        keys = self._key_vocab[self._key_ids[start:end]]
        values = self._values[start:end]
        return list(zip(keys, values))


class FeatureBatch:
    """
    A batch of Features, stored in a columnar format.
    """
    __slots__ = ('_intervals', '_kind_vocab', '_kind_ids', '_qualifiers')

    def __init__(self, intervals: IntervalIndex, kind_vocab, kind_ids, qualifiers: QualifierBatch):
        self._intervals = intervals
        self._kind_vocab = kind_vocab
        self._kind_ids = kind_ids
        self._qualifiers = qualifiers

    @classmethod
    def from_features(cls, features: list[Feature]) -> 'FeatureBatch':
        n = len(features)
        
        # Intervals
        starts = np.empty(n, dtype=np.int32)
        ends = np.empty(n, dtype=np.int32)
        strands = np.empty(n, dtype=np.int32)
        
        # Kinds
        kind_to_id = {}
        kind_vocab = []
        kind_ids = np.empty(n, dtype=np.int32)
        
        for i, f in enumerate(features):
            starts[i] = f.interval.start
            ends[i] = f.interval.end
            strands[i] = f.interval.strand
            
            k = f.kind
            if k not in kind_to_id:
                kind_to_id[k] = len(kind_vocab)
                kind_vocab.append(k)
            kind_ids[i] = kind_to_id[k]
            
        intervals = IntervalIndex(starts, ends, strands)
        # Generator to avoid creating intermediate list of qualifier lists
        qualifiers = QualifierBatch.from_qualifiers((f.qualifiers for f in features))
        
        return cls(intervals, np.array(kind_vocab, dtype=object), kind_ids, qualifiers)

    @property
    def intervals(self): return self._intervals
    @property
    def kind_ids(self): return self._kind_ids
    @property
    def kind_vocab(self): return self._kind_vocab
    
    def get_kind(self, idx: int) -> bytes:
        return self._kind_vocab[self._kind_ids[idx]]
        
    def get_qualifiers(self, idx: int) -> list[tuple]:
        return self._qualifiers.get(idx)

    def get(self, idx: int) -> 'Feature':
        """
        Reconstructs a Feature object from the batch.
        """
        interval = Interval(
            self._intervals.starts[idx],
            self._intervals.ends[idx],
            self._intervals.strands[idx]
        )
        return Feature(interval, self.get_kind(idx), self.get_qualifiers(idx))


class RecordBatch:
    """
    A high-performance, read-only container for a collection of Records.
    
    Uses Structure-of-Arrays (SoA) layout to store sequences and features contiguously,
    enabling vectorized operations and Numba compatibility.
    """
    __slots__ = ('_seqs', '_ids', '_features', '_feature_rec_indices', '_feature_offsets')

    def __init__(self, records: Iterable[Record]):
        # 1. Materialize list to avoid multiple passes if iterator
        recs = list(records)

        # 2. Batch Sequences
        self._seqs = SeqBatch.from_seqs((r.seq for r in recs))

        # 3. Batch IDs (Fixed length numpy array for speed, or object array)
        # We use object array of bytes to handle variable length IDs safely
        self._ids = np.array([r.id for r in recs], dtype=object)

        self._build_features(recs)

    @classmethod
    def from_aligned_batch(cls, batch: SeqBatch, records: list[Record]):
        """
        Efficiently creates a RecordBatch from a pre-computed SeqBatch and a list of Records.
        Assumes records[i] corresponds to batch[i].
        """
        obj = cls.__new__(cls)
        obj._seqs = batch
        obj._ids = np.array([r.id for r in records], dtype=object)
        obj._build_features(records)
        return obj

    def _build_features(self, recs):
        # 1. Flatten features
        all_features = []
        rec_lens = []
        for r in recs:
            all_features.extend(r.features)
            rec_lens.append(len(r.features))
            
        # 2. Create FeatureBatch
        self._features = FeatureBatch.from_features(all_features)
        
        # 3. Build Record-Feature mappings
        self._feature_offsets = np.zeros(len(recs) + 1, dtype=np.int32)
        np.cumsum(rec_lens, out=self._feature_offsets[1:])
        
        # Efficiently create the mapping from feature_idx -> record_idx
        self._feature_rec_indices = np.repeat(np.arange(len(recs), dtype=np.int32), rec_lens)

    def __len__(self): return len(self._seqs)
    @property
    def ids(self) -> np.ndarray: return self._ids
    @property
    def seqs(self) -> SeqBatch: return self._seqs
    @property
    def n_features(self) -> int: return len(self._features.intervals)

    def __iter__(self) -> Generator['Record', None, None]:
        for i in range(len(self)):
            yield self.get_record(i)

    def get_record(self, idx: int) -> 'Record':
        """
        Reconstructs a full Record object with features from the batch.
        """
        seq = self.seqs[idx]
        rec_id = self.ids[idx]
        
        start = self._feature_offsets[idx]
        end = self._feature_offsets[idx+1]
        features = [self._features.get(i) for i in range(start, end)]
        
        return Record(seq, id_=rec_id, features=features)

    def features_for(self, record_idx: int) -> 'IntervalIndex':
        """Returns the IntervalIndex for a specific record."""
        start = self._feature_offsets[record_idx]
        end = self._feature_offsets[record_idx+1]
        
        intervals = self._features.intervals
        # Slicing IntervalIndex is not yet implemented in your code, 
        # but assuming standard numpy slicing on internal arrays:
        return IntervalIndex(
            intervals.starts[start:end],
            intervals.ends[start:end],
            intervals.strands[start:end]
        )

    def extract_features(self, kind: bytes = None) -> 'SeqBatch':
        """
        Extracts sequences for ALL features in the batch, optionally filtered by kind.
        Returns a new SeqBatch of the feature sequences.
        """
        if kind is not None:
            # Find ID for kind
            kind_id = -1
            for i, k in enumerate(self._features.kind_vocab):
                if k == kind:
                    kind_id = i
                    break
            if kind_id == -1: return SeqBatch.from_seqs([], self.seqs.alphabet)

            mask = self._features.kind_ids == kind_id
            starts = self._features.intervals.starts[mask]
            ends = self._features.intervals.ends[mask]
            strands = self._features.intervals.strands[mask]
            rec_idxs = self._feature_rec_indices[mask]
        else:
            starts, ends, strands = self._features.intervals.starts, self._features.intervals.ends, self._features.intervals.strands
            rec_idxs = self._feature_rec_indices

        # Optimized Extraction
        n_feats = len(starts)
        lengths = ends - starts
        # Clip negative lengths (safety)
        np.maximum(lengths, 0, out=lengths)
        
        out_starts = np.zeros(n_feats, dtype=np.int32)
        if n_feats > 1:
            np.cumsum(lengths[:-1], out=out_starts[1:])
            
        total_len = out_starts[-1] + lengths[-1] if n_feats > 0 else 0
        out_data = np.empty(total_len, dtype=self.seqs.DTYPE)
        
        rc_table = self.seqs.alphabet.complement
        if rc_table is None:
            rc_table = np.empty(0, dtype=self.seqs.DTYPE) # Dummy
            
        _batch_extract_kernel(
            self.seqs._data, self.seqs._starts,
            starts, ends, strands, rec_idxs,
            out_data, out_starts, lengths,
            rc_table
        )
        
        return SeqBatch(out_data, out_starts, lengths, self.seqs.alphabet)

    def get_feature_kind(self, idx: int) -> bytes:
        """Returns the kind of the feature at the global index."""
        return self._features.get_kind(idx)

    def get_feature_qualifiers(self, idx: int) -> list[tuple]:
        """Returns the qualifiers of the feature at the global index."""
        return self._features.get_qualifiers(idx)


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


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_extract_kernel(seq_data, seq_starts, 
                          feat_starts, feat_ends, feat_strands, feat_rec_idxs,
                          out_data, out_starts, out_lens,
                          rc_table):
    n_feats = len(feat_starts)
    has_rc = len(rc_table) > 0
    
    for i in prange(n_feats):
        rec_idx = feat_rec_idxs[i]
        rec_start = seq_starts[rec_idx]
        
        f_s = feat_starts[i]
        strand = feat_strands[i]
        l = out_lens[i]
        
        dst_start = out_starts[i]
        src_ptr = rec_start + f_s
        
        if strand == -1 and has_rc:
            for j in range(l):
                val = seq_data[src_ptr + (l - 1 - j)]
                out_data[dst_start + j] = rc_table[val]
        else:
            out_data[dst_start : dst_start + l] = seq_data[src_ptr : src_ptr + l]
