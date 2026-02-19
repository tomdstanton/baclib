"""Container for biological sequence records with metadata, features, and batch support."""
from binascii import hexlify
from operator import attrgetter
from typing import Generator, Union, Iterator, Iterable
from uuid import uuid4

import numpy as np

from baclib.core.alphabet import Alphabet
from baclib.containers.seq import Seq, SeqBatch
from baclib.core.interval import Interval, IntervalBatch
from baclib.containers import Batch, Batchable
from baclib.containers.feature import FeatureKey, Feature, FeatureList, FeatureBatch
from baclib.containers.qualifier import QualifierList, QualifierBatch
from baclib.lib.resources import RESOURCES, jit
from baclib.lib.protocols import HasIntervals

if RESOURCES.has_module('numba'): 
    from numba import prange
else: 
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Record(HasIntervals, Batchable):
    """
    A biological sequence record with ID, description, qualifiers, and features.

    Typically represents a contig, chromosome, or plasmid — a named sequence
    annotated with genomic features (CDS, tRNA, etc.) and key-value qualifiers.

    Args:
        seq: The underlying ``Seq`` object.
        id_: Record identifier as bytes (auto-generated UUID if omitted).
        desc: Optional description line.
        qualifiers: Optional iterable of ``(key, value)`` qualifier tuples.
        features: Optional list of ``Feature`` objects.

    Examples:
        >>> rec = Record(Alphabet.DNA.seq(b'ATGCGA'), id_=b'contig_1')
        >>> len(rec)
        6
        >>> rec.id
        b'contig_1'
    """
    __slots__ = ('_seq', 'id', 'description', '_qualifiers', '_features')
    def __init__(self, seq: Seq, id_: bytes = None, desc: bytes = None, qualifiers: Iterable[tuple] = None,
                 features: list['Feature'] = None):
        self._seq: Seq = seq
        self.id: bytes = id_ or hexlify(uuid4().bytes)
        self.description: bytes = desc or b''
        self.qualifiers = qualifiers
        # FeatureList is a list subclass that likely handles parent linkage
        self._features = FeatureList(features)
    def __str__(self): return self.id.decode(errors='ignore')
    def __repr__(self) -> str: return f'{self.id} {self._seq.__repr__()}'
    def __len__(self) -> int: return len(self._seq)
    def __hash__(self) -> int: return hash(self.id)
    def __iter__(self) -> Iterator['Feature']: return iter(self.features)
    def __eq__(self, other) -> bool: return self.id == other.id if isinstance(other, Record) else False

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``RecordBatch`` class.
        """
        return RecordBatch

    @property
    def seq(self) -> Seq:
        """Returns the underlying sequence.

        Returns:
            The ``Seq`` object.
        """
        return self._seq

    @property
    def features(self) -> 'FeatureList':
        """Returns the list of features annotated on this record.

        Returns:
            A ``FeatureList``.
        """
        return self._features

    @features.setter
    def features(self, value): self._features = FeatureList(value)

    @property
    def qualifiers(self) -> QualifierList:
        """Returns the record-level qualifiers.

        Returns:
            A ``QualifierList`` of ``(key, value)`` tuples.
        """
        return self._qualifiers

    @qualifiers.setter
    def qualifiers(self, value): self._qualifiers = QualifierList(value)

    @property
    def intervals(self) -> 'IntervalBatch':
        """Returns the intervals of all features on this record.

        Returns:
            An ``IntervalBatch``.
        """
        return self.features.intervals

    def add_features(self, *features: 'Feature'):
        """Adds features to this record, re-sorting by start position.

        Args:
            features: One or more ``Feature`` objects to add.
        """
        self.features.extend(features)
        self.features.sort(key=attrgetter('interval.start'))

    def extract_feature(self, feature: Feature) -> Seq:
        """Extracts the subsequence corresponding to a feature.

        Uses the feature's interval (including strand) to slice the record's
        sequence, applying reverse complement if on the minus strand.

        Args:
            feature: The ``Feature`` whose sequence to extract.

        Returns:
            A ``Seq`` of the feature's sequence.

        Examples:
            >>> cds = rec.features[0]
            >>> rec.extract_feature(cds)
            ATGCGA
        """
        return feature.extract(self._seq)

    def reverse_complement(self) -> 'Record':
        """Returns the reverse complement of this record and all its features.

        Feature coordinates and strands are flipped relative to the new
        sequence orientation.

        Returns:
            A new ``Record`` with the reverse-complemented sequence and
            transformed feature coordinates.
        """
        new_seq = self._seq.alphabet.reverse_complement(self._seq)
        parent_len = len(self)
        # Transform features: coordinates flip, strand flips
        new_feats = [f.reverse_complement(parent_len) for f in self.features]
        new_feats.sort(key=attrgetter('interval.start'))
        return Record(new_seq, id_=self.id, desc=self.description,
                      qualifiers=list(self.qualifiers), features=new_feats)

    # --- Slicing & Access ---
    def __getitem__(self, item) -> 'Record':
        """Slices the record, truncating overlapping features to fit the new bounds.

        Args:
            item: A slice, integer, or ``Interval``.

        Returns:
            A new ``Record`` with the sliced sequence and adjusted features.
        """
        item = Interval.from_item(item, length=len(self._seq))
        new_record = Record(self._seq[item])

        # Overlapping features are already sorted by start due to IntervalBatch properties
        overlapping = self.features.get_overlapping(item.start, item.end)

        new_features = []
        offset = item.start
        limit = len(new_record)

        for feature in overlapping:
            # Direct calculation of new coordinates
            s = max(0, feature.interval.start - offset)
            e = min(limit, feature.interval.end - offset)

            if s < e:
                new_f = Feature(Interval(s, e, feature.interval.strand), feature.key, feature.qualifiers)
                new_features.append(new_f)

        new_record.features.extend(new_features)
        return new_record

    # --- Modification ---
    def __add__(self, other: 'Record') -> 'Record':
        if not isinstance(other, Record): raise TypeError(other)
        # 1. Merge Sequence
        new_seq = self._seq + other.seq
        
        # 2. Handle Features
        feats_self = self.features
        feats_other = other.features
        features = [f.copy() for f in feats_self]
        offset = len(self)

        # Merge Boundary Features Logic
        # (Only if adjacent and same kind)
        if (feats_self and feats_other and
                feats_self[-1].interval.end == offset and
                feats_other[0].interval.start == 0 and
                feats_self[-1].key == feats_other[0].key):

            last = features.pop()  # Remove last from self
            first = feats_other[0]  # Get first from other

            # Extend the last feature
            # FIX: Interval is immutable, must create new instance
            new_end = last.interval.end + (first.interval.end - first.interval.start)
            last.interval = Interval(last.interval.start, new_end, last.interval.strand)
            # Qualifiers merge logic? (For now, kept self's qualifiers)
            features.append(last)

            # Add remaining other features
            for f in feats_other[1:]:
                features.append(f.shift(offset))
        else:
            # Standard append
            for f in feats_other:
                features.append(f.shift(offset))

        return Record(new_seq, features=features)

    def __delitem__(self, key: Union[slice, int]):
        """Deletes a slice from the record, truncating overlapping features.

        Args:
            key: A slice specifying the region to delete.

        Raises:
            TypeError: If *key* is not a slice.
        """
        if not isinstance(key, slice):
            raise TypeError("Deletion supported for slices only.")

        start, stop, step = key.indices(len(self))
        if step != 1: raise ValueError("Deletion step not supported.")

        slice_len = stop - start
        if slice_len <= 0: return

        new_features = []

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
            ov_start = max(start, f_start)
            ov_end = min(stop, f_end)
            overlap_len = max(0, ov_end - ov_start)

            if overlap_len == len(feature):
                # Entirely deleted
                continue

            new_f = feature.copy()

            if f_start < start:
                new_f.interval = Interval(new_f.interval.start, new_f.interval.end - overlap_len, new_f.interval.strand)
            else:
                new_start = start
                new_end = start + (len(feature) - overlap_len)
                new_f.interval = Interval(new_start, new_end, new_f.interval.strand)

            new_features.append(new_f)

        self.features = new_features
        self._seq = self._seq[:start] + self._seq[stop:]  # This delegates to Seq slicing

    # --- Utilities ---
    def insert(self, other: 'Record', at: int, replace: bool = False) -> 'Record':
        """Inserts another record's content at a given position.

        Args:
            other: The ``Record`` to insert.
            at: The insertion position (0-indexed).
            replace: If ``True``, overwrite ``len(other)`` bases starting at *at*.

        Returns:
            A new ``Record`` with the insertion applied.

        Raises:
            IndexError: If *at* is out of range.

        Examples:
            >>> result = rec.insert(insertion, at=100)
        """
        if not 0 <= at <= len(self): raise IndexError(f"Insert index {at} out of range.")
        suffix_start = at + (len(other) if replace else 0)
        return self[:at] + other + self[suffix_start:]

    def shred(self, rng: np.random.Generator = None, n_breaks: int = None, break_points: list[int] = None
              ) -> Generator['Record', None, None]:
        """Randomly shreds the record into fragments at break points.

        Args:
            rng: NumPy random generator (uses global default if omitted).
            n_breaks: Number of random break points.
            break_points: Explicit list of break positions (overrides *n_breaks*).

        Yields:
            ``Record`` fragments.

        Examples:
            >>> fragments = list(rec.shred(n_breaks=3))
            >>> len(fragments)
            4
        """
        if rng is None: rng = RESOURCES.rng
        if not n_breaks and not break_points:
            n_breaks = rng.integers(1, max(2, len(self) // 1000))  # improved default

        if break_points is None:
            # Use np.sort on the array directly for efficiency
            break_points = np.sort(rng.choice(len(self), size=n_breaks, replace=False))
        else:
            break_points = np.sort(break_points)

        previous_end = 0
        for break_point in break_points:
            # Yield slice
            yield self[previous_end:break_point]
            previous_end = break_point
        yield self[previous_end:]


class RecordBatch(Batch, HasIntervals):
    """
    High-performance, read-only columnar container for a collection of Records.

    Uses Structure-of-Arrays (SoA) layout: sequences, IDs, qualifiers, and features
    are stored in contiguous arrays for vectorized operations and Numba compatibility.

    Args:
        records: An iterable of ``Record`` objects to batch.
        deduplicate: If ``True``, deduplicates identical sequences.

    Examples:
        >>> batch = RecordBatch([rec1, rec2, rec3])
        >>> len(batch)
        3
        >>> batch[0]
        Record(...)
    """
    __slots__ = ('_seqs', '_ids', '_features', '_feature_rec_indices', '_feature_offsets', '_qualifiers')


    def __init__(self, records: Iterable[Record], deduplicate: bool = False):
        # 1. Materialize list to avoid multiple passes if iterator
        recs = list(records)
        # 2. Batch Sequences
        self._seqs = recs[0].seq.alphabet.batch_from((r.seq for r in recs), deduplicate=deduplicate)
        # 3. Batch IDs (Fixed length numpy array for speed, or object array)
        # We use object array of bytes to handle variable length IDs safely
        self._ids = np.array([r.id for r in recs])
        # 4. Batch Qualifiers
        self._qualifiers = QualifierBatch.build(r.qualifiers for r in recs)
        self._build_features(recs)

    @classmethod
    def random(cls, alphabet: Alphabet, rng: np.random.Generator = None, n_seqs: int = None, min_n: int = 1,
               max_n: int = 1000, length: int = None, min_length: int = 10, max_length: int = 5_000_000, weights=None):
        """Generates a random RecordBatch for testing purposes.

        Args:
            alphabet: The ``Alphabet`` to use (e.g. ``Alphabet.DNA``).
            rng: NumPy random generator.
            n_seqs: Exact number of records (overrides *min_n*/*max_n*).
            min_n: Minimum number of random records.
            max_n: Maximum number of random records.
            length: Exact sequence length (overrides *min_length*/*max_length*).
            min_length: Minimum random sequence length.
            max_length: Maximum random sequence length.
            weights: Symbol frequency weights for random generation.

        Returns:
            A new ``RecordBatch`` with random sequences and UUID IDs.
        """
        if rng is None: rng = RESOURCES.rng
        seqs = alphabet.random_batch(rng=rng, n_seqs=n_seqs, min_seqs=min_n, max_seqs=max_n, length=length, min_len=min_length,
                                     max_len=max_length, weights=weights)

        # Generate random UUIDs (Standard for Records)
        ids = np.array([hexlify(uuid4().bytes) for _ in range(len(seqs))])

        # Construct the batch manually for performance
        obj = cls.__new__(cls)
        obj._seqs = seqs
        obj._ids = ids
        obj._features = FeatureBatch.empty()
        obj._feature_offsets = np.zeros(len(seqs) + 1, dtype=np.int32)
        obj._feature_rec_indices = np.empty(0, dtype=np.int32)
        obj._qualifiers = QualifierBatch.zeros(len(seqs))
        return obj


    @classmethod
    def from_aligned_batch(cls, batch: SeqBatch, records: list[Record]):
        """Creates a RecordBatch from a pre-computed SeqBatch and matching Records.

        Assumes ``records[i]`` corresponds to ``batch[i]``. Useful when sequences
        have already been aligned or deduplicated externally.

        Args:
            batch: A pre-computed ``SeqBatch``.
            records: A list of ``Record`` objects providing IDs, qualifiers,
                and features.

        Returns:
            A new ``RecordBatch``.
        """
        obj = cls.__new__(cls)
        obj._seqs = batch
        obj._ids = np.array([r.id for r in records])
        obj._qualifiers = QualifierBatch.build(r.qualifiers for r in records)
        obj._build_features(records)
        return obj

    @classmethod
    def build(cls, components: Iterable[Record]) -> 'RecordBatch':
        """Constructs a RecordBatch from an iterable of Record objects.

        Args:
            components: An iterable of ``Record`` objects.

        Returns:
            A new ``RecordBatch``.
        """
        return cls(components)

    def _build_features(self, recs):
        # 1. Count total features
        n_feats = sum(len(r.features) for r in recs)
        
        # 2. Allocate Arrays
        starts = np.empty(n_feats, dtype=np.int32)
        ends = np.empty(n_feats, dtype=np.int32)
        strands = np.empty(n_feats, dtype=np.int32)
        keys = np.empty(n_feats, dtype=np.int16)
        
        # 4. Record offsets & Indices
        self._feature_offsets = np.zeros(len(recs) + 1, dtype=np.int32)
        self._feature_rec_indices = np.empty(n_feats, dtype=np.int32)
        
        # 5. Fill Arrays
        curr_idx = 0
        for i, r in enumerate(recs):
            self._feature_offsets[i+1] = self._feature_offsets[i] + len(r.features)
            for f in r.features:
                starts[curr_idx] = f.interval.start
                ends[curr_idx] = f.interval.end
                strands[curr_idx] = f.interval.strand
                
                keys[curr_idx] = f.key.value
                
                self._feature_rec_indices[curr_idx] = i
                curr_idx += 1

        # 6. Create Batches
        # IMPORTANT: Do not sort! We must preserve the record-grouped order.
        intervals = IntervalBatch(starts, ends, strands, sort=False)
        
        # Generator to avoid creating intermediate list of qualifier lists
        # We iterate records again, but this is cheap compared to list allocation overhead
        def qual_gen():
            for r in recs:
                for f in r.features:
                    yield f.qualifiers

        qualifiers = QualifierBatch.build(qual_gen())
        
        self._features = FeatureBatch(intervals, keys, qualifiers)

    @classmethod
    def empty(cls, alphabet: Alphabet) -> 'RecordBatch':
        """Creates an empty RecordBatch with zero records.

        Args:
            alphabet: The ``Alphabet`` for the (empty) sequence batch.

        Returns:
            An empty ``RecordBatch``.
        """
        return cls.zeros(0, alphabet)

    @classmethod
    def zeros(cls, n: int, alphabet: Alphabet) -> 'RecordBatch':
        """Creates a batch of *n* placeholder records with empty sequences.

        Args:
            n: Number of placeholder records.
            alphabet: The ``Alphabet`` for the sequences.

        Returns:
            A ``RecordBatch`` with empty sequences and no features.

        Examples:
            >>> batch = RecordBatch.zeros(5, Alphabet.DNA)
            >>> len(batch)
            5
        """
        obj = cls.__new__(cls)
        obj._seqs = alphabet.zeros_batch(n)
        obj._ids = np.full(n, b'', dtype='S1') # Empty bytes, will grow if needed or stay empty
        obj._qualifiers = QualifierBatch.zeros(n)
        obj._features = FeatureBatch.empty()
        obj._feature_offsets = np.zeros(n + 1, dtype=np.int32)
        obj._feature_rec_indices = np.empty(0, dtype=np.int32)
        return obj

    @classmethod
    def concat(cls, batches: Iterable['RecordBatch'], deduplicate: bool = False) -> 'RecordBatch':
        """Concatenates multiple RecordBatch objects into one.

        Correctly adjusts feature-to-record index mappings and offset arrays
        across batch boundaries.

        Args:
            batches: An iterable of ``RecordBatch`` objects.
            deduplicate: Whether to deduplicate identical records.

        Returns:
            A single concatenated ``RecordBatch``.

        Raises:
            ValueError: If the list is empty.

        Examples:
            >>> combined = RecordBatch.concat([batch_a, batch_b])
        """
        batches = list(batches)
        if not batches: raise ValueError("Cannot concatenate empty list")
        
        # 1. Simple Concatenations
        seqs = SeqBatch.concat([b.seqs for b in batches])
        ids = np.concatenate([b.ids for b in batches])
        qualifiers = QualifierBatch.concat([b._qualifiers for b in batches])
        
        # 2. Feature Concatenation (Complex)
        # We must adjust rec_indices and offsets
        
        # Calculate shifts for record indices (how many records were in previous batches)
        rec_counts = [len(b) for b in batches]
        rec_shifts = np.cumsum([0] + rec_counts[:-1])
        
        # Adjust indices
        new_rec_indices = np.concatenate([
            b._feature_rec_indices + shift 
            for b, shift in zip(batches, rec_shifts)
        ])
        
        # Stack offsets (using RaggedBatch logic manually here as this isn't a RaggedBatch subclass)
        feat_counts = [b.n_features for b in batches]
        feat_shifts = np.cumsum([0] + feat_counts[:-1])
        
        offset_parts = [batches[0]._feature_offsets]
        for i in range(1, len(batches)):
            # Take offsets[1:] and add the cumulative feature count
            offset_parts.append(batches[i]._feature_offsets[1:] + feat_shifts[i])
            
        new_offsets = np.concatenate(offset_parts)
        
        # Merge FeatureBatches
        features = FeatureBatch.concat([b._features for b in batches])
        
        return cls._reconstruct(seqs, ids, features, new_rec_indices, new_offsets, qualifiers)

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by all internal arrays.
        """
        return (self._seqs.nbytes + self._ids.nbytes + self._features.nbytes + 
                self._qualifiers.nbytes + self._feature_offsets.nbytes + self._feature_rec_indices.nbytes)

    def copy(self) -> 'RecordBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``RecordBatch`` with copied arrays.
        """
        return self._reconstruct(
            self._seqs.copy(), self._ids.copy(), self._features.copy(), 
            self._feature_rec_indices.copy(), self._feature_offsets.copy(), self._qualifiers.copy())

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``Record`` class.
        """
        return Record

    def __repr__(self): return f"<RecordBatch: {len(self)} records>"
    def __len__(self): return len(self._seqs)

    @property
    def ids(self) -> np.ndarray:
        """Returns the record IDs as a numpy object array.

        Returns:
            A numpy array of ``bytes`` IDs.
        """
        return self._ids

    @property
    def seqs(self) -> SeqBatch:
        """Returns the underlying sequence batch.

        Returns:
            A ``SeqBatch``.
        """
        return self._seqs

    @property
    def intervals(self) -> IntervalBatch:
        """Returns the intervals of all features across all records.

        Returns:
            An ``IntervalBatch``.
        """
        return self._features.intervals

    @property
    def n_features(self) -> int:
        """Returns the total number of features across all records.

        Returns:
            Feature count.
        """
        return len(self._features)

    def get_record(self, idx: int) -> 'Record':
        """Reconstructs a full Record object from the batch at index *idx*.

        Args:
            idx: Zero-based record index.

        Returns:
            A ``Record`` with its sequence, features, and qualifiers.

        Examples:
            >>> rec = batch.get_record(0)
            >>> rec.id
            b'contig_1'
        """
        seq = self.seqs[idx]
        rec_id = self.ids[idx]
        
        start = self._feature_offsets[idx]
        end = self._feature_offsets[idx+1]
        features = [self._features[i] for i in range(start, end)]
        quals = self._qualifiers[idx]
        
        return Record(seq, id_=rec_id, features=features, qualifiers=quals)

    def get_qualifiers(self, idx: int) -> list[tuple]:
        """Returns the qualifiers for the record at *idx*.

        Args:
            idx: Zero-based record index.

        Returns:
            A list of ``(key, value)`` tuples.
        """
        return self._qualifiers[idx]
    
    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)): return self.get_record(item)
        
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1: raise NotImplementedError("RecordBatch slicing with step != 1 not supported")
            
            # 1. Slice simple arrays
            new_seqs = self._seqs[item]
            new_ids = self._ids[item]
            new_quals = self._qualifiers[item]
            
            # 2. Slice Features
            feat_start = self._feature_offsets[start]
            feat_end = self._feature_offsets[stop]
            
            new_features = self._features[feat_start:feat_end]
            new_offsets = self._feature_offsets[start:stop+1] - feat_start
            new_rec_indices = self._feature_rec_indices[feat_start:feat_end] - start
            
            # 3. Construct
            return self._reconstruct(new_seqs, new_ids, new_features, new_rec_indices, new_offsets, new_quals)
            
        raise TypeError(f"Invalid index type: {type(item)}")
        
    @classmethod
    def _reconstruct(cls, seqs, ids, features, rec_indices, offsets, qualifiers):
        obj = cls.__new__(cls)
        obj._seqs = seqs
        obj._ids = ids
        obj._features = features
        obj._feature_rec_indices = rec_indices
        obj._feature_offsets = offsets
        obj._qualifiers = qualifiers
        return obj

    def add_features(self, features: 'FeatureBatch', pivot_key: FeatureKey = FeatureKey.SOURCE) -> 'RecordBatch':
        """Returns a new RecordBatch with additional features mapped via a pivot qualifier.

        Each feature is assigned to a record by matching a qualifier value
        (identified by *pivot_key*) against record IDs. Uses vectorized
        qualifier lookup through ``QualifierBatch._key_ids``.

        Args:
            features: A ``FeatureBatch`` of features to add.
            pivot_key: The ``FeatureKey`` whose qualifier value identifies
                the target record (default ``SOURCE``).

        Returns:
            A new ``RecordBatch`` with the matched features merged in.

        Raises:
            ValueError: If *pivot_key* is not found in the feature qualifiers.

        Examples:
            >>> annotated = batch.add_features(gff_features, pivot_key=FeatureKey.SOURCE)
        """
        if len(features) == 0: return self.copy()

        # 1. Build id → record index lookup
        id_to_idx = {self._ids[i]: i for i in range(len(self))}

        # 2. Find the pivot key in the qualifier vocabulary
        quals = features._qualifiers
        pivot_bytes = pivot_key.bytes
        pivot_vocab_idx = -1
        for i, k in enumerate(quals._key_vocab):
            if k == pivot_bytes:
                pivot_vocab_idx = i
                break

        if pivot_vocab_idx == -1:
            raise ValueError(f"Pivot key '{pivot_key}' not found in feature qualifiers")

        # 3. Vectorized: find all qualifier entries matching the pivot key
        is_pivot = (quals._key_ids == pivot_vocab_idx)

        # 4. Map each qualifier position to its feature index
        n_new = len(features)
        feat_of_qual = np.repeat(np.arange(n_new, dtype=np.int32), np.diff(quals._offsets))

        # 5. Extract the first pivot value per feature
        pivot_positions = np.where(is_pivot)[0]
        pivot_feat_indices = feat_of_qual[pivot_positions]
        pivot_values = quals._values[pivot_positions]

        # Take only first match per feature
        _, first_idx = np.unique(pivot_feat_indices, return_index=True)
        matched_feat_indices = pivot_feat_indices[first_idx]
        matched_values = pivot_values[first_idx]

        # 6. Map pivot values to record indices
        rec_indices = np.array([id_to_idx.get(v, -1) for v in matched_values], dtype=np.int32)
        valid = rec_indices >= 0
        matched_feat_indices = matched_feat_indices[valid]
        new_rec_indices = rec_indices[valid]

        if len(matched_feat_indices) == 0: return self.copy()

        # 7. Slice matched features from the input batch
        new_features = FeatureBatch.concat([features[int(i):int(i)+1] for i in matched_feat_indices])

        # 8. Merge with existing features, grouped by record
        # Sort new features by record index for offset construction
        sort_order = np.argsort(new_rec_indices)
        new_rec_indices = new_rec_indices[sort_order]
        sorted_feat_indices = np.arange(len(new_rec_indices))
        # Rebuild new_features in sorted order
        reordered = [matched_feat_indices[sort_order[j]] for j in range(len(sort_order))]
        new_features = FeatureBatch.concat([features[int(i):int(i)+1] for i in reordered])

        # 9. Combine: interleave existing and new features per record
        n_recs = len(self)
        combined_features = FeatureBatch.concat([self._features, new_features])
        combined_rec_indices = np.concatenate([self._feature_rec_indices, new_rec_indices])

        # Rebuild offsets from combined rec_indices
        new_offsets = np.zeros(n_recs + 1, dtype=np.int32)
        if len(combined_rec_indices) > 0:
            counts = np.bincount(combined_rec_indices, minlength=n_recs)
            np.cumsum(counts, out=new_offsets[1:])

        # Sort combined features by (rec_index, position) for proper grouping
        order = np.argsort(combined_rec_indices, kind='stable')
        combined_rec_indices = combined_rec_indices[order]

        # Reorder the FeatureBatch by the sort order
        reordered_intervals = IntervalBatch(
            combined_features.intervals.starts[order],
            combined_features.intervals.ends[order],
            combined_features.intervals.strands[order],
            sort=False
        )
        reordered_keys = combined_features.keys[order]
        reordered_quals = QualifierBatch.build(combined_features._qualifiers[int(i)] for i in order)
        combined_features = FeatureBatch(reordered_intervals, reordered_keys, reordered_quals)

        return self._reconstruct(
            self._seqs, self._ids, combined_features,
            combined_rec_indices, new_offsets, self._qualifiers
        )

    def extract_features(self, kind: bytes = None) -> 'SeqBatch':
        """Extracts sequences for all features, optionally filtered by type.

        Uses a Numba-accelerated kernel for parallel extraction with
        automatic reverse complement for minus-strand features.

        Args:
            kind: Optional ``FeatureKey`` bytes value to filter by
                (e.g. ``FeatureKey.CDS.bytes``). Extracts all features
                if ``None``.

        Returns:
            A ``SeqBatch`` of the extracted feature sequences.

        Examples:
            >>> cds_seqs = batch.extract_features(FeatureKey.CDS.bytes)
            >>> len(cds_seqs)
            42
        """
        if kind is not None:
            # Find ID for kind
            key_obj = FeatureKey.from_bytes(kind)
            target_val = key_obj.value
            mask = self._features.keys == target_val
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
        out_data = np.empty(total_len, dtype=np.uint8)
        
        rc_table = self.seqs.alphabet.complement
        if rc_table is None:
            rc_table = np.empty(0, dtype=np.uint8) # Dummy
            
        _batch_extract_kernel(
            self.seqs.encoded, self.seqs.starts,
            starts, strands, rec_idxs,
            out_data, out_starts, lengths,
            rc_table
        )
        
        return self.seqs.alphabet.new_batch(out_data, out_starts, lengths)


# TODO: There should be no kernels in the containers module - move elsewhere or consider moving module
# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_extract_kernel(seq_data, seq_starts, 
                          feat_starts, feat_strands, feat_rec_idxs,
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
