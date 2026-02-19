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
    Represents a sequence record, such as a contig or chromosome.
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
    def __str__(self): return self.id.decode(Alphabet.ENCODING)
    def __repr__(self) -> str: return f'{self.id} {self._seq.__repr__()}'
    def __len__(self) -> int: return len(self._seq)
    def __hash__(self) -> int: return hash(self.id)
    def __iter__(self) -> Iterator['Feature']: return iter(self.features)
    @property
    def name(self) -> bytes: return self.id
    @name.setter
    def name(self, value: bytes): self.id = value

    def __eq__(self, other) -> bool: return self.id == other.id if isinstance(other, Record) else False

    @property
    def batch(self) -> type['Batch']: return RecordBatch

    @classmethod
    def from_seq(cls, seq: Seq, id_: bytes = None) -> 'Record': return cls(seq, id_=id_)
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
    def seq(self) -> Seq: return self._seq
    @property
    def features(self) -> 'FeatureList': return self._features
    @features.setter
    def features(self, value): self._features = FeatureList(value)
    @property
    def qualifiers(self) -> QualifierList: return self._qualifiers
    @qualifiers.setter
    def qualifiers(self, value): self._qualifiers = QualifierList(value)
    @property
    def intervals(self) -> 'IntervalBatch': return self.features.intervals
    @property
    def interval_batch(self) -> 'IntervalBatch': return self.features.intervals

    def add_features(self, *features: 'Feature'):
        """Adds features and sorts them by start position."""
        self.features.extend(features)
        self.features.sort(key=attrgetter('interval.start'))

    def extract_feature(self, feature: Feature) -> Seq:
        """Extracts the sequence of a specific feature."""
        return feature.extract(self._seq)

    def reverse_complement(self) -> 'Record':
        """Returns the reverse complement of the record and all its features."""
        new_seq = self._seq.alphabet.reverse_complement(self._seq)
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
                # Construct new feature directly, avoiding intermediate objects
                # feature.qualifiers is a QualifierList, passing it to Feature constructor
                # creates a shallow copy of the list structure, which is what we want.
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

            new_f = feature.copy()

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
        self._seq = self._seq[:start] + self._seq[stop:]  # This delegates to Seq slicing

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
    A high-performance, read-only container for a collection of Records.
    
    Uses Structure-of-Arrays (SoA) layout to store sequences and features contiguously,
    enabling vectorized operations and Numba compatibility.
    """
    __slots__ = ('_seqs', '_ids', '_features', '_feature_rec_indices', '_feature_offsets', '_qualifiers')

    @classmethod
    def empty(cls) -> 'RecordBatch':
        obj = cls.__new__(cls)
        obj._seqs = SeqBatch.empty()
        obj._ids = np.empty(0, dtype=object)
        obj._features = FeatureBatch.empty()
        obj._feature_offsets = np.zeros(1, dtype=np.int32)
        obj._feature_rec_indices = np.empty(0, dtype=np.int32)
        obj._qualifiers = QualifierBatch.empty()
        return obj

    def __init__(self, records: Iterable[Record], deduplicate: bool = False):
        # 1. Materialize list to avoid multiple passes if iterator
        recs = list(records)
        # 2. Batch Sequences
        self._seqs = recs[0].seq.alphabet.batch_from((r.seq for r in recs), deduplicate=deduplicate)
        # 3. Batch IDs (Fixed length numpy array for speed, or object array)
        # We use object array of bytes to handle variable length IDs safely
        self._ids = np.array([r.id for r in recs], dtype=object)
        # 4. Batch Qualifiers
        self._qualifiers = QualifierBatch.from_qualifiers((r.qualifiers for r in recs))
        self._build_features(recs)

    @classmethod
    def random(cls, alphabet: Alphabet, rng: np.random.Generator = None, n: int = None, min_n: int = 1,
               max_n: int = 1000, l: int = None, min_l: int = 10, max_l: int = 5_000_000, weights=None):
        """
        Generates a random genome assembly for testing purposes.
        """
        if rng is None: rng = RESOURCES.rng
        seqs = alphabet.random_batch(rng=rng, n_seqs=n, min_seqs=min_n, max_seqs=max_n, length=l, min_len=min_l,
                                     max_len=max_l, weights=weights)

        # Generate random UUIDs (Standard for Records)
        ids = np.array([hexlify(uuid4().bytes) for _ in range(len(seqs))], dtype=object)

        # Construct the batch manually for performance
        obj = cls.__new__(cls)
        obj._seqs = seqs
        obj._ids = ids
        obj._features = FeatureBatch.empty()
        obj._feature_offsets = np.zeros(len(seqs) + 1, dtype=np.int32)
        obj._feature_rec_indices = np.empty(0, dtype=np.int32)
        obj._qualifiers = QualifierBatch.from_qualifiers((() for _ in range(len(seqs))))
        return obj


    @classmethod
    def from_aligned_batch(cls, batch: SeqBatch, records: list[Record]):
        """
        Efficiently creates a RecordBatch from a pre-computed SeqBatch and a list of Records.
        Assumes records[i] corresponds to batch[i].
        """
        obj = cls.__new__(cls)
        obj._seqs = batch
        obj._ids = np.array([r.id for r in records], dtype=object)
        obj._qualifiers = QualifierBatch.from_qualifiers((r.qualifiers for r in records))
        obj._build_features(records)
        return obj

    @classmethod
    def build(cls, components: Iterable[Record]) -> 'RecordBatch':
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

        qualifiers = QualifierBatch.from_qualifiers(qual_gen())
        
        self._features = FeatureBatch(intervals, keys, qualifiers)

    @classmethod
    def empty(cls) -> 'RecordBatch':
        return cls.zeros(0)

    @classmethod
    def zeros(cls, n: int) -> 'RecordBatch':
        """
        Creates a batch of n empty records.
        """
        obj = cls.__new__(cls)
        obj._seqs = SeqBatch.zeros(n)
        obj._ids = np.full(n, b'', dtype=object)
        obj._qualifiers = QualifierBatch.zeros(n)
        obj._features = FeatureBatch.empty()
        obj._feature_offsets = np.zeros(n + 1, dtype=np.int32)
        obj._feature_rec_indices = np.empty(0, dtype=np.int32)
        return obj

    @classmethod
    def concat(cls, batches: Iterable['RecordBatch']) -> 'RecordBatch':
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
        # offsets[0] is 0. We take offsets[1:] + total_feats_prev
        feat_counts = [b.n_features for b in batches]
        feat_shifts = np.cumsum([0] + feat_counts[:-1])
        
        # [0, 2, 5] + [0, 3] -> [0, 2, 5, 8]
        # We need to construct the full offset array
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
        return (self._seqs.nbytes + self._ids.nbytes + self._features.nbytes + 
                self._qualifiers.nbytes + self._feature_offsets.nbytes + self._feature_rec_indices.nbytes)

    def copy(self) -> 'RecordBatch':
        return self._reconstruct(
            self._seqs.copy(), self._ids.copy(), self._features.copy(), 
            self._feature_rec_indices.copy(), self._feature_offsets.copy(), self._qualifiers.copy())

    @property
    def component(self): return Record

    def __repr__(self): return f"<RecordBatch: {len(self)} records>"

    def __len__(self): return len(self._seqs)
    @property
    def ids(self) -> np.ndarray: return self._ids
    @property
    def seqs(self) -> SeqBatch: return self._seqs
    @property
    def intervals(self) -> IntervalBatch:
        """Returns the intervals of all features in this batch."""
        return self._features.intervals

    @property
    def n_features(self) -> int: return len(self._features)

    def get_record(self, idx: int) -> 'Record':
        """
        Reconstructs a full Record object with features from the batch.
        """
        seq = self.seqs[idx]
        rec_id = self.ids[idx]
        
        start = self._feature_offsets[idx]
        end = self._feature_offsets[idx+1]
        features = [self._features[i] for i in range(start, end)]
        quals = self._qualifiers[idx]
        
        return Record(seq, id_=rec_id, features=features, qualifiers=quals)

    def get_qualifiers(self, idx: int) -> list[tuple]:
        """Returns the qualifiers for the record at idx."""
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

    def features_for(self, record_idx: int) -> 'IntervalBatch':
        """Returns the IntervalBatch for a specific record."""
        start = self._feature_offsets[record_idx]
        end = self._feature_offsets[record_idx+1]
        
        intervals = self._features.intervals
        # Slicing IntervalBatch is not yet implemented in your code, 
        # but assuming standard numpy slicing on internal arrays:
        return IntervalBatch(
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

    def get_feature_key(self, idx: int) -> FeatureKey:
        """Returns the kind of the feature at the global index."""
        return self._features.get_key(idx)

    def get_feature_qualifiers(self, idx: int) -> list[tuple]:
        """Returns the qualifiers of the feature at the global index."""
        return self._features.get_qualifiers(idx)


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
