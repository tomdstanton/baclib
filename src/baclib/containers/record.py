from binascii import hexlify
from operator import attrgetter
from typing import Generator, Union, Iterator, MutableSequence, Iterable, ClassVar
from uuid import uuid4
from enum import IntEnum, auto

import numpy as np

from baclib.core.alphabet import Alphabet
from baclib.core.seq import Seq, SeqBatch, SparseSeq
from baclib.core.interval import Interval, IntervalBatch
from baclib.utils import Batch, RaggedBatch
from baclib.utils.resources import RESOURCES, jit
from baclib.utils.protocols import HasInterval, HasIntervals

if RESOURCES.has_module('numba'): 
    from numba import prange
else: 
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
QualifierType = Union[int, float, bytes, bool]


class QualifierList(MutableSequence):
    """
    A list-like container for (key, value) tuples that also supports dictionary-style access.
    Maintains insertion order and allows duplicate keys.
    """
    __slots__ = ('_data',)

    def __init__(self, items: Iterable[tuple[bytes, QualifierType]] = None):
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
    def __repr__(self):
        if len(self) > 6:
            return f"[{', '.join(repr(x) for x in self[:3])}, ..., {', '.join(repr(x) for x in self[-3:])}]"
        return repr(self._data)
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

    def get_all(self, key: bytes) -> list[QualifierType]:
        """Returns all values for a key."""
        return [v for k, v in self._data if k == key]

    def add(self, key: bytes, value: QualifierType):
        """Adds a new key-value pair."""
        self._data.append((key, value))

    def to_dict(self) -> dict[bytes, QualifierType]:
        """Converts the list to a standard dictionary (lossy for duplicates)."""
        return {k: v for k, v in self._data}


class FeatureKey(IntEnum):
    """
    Valid INSDC Feature Table Keys.

    Includes a __str__ method to return the exact INSDC string representation,
    handling special characters (like 5'UTR and D-loop) that are not valid
    Python identifiers.
    """
    # --- Genes, Coding & Transcriptional Products ---
    GENE = auto()
    CDS = auto()
    MRNA = auto()
    TRNA = auto()
    RRNA = auto()
    NCRNA = auto()
    TMRNA = auto()
    PRECURSOR_RNA = auto()
    PRIM_TRANSCRIPT = auto()
    EXON = auto()
    INTRON = auto()
    # Handled specifically in __str__
    FIVE_PRIME_UTR = auto()
    THREE_PRIME_UTR = auto()
    # --- Protein Maturation & Signaling ---
    SIG_PEPTIDE = auto()
    MAT_PEPTIDE = auto()
    PROPEPTIDE = auto()
    TRANSIT_PEPTIDE = auto()
    # --- Immunoglobulin & T-cell Receptor ---
    V_REGION = auto()
    D_SEGMENT = auto()
    J_SEGMENT = auto()
    C_REGION = auto()
    N_REGION = auto()
    S_REGION = auto()
    V_SEGMENT = auto()
    IDNA = auto()
    # --- Genomic Structure & Regulation ---
    REGULATORY = auto()  # Replaces promoter, enhancer, terminator, etc.
    OPERON = auto()
    POLYA_SITE = auto()
    REP_ORIGIN = auto()
    ORIT = auto()
    CENTROMERE = auto()
    TELOMERE = auto()
    MOBILE_ELEMENT = auto()
    REPEAT_REGION = auto()
    # --- Sequence, Variation & Binding ---
    GAP = auto()
    ASSEMBLY_GAP = auto()
    VARIATION = auto()
    MISC_DIFFERENCE = auto()
    MODIFIED_BASE = auto()
    PROTEIN_BIND = auto()
    PRIMER_BIND = auto()
    MISC_BINDING = auto()
    # --- Secondary Structure ---
    STEM_LOOP = auto()
    D_LOOP = auto()  # Handled specifically in __str__
    MISC_STRUCTURE = auto()
    # --- Miscellaneous / Other ---
    SOURCE = auto()
    MISC_FEATURE = auto()
    MISC_RECOMB = auto()
    MISC_RNA = auto()
    OLD_SEQUENCE = auto()
    MOTIF = auto()
    STS = auto()
    UNSURE = auto()

    _STR_CACHE: ClassVar[dict]
    _BYTES_CACHE: ClassVar[dict]
    _FROM_BYTES_CACHE: ClassVar[dict]

    def __str__(self): return self._STR_CACHE[self]
    @property
    def bytes(self) -> bytes: return self._BYTES_CACHE[self]
    @classmethod
    def from_bytes(cls, b: bytes) -> 'FeatureKey': return cls._FROM_BYTES_CACHE.get(b, cls.MISC_FEATURE)
    @classmethod
    def _init_caches(cls):
        cls._STR_CACHE = {}
        cls._BYTES_CACHE = {}
        cls._FROM_BYTES_CACHE = {}
        special = {
            cls.FIVE_PRIME_UTR: "5'UTR",
            cls.THREE_PRIME_UTR: "3'UTR",
            cls.D_LOOP: "D-loop",
            cls.IDNA: "iDNA",
            cls.ORIT: "oriT",
            # Standard INSDC Keys with specific casing
            cls.CDS: "CDS",
            cls.MRNA: "mRNA",
            cls.TRNA: "tRNA",
            cls.RRNA: "rRNA",
            cls.NCRNA: "ncRNA",
            cls.TMRNA: "tmRNA",
            cls.PRECURSOR_RNA: "precursor_RNA",
            cls.MISC_RNA: "misc_RNA",
            cls.STS: "STS",
            cls.V_REGION: "V_region",
            cls.D_SEGMENT: "D_segment",
            cls.J_SEGMENT: "J_segment",
            cls.C_REGION: "C_region",
            cls.N_REGION: "N_region",
            cls.S_REGION: "S_region",
            cls.V_SEGMENT: "V_segment",
            cls.POLYA_SITE: "polyA_site",
            cls.SIG_PEPTIDE: "sig_peptide",
            cls.MAT_PEPTIDE: "mat_peptide",
            cls.TRANSIT_PEPTIDE: "transit_peptide",
            cls.REP_ORIGIN: "rep_origin",
            cls.MOTIF: "motif",
        }
        for k in cls:
            s = special.get(k, k.name.lower())
            cls._STR_CACHE[k] = s
            b = s.encode('ascii')
            cls._BYTES_CACHE[k] = b
            cls._FROM_BYTES_CACHE[b] = k

FeatureKey._init_caches()


class Feature(HasInterval):
    """
    Represents a genomic feature, such as a gene or CDS.
    """
    Key = FeatureKey  # Alias for convenience (e.g. Feature.Key.CDS)
    __slots__ = ('_interval', '_key', '_qualifiers')
    def __init__(self, interval: 'Interval', key: Union[FeatureKey, bytes] = FeatureKey.MISC_FEATURE,
                 qualifiers: Iterable[tuple[bytes, QualifierType]] = None):
        self._interval = interval
        self._key = FeatureKey.from_bytes(key) if isinstance(key, bytes) else key
        self._qualifiers = QualifierList(qualifiers)
        
    @property
    def interval(self) -> 'Interval': return self._interval
    @property
    def key(self) -> FeatureKey: return self._key
    @property
    def qualifiers(self) -> QualifierList: return self._qualifiers

    def __len__(self) -> int: return len(self.interval)
    def __iter__(self): return self.interval.__iter__()
    def __contains__(self, item) -> bool: return self.interval.__contains__(item)
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.interval == other.interval and self.key == other.key and
                    self.qualifiers == other.qualifiers)
        return False
    
    def __repr__(self): return f"Feature({self.key.name}, {self.interval})"
    # Delegate dict-like access to qualifiers, but ensure we use .get() for lookups
    # to avoid ambiguity with integer indices if Feature were to support them.
    def __getitem__(self, item): return self.qualifiers.get(item)
    def __setitem__(self, key: bytes, value: QualifierType): self.qualifiers[key] = value
    def overlap(self, other) -> int: return self.interval.overlap(other)
    def get(self, item, default = None): return self.qualifiers.get(item, default)
    def get_all(self, key: bytes) -> list[QualifierType]: return self.qualifiers.get_all(key)
    def add_qualifier(self, key: bytes, value: QualifierType = True): self.qualifiers.add(key, value)
    def shift(self, x: int, y: int = None) -> 'Feature':
        return Feature(self.interval.shift(x, y), self.key, list(self.qualifiers))
    def reverse_complement(self, parent_length: int) -> 'Feature':
        return Feature(self.interval.reverse_complement(parent_length), self.key, list(self.qualifiers))
    def extract(self, parent_seq: Seq) -> Seq:
        """Extracts the feature's sequence from the parent."""
        sub = parent_seq[self.interval.start:self.interval.end]
        if self.interval.strand == -1: return parent_seq.alphabet.reverse_complement(sub)
        return sub
    def copy(self) -> 'Feature':
        """Creates a shallow copy of the feature (qualifiers are copied)."""
        return Feature(self.interval, self.key, list(self.qualifiers))


class FeatureList(MutableSequence, HasIntervals):
    """
    A list-like object that manages genomic features and their spatial index.
    
    It automatically invalidates the spatial index when features are modified.
    """
    __slots__ = ('_data', '_interval_batch')

    def __init__(self, features: Iterable['Feature'] = None):
        self._data: list[Feature] = list(features) if features else []
        self._interval_batch = None

    def __getitem__(self, index): return self._data[index]
    def __repr__(self):
        if len(self) > 6:
            return f"[{', '.join(repr(x) for x in self[:3])}, ..., {', '.join(repr(x) for x in self[-3:])}]"
        return repr(self._data)
    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)
    def _flag_dirty(self): self._interval_batch = None
    @property
    def intervals(self) -> 'IntervalBatch':
        """Alias for interval_batch to satisfy HasIntervals protocol."""
        return self.interval_batch

    @property
    def interval_batch(self) -> 'IntervalBatch':
        if self._interval_batch is None:
            self._interval_batch = IntervalBatch.from_features(self._data)
        return self._interval_batch

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
        Delegates to the IntervalBatch for robust and fast querying.
        """
        # self.interval_batch ensures the index is built and up-to-date
        indices = self.interval_batch.query(start, end)
        for i in indices: yield self._data[i]


class Record(HasIntervals):
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
    def __eq__(self, other) -> bool: return self.id == other.id if isinstance(other, Record) else False

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
    def intervals(self) -> 'IntervalBatch': return self.features.interval_batch
    @property
    def interval_batch(self) -> 'IntervalBatch': return self.features.interval_batch

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


class QualifierBatch(RaggedBatch):
    """
    A batch of qualifier lists, stored in a columnar format (Structure-of-Arrays).
    """
    __slots__ = ('_key_vocab', '_key_ids', '_values')

    def __init__(self, key_vocab, key_ids, values, offsets):
        super().__init__(offsets)
        self._key_vocab = key_vocab
        self._key_ids = key_ids
        self._values = values

    @classmethod
    def from_qualifiers(cls, qualifiers_list: Iterable[Iterable[tuple[bytes, QualifierType]]]) -> 'QualifierBatch':
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

    @classmethod
    def empty_batch(cls) -> 'QualifierBatch':
        return cls(
            np.array([], dtype=object),
            np.array([], dtype=np.int32),
            np.array([], dtype=object),
            np.array([0], dtype=np.int32)
        )

    def empty(self) -> 'QualifierBatch':
        return self.empty_batch()

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            start = self._offsets[item]
            end = self._offsets[item+1]
            if start == end: return []
            keys = self._key_vocab[self._key_ids[start:end]]
            values = self._values[start:end]
            return list(zip(keys, values))
        
        if isinstance(item, slice):
            new_offsets, val_start, val_end = self._get_slice_info(item)
            new_key_ids = self._key_ids[val_start:val_end]
            new_values = self._values[val_start:val_end]
            
            # Create new batch (Zero Copy views where possible)
            obj = object.__new__(QualifierBatch)
            obj._key_vocab = self._key_vocab
            obj._key_ids = new_key_ids
            obj._values = new_values
            obj._offsets = new_offsets
            return obj
            
        raise TypeError(f"Invalid index type: {type(item)}")


class FeatureBatch(Batch, HasIntervals):
    """
    A batch of Features, stored in a columnar format.
    """
    __slots__ = ('_intervals', '_keys', '_qualifiers')

    def __init__(self, intervals: IntervalBatch, keys: np.ndarray, qualifiers: QualifierBatch):
        self._intervals = intervals
        self._keys = keys
        self._qualifiers = qualifiers

    @classmethod
    def from_features(cls, features: list[Feature]) -> 'FeatureBatch':
        n = len(features)
        
        # Manual extraction to ensure sort=False (IntervalBatch.from_features sorts by default)
        starts = np.empty(n, dtype=np.int32)
        ends = np.empty(n, dtype=np.int32)
        strands = np.empty(n, dtype=np.int32)
        keys = np.empty(n, dtype=np.int16)
        
        for i, f in enumerate(features):
            iv = f.interval
            starts[i] = iv.start
            ends[i] = iv.end
            strands[i] = iv.strand
            keys[i] = f.key.value
            
        intervals = IntervalBatch(starts, ends, strands, sort=False)
        
        # Generator to avoid creating intermediate list of qualifier lists
        qualifiers = QualifierBatch.from_qualifiers((f.qualifiers for f in features))
        return cls(intervals, keys, qualifiers)

    @classmethod
    def empty_batch(cls) -> 'FeatureBatch':
        return cls(
            IntervalBatch(),
            np.array([], dtype=np.int16),
            QualifierBatch.empty_batch()
        )

    def empty(self) -> 'FeatureBatch':
        return self.empty_batch()

    def __repr__(self): return f"<FeatureBatch: {len(self)} features>"

    def __len__(self): return len(self._intervals)

    @property
    def intervals(self) -> IntervalBatch: return self._intervals
    @property
    def keys(self): return self._keys
    
    def get_key(self, idx: int) -> FeatureKey:
        return FeatureKey(self._keys[idx])
        
    def get_qualifiers(self, idx: int) -> list[tuple]:
        return self._qualifiers[idx]

    def __getitem__(self, item) -> Union['Feature', 'FeatureBatch']:
        """
        Reconstructs a Feature object from the batch.
        """
        if isinstance(item, (int, np.integer)):
            interval = Interval(
                self._intervals.starts[item],
                self._intervals.ends[item],
                self._intervals.strands[item]
            )
            return Feature(interval, self.get_key(item), self._qualifiers[item])
        
        if isinstance(item, slice):
            # Slice components
            new_intervals = self._intervals[item]
            new_keys = self._keys[item]
            new_quals = self._qualifiers[item]
            
            obj = object.__new__(FeatureBatch)
            obj._intervals = new_intervals
            obj._keys = new_keys
            obj._qualifiers = new_quals
            return obj
            
        raise TypeError(f"Invalid index type: {type(item)}")


class RecordBatch(Batch, HasIntervals):
    """
    A high-performance, read-only container for a collection of Records.
    
    Uses Structure-of-Arrays (SoA) layout to store sequences and features contiguously,
    enabling vectorized operations and Numba compatibility.
    """
    __slots__ = ('_seqs', '_ids', '_features', '_feature_rec_indices', '_feature_offsets', '_qualifiers')

    def __init__(self, records: Iterable[Record]):
        # 1. Materialize list to avoid multiple passes if iterator
        recs = list(records)

        # 2. Batch Sequences
        self._seqs = recs[0].seq.alphabet.batch_from((r.seq for r in recs))

        # 3. Batch IDs (Fixed length numpy array for speed, or object array)
        # We use object array of bytes to handle variable length IDs safely
        self._ids = np.array([r.id for r in recs], dtype=object)
        
        # 4. Batch Qualifiers
        self._qualifiers = QualifierBatch.from_qualifiers((r.qualifiers for r in recs))

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
        obj._qualifiers = QualifierBatch.from_qualifiers((r.qualifiers for r in records))
        obj._build_features(records)
        return obj

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

    def empty(self) -> 'RecordBatch':
        # Use __new__ to bypass __init__ logic which expects iterable of Records
        obj = self.__class__.__new__(self.__class__)
        obj._seqs = self._seqs.empty()
        obj._ids = np.array([], dtype=object)
        obj._qualifiers = self._qualifiers.empty()
        obj._features = self._features.empty()
        obj._feature_offsets = np.array([0], dtype=np.int32)
        obj._feature_rec_indices = np.array([], dtype=np.int32)
        return obj

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


class MutationEffect(IntEnum):
    """
    Enum for efficient integer storage of mutation effects.
    """
    UNKNOWN = auto()
    SYNONYMOUS = auto()
    MISSENSE = auto()
    NONSENSE = auto()
    FRAMESHIFT = auto()
    INTERGENIC = auto()
    NON_CODING = auto()


class Mutation(Feature):
    """
    Represents a discrete change: SNP, Insertion, or Deletion.

    Attributes:
        interval (Interval): The location on the REFERENCE sequence.
        ref_seq (Seq): The reference sequence content.
        alt_seq (Seq): The alternative sequence content.
        effect (MutationEffect): The predicted functional impact.
    """
    __slots__ = ('ref_seq', 'alt_seq', 'effect', 'aa_change')

    def __init__(self, interval: Interval, ref_seq: Seq, alt_seq: Seq,
                 effect: MutationEffect = MutationEffect.UNKNOWN,
                 aa_change: bytes = None):
        super().__init__(interval, key=b'mutation')
        self.ref_seq = ref_seq
        self.alt_seq = alt_seq
        self.effect = effect
        self.aa_change = aa_change

    def __repr__(self):
        # VCF-style notation: Pos Ref>Alt (1-based for display)
        return f"{self.interval.start + 1}:{self.ref_seq}>{self.alt_seq}"

    @property
    def is_snp(self): return len(self.ref_seq) == 1 and len(self.alt_seq) == 1
    @property
    def is_indel(self): return len(self.ref_seq) != len(self.alt_seq)
    @property
    def diff(self) -> int:
        """Returns the net change in sequence length (Alt - Ref)."""
        return len(self.alt_seq) - len(self.ref_seq)


class MutationBatch(Batch, HasIntervals):
    """
    Efficient storage for a collection of mutations.
    Can be used to reconstruct sequences via SparseSeq.
    """
    __slots__ = ('_intervals', '_ref_seqs', '_alt_seqs', '_effects', '_aa_changes')

    def __init__(self, intervals: IntervalBatch, ref_seqs: SeqBatch, alt_seqs: SeqBatch,
                 effects: np.ndarray = None, aa_changes: np.ndarray = None):
        self._intervals = intervals
        self._ref_seqs = ref_seqs
        self._alt_seqs = alt_seqs
        n = len(intervals)
        self._effects = effects if effects is not None else np.zeros(n, dtype=np.uint8)
        self._aa_changes = aa_changes if aa_changes is not None else np.full(n, None, dtype=object)

    def empty(self) -> 'MutationBatch':
        return MutationBatch(
            self._intervals.empty(),
            self._ref_seqs.empty(),
            self._alt_seqs.empty()
        )

    def __repr__(self): return f"<MutationBatch: {len(self)} mutations>"

    def __len__(self): return len(self._intervals)
    
    @property
    def intervals(self) -> IntervalBatch: return self._intervals
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            interval = self._intervals[item]
            ref = self._ref_seqs[item]
            alt = self._alt_seqs[item]
            eff = MutationEffect(self._effects[item])
            aa = self._aa_changes[item]
            return Mutation(interval, ref, alt, eff, aa)
        
        if isinstance(item, slice):
            return MutationBatch(
                self._intervals[item],
                self._ref_seqs[item],
                self._alt_seqs[item],
                self._effects[item],
                self._aa_changes[item]
            )
        raise TypeError(f"Invalid index type: {type(item)}")

    def apply_to(self, reference: Seq) -> SparseSeq:
        """
        Applies the mutations to a reference sequence, returning a SparseSeq.
        """
        # We can iterate self because __iter__ yields Mutation objects
        # which SparseSeq accepts.
        return SparseSeq(reference, self)


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
