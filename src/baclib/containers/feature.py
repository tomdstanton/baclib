from typing import Generator, Union, MutableSequence, Iterable, ClassVar
from enum import IntEnum, auto

import numpy as np

from baclib.containers import Batch, Batchable
from baclib.containers.seq import Seq
from baclib.core.interval import Interval, IntervalBatch
from baclib.containers.qualifier import QualifierList, QualifierBatch, QualifierType
from baclib.lib.protocols import HasInterval, HasIntervals


# Classes --------------------------------------------------------------------------------------------------------------
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


class Feature(HasInterval, Batchable):
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
    def batch(self) -> type['Batch']: return FeatureBatch
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
    __slots__ = ('_data', '_intervals')

    def __init__(self, features: Iterable['Feature'] = None):
        self._data: list[Feature] = list(features) if features else []
        self._intervals = None

    @property
    def intervals(self) -> 'IntervalBatch':
        if self._intervals is None:
            self._intervals = IntervalBatch.from_features(self._data)
        return self._intervals

    def __getitem__(self, index): return self._data[index]
    def __repr__(self):
        if len(self) > 6:
            return f"[{', '.join(repr(x) for x in self[:3])}, ..., {', '.join(repr(x) for x in self[-3:])}]"
        return repr(self._data)
    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)
    def _flag_dirty(self): self._intervals = None
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
        indices = self.intervals.query(start, end)
        for i in indices: yield self._data[i]


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
    def empty(cls) -> 'FeatureBatch':
        return cls(IntervalBatch.empty(), np.empty(0, dtype=np.int16), QualifierBatch.empty())

    @classmethod
    def build(cls, features: list[Feature]) -> 'FeatureBatch':
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
    def concat(cls, batches: Iterable['FeatureBatch']) -> 'FeatureBatch':
        batches = list(batches)
        if not batches: return cls.empty()
        
        intervals = IntervalBatch.concat([b.intervals for b in batches])
        keys = np.concatenate([b.keys for b in batches])
        qualifiers = QualifierBatch.concat([b._qualifiers for b in batches])
        
        return cls(intervals, keys, qualifiers)

    @property
    def nbytes(self) -> int: return self._intervals.nbytes + self._keys.nbytes + self._qualifiers.nbytes

    def copy(self) -> 'FeatureBatch':
        return self.__class__(self._intervals.copy(), self._keys.copy(), self._qualifiers.copy())

    @property
    def component(self): return Feature

    @property
    def intervals(self) -> IntervalBatch: return self._intervals
    @property
    def keys(self): return self._keys
    def get_key(self, idx: int) -> FeatureKey: return FeatureKey(self._keys[idx])
    def get_qualifiers(self, idx: int) -> list[tuple]: return self._qualifiers[idx]
    def __repr__(self): return f"<FeatureBatch: {len(self)} features>"
    def __len__(self): return len(self._intervals)
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

    @classmethod
    def zeros(cls, n: int) -> 'FeatureBatch':
        """Creates a batch of n 'empty' features (0-length, 0-start)."""
        return cls(
            IntervalBatch.build([Interval(0, 0)] * n),
            np.zeros(n, dtype=np.int16),
            QualifierBatch.zeros(n)
        )

    @classmethod
    def empty(cls) -> 'FeatureBatch':
        return cls(
            IntervalBatch.empty(),
            np.array([], dtype=np.int16),
            QualifierBatch.empty()
        )


# Cache initialisations ------------------------------------------------------------------------------------------------
FeatureKey._init_caches()
