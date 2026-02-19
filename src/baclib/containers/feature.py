"""Containers for genomic features (genes, CDS, etc.) with interval-based coordinates and batch support."""
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

    Each member maps to a string representation via ``__str__`` and a bytes
    representation via the ``bytes`` property. Special characters
    (like ``5'UTR`` and ``D-loop``) that are not valid Python identifiers
    are handled automatically.

    Examples:
        >>> FeatureKey.CDS
        <FeatureKey.CDS: 2>
        >>> str(FeatureKey.CDS)
        'CDS'
        >>> FeatureKey.from_bytes(b'CDS')
        <FeatureKey.CDS: 2>
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
    def bytes(self) -> bytes:
        """Returns the INSDC byte-string representation of this key.

        Returns:
            The key name as ASCII bytes (e.g. ``b'CDS'``, ``b"5'UTR"``).
        """
        return self._BYTES_CACHE[self]

    @classmethod
    def from_bytes(cls, b: bytes) -> 'FeatureKey':
        """Looks up a FeatureKey by its byte-string representation.

        Falls back to ``MISC_FEATURE`` for unrecognised keys.

        Args:
            b: INSDC key as bytes (e.g. ``b'CDS'``).

        Returns:
            The matching FeatureKey member.

        Examples:
            >>> FeatureKey.from_bytes(b'CDS')
            <FeatureKey.CDS: 2>
            >>> FeatureKey.from_bytes(b'unknown')
            <FeatureKey.MISC_FEATURE: ...>
        """
        return cls._FROM_BYTES_CACHE.get(b, cls.MISC_FEATURE)

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
    A single genomic feature with an interval, type key, and qualifiers.

    Features represent annotated regions of a sequence (genes, CDS, regulatory
    elements, etc.) and carry metadata as key-value qualifier pairs.

    Args:
        interval: Genomic coordinates and strand.
        key: The INSDC feature type. Accepts a ``FeatureKey`` enum or raw bytes.
        qualifiers: Optional iterable of ``(key, value)`` qualifier tuples.

    Examples:
        >>> feat = Feature(Interval(100, 500, 1), FeatureKey.CDS,
        ...                [(b'gene', b'dnaA'), (b'product', b'initiator')])
        >>> feat.key
        <FeatureKey.CDS: 2>
        >>> feat[b'gene']
        b'dnaA'
    """
    Key = FeatureKey  # Alias for convenience (e.g. Feature.Key.CDS)
    __slots__ = ('_interval', '_key', '_qualifiers')
    def __init__(self, interval: 'Interval', key: Union[FeatureKey, bytes] = FeatureKey.MISC_FEATURE,
                 qualifiers: Iterable[tuple[bytes, QualifierType]] = None):
        self._interval = interval
        self._key = FeatureKey.from_bytes(key) if isinstance(key, bytes) else key
        self._qualifiers = QualifierList(qualifiers)

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``FeatureBatch`` class.
        """
        return FeatureBatch

    @property
    def interval(self) -> 'Interval':
        """Returns the genomic interval (start, end, strand).

        Returns:
            The feature's ``Interval``.
        """
        return self._interval

    @property
    def key(self) -> FeatureKey:
        """Returns the INSDC feature type key.

        Returns:
            A ``FeatureKey`` enum member.
        """
        return self._key

    @property
    def qualifiers(self) -> QualifierList:
        """Returns the qualifier list for this feature.

        Returns:
            A ``QualifierList`` of ``(key, value)`` tuples.
        """
        return self._qualifiers

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

    def overlap(self, other) -> int:
        """Returns the number of overlapping bases with another interval.

        Args:
            other: An ``Interval``, ``Feature``, or other ``HasInterval``.

        Returns:
            Number of overlapping bases (0 if none).
        """
        return self.interval.overlap(other)

    def get(self, item: bytes, default=None) -> QualifierType:
        """Returns the first qualifier value for a key, or *default*.

        Args:
            item: The qualifier key to look up.
            default: Value to return if key is absent.

        Returns:
            The first matching value, or *default*.

        Examples:
            >>> feat.get(b'gene')
            b'dnaA'
        """
        return self.qualifiers.get(item, default)

    def get_all(self, key: bytes) -> list[QualifierType]:
        """Returns all qualifier values for a key.

        Args:
            key: The qualifier key to look up.

        Returns:
            A list of all matching values.
        """
        return self.qualifiers.get_all(key)

    def add_qualifier(self, key: bytes, value: QualifierType = True):
        """Appends a qualifier to this feature.

        Args:
            key: The qualifier key.
            value: The qualifier value (defaults to ``True`` for flag qualifiers).

        Examples:
            >>> feat.add_qualifier(b'pseudo')
            >>> feat[b'pseudo']
            True
        """
        self.qualifiers.add(key, value)

    def shift(self, x: int, y: int = None) -> 'Feature':
        """Returns a new Feature with shifted coordinates.

        Args:
            x: Offset to add to the start (and end, unless *y* is given).
            y: Optional separate offset for the end.

        Returns:
            A new ``Feature`` with adjusted interval.
        """
        return Feature(self.interval.shift(x, y), self.key, list(self.qualifiers))

    def reverse_complement(self, parent_length: int) -> 'Feature':
        """Returns a new Feature with reverse-complemented coordinates.

        Args:
            parent_length: Length of the parent sequence.

        Returns:
            A new ``Feature`` on the opposite strand.
        """
        return Feature(self.interval.reverse_complement(parent_length), self.key, list(self.qualifiers))

    def extract(self, parent_seq: Seq) -> Seq:
        """Extracts the feature's sequence from the parent, respecting strand.

        Args:
            parent_seq: The full parent ``Seq`` to slice from.

        Returns:
            The extracted subsequence (reverse-complemented if on minus strand).

        Examples:
            >>> seq = Alphabet.DNA.from_bytes(b'ATGCGATCGA')
            >>> feat = Feature(Interval(0, 3, 1), FeatureKey.CDS)
            >>> feat.extract(seq)  # returns Seq('ATG')
        """
        sub = parent_seq[self.interval.start:self.interval.end]
        if self.interval.strand == -1: return parent_seq.alphabet.reverse_complement(sub)
        return sub

    def copy(self) -> 'Feature':
        """Creates a shallow copy of the feature (qualifiers are copied).

        Returns:
            A new ``Feature`` with the same interval, key, and copied qualifiers.
        """
        return Feature(self.interval, self.key, list(self.qualifiers))


class FeatureList(MutableSequence, HasIntervals):
    """
    A mutable list of genomic features with a lazily-built spatial index.

    The spatial index is automatically invalidated when features are added,
    removed, or reordered, and rebuilt on next access via the ``intervals``
    property.

    Args:
        features: Optional iterable of ``Feature`` objects.

    Examples:
        >>> fl = FeatureList([feat1, feat2])
        >>> fl.append(feat3)
        >>> overlapping = list(fl.get_overlapping(100, 500))
    """
    __slots__ = ('_data', '_intervals')

    def __init__(self, features: Iterable['Feature'] = None):
        self._data: list[Feature] = list(features) if features else []
        self._intervals = None

    @property
    def intervals(self) -> 'IntervalBatch':
        """Returns the spatial index over all feature intervals.

        Lazily built on first access and invalidated when the list is modified.

        Returns:
            An ``IntervalBatch`` covering all features.
        """
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

    def insert(self, index: int, value: 'Feature'):
        """Inserts a feature at the given index.

        Args:
            index: Position to insert at.
            value: The ``Feature`` to insert.
        """
        self._data.insert(index, value)
        self._flag_dirty()

    def extend(self, values: Iterable['Feature']):
        """Appends multiple features to the list.

        Args:
            values: An iterable of ``Feature`` objects.
        """
        self._data.extend(values)
        self._flag_dirty()

    def reverse(self):
        """Reverses the feature list in place."""
        self._data.reverse()
        self._flag_dirty()

    def sort(self, key=None, reverse: bool = False):
        """Sorts the feature list in place.

        Args:
            key: Optional sort key function.
            reverse: If ``True``, sort in descending order.
        """
        self._data.sort(key=key, reverse=reverse)
        self._flag_dirty()

    def get_overlapping(self, start: int, end: int) -> Generator['Feature', None, None]:
        """Yields features overlapping the interval ``[start, end)``.

        Uses the spatial index for fast querying.

        Args:
            start: Start coordinate (inclusive).
            end: End coordinate (exclusive).

        Yields:
            ``Feature`` objects whose intervals overlap the query range.

        Examples:
            >>> for feat in fl.get_overlapping(100, 500):
            ...     print(feat.key)
        """
        # self.intervals ensures the index is built and up-to-date
        indices = self.intervals.query(start, end)
        for i in indices: yield self._data[i]


class FeatureBatch(Batch, HasIntervals):
    """
    A columnar batch of features for efficient bulk operations.

    Stores intervals, keys, and qualifiers in separate numpy arrays,
    enabling vectorized filtering and slicing.

    Args:
        intervals: An ``IntervalBatch`` of feature coordinates.
        keys: An ``int16`` numpy array of ``FeatureKey`` values.
        qualifiers: A ``QualifierBatch`` of per-feature qualifiers.

    Examples:
        >>> batch = FeatureBatch.build([feat1, feat2, feat3])
        >>> len(batch)
        3
        >>> batch[0]  # reconstructs a Feature
        Feature(CDS, Interval(100, 500, +))
    """
    __slots__ = ('_intervals', '_keys', '_qualifiers')

    def __init__(self, intervals: IntervalBatch, keys: np.ndarray, qualifiers: QualifierBatch):
        self._intervals = intervals
        self._keys = keys
        self._qualifiers = qualifiers

    @classmethod
    def empty(cls) -> 'FeatureBatch':
        """Creates an empty FeatureBatch with zero features.

        Returns:
            An empty ``FeatureBatch``.
        """
        return cls(IntervalBatch.empty(), np.empty(0, dtype=np.int16), QualifierBatch.empty())

    @classmethod
    def build(cls, features: list[Feature]) -> 'FeatureBatch':
        """Constructs a FeatureBatch from a list of Feature objects.

        Args:
            features: List of ``Feature`` objects to batch.

        Returns:
            A new ``FeatureBatch``.

        Examples:
            >>> batch = FeatureBatch.build([feat1, feat2])
            >>> len(batch)
            2
        """
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
        qualifiers = QualifierBatch.build(f.qualifiers for f in features)
        return cls(intervals, keys, qualifiers)

    @classmethod
    def concat(cls, batches: Iterable['FeatureBatch']) -> 'FeatureBatch':
        """Concatenates multiple FeatureBatch objects into one.

        Args:
            batches: An iterable of ``FeatureBatch`` objects.

        Returns:
            A single concatenated ``FeatureBatch``.

        Examples:
            >>> combined = FeatureBatch.concat([batch_a, batch_b])
        """
        batches = list(batches)
        if not batches: return cls.empty()
        
        intervals = IntervalBatch.concat([b.intervals for b in batches])
        keys = np.concatenate([b.keys for b in batches])
        qualifiers = QualifierBatch.concat([b._qualifiers for b in batches])
        
        return cls(intervals, keys, qualifiers)

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by intervals, keys, and qualifiers.
        """
        return self._intervals.nbytes + self._keys.nbytes + self._qualifiers.nbytes

    def copy(self) -> 'FeatureBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``FeatureBatch`` with copied arrays.
        """
        return self.__class__(self._intervals.copy(), self._keys.copy(), self._qualifiers.copy())

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``Feature`` class.
        """
        return Feature

    @property
    def intervals(self) -> IntervalBatch:
        """Returns the interval array for all features.

        Returns:
            An ``IntervalBatch`` of feature coordinates.
        """
        return self._intervals

    @property
    def keys(self) -> np.ndarray:
        """Returns the raw ``int16`` array of feature key values.

        Returns:
            A numpy array of ``FeatureKey`` integer values.
        """
        return self._keys

    def get_key(self, idx: int) -> FeatureKey:
        """Returns the FeatureKey for the feature at *idx*.

        Args:
            idx: Feature index.

        Returns:
            The ``FeatureKey`` enum member.

        Examples:
            >>> batch.get_key(0)
            <FeatureKey.CDS: 2>
        """
        return FeatureKey(self._keys[idx])

    def get_qualifiers(self, idx: int) -> list[tuple]:
        """Returns the qualifier list for the feature at *idx*.

        Args:
            idx: Feature index.

        Returns:
            A list of ``(key, value)`` tuples.

        Examples:
            >>> batch.get_qualifiers(0)
            [(b'gene', b'dnaA'), (b'product', b'initiator')]
        """
        return self._qualifiers[idx]

    def __repr__(self): return f"<FeatureBatch: {len(self)} features>"
    def __len__(self): return len(self._intervals)
    def __getitem__(self, item) -> Union['Feature', 'FeatureBatch']:
        """Reconstructs a Feature or slices a sub-batch.

        Args:
            item: Integer index (returns ``Feature``) or slice (returns ``FeatureBatch``).

        Returns:
            A single ``Feature`` or a sliced ``FeatureBatch``.
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
    def random(cls, n: int, rng: np.random.Generator = None, length: int = None, min_len: int = 1, max_len: int = 1000,
               min_start: int = 0, max_start: int = 1_000_000) -> 'FeatureBatch':
        """Creates a batch of *n* random features for testing.

        Args:
            n: Number of features to generate.
            rng: Random number generator (optional).
            length: Fixed length for all features (optional).
            min_len: Minimum feature length (default 1).
            max_len: Maximum feature length (default 1000).
            min_start: Minimum start coordinate (default 0).
            max_start: Maximum start coordinate (default 1,000,000).

        Returns:
            A new ``FeatureBatch`` with random keys and intervals.

        Examples:
            >>> batch = FeatureBatch.random(10, rng=np.random.default_rng(42))
            >>> len(batch)
            10
        """
        intervals = IntervalBatch.random(n, rng, length, min_len, max_len, min_start, max_start)
        if rng is None: rng = np.random.default_rng()
        
        # Select random keys from available FeatureKey values
        valid_keys = np.array([k.value for k in FeatureKey], dtype=np.int16)
        keys = rng.choice(valid_keys, size=n)
        
        qualifiers = QualifierBatch.zeros(n)
        
        return cls(intervals, keys, qualifiers)

    @classmethod
    def zeros(cls, n: int) -> 'FeatureBatch':
        """Creates a batch of *n* zero-length placeholder features.

        Args:
            n: Number of features to create.

        Returns:
            A ``FeatureBatch`` where every feature has interval ``(0, 0)``
            and key ``0``.

        Examples:
            >>> batch = FeatureBatch.zeros(3)
            >>> len(batch)
            3
        """
        return cls(
            IntervalBatch.build([Interval(0, 0)] * n),
            np.zeros(n, dtype=np.int16),
            QualifierBatch.zeros(n)
        )


# Cache initialisations ------------------------------------------------------------------------------------------------
FeatureKey._init_caches()
