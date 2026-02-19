"""Containers for sequence motifs (PSSMs), background models, and motif scan hits."""
from typing import Union, Optional, Iterable, Any

import numpy as np

from baclib.core.interval import Interval
from baclib.core.alphabet import Alphabet
from baclib.containers import Batch
from baclib.containers.seq import Seq, SeqBatch
from baclib.containers.record import Feature, FeatureKey
from baclib.lib.resources import jit, RESOURCES
from baclib.lib.protocols import HasAlphabet


# Classes --------------------------------------------------------------------------------------------------------------
class Background(HasAlphabet):
    """
    Represents the background nucleotide frequencies.

    Used to calculate log-odds scores for motifs. Can be uniform or estimated
    from sequences or counts.

    Attributes:
        alphabet (Alphabet): The alphabet used.
        data (np.ndarray): Array of probabilities summing to 1.0.

    Examples:
        >>> bg = Background.uniform(Alphabet.DNA)
        >>> bg.data
        array([0.25, 0.25, 0.25, 0.25], dtype=float32)
    """
    __slots__ = ('_alphabet', '_data')
    _DEFAULT_ALPHABET = Alphabet.DNA
    _DTYPE = np.float32

    def __init__(self):
        self._data = None
        self._alphabet = None

    @property
    def alphabet(self) -> Alphabet:
        """Returns the alphabet.

        Returns:
            The ``Alphabet`` object.
        """
        return self._alphabet

    @property
    def data(self) -> np.ndarray:
        """Returns the background probabilities.

        Returns:
            A read-only numpy array of floats.
        """
        return self._data

    @classmethod
    def from_counts(cls, counts: np.ndarray, alphabet: Alphabet = None) -> 'Background':
        """Creates a Background model from raw counts.

        Args:
            counts: Array of counts corresponding to alphabet symbols.
            alphabet: The ``Alphabet`` (defaults to DNA).

        Returns:
            A new ``Background`` instance.
        """
        alphabet = alphabet or cls._DEFAULT_ALPHABET
        assert len(counts) == len(alphabet)
        new = cls()
        new._alphabet = alphabet
        new._data = counts / np.sum(counts, dtype=cls._DTYPE)
        new._data.flags.writeable = False
        return new

    @classmethod
    def from_seq(cls, seq: Union[Seq, SeqBatch]) -> 'Background':
        """Estimates background frequencies from a sequence or batch.

        Args:
            seq: A ``Seq`` or ``SeqBatch``.

        Returns:
            A new ``Background`` instance.
        """
        return cls.from_counts(np.bincount(seq.encoded), seq.alphabet)

    @classmethod
    def uniform(cls, alphabet: Alphabet = None) -> 'Background':
        """Creates a uniform Background model (equal probability for all symbols).

        Args:
            alphabet: The ``Alphabet`` (defaults to DNA).

        Returns:
            A new ``Background`` instance.
        """
        alphabet = alphabet or cls._DEFAULT_ALPHABET
        new = cls()
        new._alphabet = alphabet
        n = len(alphabet)
        new._data = np.ones(n, dtype=cls._DTYPE) / n
        new._data.flags.writeable = False
        return new

    def __array__(self, dtype=None) -> np.ndarray:
        return self._data if dtype is None else self._data.astype(dtype)


class Motif:
    """
    Position Specific Scoring Matrix (PSSM) for sequence motifs.

    Represents a motif model with log-odds scores relative to a background.
    Supports P-value calculation using discretized dynamic programming.

    Attributes:
        name (bytes): Motif name.
        background (Background): Background model used for scoring.
        pssm (np.ndarray): The log-odds scoring matrix (Bases x Positions).

    Examples:
        >>> m = Motif.sigma70_35()
        >>> len(m)
        6
    """
    _DTYPE = np.float32
    _DEFAULT_GRANULARITY = 1000
    _DEFAULT_PSEUDOCOUNT = 0.1
    __slots__ = ('_name', '_background', '_count', '_discrete', '_frequency', '_scoring', '_weight',
                 '_granularity', '_pseudocount', '_min_score', '_max_score', '_pdf', '_max_suffix', '_max_suffix_rc')

    def __init__(self, name: bytes, scores: np.ndarray, background: Background, granularity: int = _DEFAULT_GRANULARITY,
                 pseudocount: float = _DEFAULT_PSEUDOCOUNT, counts: np.ndarray = None, frequencies: np.ndarray = None,
                 weights: np.ndarray = None):
        self._name: bytes = name
        self._background: Background = background
        self._granularity: int = granularity
        self._pseudocount: float = pseudocount
        self._count: Optional[np.ndarray] = counts  # A matrix storing symbol occurrences at each position.
        self._frequency: Optional[np.ndarray] = frequencies  # A matrix storing symbol frequencies at each position.
        self._weight: Optional[
            np.ndarray] = weights  # A matrix storing odds ratio of symbol occurrences at each position.
        self._scoring: np.ndarray = scores  # A matrix storing log-odds ratio of symbol occurrences at each position.
        self._discrete = np.round(self._scoring * self._granularity).astype(
            np.int32)  # A position-specific scoring matrix discretized over u8::MIN..u8::MAX.
        self._min_score = self._discrete.min(axis=0).sum()
        self._max_score = self._discrete.max(axis=0).sum()
        self._pdf = None
        self._max_suffix = _calc_max_suffix(self._scoring)
        self._max_suffix_rc = _calc_max_suffix(self.pssm_rc) if self.pssm_rc is not None else None
        # Enforce immutability for thread safety
        for array in [self._count, self._frequency, self._weight, self._scoring, self._discrete]:
            if array is not None: array.flags.writeable = False

    def __repr__(self):
        return f"<Motif: {self.name.decode(errors='ignore')}, len={len(self)}>"

    def __len__(self) -> int:
        return len(self._scoring.shape[1])

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``MotifBatch`` class.
        """
        return MotifBatch

    @property
    def name(self) -> bytes:
        """Returns the motif name as bytes.

        Returns:
            Motif name.
        """
        return self._name

    @property
    def background(self) -> Background:
        """Returns the background model.

        Returns:
            The ``Background`` object.
        """
        return self._background

    @property
    def pssm(self) -> np.ndarray:
        """Returns the log-odds scoring matrix.

        Returns:
            A (Bases, Positions) numpy array.
        """
        return self._scoring

    @property
    def pssm_rc(self) -> Optional[np.ndarray]:
        """Returns the reverse complement PSSM.

        Returns:
            A (Bases, Positions) numpy array or ``None`` if alphabet has no complement.
        """
        comp = self.background.alphabet.complement
        if comp is None: return None
        return np.ascontiguousarray(self._scoring[comp, ::-1])

    @classmethod
    def from_counts(cls, name: bytes, counts: np.ndarray, background: Background,
                    granularity: int = _DEFAULT_GRANULARITY, pseudocount: float = _DEFAULT_PSEUDOCOUNT) -> 'Motif':
        """Creates a Motif from a count matrix (Rows=Bases, Cols=Positions).

        Args:
            name: Motif name.
            counts: Matrix of observation counts.
            background: Background model.
            granularity: Discretization factor for P-value calculation.
            pseudocount: Pseudocount added to avoid zero probabilities.

        Returns:
            A new ``Motif``.
        """
        adjusted = counts + pseudocount
        frequencies = adjusted / adjusted.sum(axis=0)
        return cls.from_frequencies(name, frequencies, background, granularity, pseudocount, counts)

    @classmethod
    def from_frequencies(cls, name: bytes, frequencies: np.ndarray, background: Background,
                         granularity: int = _DEFAULT_GRANULARITY, pseudocount: float = _DEFAULT_PSEUDOCOUNT,
                         counts: np.ndarray = None) -> 'Motif':
        """Creates a Motif from a frequency matrix.

        Args:
            name: Motif name.
            frequencies: Matrix of probabilities (columns sum to 1).
            background: Background model.
            granularity: Discretization factor for P-value calculation.
            pseudocount: Pseudocount used in derivation (optional tracking).
            counts: Original counts (optional tracking).

        Returns:
            A new ``Motif``.
        """
        weights = frequencies / background.data[:, None]
        return cls.from_weights(name, weights, background, granularity, pseudocount, counts, frequencies)

    @classmethod
    def from_weights(cls, name: bytes, weights: np.ndarray, background: Background,
                     granularity: int = _DEFAULT_GRANULARITY, pseudocount: float = _DEFAULT_PSEUDOCOUNT,
                     counts: np.ndarray = None,
                     frequencies: np.ndarray = None) -> 'Motif':
        """Creates a Motif from a weight matrix (odds ratios).

        Args:
            name: Motif name.
            weights: Matrix of odds ratios.
            background: Background model.

        Returns:
            A new ``Motif``.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.log2(weights, dtype=cls._DTYPE)

        # Clamp lower bound to -100.0 bits to prevent:
        # 1. -inf causing NaN/Errors in integer conversion
        # 2. Massive DP tables in p-value calculation
        min_score = -100.0
        scores[scores < min_score] = min_score
        scores[np.isnan(scores)] = min_score

        return cls.from_scores(name, scores, background, granularity, pseudocount, counts, frequencies, weights)

    @classmethod
    def from_scores(cls, name: bytes, scores: np.ndarray, background: Background,
                    granularity: int = _DEFAULT_GRANULARITY, pseudocount: float = _DEFAULT_PSEUDOCOUNT,
                    counts: np.ndarray = None, frequencies: np.ndarray = None, weights: np.ndarray = None) -> 'Motif':
        """Creates a Motif directly from a log-odds score matrix.

        Args:
            name: Motif name.
            scores: Log-odds score matrix.
            background: Background model.

        Returns:
            A new ``Motif``.
        """
        return cls(name, scores, background, granularity, pseudocount, counts, frequencies, weights)

    def pdf(self) -> np.ndarray:
        """Computes the probability density function of scores.

        Uses dynamic programming to compute the exact distribution of
        discretized scores.

        Returns:
            An array where index corresponds to (score - min_score).
        """
        if self._pdf is None:
            bg = self._background.data.astype(np.float64)
            self._pdf = _score_distribution_kernel(self._discrete, bg, self._min_score, self._max_score)
            self._pdf.flags.writeable = False
        return self._pdf

    def get_score(self, p_value: float) -> float:
        """Calculates the score threshold for a given P-value.

        Args:
            p_value: Target P-value (e.g. 1e-4).

        Returns:
            The score threshold.
        """
        # Calculate CCDF (Survival Function) to find P-value
        # We walk backwards from the max score until sum(probs) >= p_value
        cumulative = 0.0
        threshold_int = self._min_score
        pdf = self.pdf()
        for s in range(len(pdf) - 1, -1, -1):
            cumulative += pdf[s]
            if cumulative > p_value:
                threshold_int = s + 1 + self._min_score
                break
        if threshold_int > self._max_score: threshold_int = self._max_score
        return threshold_int / self._granularity

    def get_pvalue(self, score: float) -> float:
        """Calculates the P-value for a given score.

        Args:
            score: The raw score.

        Returns:
            The probability of observing a score >= *score* by chance.
        """
        target_int = int(round(score * self._granularity))
        idx = target_int - self._min_score
        if idx < 0: return 1.0
        pdf = self.pdf()
        if idx >= len(pdf): return 0.0
        return pdf[idx:].sum()

    def scan(self, seqs: Union[Seq, SeqBatch], pvalue_threshold: float = 1e-4) -> 'MotifHitBatch':
        """Scans sequences for this motif. Use ``MotifScanner`` for batch scanning.

        Raises:
            TypeError: Direct scanning not supported here.
        """
        raise TypeError("Use baclib.engines.motif.MotifScanner(motif).scan(seqs) instead.")

    @classmethod
    def sigma70_35(cls, bg: Background = None) -> 'Motif':
        """Returns a built-in Sigma70 -35 box motif (E. coli)."""
        # -35 Motif Consensus: T T G A C A
        # Strong conservation at pos 0, 1, 2. Weaker at 3, 4, 5.
        counts = np.array([
            [820, 840, 90, 240, 180, 180],  # T
            [50, 50, 60, 100, 540, 140],  # C
            [69, 50, 60, 560, 140, 540],  # A
            [61, 60, 790, 100, 140, 140],  # G
        ], dtype=cls._DTYPE)
        dna = Alphabet.DNA
        if bg is not None:
            if bg.alphabet != dna: raise ValueError('Background alphabet must be dna for this motif')
        else:
            bg = Background.uniform(dna)
        return cls.from_counts(b"Sigma70_-35", counts, bg)

    @classmethod
    def sigma70_10(cls, bg: Background = None) -> 'Motif':
        """Returns a built-in Sigma70 -10 box motif (E. coli)."""
        # -10 Motif Consensus: T A T A A T
        # Very strong conservation at pos 1 (A) and 5 (T).
        # Pos 2 (T) is often variable (T or C).
        counts = np.array([
            [770, 30, 450, 140, 120, 960],  # T
            [80, 10, 140, 140, 230, 20],  # C
            [50, 950, 260, 590, 510, 10],  # A
            [100, 10, 150, 130, 140, 10],  # G
        ], dtype=cls._DTYPE)
        dna = Alphabet.DNA
        if bg is not None:
            if bg.alphabet != dna: raise ValueError('Background alphabet must be dna for this motif')
        else:
            bg = Background.uniform(dna)
        return cls.from_counts(b"Sigma70_-10", counts, bg)


class MotifBatch(Batch):
    """
    Efficiently stores and scans multiple motifs.

    Concatenates PSSMs into a single large matrix for vectorized scanning.

    Args:
        motifs: Iterable of ``Motif`` objects (must share alphabet).
    """
    __slots__ = ('_motifs', '_pssm_combined', '_pssm_rc_combined', '_offsets', '_max_suffixes', '_max_suffixes_rc')

    @classmethod
    def empty(cls) -> 'MotifBatch':
        """Creates an empty MotifBatch."""
        return cls([])

    def __init__(self, motifs: Iterable[Motif]):
        self._motifs = list(motifs)
        n = len(self._motifs)
        if n == 0:
            self._offsets = np.zeros(1, dtype=np.int32)
            return
        
        # If created via empty(), we might have no motifs but valid state
        if n == 0 and not hasattr(self, '_offsets'):
             self._offsets = np.zeros(1, dtype=np.int32)

        # 1. Validate Alphabets
        alpha = self._motifs[0].background.alphabet
        for m in self._motifs:
            if m.background.alphabet != alpha:
                raise ValueError("All motifs in a batch must share the same alphabet.")

        # 2. Concatenate PSSMs (4, Total_Len)
        pssm_list = [m.pssm for m in self._motifs]
        self._pssm_combined = np.concatenate(pssm_list, axis=1)

        # 3. Concatenate RC PSSMs
        pssm_rc_list = [m.pssm_rc for m in self._motifs]
        if any(p is None for p in pssm_rc_list):
            self._pssm_rc_combined = None
        else:
            self._pssm_rc_combined = np.concatenate(pssm_rc_list, axis=1)

        # 4. Offsets
        lengths = [m.pssm.shape[1] for m in self._motifs]
        self._offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lengths, out=self._offsets[1:])

        # 5. Max Suffixes
        suffixes = [_calc_max_suffix(p) for p in pssm_list]
        self._max_suffixes = np.concatenate(suffixes)

        if self._pssm_rc_combined is not None:
            suffixes_rc = [_calc_max_suffix(p) for p in pssm_rc_list]
            self._max_suffixes_rc = np.concatenate(suffixes_rc)
        else:
            self._max_suffixes_rc = None

    @classmethod
    def build(cls, components: Iterable[object]) -> 'Batch':
        """Constructs a MotifBatch from an iterable of Motifs."""
        return cls(components)

    @classmethod
    def concat(cls, batches: Iterable['MotifBatch']) -> 'MotifBatch':
        """Concatenates multiple MotifBatch objects."""
        # Simple list concatenation and rebuild
        all_motifs = []
        for b in batches:
            all_motifs.extend(b._motifs)
        return cls(all_motifs)

    @property
    def n_motifs(self) -> int:
        """Returns the number of motifs in the batch."""
        return len(self._motifs)

    def scan(self, seqs: SeqBatch, pvalue_threshold: float = 1e-4) -> 'MotifHitBatch':
        """Scans sequences for these motifs. Use ``MotifScanner`` for batch scanning.

        Raises:
            TypeError: Direct scanning not supported.
        """
        raise TypeError("Use baclib.engines.motif.MotifScanner(batch).scan(seqs) instead.")

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes."""
        size = self._offsets.nbytes + self._max_suffixes.nbytes + self._pssm_combined.nbytes
        if self._pssm_rc_combined is not None: size += self._pssm_rc_combined.nbytes
        if self._max_suffixes_rc is not None: size += self._max_suffixes_rc.nbytes
        return size

    def copy(self) -> 'MotifBatch':
        """Returns a copy of the batch."""
        # Motifs are effectively immutable, but we can return a new wrapper
        return self.__class__(list(self._motifs))

    @property
    def component(self): return Motif

    def __repr__(self):
        return f"<MotifBatch: {len(self)} motifs>"

    def __len__(self):
        return len(self._motifs)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            return self._motifs[item]

        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1: raise NotImplementedError("Batch slicing with step != 1 not supported")

            # Zero-copy slicing of the batch
            obj = object.__new__(MotifBatch)
            obj._motifs = self._motifs[item]

            val_start = self._offsets[start]
            val_end = self._offsets[stop]
            obj._offsets = self._offsets[start:stop + 1] - val_start

            obj._pssm_combined = self._pssm_combined[:, val_start:val_end]
            obj._pssm_rc_combined = self._pssm_rc_combined[
                :, val_start:val_end] if self._pssm_rc_combined is not None else None

            # Suffix offset = PSSM offset + Motif Index
            s_start = val_start + start
            s_end = val_end + stop
            obj._max_suffixes = self._max_suffixes[s_start:s_end]
            obj._max_suffixes_rc = self._max_suffixes_rc[s_start:s_end] if self._max_suffixes_rc is not None else None

            return obj

        if isinstance(item, (list, np.ndarray)):
            return MotifBatch([self._motifs[i] for i in item])

        raise TypeError(f"Invalid index type: {type(item)}")

    def __iter__(self):
        return iter(self._motifs)

    @classmethod
    def sigma70(cls, bg: Background = None) -> 'MotifBatch':
        """Returns a batch containing Sigma70 -35 and -10 motifs."""
        dna = Alphabet.DNA
        if bg is not None:
            if bg.alphabet != dna: raise ValueError('Background alphabet must be dna for this motif')
        else:
            bg = Background.uniform(dna)
        return MotifBatch([Motif.sigma70_35(bg), Motif.sigma70_10(bg)])


class MotifHit(Feature):
    """
    Represents a single occurrence of a Motif in a sequence.

    Args:
        interval: The location and strand of the hit.
        score: The log-odds score.
        motif: The matching ``Motif`` object.
        pvalue: The P-value of the score (calculated lazily if ``None``).
        qualifiers: Optional qualifiers.

    Examples:
        >>> hit = MotifHit(Interval(10, 20), 12.5, motif)
        >>> hit.pvalue
        1.2e-5
    """
    __slots__ = ('score', 'motif', '_pvalue')

    def __init__(self, interval: Interval, score: float, motif: 'Motif',
                 pvalue: float = None, qualifiers: Iterable[tuple[bytes, Any]] = None):
        super().__init__(interval, key=FeatureKey.MOTIF, qualifiers=qualifiers)
        self.score = score
        self.motif = motif
        self._pvalue = pvalue

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``MotifHitBatch`` class.
        """
        return MotifHitBatch

    @property
    def name(self) -> bytes:
        """Returns the motif name."""
        return self.motif.name

    @property
    def pvalue(self) -> float:
        """Returns the P-value of the hit score."""
        if self._pvalue is None: self._pvalue = self.motif.get_pvalue(self.score)
        return self._pvalue

    def __getitem__(self, item):
        if item == b'score': return self.score
        if item == b'motif': return self.name
        if item == b'pvalue': return self.pvalue
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if key == b'score':
            self.score = value
        elif key == b'pvalue':
            self._pvalue = value
        elif key == b'motif':
            raise AttributeError("Cannot set motif via dict access")
        else:
            super().__setitem__(key, value)

    def __repr__(self):
        return f"MotifHit({self.name.decode(Alphabet.ENCODING)}, score={self.score:.2f}, {self.interval})"

    def copy(self) -> 'MotifHit':
        """Returns a deep copy of the hit."""
        return MotifHit(self.interval, self.score, self.motif, self._pvalue, list(self.qualifiers))

    def shift(self, x: int, y: int = None) -> 'MotifHit':
        """Shifts the hit coordinates."""
        return MotifHit(self.interval.shift(x, y), self.score, self.motif, self._pvalue, list(self.qualifiers))

    def reverse_complement(self, parent_length: int) -> 'MotifHit':
        """Reverse complements the hit."""
        return MotifHit(self.interval.reverse_complement(parent_length), self.score, self.motif, self._pvalue,
                        list(self.qualifiers))


class MotifHitBatch(Batch):
    """
    Columnar container for motif scan results.

    Attributes:
        seq_indices: Index of the sequence in the scanned batch.
        positions: Start position of the hit.
        scores: Log-odds score.
        strands: Strand of the hit.
        motif_indices: Index of the motif in the scanner's batch (optional).
    """
    __slots__ = ('seq_indices', 'positions', 'scores', 'strands', 'motif_indices', '_source')

    def __init__(self, seq_indices, positions, scores, strands, source, motif_indices=None):
        self.seq_indices: np.ndarray = seq_indices
        self.positions: np.ndarray = positions
        self.scores: np.ndarray = scores
        self.strands: np.ndarray = strands
        self._source = source  # Motif or MotifBatch
        self.motif_indices: Optional[np.ndarray] = motif_indices

    @classmethod
    def new_empty(cls, source):
        """Creates an empty batch for a given source."""
        return cls(np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                   np.array([], dtype=np.float32), np.array([], dtype=np.int8), source)

    @classmethod
    def build(cls, components: Iterable[object]) -> 'Batch':
        """
        Raises:
            NotImplementedError: Hits should be created via scanning.
        """
        raise NotImplementedError("MotifHitBatch should be created via Motif.scan()")

    @classmethod
    def concat(cls, batches: Iterable['MotifHitBatch']) -> 'MotifHitBatch':
        """Concatenates multiple hit batches."""
        batches = list(batches)
        if not batches: raise ValueError("Cannot concat empty list")
        # Find first valid source
        source = next((b._source for b in batches if b._source is not None), None)
        
        seq_indices = np.concatenate([b.seq_indices for b in batches])
        positions = np.concatenate([b.positions for b in batches])
        scores = np.concatenate([b.scores for b in batches])
        strands = np.concatenate([b.strands for b in batches])
        
        # Handle motif indices if present
        mi = None
        # Check first non-empty batch for motif indices logic
        first = batches[0]
        if first.motif_indices is not None:
             mi = np.concatenate([b.motif_indices for b in batches])
             
        return cls(seq_indices, positions, scores, strands, source, mi)

    @property
    def nbytes(self) -> int:
        """Returns total memory usage in bytes."""
        base = self.seq_indices.nbytes + self.positions.nbytes + self.scores.nbytes + self.strands.nbytes
        return base + (self.motif_indices.nbytes if self.motif_indices is not None else 0)

    def copy(self) -> 'MotifHitBatch':
        """Returns a copy of the batch."""
        return self.__class__(self.seq_indices.copy(), self.positions.copy(), self.scores.copy(), self.strands.copy(), self._source, self.motif_indices.copy() if self.motif_indices is not None else None)

    @classmethod
    def empty(cls) -> 'MotifHitBatch':
        """Creates an empty hit batch."""
        return cls.new_empty(None)

    @property
    def component(self): return MotifHit

    def __repr__(self):
        return f"<MotifHitBatch: {len(self)} hits>"

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            mi = self.motif_indices[item] if self.motif_indices is not None else None
            return (self.seq_indices[item], self.positions[item], self.scores[item], self.strands[item], mi)

        if isinstance(item, slice):
            mi = self.motif_indices[item] if self.motif_indices is not None else None
            return MotifHitBatch(
                self.seq_indices[item], self.positions[item], self.scores[item], self.strands[item], self._source, mi
            )

        if isinstance(item, (np.ndarray, list)):
            mi = self.motif_indices[item] if self.motif_indices is not None else None
            return MotifHitBatch(
                self.seq_indices[item], self.positions[item], self.scores[item], self.strands[item], self._source, mi
            )

        raise TypeError(f"Invalid index type: {type(item)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_features(self, seq_batch: SeqBatch) -> list[list[MotifHit]]:
        """
        Converts hits to Feature objects, grouped by sequence index.

        Args:
            seq_batch: The ``SeqBatch`` that was scanned (for context).

        Returns:
            A list of lists, where `list[i]` contains features for `seq_batch[i]`.
        """
        n_seqs = len(seq_batch)
        features_by_seq = [[] for _ in range(n_seqs)]

        for i in range(len(self.scores)):
            s_idx = self.seq_indices[i]
            pos = self.positions[i]
            score = self.scores[i]
            strand = self.strands[i]

            if self.motif_indices is not None:
                motif = self._source[self.motif_indices[i]]
            else:
                motif = self._source

            features_by_seq[s_idx].append(
                MotifHit(Interval(pos, pos + len(motif), strand), score, motif)
            )
        return features_by_seq
