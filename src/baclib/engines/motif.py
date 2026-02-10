from typing import Union, Optional, Iterable, Any

import numpy as np

from baclib.utils.protocols import HasAlphabet
from baclib.core.alphabet import Alphabet
from baclib.core.seq import Seq, SeqBatch
from baclib.containers.record import Feature, FeatureKey
from baclib.core.interval import Interval
from baclib.engines.index import SparseMapIndex, SparseMapIndexMode
from baclib.utils.resources import RESOURCES, jit
from baclib.utils import Batch

if RESOURCES.has_module('numba'): 
    from numba import prange
else: 
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Background(HasAlphabet):
    """
    Represents the background nucleotide frequencies.

    Attributes:
        alphabet (Alphabet): The alphabet used.
        data (np.ndarray): Array of probabilities.

    Examples:
        >>> bg = Background.uniform(Alphabet.dna())
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
    def alphabet(self) -> Alphabet: return self._alphabet
    @property
    def data(self) -> np.ndarray: return self._data
    @classmethod
    def from_counts(cls, counts: np.ndarray, alphabet: Alphabet = None) -> 'Background':
        """Creates a Background from raw counts."""
        alphabet = alphabet or cls._DEFAULT_ALPHABET
        assert len(counts) == len(alphabet)
        new = cls()
        new._alphabet = alphabet
        new._data = counts / np.sum(counts, dtype=cls._DTYPE)
        new._data.flags.writeable = False
        return new
    @classmethod
    def from_seq(cls, seq: Union[Seq, SeqBatch]) -> 'Background':
        """Creates a Background from a sequence or batch."""
        return cls.from_counts(np.bincount(seq.encoded), seq.alphabet)
    @classmethod
    def uniform(cls, alphabet: Alphabet = None) -> 'Background':
        """Creates a uniform Background."""
        alphabet = alphabet or cls._DEFAULT_ALPHABET
        new = cls()
        new._alphabet = alphabet
        n = len(alphabet)
        new._data =  np.ones(n, dtype=cls._DTYPE) / n
        new._data.flags.writeable = False
        return new

    def __array__(self, dtype=None) -> np.ndarray:
        return self._data if dtype is None else self._data.astype(dtype)


class Motif:
    """
    Standard Position Specific Scoring Matrix (PSSM).

    Attributes:
        name (bytes): Motif name.
        background (Background): Background model.
        pssm (np.ndarray): The scoring matrix.

    Examples:
        >>> bg = Background.uniform(Alphabet.dna())
        >>> counts = np.array([[10, 0, 0, 0], [0, 10, 0, 0]]) # AA
        >>> m = Motif.from_counts(b'polyA', counts.T, bg)
        >>> len(m)
        2
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

    def __len__(self) -> int: return len(self._scoring.shape[1])
    @property
    def name(self) -> bytes: return self._name
    @property
    def background(self) -> Background: return self._background
    @property
    def pssm(self) -> np.ndarray: return self._scoring
    @property
    def pssm_rc(self) -> Optional[np.ndarray]:
        comp = self.background.alphabet.complement
        if comp is None: return None
        return np.ascontiguousarray(self._scoring[comp, ::-1])
    
    @classmethod
    def from_counts(cls, name: bytes, counts: np.ndarray, background: Background,
                    granularity: int = _DEFAULT_GRANULARITY, pseudocount: float = _DEFAULT_PSEUDOCOUNT) -> 'Motif':
        """Creates a Motif from a count matrix (Rows=Bases, Cols=Positions)."""
        adjusted = counts + pseudocount
        frequencies = adjusted / adjusted.sum(axis=0)
        return cls.from_frequencies(name, frequencies, background, granularity, pseudocount, counts)

    @classmethod
    def from_frequencies(cls, name: bytes, frequencies: np.ndarray, background: Background,
                        granularity: int = _DEFAULT_GRANULARITY, pseudocount: float = _DEFAULT_PSEUDOCOUNT,
                        counts: np.ndarray = None) -> 'Motif':
        """Creates a Motif from a frequency matrix."""
        weights = frequencies / background.data[:, None]
        return cls.from_weights(name, weights, background, granularity, pseudocount, counts, frequencies)

    @classmethod
    def from_weights(cls, name: bytes, weights: np.ndarray, background: Background,
                     granularity: int = _DEFAULT_GRANULARITY, pseudocount: float = _DEFAULT_PSEUDOCOUNT,
                     counts: np.ndarray = None,
                     frequencies: np.ndarray = None) -> 'Motif':
        """Creates a Motif from a weight matrix."""
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
        """Creates a Motif from a log-odds score matrix."""
        return cls(name, scores, background, granularity, pseudocount, counts, frequencies, weights)

    @classmethod
    def sigma70_35(cls, bg: Background = None) -> 'Motif':
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
    
    def pdf(self):
        """Computes the probability density function of scores."""
        if self._pdf is None:
            bg = self._background.data.astype(np.float64)
            self._pdf = _score_distribution_kernel(self._discrete, bg, self._min_score, self._max_score)
            self._pdf.flags.writeable = False
        return self._pdf
    
    def get_score(self, p_value: float) -> float:
        """
        Calculates the score threshold for a given P-value.
        Uses discretized Dynamic Programming (TFM-PVALUE style).
        """
        # 5. Calculate CCDF (Survival Function) to find P-value
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
        """
        Calculates the P-value for a given score.
        Uses discretized Dynamic Programming (TFM-PVALUE style).
        """
        target_int = int(round(score * self._granularity))
        idx = target_int - self._min_score
        if idx < 0: return 1.0
        pdf = self.pdf()
        if idx >= len(pdf): return 0.0
        return pdf[idx:].sum()

    def scan(self, seqs: SeqBatch, pvalue_threshold: float = 0.0) -> 'MotifHitBatch':
        """
        Scans the sequence batch for instances of this motif.

        Args:
            seqs: The SeqBatch to scan.
            pvalue_threshold: The maximum P-value to report.

        Returns:
            A MotifHitBatch containing the hits.
        """
        if seqs.alphabet != self.background.alphabet:
            raise ValueError(f"Motif alphabet ({self.background.alphabet}) does not match Sequence alphabet ({seqs.alphabet})")
        data, starts, lengths = seqs.arrays
        seq_idxs_list = []
        pos_list = []
        scores_list = []
        strands_list = []
        # --- Forward ---
        pssm = self.pssm
        score_threshold = self.get_score(pvalue_threshold)

        def _process(pssm_ptr, max_suffix_ptr, strand):
            counts = _scan_count_kernel(data, starts, lengths, pssm_ptr, score_threshold, max_suffix_ptr)
            total = counts.sum()
            if total > 0:
                offsets = np.zeros(len(counts) + 1, dtype=np.int32)
                np.cumsum(counts, out=offsets[1:])
                out_pos = np.empty(total, dtype=np.int32)
                out_scores = np.empty(total, dtype=np.float32)
                out_seq_idxs = np.empty(total, dtype=np.int32)
                _scan_fill_kernel(data, starts, lengths, pssm_ptr, score_threshold, max_suffix_ptr, offsets, out_pos, out_scores, out_seq_idxs)
                seq_idxs_list.append(out_seq_idxs)
                pos_list.append(out_pos)
                scores_list.append(out_scores)
                strands_list.append(np.full(total, strand, dtype=np.int8))

        _process(pssm, self._max_suffix, 1)
        if self.pssm_rc is not None:
            _process(self.pssm_rc, self._max_suffix_rc, -1)

        if not seq_idxs_list: return MotifHitBatch.new_empty(self)

        return MotifHitBatch(
            np.concatenate(seq_idxs_list), np.concatenate(pos_list), np.concatenate(scores_list),
            np.concatenate(strands_list), self
        )


class MotifBatch(Batch):
    """
    Efficiently stores and scans multiple motifs.
    """
    __slots__ = ('_motifs', '_pssm_combined', '_pssm_rc_combined', '_offsets', '_max_suffixes', '_max_suffixes_rc')

    def __init__(self, motifs: Iterable[Motif]):
        self._motifs = list(motifs)
        n = len(self._motifs)
        if n == 0:
            self._offsets = np.zeros(1, dtype=np.int32)
            return

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

    def empty(self) -> 'MotifBatch':
        return MotifBatch([])

    def __repr__(self): return f"<MotifBatch: {len(self)} motifs>"

    def __len__(self): return len(self._motifs)

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
        dna = Alphabet.DNA
        if bg is not None:
            if bg.alphabet != dna: raise ValueError('Background alphabet must be dna for this motif')
        else:
            bg = Background.uniform(dna)
        return MotifBatch([Motif.sigma70_35(bg), Motif.sigma70_10(bg)])

    def scan(self, seqs: SeqBatch, pvalue_threshold: float = 0.0) -> 'MotifHitBatch':
        if not self._motifs: return MotifHitBatch.new_empty(self)
        if seqs.alphabet != self._motifs[0].background.alphabet:
            raise ValueError(f"Motif alphabet does not match Sequence alphabet")

        # Calculate thresholds for all motifs
        thresholds = np.array([m.get_score(pvalue_threshold) for m in self._motifs], dtype=np.float32)
        data, starts, lengths = seqs.arrays

        seq_idxs_list, pos_list, scores_list, strands_list, motif_idxs_list = [], [], [], [], []

        def _process(pssm_comb, max_suff, strand):
            counts = _scan_batch_count_kernel(
                data, starts, lengths, pssm_comb, self._offsets, thresholds, max_suff
            )
            total = counts.sum()
            if total > 0:
                offsets = np.zeros(len(counts) + 1, dtype=np.int32)
                np.cumsum(counts, out=offsets[1:])
                out_pos = np.empty(total, dtype=np.int32)
                out_scores = np.empty(total, dtype=np.float32)
                out_seq_idxs = np.empty(total, dtype=np.int32)
                out_motif_idxs = np.empty(total, dtype=np.int32)
                _scan_batch_fill_kernel(
                    data, starts, lengths, pssm_comb, self._offsets, thresholds, max_suff,
                    offsets, out_pos, out_scores, out_seq_idxs, out_motif_idxs
                )
                seq_idxs_list.append(out_seq_idxs)
                pos_list.append(out_pos)
                scores_list.append(out_scores)
                motif_idxs_list.append(out_motif_idxs)
                strands_list.append(np.full(total, strand, dtype=np.int8))

        _process(self._pssm_combined, self._max_suffixes, 1)
        if self._pssm_rc_combined is not None:
            _process(self._pssm_rc_combined, self._max_suffixes_rc, -1)

        if not seq_idxs_list: return MotifHitBatch.new_empty(self)

        return MotifHitBatch(
            np.concatenate(seq_idxs_list), np.concatenate(pos_list), np.concatenate(scores_list),
            np.concatenate(strands_list), self, np.concatenate(motif_idxs_list)
        )


class MotifHit(Feature):
    """
    Represents a single occurrence of a Motif.
    """
    __slots__ = ('score', 'motif', '_pvalue')

    def __init__(self, interval: Interval, score: float, motif: 'Motif', 
                 pvalue: float = None, qualifiers: Iterable[tuple[bytes, Any]] = None):
        super().__init__(interval, key=FeatureKey.MOTIF, qualifiers=qualifiers)
        self.score = score
        self.motif = motif
        self._pvalue = pvalue

    @property
    def name(self) -> bytes: return self.motif.name

    @property
    def pvalue(self) -> float:
        if self._pvalue is None: self._pvalue = self.motif.get_pvalue(self.score)
        return self._pvalue

    def __getitem__(self, item):
        if item == b'score': return self.score
        if item == b'motif': return self.name
        if item == b'pvalue': return self.pvalue
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if key == b'score': self.score = value
        elif key == b'pvalue': self._pvalue = value
        elif key == b'motif': raise AttributeError("Cannot set motif via dict access")
        else: super().__setitem__(key, value)

    def __repr__(self):
        return f"MotifHit({self.name.decode(Alphabet.ENCODING)}, score={self.score:.2f}, {self.interval})"

    def copy(self) -> 'MotifHit':
        return MotifHit(self.interval, self.score, self.motif, self._pvalue, list(self.qualifiers))

    def shift(self, x: int, y: int = None) -> 'MotifHit':
        return MotifHit(self.interval.shift(x, y), self.score, self.motif, self._pvalue, list(self.qualifiers))

    def reverse_complement(self, parent_length: int) -> 'MotifHit':
        return MotifHit(self.interval.reverse_complement(parent_length), self.score, self.motif, self._pvalue, list(self.qualifiers))


class MotifHitBatch(Batch):
    """
    Container for motif scan results.

    Attributes:
        seq_indices (np.ndarray): Indices of sequences in the batch.
        positions (np.ndarray): Start positions of hits.
        scores (np.ndarray): Scores of hits.
        strands (np.ndarray): Strands of hits.
        motif_indices (np.ndarray): Indices of the motif in the MotifBatch (optional).
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
        return cls(np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                   np.array([], dtype=np.float32), np.array([], dtype=np.int8), source)

    def empty(self) -> 'MotifHitBatch':
        return self.new_empty(self._source)

    def __repr__(self): return f"<MotifHitBatch: {len(self)} hits>"

    def __len__(self): return len(self.scores)

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
        Returns a list of lists, where list[i] contains features for seq_batch[i].
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


class MotifFinder:
    """
    Engine for de novo motif discovery.
    Implements a STREME-like pipeline:
    1. Efficient K-mer counting (using Dense Index).
    2. Statistical Enrichment Testing (Binomial/Fisher).
    3. Iterative Refinement (Seed -> Scan -> Update PSSM).
    """
    def __init__(self, foreground: SeqBatch, background: Union[SeqBatch, Background] = None, k: int = 8):
        self.k = k
        self.fg = foreground
        self.alphabet = foreground.alphabet

        # 1. Build Foreground Index (The "Observed" Data)
        self._fg_idx = SparseMapIndex(k=k, mode=SparseMapIndexMode.DENSE, alphabet=self.alphabet)
        self._fg_idx.build(foreground)

        # 2. Count Foreground K-mers
        # Sorts by hash, allowing O(log N) lookups
        self._fg_kmers, self._fg_counts = np.unique(self._fg_idx.hashes, return_counts=True)
        self._n_fg = self._fg_counts.sum()
        
        # Symbol mask for decoding (e.g. 3 for DNA, 31 for Amino)
        self._symbol_mask = (1 << self._fg_idx.bps) - 1

        # 3. Setup Background Model
        self._bg_probs = None  # Probability of finding each unique k-mer in the background
        self._bg_model = None  # The Background object (for PSSM creation)

        if isinstance(background, SeqBatch):
            # Discriminative Mode (vs Control Sequences)
            bg_idx = SparseMapIndex(k=k, mode=SparseMapIndexMode.DENSE, alphabet=self.alphabet)
            bg_idx.build(background)
            bg_kmers_raw, bg_counts_raw = np.unique(bg_idx.hashes, return_counts=True)
            n_bg = bg_counts_raw.sum()

            # Align counts to Foreground k-mers
            bg_counts_aligned = _align_counts(self._fg_kmers, bg_kmers_raw, bg_counts_raw)

            # P(kmer) = (Count_BG + 1) / (Total_BG + Unique_Kmers) - Laplace Smoothing
            denom = n_bg + len(bg_kmers_raw)
            self._bg_probs = (bg_counts_aligned + 1) / denom
            self._bg_model = Background.from_seq(background)

        elif isinstance(background, Background):
            # Provided Background Model
            self._bg_model = background
            self._bg_probs = _calc_kmer_probs_kernel(
                self._fg_kmers, k, background.data, self._fg_idx.bps, self._symbol_mask
            )
        else:
            # Null Model (Learn 0-order from Foreground)
            self._bg_model = Background.from_seq(foreground)
            self._bg_probs = _calc_kmer_probs_kernel(
                self._fg_kmers, k, self._bg_model.data, self._fg_idx.bps, self._symbol_mask
            )

    def run(self, n_motifs: int = 5, max_pvalue: float = 0.05, refine: bool = True) -> MotifBatch:
        """
        Executes the discovery pipeline.

        Args:
            n_motifs: Number of top motifs to return.
            max_pvalue: Significance threshold for seeds.
            refine: If True, performs 1 round of EM-refinement (Scan -> Rebuild PSSM).
        """
        try:
            from scipy.special import bdtrc
        except ImportError:
            raise ImportError("Motif discovery requires 'scipy' to be installed.")

        # 1. Statistical Test (Binomial Survival Function)
        # H0: k-mer occurrences follow Binomial(n=total_fg, p=bg_probs)
        pvalues = bdtrc(self._fg_counts - 1, self._n_fg, self._bg_probs)

        # 2. Filter & Sort Seeds
        sig_mask = pvalues < max_pvalue
        if not np.any(sig_mask): return MotifBatch([])

        sig_indices = np.where(sig_mask)[0]
        sorted_indices = sig_indices[np.argsort(pvalues[sig_indices])]

        # 3. Select unique seeds (Naive approach: just take top N)
        # In a full implementation, you'd mask occurrences of the top motif before finding the next.
        top_indices = sorted_indices[:n_motifs]

        final_motifs = []

        for i, idx in enumerate(top_indices):
            kmer_int = self._fg_kmers[idx]
            pval = pvalues[idx]

            # 3a. Create Seed Motif
            # We initialize the PSSM with the k-mer sequence itself (count=100 equiv)
            seed_counts = _int_to_onehot(kmer_int, self.k, self._fg_idx.bps, self._symbol_mask) * 100.0
            name = f"rank_{i + 1}_p{pval:.1e}".encode(Alphabet.ENCODING)
            m = Motif.from_counts(name, seed_counts, self._bg_model)

            if refine:
                # 3b. Refine (The EM Step)
                # Scan with a modest threshold to find variations of the motif
                hits = m.scan(self.fg, pvalue_threshold=1e-3)

                if len(hits) > 5:  # Only update if we found decent support
                    # Accumulate actual sequences from hits into a new Count Matrix
                    # This is much faster than extracting strings
                    data, starts, _ = self.fg.arrays
                    new_counts = _accumulate_hits_kernel(
                        data, hits.seq_indices, hits.positions, hits.strands,
                        starts, self.k, self.alphabet.complement
                    )
                    # Create new, refined motif
                    m = Motif.from_counts(name, new_counts, self._bg_model)

            final_motifs.append(m)

        return MotifBatch(final_motifs)


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _score_distribution_kernel(int_pssm, bg_probs, min_score, max_score):
    """
    Computes the exact distribution of PSSM scores using Dynamic Programming.
    Complexity: O(L * ScoreRange * 4)
    """
    n_bases, length = int_pssm.shape

    # dp[s] = probability of getting score 's' (shifted by current_min)
    # Start with score 0 having prob 1.0
    current_dist = np.zeros(1, dtype=np.float64)
    current_dist[0] = 1.0

    current_min = 0

    for col in range(length):
        # Determine the range of the new distribution
        col_scores = int_pssm[:, col]
        col_min = np.min(col_scores)
        col_max = np.max(col_scores)

        # New distribution size = Old_Size + (Col_Max - Col_Min)
        new_len = len(current_dist) + (col_max - col_min)
        next_dist = np.zeros(new_len, dtype=np.float64)

        # Convolve current_dist with this column's probabilities
        for b in range(n_bases):
            score = col_scores[b]
            prob = bg_probs[b]

            # Where does this base land relative to the new minimum?
            # shift = score - col_min
            shift = score - col_min

            # Add probabilities (Vectorized operation in loop)
            # Numba handles slice assignment efficiently (SIMD)
            next_dist[shift : shift + len(current_dist)] += current_dist * prob

        current_dist = next_dist
        current_min += col_min

    return current_dist


def _calc_max_suffix(pssm: np.ndarray) -> np.ndarray:
    """Calculates the maximum possible remaining score for lookahead pruning."""
    max_per_col = pssm.max(axis=0)
    suffix = np.cumsum(max_per_col[::-1])[::-1]
    return np.concatenate((suffix, np.array([0.0], dtype=pssm.dtype)))


@jit(nopython=True, cache=True, nogil=True, parallel=True, fastmath=True)
def _scan_count_kernel(data, starts, lengths, pssm, score_threshold, max_suffix):
    n_seqs = len(starts)
    counts = np.zeros(n_seqs, dtype=np.int32)
    pssm_len = pssm.shape[1]

    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        if l < pssm_len: continue

        c = 0
        for j in range(l - pssm_len + 1):
            score = 0.0
            for k in range(pssm_len):
                base = data[s + j + k]
                score += pssm[base, k]
                # Lookahead Pruning:
                # If current score + max possible future score < threshold, give up.
                if score + max_suffix[k + 1] < score_threshold:
                    score = -1.0e9 # Ensure failure
                    break
            if score >= score_threshold: c += 1
        counts[i] = c
    return counts


@jit(nopython=True, cache=True, nogil=True, parallel=True, fastmath=True)
def _scan_fill_kernel(data, starts, lengths, pssm, score_threshold, max_suffix, offsets, out_pos, out_scores, out_seq_idxs):
    n_seqs = len(starts)
    pssm_len = pssm.shape[1]

    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        out_idx = offsets[i]
        if l < pssm_len: continue

        for j in range(l - pssm_len + 1):
            score = 0.0
            for k in range(pssm_len):
                base = data[s + j + k]
                score += pssm[base, k]
                if score + max_suffix[k + 1] < score_threshold:
                    score = -1.0e9
                    break
            if score >= score_threshold:
                out_pos[out_idx] = j
                out_scores[out_idx] = score
                out_seq_idxs[out_idx] = i
                out_idx += 1


@jit(nopython=True, cache=True, nogil=True, parallel=True, fastmath=True)
def _scan_batch_count_kernel(data, starts, lengths, pssm_combined, offsets, thresholds, max_suffixes):
    n_seqs = len(starts)
    n_motifs = len(thresholds)
    counts = np.zeros(n_seqs, dtype=np.int32)

    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        c = 0
        for m in range(n_motifs):
            m_start = offsets[m]
            m_end = offsets[m+1]
            pssm_len = m_end - m_start
            if l < pssm_len: continue
            
            thresh = thresholds[m]
            suffix_base = m_start + m  # Offset in max_suffixes array
            
            for j in range(l - pssm_len + 1):
                score = 0.0
                for k in range(pssm_len):
                    base = data[s + j + k]
                    score += pssm_combined[base, m_start + k]
                    if score + max_suffixes[suffix_base + k + 1] < thresh:
                        score = -1.0e9
                        break
                if score >= thresh: c += 1
        counts[i] = c
    return counts


@jit(nopython=True, cache=True, nogil=True, parallel=True, fastmath=True)
def _scan_batch_fill_kernel(data, starts, lengths, pssm_combined, offsets, thresholds, max_suffixes, 
                            out_offsets, out_pos, out_scores, out_seq_idxs, out_motif_idxs):
    n_seqs = len(starts)
    n_motifs = len(thresholds)

    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        out_idx = out_offsets[i]
        
        for m in range(n_motifs):
            m_start = offsets[m]
            m_end = offsets[m+1]
            pssm_len = m_end - m_start
            if l < pssm_len: continue
            
            thresh = thresholds[m]
            suffix_base = m_start + m
            
            for j in range(l - pssm_len + 1):
                score = 0.0
                for k in range(pssm_len):
                    base = data[s + j + k]
                    score += pssm_combined[base, m_start + k]
                    if score + max_suffixes[suffix_base + k + 1] < thresh:
                        score = -1.0e9
                        break
                if score >= thresh:
                    out_pos[out_idx] = j
                    out_scores[out_idx] = score
                    out_seq_idxs[out_idx] = i
                    out_motif_idxs[out_idx] = m
                    out_idx += 1


@jit(nopython=True, cache=True, nogil=True)
def _calc_kmer_probs_kernel(kmer_ints, k, bg_probs, bits, mask):
    """
    Calculates the expected probability of k-mers based on 0-order background.
    """
    n = len(kmer_ints)
    probs = np.empty(n, dtype=np.float32)

    # Pre-compute logs to prevent underflow for large K, or just use direct mul for small K
    # For speed on k<=15, direct multiplication is usually fine.

    for i in prange(n):
        val = kmer_ints[i]
        p = 1.0
        # Iterate over the k bases in the integer
        for j in range(k):
            # Extract lowest base (or highest, order doesn't matter for 0-order multiplication)
            base = val & mask
            p *= bg_probs[base]
            val >>= bits
        probs[i] = p
    return probs


def _align_counts(fg_hashes, bg_hashes, bg_counts):
    """Matches background counts to foreground hashes."""
    indices = np.searchsorted(bg_hashes, fg_hashes)
    indices[indices == len(bg_hashes)] = 0
    mask = bg_hashes[indices] == fg_hashes
    aligned = np.zeros(len(fg_hashes), dtype=np.int32)
    aligned[mask] = bg_counts[indices[mask]]
    return aligned


@jit(nopython=True, cache=True, nogil=True)
def _int_to_onehot(val, k, bits, mask):
    """Decodes a k-mer integer into a One-Hot count matrix (4, k)."""
    mat = np.zeros((4, k), dtype=np.float32)
    for i in range(k):
        col = k - 1 - i
        base = val & mask
        mat[base, col] = 1.0
        val >>= bits
    return mat


@jit(nopython=True, cache=True, nogil=True)
def _accumulate_hits_kernel(data, seq_idxs, pos, strands, seq_starts, k, rc_table):
    """
    Extracts sequences from hits and builds a Count Matrix (PSSM).
    Replaces the need to extract string objects for EM refinement.
    """
    # 4 rows (A,C,G,T), k columns
    counts = np.zeros((4, k), dtype=np.float32)
    n_hits = len(seq_idxs)

    for i in range(n_hits):
        s_idx = seq_idxs[i]
        p = pos[i]
        strand = strands[i]
        global_start = seq_starts[s_idx] + p

        if strand == 1:
            # Forward Strand: Add bases directly
            for j in range(k):
                base = data[global_start + j]
                counts[base, j] += 1.0
        else:
            # Reverse Strand: Add complement bases in reverse order
            # Index j (0..k) maps to column (k-1-j)
            for j in range(k):
                base = data[global_start + j]
                comp_base = rc_table[base]
                counts[comp_base, k - 1 - j] += 1.0

    return counts
