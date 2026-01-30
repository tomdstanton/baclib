from typing import Union, Optional

import numpy as np

from baclib.core.seq import Seq, SeqBatch, Alphabet
from baclib.containers.record import Feature
from baclib.core.interval import Interval
from baclib.utils.resources import RESOURCES, jit

if RESOURCES.has_module('numba'): from numba import prange
else: prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Background:
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
    _DNA = Alphabet.dna()
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
        alphabet = alphabet or cls._DNA
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
                 '_granularity', '_pseudocount', '_min_score', '_max_score', '_pdf')
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
        # Enforce immutability for thread safety
        for array in [self._count, self._frequency, self._weight, self._scoring, self._discrete]:
            if array is not None: array.flags.writeable = False

    def __len__(self) -> int: return len(self._scoring.shape[1])
    @property
    def name(self) -> bytes: return self._name
    @property
    def background(self) -> Background: return self._background
    @staticmethod
    def _pad_matrix(matrix: np.ndarray, target_rows, dtype) -> np.ndarray:
        rows, cols = matrix.shape
        if rows == target_rows: return matrix
        padded = np.full((target_rows, cols), -1.0e9, dtype=dtype)
        padded[:rows, :] = matrix
        return padded
    @property
    def pssm(self) -> np.ndarray: return self._pad_matrix(self._scoring, 256, self._DTYPE)
    @property
    def pssm_rc(self) -> Optional[np.ndarray]:
        comp = self.background.alphabet.complement
        if comp is None: return None
        rc = self._scoring[comp, ::-1]
        return self._pad_matrix(rc, 256, self._DTYPE)
    
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
        data, starts, lengths = seqs.arrays
        seq_idxs_list = []
        pos_list = []
        scores_list = []
        strands_list = []
        # --- Forward ---
        pssm = self.pssm
        score_threshold = self.get_score(pvalue_threshold)
        
        # Pre-calculate max suffix scores for Lookahead Pruning
        max_suffix_f = _calc_max_suffix(pssm)
        
        counts_f = _scan_count_kernel(data, starts, lengths, pssm, score_threshold, max_suffix_f)
        total_f = counts_f.sum()
        if total_f > 0:
            offsets_f = np.zeros(len(counts_f) + 1, dtype=np.int32)
            np.cumsum(counts_f, out=offsets_f[1:])
            out_pos = np.empty(total_f, dtype=np.int32)
            out_scores = np.empty(total_f, dtype=np.float32)
            out_seq_idxs = np.empty(total_f, dtype=np.int32)
            _scan_fill_kernel(data, starts, lengths, pssm, score_threshold, max_suffix_f, offsets_f, out_pos, out_scores, out_seq_idxs)
            seq_idxs_list.append(out_seq_idxs)
            pos_list.append(out_pos)
            scores_list.append(out_scores)
            strands_list.append(np.full(total_f, 1, dtype=np.int8))
        # --- Reverse ---
        pssm_rc = self.pssm_rc
        if pssm_rc is not None:
            max_suffix_r = _calc_max_suffix(pssm_rc)
            counts_r = _scan_count_kernel(data, starts, lengths, pssm_rc, score_threshold, max_suffix_r)
            total_r = counts_r.sum()
            if total_r > 0:
                offsets_r = np.zeros(len(counts_r) + 1, dtype=np.int32)
                np.cumsum(counts_r, out=offsets_r[1:])
                out_pos = np.empty(total_r, dtype=np.int32)
                out_scores = np.empty(total_r, dtype=np.float32)
                out_seq_idxs = np.empty(total_r, dtype=np.int32)
                _scan_fill_kernel(data, starts, lengths, pssm_rc, score_threshold, max_suffix_r, offsets_r, out_pos, out_scores, out_seq_idxs)
                seq_idxs_list.append(out_seq_idxs)
                pos_list.append(out_pos)
                scores_list.append(out_scores)
                strands_list.append(np.full(total_r, -1, dtype=np.int8))

        if not seq_idxs_list: return MotifHitBatch.empty(self)

        return MotifHitBatch(
            np.concatenate(seq_idxs_list), np.concatenate(pos_list), np.concatenate(scores_list),
            np.concatenate(strands_list), self
        )


class MotifHitBatch:
    """
    Container for motif scan results.

    Attributes:
        seq_indices (np.ndarray): Indices of sequences in the batch.
        positions (np.ndarray): Start positions of hits.
        scores (np.ndarray): Scores of hits.
        strands (np.ndarray): Strands of hits.
    """
    __slots__ = ('seq_indices', 'positions', 'scores', 'strands', '_motif')

    def __init__(self, seq_indices, positions, scores, strands, motif):
        self.seq_indices: np.ndarray = seq_indices
        self.positions: np.ndarray = positions
        self.scores: np.ndarray = scores
        self.strands: np.ndarray = strands
        self._motif: Motif = motif

    @classmethod
    def empty(cls, motif):
        return cls(np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                   np.array([], dtype=np.float32), np.array([], dtype=np.int8), motif)

    def __len__(self): return len(self.scores)

    def to_features(self, seq_batch: SeqBatch, feature_kind: bytes = b'misc_feature') -> list[list[Feature]]:
        """
        Converts hits to Feature objects, grouped by sequence index.
        Returns a list of lists, where list[i] contains features for seq_batch[i].
        """
        n_seqs = len(seq_batch)
        features_by_seq = [[] for _ in range(n_seqs)]
        motif = self._motif
        for i in range(len(self.scores)):
            s_idx = self.seq_indices[i]
            pos = self.positions[i]
            score = self.scores[i]
            strand = self.strands[i]
            features_by_seq[s_idx].append(
                Feature(Interval(pos, pos + len(motif), strand), kind=feature_kind,
                        qualifiers=[(b'score', float(score)), (b'motif', motif.name)])
            )
        return features_by_seq



# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _score_distribution_kernel(int_pssm, bg_probs, min_score, max_score):
    """
    Computes the exact distribution of PSSM scores using Dynamic Programming.
    Complexity: O(L * ScoreRange * 4)
    """
    n_bases, length = int_pssm.shape

    # Offset everything to start at 0 to use as array indices
    offset_range = max_score - min_score + 1

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
            # next_dist[shift : shift + len(current_dist)] += current_dist * prob
            for i in range(len(current_dist)):
                next_dist[shift + i] += current_dist[i] * prob

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
