"""Engines for motif scanning and de novo motif discovery using k-mer enrichment analysis."""
from typing import Union, Iterable

import numpy as np

from baclib.core.alphabet import Alphabet
from baclib.containers.seq import SeqBatch
from baclib.containers.motif import Motif, MotifBatch, MotifHitBatch, Background
from baclib.engines.index import SparseMapIndex, SparseMapIndexMode
from baclib.lib.resources import RESOURCES, jit

if RESOURCES.has_module('numba'): 
    from numba import prange
else: 
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class MotifScanner:
    """
    High-performance scanner for finding motif occurrences in sequence batches.
    """
    def __init__(self, motifs: Union[Motif, Iterable[Motif], MotifBatch]):
        if isinstance(motifs, Motif): motifs = [motifs]
        self.batch = motifs if isinstance(motifs, MotifBatch) else MotifBatch(motifs)

    def scan(self, seqs: SeqBatch, pvalue_threshold: float = 1e-4) -> 'MotifHitBatch':
        if len(self.batch) == 0: return MotifHitBatch.new_empty(self.batch)
        if seqs.alphabet != self.batch._motifs[0].background.alphabet:
            raise ValueError(f"Motif alphabet does not match Sequence alphabet")

        # Calculate thresholds for all motifs
        thresholds = np.array([m.get_score(pvalue_threshold) for m in self.batch._motifs], dtype=np.float32)
        data, starts, lengths = seqs.arrays

        seq_idxs_list, pos_list, scores_list, strands_list, motif_idxs_list = [], [], [], [], []

        def _process(pssm_comb, max_suff, strand):
            counts = _scan_batch_count_kernel(
                data, starts, lengths, pssm_comb, self.batch._offsets, thresholds, max_suff
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
                    data, starts, lengths, pssm_comb, self.batch._offsets, thresholds, max_suff,
                    offsets, out_pos, out_scores, out_seq_idxs, out_motif_idxs
                )
                seq_idxs_list.append(out_seq_idxs)
                pos_list.append(out_pos)
                scores_list.append(out_scores)
                motif_idxs_list.append(out_motif_idxs)
                strands_list.append(np.full(total, strand, dtype=np.int8))

        _process(self.batch._pssm_combined, self.batch._max_suffixes, 1)
        if self.batch._pssm_rc_combined is not None:
            _process(self.batch._pssm_rc_combined, self.batch._max_suffixes_rc, -1)

        if not seq_idxs_list: return MotifHitBatch.new_empty(self.batch)

        return MotifHitBatch(
            np.concatenate(seq_idxs_list), np.concatenate(pos_list), np.concatenate(scores_list),
            np.concatenate(strands_list), self.batch, np.concatenate(motif_idxs_list)
        )


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
                hits = MotifScanner(m).scan(self.fg, pvalue_threshold=1e-3)

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
