"""Pairwise sequence alignment engine with seeding, chaining, and dynamic programming."""
from typing import Union, Iterable
from enum import IntEnum

import numpy as np

from baclib.containers.seq import Seq, SeqBatch
from baclib.engines.index import SparseMapIndex, _find_hits_kernel
from baclib.containers.alignment import AlignmentBatch, Cigar
from baclib.lib.resources import RESOURCES, jit

if RESOURCES.has_module('numba'):
    from numba import prange
else:
    prange = range

# Constants ------------------------------------------------------------------------------------------------------------
class AlignmentMode(IntEnum):
    """Alignment strategy controlling how terminal gaps are scored."""
    LOCAL = 0
    GLOBAL = 1
    GLOCAL = 2


class Trace(IntEnum):
    """Traceback pointer flags stored in the DP matrix during alignment."""
    MATCH = 0
    E = 1
    F = 2
    STOP = 3
    E_EXT = 4
    F_EXT = 8


# Classes --------------------------------------------------------------------------------------------------------------
class ScoreMatrix:
    """
    Represents a substitution matrix for alignment.

    Attributes:
        _data (np.ndarray): The raw matrix data.

    Examples:
        >>> m = ScoreMatrix.build(4, match=2, mismatch=-2)
    """
    _DTYPE = np.int8
    __slots__ = ('_data',)
    
    def __init__(self, data: Union[np.ndarray, Iterable]):
        self._data = np.ascontiguousarray(data, dtype=self._DTYPE)
        self._data.flags.writeable = False

    def __getitem__(self, item): return self._data[item]
    def __array__(self, dtype=None): return self._data.astype(dtype, copy=False) if dtype else self._data
    def __repr__(self): return f"ScoreMatrix{self._data.shape}"
    @property
    def shape(self): return self._data.shape

    @classmethod
    def blosum62(cls):
        """Returns the BLOSUM62 matrix."""
        data = [
            4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2,
            0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2,
            -2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3,
            -1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2,
            -2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3,
            0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3,
            -2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2,
            -1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1,
            -1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -3, 2, 0, -3, -3, 1, 0, -3, -1, 2,
            -1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1,
            -1, -1, -3, -2, 0, -3, -2, 1, -3, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1,
            -2, -3, 1, 0, -3, 0, 1, -3, 2, -3, -2, 6, -2, -4, -4, -1, 0, -3, -1, -3,
            -1, -3, -1, -1, -4, -2, -2, -3, 0, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3,
            -1, -3, 0, 2, -3, -2, 0, -3, -3, -2, 0, -4, -1, 5, 1, 0, -1, -2, -2, -1,
            -1, -3, -2, 0, -3, -2, 0, -3, -3, -2, -1, -4, -2, 1, 5, -1, -1, -3, -3, -2,
            1, -1, 0, 0, -2, 0, -1, -2, 1, -2, -1, -1, -1, 0, -1, 4, 1, -2, -3, -2,
            0, -1, -1, -1, -2, -2, -2, -1, 0, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2,
            0, -1, -3, -2, -1, -3, -3, 3, -3, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1,
            -3, -2, -4, -3, 1, -2, -2, -3, -1, -2, -1, -1, -4, -2, -3, -3, -2, -3, 11, 2,
            -2, -2, -3, -2, 3, -3, 2, -1, 2, -1, -1, -3, -3, -1, -2, -2, -2, -1, 2, 7
        ]
        return cls(np.array(data, dtype=cls._DTYPE).reshape(20, 20))

    @classmethod
    def build(cls, n, match=1, mismatch=-1):
        """Builds a simple match/mismatch matrix."""
        M = np.full((n, n), mismatch, dtype=cls._DTYPE)
        np.fill_diagonal(M, match)
        return cls(M)

    def pad(self, shape=(256, 256), fill_value=None) -> np.ndarray:
        """Pads the matrix to handle full uint8 range (0-255) without bounds checks."""
        if self.shape == shape: return self._data
        if fill_value is None: fill_value = np.min(self)
        new_data = np.full(shape, fill_value, dtype=self._data.dtype)
        r, c = self.shape
        new_data[:r, :c] = self._data
        return new_data


class Aligner:
    """
    High-performance pairwise aligner.
    Optimized for Seeding -> Window Extraction -> DP.

    Examples:
        >>> aligner = Aligner()
        >>> aligner.build(target_batch)
        >>> hits = aligner.map(query_batch)
    """
    _REGISTRY = {}

    @classmethod
    def register(cls, mode: int):
        def decorator(func):
            cls._REGISTRY[mode] = func
            return func
        return decorator
        
    __slots__ = ('_index', '_score_matrix', '_data', 'gap_open', 'gap_extend', '_mode', 
                 '_compute_traceback', '_traceback_kernel', '_pad_val')
    def __init__(self, targets: Union[list[Seq], SeqBatch], index: SparseMapIndex,
                 mode: Union[str, AlignmentMode] = AlignmentMode.GLOCAL,
                 score_matrix: ScoreMatrix = None, match: int = 1,  mismatch: int = -1, gap_open: int = 5,
                 gap_extend: int = 2, compute_traceback: bool = False):

        if index.built: raise ValueError('Index already built')
        self._index = index
        self._data: SeqBatch = targets if isinstance(targets, SeqBatch) else index.alphabet.batch_from(targets)
        self._index.build(self._data)

        n_sym = len(self._index.alphabet)
        self._pad_val = n_sym

        if score_matrix is None:
            score_matrix = ScoreMatrix.build(n_sym, match, mismatch)
            fill = mismatch
        else:
            fill = np.min(score_matrix)

        self._score_matrix = score_matrix.pad(shape=(n_sym + 1, n_sym + 1), fill_value=fill)
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self._mode = AlignmentMode[mode.upper() if isinstance(mode, str) else mode]
        self._compute_traceback = compute_traceback
        # Registry lookup
        self._traceback_kernel = self._REGISTRY[self._mode]

    def map(self, queries: Union[list[Seq], SeqBatch], min_score: int = 0, min_seeds: int = 2,
            padding: int = 50, chain_gap: int = 500, chain_bandwidth: int = 100) -> AlignmentBatch:
        """
        Aligns queries to the built index.

        Args:
            queries: Query sequences.
            min_score: Minimum alignment score to report.
            min_seeds: Minimum seeds to trigger alignment.
            padding: Padding around seed clusters.
            chain_gap: Gap around seed clusters.
            chain_bandwidth: Bandwidth around seed clusters.

        Returns:
            AlignmentBatch containing the alignments.
        """
        # 1. Access Index Internals to get Seed Coordinates
        if not self._index.built: raise RuntimeError("Index not built")
        
        # Helper to compute hashes for the query batch
        q_hashes, q_starts, q_ids = self._index.compute_seeds(queries)
        
        # Pack Query ID and Position into uint64 to use the generic index kernel
        q_packed = (q_ids.astype(np.uint64) << 32) | q_starts.astype(np.uint64)
        
        raw_packed, raw_t_pos, raw_t_ids = _find_hits_kernel(
            q_hashes, q_packed,
            self._index._hashes, self._index._positions, self._index._ids
        )
        raw_q_ids = (raw_packed >> 32).astype(np.uint32)
        raw_q_pos = raw_packed.astype(np.uint32)
        
        if len(raw_q_ids) == 0: return AlignmentBatch.empty()

        # 2. Cluster & Bounding Box (Crucial Optimization)
        # Groups seeds by (q_id, t_id) and finds min/max coords
        k = self._index.k
        candidates = _chain_seeds_kernel(
            raw_q_ids, raw_t_ids, raw_q_pos, raw_t_pos,
            k, min_seeds, chain_gap, chain_bandwidth, padding,
            queries.arrays[2], self._data.arrays[2],
            is_global=(self._mode == AlignmentMode.GLOBAL)
        )
        if len(candidates) == 0: return AlignmentBatch.empty()

        # Unpack candidates: [q_idx, t_idx, qs, qe, ts, te]
        c_q_idx = candidates[:, 0]
        c_t_idx = candidates[:, 1]
        c_q_start = candidates[:, 2]
        c_q_end = candidates[:, 3]
        c_t_start = candidates[:, 4]
        c_t_end = candidates[:, 5]

        # 3. Batch Extension (DP on Windows)
        n_cand = len(candidates)
        out_scores = np.empty(n_cand, dtype=np.int32)
        out_end_coords = np.empty((n_cand, 2), dtype=np.int32)

        _batch_score_driver(
            queries.arrays, self._data.arrays,
            c_q_idx, c_t_idx, c_q_start, c_q_end, c_t_start, c_t_end,
            self._score_matrix, self.gap_open, self.gap_extend,
            out_scores, out_end_coords,
            mode=self._mode,
            pad_val=self._pad_val
        )
        # 4. Filter & Refine
        mask = out_scores >= min_score
        if not np.any(mask): return AlignmentBatch.empty()

        v_q_idx = c_q_idx[mask]
        v_t_idx = c_t_idx[mask]
        v_scores = out_scores[mask]

        # Adjust local coordinates back to global
        # out_end_coords are relative to the window start (c_q_start, c_t_start)
        v_ends = out_end_coords[mask]
        v_ends[:, 0] += c_q_start[mask]
        v_ends[:, 1] += c_t_start[mask]

        n_valid = len(v_scores)
        q_coords = np.zeros((n_valid, 2), dtype=np.int32)
        t_coords = np.zeros((n_valid, 2), dtype=np.int32)

        q_coords[:, 1] = v_ends[:, 0]
        t_coords[:, 1] = v_ends[:, 1]

        cigars = None
        if self._compute_traceback:
            cigars, q_starts, t_starts = _batch_traceback_driver(
                queries.arrays, self._data.arrays,
                v_q_idx, v_t_idx,
                c_q_start[mask], c_q_end[mask], c_t_start[mask], c_t_end[mask],
                v_ends,  # These are global ends
                self._score_matrix, self.gap_open, self.gap_extend,
                self._traceback_kernel,
                mode=self._mode.value
            )
            q_coords[:, 0] = q_starts
            t_coords[:, 0] = t_starts
        else:
            # Approx starts logic relies on window start for Local/Glocal
            if self._mode == AlignmentMode.GLOBAL:
                q_coords[:, 0] = 0
                t_coords[:, 0] = 0
            else:
                # Approximate start is the Window Start
                # (Ideally traceback is used for exact start)
                q_coords[:, 0] = c_q_start[mask]
                t_coords[:, 0] = c_t_start[mask]

        return AlignmentBatch.from_data(
            q_idx=v_q_idx, t_idx=v_t_idx, score=v_scores.astype(np.float32),
            q_coords=q_coords, t_coords=t_coords,
            q_lens=queries.arrays[2][v_q_idx], t_lens=self._data.arrays[2][v_t_idx],
            cigars=cigars
        )


# Drivers --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_score_driver(q_batch, t_batch, q_idxs, t_idxs,
                        q_starts, q_ends, t_starts, t_ends,
                        matrix, go, ge, scores, coords, pad_val,
                        mode=AlignmentMode.GLOCAL, batch_size=16):
    n = len(q_idxs)

    # 1. Processing in Thread Chunks (Parallelism)
    thread_chunk = 1024
    n_chunks = (n + thread_chunk - 1) // thread_chunk

    for chunk_i in prange(n_chunks):
        start_idx = chunk_i * thread_chunk
        end_idx = min(start_idx + thread_chunk, n)

        # OPTIMIZATION: Sort chunk by length to minimize padding overhead in micro-batches
        # This prevents a single long sequence from forcing a large DP matrix for short sequences
        chunk_len = end_idx - start_idx
        local_indices = np.arange(start_idx, end_idx)
        # Heuristic: Sort by Query Length
        lengths = q_ends[start_idx:end_idx] - q_starts[start_idx:end_idx]
        sort_order = np.argsort(lengths)
        sorted_indices = local_indices[sort_order]

        # 2. Processing in Micro-Batches (SIMD)
        for b_i in range(0, chunk_len, batch_size):
            b_end_offset = min(b_i + batch_size, chunk_len)
            curr_bs = b_end_offset - b_i

            # --- PASS 1: Calculate Max Dimensions ---
            max_len_q = 0
            max_len_t = 0

            for k in range(curr_bs):
                job_idx = sorted_indices[b_i + k]
                # Calculate lengths
                l_q = q_ends[job_idx] - q_starts[job_idx]
                l_t = t_ends[job_idx] - t_starts[job_idx]

                if l_q > max_len_q: max_len_q = l_q
                if l_t > max_len_t: max_len_t = l_t

            # --- ALLOCATION: Dynamic & Exact ---
            # Allocating small arrays in Numba is very fast (stack-like allocation)
            # Use pad_val (Sentinel) for padding to ensure mismatches in ScoreMatrix
            # OPTIMIZATION: Transpose to (Len, Batch) for contiguous memory access in kernel
            seq_q_buf = np.full((max_len_q, curr_bs), pad_val, dtype=np.uint8)
            seq_t_buf = np.full((max_len_t, curr_bs), pad_val, dtype=np.uint8)
            batch_q_lens = np.empty(curr_bs, dtype=np.int32)
            batch_t_lens = np.empty(curr_bs, dtype=np.int32)

            # --- PASS 2: Fill Buffers ---
            for k in range(curr_bs):
                job_idx = sorted_indices[b_i + k]
                qi, ti = q_idxs[job_idx], t_idxs[job_idx]
                qs, ts = q_starts[job_idx], t_starts[job_idx]

                # Lengths again (cheap integer math)
                l_q = q_ends[job_idx] - qs
                l_t = t_ends[job_idx] - ts
                batch_q_lens[k] = l_q
                batch_t_lens[k] = l_t

                p_q = q_batch[1][qi] + qs
                p_t = t_batch[1][ti] + ts

                # Safe Copy
                seq_q_buf[:l_q, k] = q_batch[0][p_q: p_q + l_q]
                seq_t_buf[:l_t, k] = t_batch[0][p_t: p_t + l_t]

                # --- EXECUTE ---
                # We need to map output back to the original indices
                # The kernel writes to scores[0..curr_bs], we need to scatter them back
                # But the kernel takes slices. We can pass a temporary buffer or write directly if contiguous.
                # Since we reordered, we cannot write directly to a slice of 'scores'.

                batch_scores = np.empty(curr_bs, dtype=np.int32)
                batch_coords = np.empty((curr_bs, 2), dtype=np.int32)

                _micro_batch_kernel(
                    seq_q_buf,  # Passing full buffer is fine, it is sized exactly
                    seq_t_buf,
                    matrix, go, ge,
                    batch_scores,
                    batch_coords,
                    batch_q_lens, batch_t_lens, mode
                )

                # Scatter results back
                for k in range(curr_bs):
                    job_idx = sorted_indices[b_i + k]
                    scores[job_idx] = batch_scores[k]
                    coords[job_idx] = batch_coords[k]
            


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _micro_batch_kernel(Q, T, matrix, go, ge, out_scores, out_coords, q_lens, t_lens, mode):
    """
    Computes generic affine gap alignment for a batch of sequences.
    Q, T: 2D arrays (Len, Batch) -- Transposed for contiguous access
    """
    q_len, bs = Q.shape
    t_len, _ = T.shape

    # Initialize Score Vectors (SIMD)
    _go = np.int32(go)
    _ge = np.int32(ge)
    INF = 1_000_000_000

    # --- Memory Layout Optimization ---
    # Transpose H and F to (QLen, Batch) so that H[:, r] is contiguous.
    # This allows efficient SIMD loads/stores for the batch.
    H_col = np.empty((q_len + 1, bs), dtype=np.int32)
    F_col = np.empty((q_len + 1, bs), dtype=np.int32)

    # --- Initialization (Column 0) ---
    # Local: 0
    # Global/Glocal: Gap penalties (Vertical/Query gaps)
    if mode == AlignmentMode.LOCAL:
        H_col[:, :] = 0
        F_col[:, :] = -INF
    else:
        # Global/Glocal: Initialize first column with gap penalties
        H_col[0, :] = 0
        F_col[0, :] = -INF
        for r in range(1, q_len + 1):
            val = -(_go + (r - 1) * _ge)
            H_col[r, :] = val
            F_col[r, :] = -INF

    # Output tracking
    max_scores = np.full(bs, -INF, dtype=np.int32)
    max_r = np.zeros(bs, dtype=np.int32)
    max_c = np.zeros(bs, dtype=np.int32)

    # Pre-allocate match buffer to avoid allocation in inner loop
    match_scores = np.empty(bs, dtype=np.int32)

    # --- Outer Loop: Target (Columns) ---
    for c in range(t_len):
        chars_t = T[c, :] # Contiguous load

        # Init E (Horizontal Gap) for the batch
        E = np.full(bs, -INF, dtype=np.int32)

        # H_diag starts as H_col[:, 0]
        # Note: H_col is (QLen, Batch), so H_col[0] is the top row (Batch size)
        H_diag = H_col[0, :].copy()

        # Update 0-th Row (Horizontal/Target Gaps)
        if mode == AlignmentMode.GLOBAL:
            # Global: Penalize gaps at start of Target
            val = -(_go + c * _ge)
            H_col[0, :] = val
        else:
            # Local/Glocal: Free gaps at start of Target
            H_col[0, :] = 0

        # --- Inner Loop: Query (Rows) ---
        for r in range(1, q_len + 1):
            chars_q = Q[r - 1, :] # Contiguous load

            # 1. Calculate Scores (Vectorized)
            # Fetch H_up (H_col[r] before update) - Contiguous load
            H_up = H_col[r, :]

            # Fetch F (Vertical Gap)
            f_score = F_col[r, :]

            # Vector Ops
            term1 = H_up - _go
            term2 = f_score - _ge
            new_F = np.maximum(term1, term2)
            F_col[r, :] = new_F

            # Calculate E (Horizontal Gap)
            h_left = H_up
            term3 = h_left - _go
            term4 = E - _ge
            new_E = np.maximum(term3, term4)
            E = new_E

            # Calculate Match
            # Random access into score matrix (Gather)
            for k in range(bs):
                match_scores[k] = matrix[chars_q[k], chars_t[k]]

            score = match_scores + H_diag

            # Maximize
            best = np.maximum(score, new_E)
            best = np.maximum(best, new_F)
            
            if mode == AlignmentMode.LOCAL:
                best = np.maximum(best, np.int32(0))

            # Update Traceback / H_diag
            H_diag = h_left  # Old H[r] becomes diag for r+1

            # Store result
            H_col[r, :] = best

            # --- Max Tracking ---
            for k in range(bs):
                val = best[k]
                
                if mode == AlignmentMode.LOCAL:
                    # Local: Max anywhere
                    if val > max_scores[k]:
                        max_scores[k] = val
                        max_r[k] = r
                        max_c[k] = c + 1
                        
                elif mode == AlignmentMode.GLOBAL:
                    # Global: Score at exact end (bottom-right)
                    if r == q_lens[k] and c == t_lens[k] - 1:
                        max_scores[k] = val
                        max_r[k] = r
                        max_c[k] = c
                        max_c[k] = c + 1
                        
                elif mode == AlignmentMode.GLOCAL:
                    # Glocal: Max at end of Query (last row), anywhere in Target
                    if r == q_lens[k]:
                        if val > max_scores[k]:
                            max_scores[k] = val
                            max_r[k] = r
                            max_c[k] = c + 1

    # Write back
    for k in range(bs):
        out_scores[k] = max_scores[k]
        out_coords[k, 0] = max_r[k]
        out_coords[k, 1] = max_c[k]


def _batch_traceback_driver(q_batch, t_batch, q_idxs, t_idxs,
                            q_starts, q_ends, t_starts, t_ends,
                            global_ends, matrix, go, ge,
                            back_kernel, mode):
    """
    Python wrapper that orchestrates the JIT kernel and formats the final CIGAR strings.
    """
    n = len(q_idxs)

    # Calculate relative ends for the window
    rel_ends = np.empty((n, 2), dtype=np.int32)
    rel_ends[:, 0] = global_ends[:, 0] - q_starts
    rel_ends[:, 1] = global_ends[:, 1] - t_starts

    cigar_counts, cigar_ops, cigar_cnts, out_qs, out_ts = _batch_traceback_loop(
        q_batch, t_batch, q_idxs, t_idxs,
        q_starts, q_ends, t_starts, t_ends,
        rel_ends, matrix, go, ge,
        back_kernel, mode
    )

    out_cigars = np.empty(n, dtype=object)
    op_lookup = Cigar._OP_BYTES_LOOKUP

    for i in range(n):
        count = cigar_counts[i]
        if count == 0:
            out_cigars[i] = b""
        else:
            # Extract valid slice from the 2D buffer
            parts = [b"%d" % cigar_cnts[i, k] + op_lookup[cigar_ops[i, k]] for k in range(count)]
            out_cigars[i] = b"".join(parts)

    # Adjust starts back to global
    final_qs = out_qs + q_starts
    final_ts = out_ts + t_starts

    return out_cigars, final_qs, final_ts


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_traceback_loop(q_batch, t_batch, q_idxs, t_idxs,
                          q_starts, q_ends, t_starts, t_ends,
                          ends, matrix, go, ge,
                          back_kernel, mode):
    n = len(q_idxs)
    out_qs = np.empty(n, dtype=np.int32)
    out_ts = np.empty(n, dtype=np.int32)
    
    # Pre-allocate buffers for parallel execution
    # We can't easily pre-allocate a single 2D array for ops/cnts if sizes vary wildly
    # without max_rows/cols. However, we can use a safe upper bound or dynamic lists.
    # For Numba parallel, dynamic lists are tricky.
    # Strategy: Use a conservative max length based on the specific alignment's window.
    # But we need a rectangular array to return.
    # We will use a list of arrays for the output ops/cnts to handle variable lengths efficiently.
    # Numba supports list of arrays.
    
    # Actually, to keep it simple and compatible with the driver, we can just use a large enough buffer
    # derived from the max window in the batch (which we removed from args, but can re-calculate or just use a safe default).
    # Better: Calculate max window here.
    
    max_q_len = 0
    max_t_len = 0
    for i in range(n):
        ql = q_ends[i] - q_starts[i]
        tl = t_ends[i] - t_starts[i]
        if ql > max_q_len: max_q_len = ql
        if tl > max_t_len: max_t_len = tl
        
    max_cigar_len = max_q_len + max_t_len + 100
    
    out_cigar_ops = np.empty((n, max_cigar_len), dtype=np.uint8)
    out_cigar_cnts = np.empty((n, max_cigar_len), dtype=np.int32)
    cigar_counts = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        qi, ti = q_idxs[i], t_idxs[i]
        qs_g, qe_g = q_starts[i], q_ends[i]
        ts_g, te_g = t_starts[i], t_ends[i]

        seq_q_ptr = q_batch[1][qi] + qs_g
        seq_t_ptr = t_batch[1][ti] + ts_g

        ql = qe_g - qs_g
        tl = te_g - ts_g

        # Thread-local trace buffer
        # Allocate exactly what is needed for this alignment
        trace_buf = np.zeros((ql + 1, tl + 1), dtype=np.uint8)

        # Slices
        seq1 = q_batch[0][seq_q_ptr: seq_q_ptr + ql]
        seq2 = t_batch[0][seq_t_ptr: seq_t_ptr + tl]
        band = abs(ql - tl) + 50

        # Re-run full DP (traceback mode) on the window
        _generic_full_align(seq1, seq2, matrix, go, ge, -band, band, trace_buf, mode)

        # Override max_pos with the actual end found in Score phase (passed in 'ends')
        # This ensures consistency between score and alignment
        # Note: ends is (r, c)
        force_end = (ends[i, 0], ends[i, 1])

        # Kernel returns ops in REVERSE order (End -> Start)
        c_ops, c_counts, rs, cs, matches = back_kernel(
            trace_buf, seq1, seq2, force_end
        )

        out_qs[i] = rs
        out_ts[i] = cs
        
        count = len(c_ops)
        cigar_counts[i] = count
        for k in range(count):
            # Reverse back to Start -> End
            out_cigar_ops[i, k] = c_ops[count - 1 - k]
            out_cigar_cnts[i, k] = c_counts[count - 1 - k]

    return cigar_counts, out_cigar_ops, out_cigar_cnts, out_qs, out_ts


@jit(nopython=True, cache=True, nogil=True)
def _chain_seeds_kernel(q_ids, t_ids, q_pos, t_pos, k, min_seeds, max_gap, bandwidth, padding, q_lens, t_lens, is_global):
    """
    Groups seeds by (q_id, t_id) and performs diagonal chaining to identify candidate regions.
    Returns array of [q_idx, t_idx, q_start, q_end, t_start, t_end].
    """
    n = len(q_ids)
    if n == 0: return np.empty((0, 6), dtype=np.int32)

    # 1. Sort by (q_id, t_id, q_pos)
    # Sort by Q_POS first (stable sort will preserve this order within groups)
    order = np.argsort(q_pos, kind='mergesort')

    # Sort by Pair (Q_ID, T_ID)
    packed = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx = order[i]
        packed[i] = (np.int64(q_ids[idx]) << 32) | np.int64(t_ids[idx])

    group_sorter = np.argsort(packed, kind='mergesort')
    final_order = order[group_sorter]

    # 2. Iterate and Merge (Pre-allocated array for speed)
    out_arr = np.empty((n, 6), dtype=np.int32)
    out_count = 0

    curr_q = -1
    curr_t = -1
    
    # Chain state
    c_min_q = 0; c_max_q = 0
    c_min_t = 0; c_max_t = 0
    c_count = 0
    
    last_q = -1
    last_t = -1

    for i in range(n):
        idx = final_order[i]
        qid = q_ids[idx]
        tid = t_ids[idx]
        qp = q_pos[idx]
        tp = t_pos[idx]

        if qid != curr_q or tid != curr_t:
            # Commit previous chain
            if c_count >= min_seeds:
                # Inline emission logic
                if is_global:
                    qs = 0
                    qe = q_lens[curr_q]
                    ts = 0
                    te = t_lens[curr_t]
                else:
                    qs = max(0, c_min_q - padding)
                    qe = min(q_lens[curr_q], c_max_q + padding)
                    ts = max(0, c_min_t - padding)
                    te = min(t_lens[curr_t], c_max_t + padding)
                
                out_arr[out_count, 0] = curr_q
                out_arr[out_count, 1] = curr_t
                out_arr[out_count, 2] = qs; out_arr[out_count, 3] = qe
                out_arr[out_count, 4] = ts; out_arr[out_count, 5] = te
                out_count += 1

            # Reset
            curr_q = qid
            curr_t = tid
            c_min_q = qp; c_max_q = qp + k
            c_min_t = tp; c_max_t = tp + k
            c_count = 1
            last_q = qp
            last_t = tp
        else:
            # Check Chain Condition
            if is_global:
                # Global: Merge everything
                if qp < c_min_q: c_min_q = qp
                if qp + k > c_max_q: c_max_q = qp + k
                if tp < c_min_t: c_min_t = tp
                if tp + k > c_max_t: c_max_t = tp + k
                c_count += 1
            else:
                # Local/Glocal: Diagonal Chaining
                dq = qp - last_q
                dt = tp - last_t
                diag_diff = abs(dt - dq)
                
                compatible = True
                if dq > max_gap: compatible = False
                elif diag_diff > bandwidth: compatible = False
                elif dt < -bandwidth: compatible = False # Allow small jitter
                
                if compatible:
                    # Extend
                    if qp < c_min_q: c_min_q = qp
                    if qp + k > c_max_q: c_max_q = qp + k
                    if tp < c_min_t: c_min_t = tp
                    if tp + k > c_max_t: c_max_t = tp + k
                    c_count += 1
                    last_q = qp
                    last_t = tp
                else:
                    # Break & Emit
                    if c_count >= min_seeds:
                        if is_global:
                            qs, qe = 0, q_lens[curr_q]
                            ts, te = 0, t_lens[curr_t]
                        else:
                            qs = max(0, c_min_q - padding)
                            qe = min(q_lens[curr_q], c_max_q + padding)
                            ts = max(0, c_min_t - padding)
                            te = min(t_lens[curr_t], c_max_t + padding)
                        
                        out_arr[out_count, 0] = curr_q
                        out_arr[out_count, 1] = curr_t
                        out_arr[out_count, 2] = qs; out_arr[out_count, 3] = qe
                        out_arr[out_count, 4] = ts; out_arr[out_count, 5] = te
                        out_count += 1
                    
                    # Start new chain
                    c_min_q = qp; c_max_q = qp + k
                    c_min_t = tp; c_max_t = tp + k
                    c_count = 1
                    last_q = qp
                    last_t = tp

    # Final Commit
    if c_count >= min_seeds:
        if is_global:
            qs, qe = 0, q_lens[curr_q]
            ts, te = 0, t_lens[curr_t]
        else:
            qs = max(0, c_min_q - padding)
            qe = min(q_lens[curr_q], c_max_q + padding)
            ts = max(0, c_min_t - padding)
            te = min(t_lens[curr_t], c_max_t + padding)
        
        out_arr[out_count, 0] = curr_q
        out_arr[out_count, 1] = curr_t
        out_arr[out_count, 2] = qs; out_arr[out_count, 3] = qe
        out_arr[out_count, 4] = ts; out_arr[out_count, 5] = te
        out_count += 1

    return out_arr[:out_count]



# --- Buffered Full Kernels ---

@jit(nopython=True, cache=True, nogil=True)
def _generic_full_align(seq1, seq2, matrix, gap_open, gap_extend, min_diag, max_diag, trace, mode):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000
    
    H = np.empty(cols, dtype=np.int32)
    F = np.empty(cols, dtype=np.int32)
    
    # --- Initialization ---
    if mode == AlignmentMode.LOCAL:
        H[:] = 0
        F[:] = NEG_INF
        trace[0, :] = 0
    else:
        # Global / Glocal
        H[0] = 0
        F[0] = NEG_INF
        for c in range(1, cols):
            H[c] = -(gap_open + (c - 1) * gap_extend)
            F[c] = NEG_INF
            trace[0, c] = Trace.E | (Trace.E_EXT if c > 1 else 0)
            
    max_score = NEG_INF
    best_c = 0
    global_max = 0
    max_r = 0
    max_c = 0
    
    for r in range(1, rows):
        h_diag = H[0]
        
        if mode == AlignmentMode.LOCAL:
            H[0] = 0
            trace[r, 0] = 0
        else:
            H[0] = -(gap_open + (r - 1) * gap_extend)
            trace[r, 0] = Trace.F | (Trace.F_EXT if r > 1 else 0)
            
        running_E = NEG_INF
        char_q = seq1[r - 1]
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            h_up = H[c]
            f_ext = F[c] - gap_extend
            f_open = h_up - gap_open
            if f_ext >= f_open:
                F[c] = f_ext
                f_bit = Trace.F_EXT
            else:
                F[c] = f_open
                f_bit = 0
            h_left = H[c - 1]
            e_ext = running_E - gap_extend
            e_open = h_left - gap_open
            if e_ext >= e_open:
                running_E = e_ext
                e_bit = Trace.E_EXT
            else:
                running_E = e_open
                e_bit = 0
            match = matrix[char_q, seq2[c - 1]] + h_diag
            best = match
            source = Trace.MATCH
            if F[c] > best:
                best = F[c]
                source = Trace.F
            if running_E > best:
                best = running_E
                source = Trace.E
            
            if mode == AlignmentMode.LOCAL:
                if best < 0:
                    best = 0
                    source = Trace.STOP # Stop
            
            h_diag = h_up
            H[c] = best
            trace[r, c] = source | e_bit | f_bit
            
            if mode == AlignmentMode.LOCAL:
                if best > global_max:
                    global_max = best
                    max_r = r
                    max_c = c

    if mode == AlignmentMode.GLOBAL:
        return H[cols - 1], rows - 1, cols - 1
    elif mode == AlignmentMode.LOCAL:
        return global_max, max_r, max_c
    else: # GLOCAL
        for c in range(start_c, end_c):
            if H[c] > max_score:
                max_score = H[c]
                best_c = c
        return max_score, rows - 1, best_c


# --- Traceback Kernels ---

@Aligner.register(AlignmentMode.GLOCAL)
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _glocal_traceback_kernel(trace, seq1, seq2, max_pos):
    r, c = max_pos
    
    # Upper bound for ops is r + c
    max_ops = r + c
    ops = np.empty(max_ops, dtype=np.uint8)
    counts = np.empty(max_ops, dtype=np.int32)
    
    k = 0
    matches = 0
    
    curr_op = 255 # Invalid
    curr_count = 0
    
    while r > 0:
        op = 255
        if c == 0:
            op = 2 # D (Deletion from Ref / Insertion to Query)
            r -= 1
        else:
            flag = trace[r, c]
            h_src = flag & 3
            if h_src == Trace.MATCH:
                op = 0 # M
                if seq1[r - 1] == seq2[c - 1]: matches += 1
                r -= 1
                c -= 1
            elif h_src == Trace.F:
                op = 2 # D
                r -= 1
            elif h_src == Trace.E:
                op = 1 # I
                c -= 1
        
        if op == curr_op:
            curr_count += 1
        else:
            if curr_op != 255:
                ops[k] = curr_op
                counts[k] = curr_count
                k += 1
            curr_op = op
            curr_count = 1
            
    if curr_op != 255:
        ops[k] = curr_op
        counts[k] = curr_count
        k += 1
        
    return ops[:k], counts[:k], 0, c, matches


@Aligner.register(AlignmentMode.LOCAL)
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _local_traceback_kernel(trace, seq1, seq2, max_pos):
    r, c = max_pos
    max_ops = r + c
    ops = np.empty(max_ops, dtype=np.uint8)
    counts = np.empty(max_ops, dtype=np.int32)
    k = 0
    matches = 0
    curr_op = 255
    curr_count = 0

    while r > 0 and c > 0:
        flag = trace[r, c]
        h_src = flag & 3
        op = 255
        if h_src == Trace.MATCH:
            op = 0
            if seq1[r - 1] == seq2[c - 1]: matches += 1
            r -= 1
            c -= 1
        elif h_src == Trace.F:
            op = 2
            r -= 1
        elif h_src == Trace.E:
            op = 1
            c -= 1
        else:
            break  # Should catch 0 case
        
        if op == curr_op:
            curr_count += 1
        else:
            if curr_op != 255:
                ops[k] = curr_op
                counts[k] = curr_count
                k += 1
            curr_op = op
            curr_count = 1
            
    if curr_op != 255:
        ops[k] = curr_op
        counts[k] = curr_count
        k += 1
    return ops[:k], counts[:k], r, c, matches


@Aligner.register(AlignmentMode.GLOBAL)
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _global_traceback_kernel(trace, seq1, seq2, max_pos):
    r, c = max_pos
    max_ops = r + c
    ops = np.empty(max_ops, dtype=np.uint8)
    counts = np.empty(max_ops, dtype=np.int32)
    k = 0
    matches = 0
    curr_op = 255
    curr_count = 0

    while r > 0 or c > 0:
        op = 255
        if r == 0:
            op = 1 # I
            c -= 1
        elif c == 0:
            op = 2 # D
            r -= 1
        else:
            flag = trace[r, c]
            h_src = flag & 3
            if h_src == Trace.MATCH:
                op = 0
                if seq1[r - 1] == seq2[c - 1]: matches += 1
                r -= 1
                c -= 1
            elif h_src == Trace.F:
                op = 2
                r -= 1
            elif h_src == Trace.E:
                op = 1
                c -= 1
        
        if op == curr_op:
            curr_count += 1
        else:
            if curr_op != 255:
                ops[k] = curr_op
                counts[k] = curr_count
                k += 1
            curr_op = op
            curr_count = 1
            
    if curr_op != 255:
        ops[k] = curr_op
        counts[k] = curr_count
        k += 1
    return ops[:k], counts[:k], 0, 0, matches
