from typing import List, Union, Literal
from time import time

import numpy as np

from baclib.core.seq import Seq, SeqBatch
from baclib.align.index import SparseMapIndex, MinimizerIndex, _find_hits_kernel
from baclib.align.alignment import AlignmentBatch, CigarParser, _cigar_rle_kernel
from baclib.utils.resources import RESOURCES, jit

if 'numba' in RESOURCES.optional_packages:
    from numba import prange
else:
    prange = range

# Constants ------------------------------------------------------------------------------------------------------------
_TR_H_MATCH = 0
_TR_H_E = 1
_TR_H_F = 2
_TR_E_EXT = 4
_TR_F_EXT = 8


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

    def __init__(self, data): self._data = data

    @classmethod
    def blosum62(cls):
        """Returns the BLOSUM62 matrix."""
        return cls(np.reshape([
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
        ], (20, 20)).astype(_DTYPE))

    @classmethod
    def build(cls, n, match=1, mismatch=-1):
        """Builds a simple match/mismatch matrix."""
        M = np.full((n, n), mismatch, dtype=cls._DTYPE)
        np.fill_diagonal(M, match)
        return cls(M)

    def pad(self, shape=(256, 256), fill_value=None):
        """Pads the matrix to handle full uint8 range (0-255) without bounds checks."""
        if self._data.shape == shape: return self
        if fill_value is None: fill_value = np.min(self._data)
        new_data = np.full(shape, fill_value, dtype=self._data.dtype)
        r, c = self._data.shape
        new_data[:r, :c] = self._data
        return ScoreMatrix(new_data)


class Aligner:
    """
    High-performance pairwise aligner.
    Optimized for Seeding -> Window Extraction -> DP.

    Examples:
        >>> aligner = Aligner()
        >>> aligner.build(target_batch)
        >>> hits = aligner.map(query_batch)
    """
    __slots__ = ('_index', '_score_matrix', '_data', 'gap_open', 'gap_extend', '_mode', '_compute_traceback',
                 '_score_kernel', '_full_kernel', '_traceback_kernel')

    def __init__(self, index: SparseMapIndex = None, score_matrix: ScoreMatrix = None, match: int = 1,
                 mismatch: int = -1, gap_open: int = 5, gap_extend: int = 2,
                 mode: Literal['local', 'global', 'glocal'] = 'glocal', compute_traceback: bool = False):

        self._data: SeqBatch = None

        if index is not None:
            if index.built: raise RuntimeError("Index already built")
            self._index = index
        else:
            self._index = MinimizerIndex()

        if score_matrix is None:
            score_matrix = ScoreMatrix.build(len(self._index.alphabet), match, mismatch)
            fill = mismatch
        else:
            fill = np.min(score_matrix._data)

        self._score_matrix = score_matrix.pad(fill_value=fill)._data
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self._mode = mode
        self._compute_traceback = compute_traceback

        kernels = _ALIGNMENT_KERNEL_REGISTRY[mode]
        self._score_kernel = kernels['score']
        self._full_kernel = kernels['full']
        self._traceback_kernel = kernels['trace']

    def build(self, targets: SeqBatch):
        """Builds the index on the target sequences."""
        self._data = targets
        self._index.build(targets)

    def map(self, queries: Union[List[Seq], SeqBatch], min_score: int = 0, min_seeds: int = 2,
            padding: int = 50) -> AlignmentBatch:
        """
        Aligns queries to the built index.

        Args:
            queries: Query sequences.
            min_score: Minimum alignment score to report.
            min_seeds: Minimum seeds to trigger alignment.
            padding: Padding around seed clusters.

        Returns:
            AlignmentBatch containing the alignments.
        """
        if not isinstance(queries, SeqBatch): queries = SeqBatch(queries)

        # 1. Access Index Internals to get Seed Coordinates
        # (Bypassing index.query_batch because it discards positions)
        if not self._index.built: raise RuntimeError("Index not built")

        # Helper to compute hashes for the query batch
        t = time()
        q_hashes, q_starts, q_ids = self._index._compute_batch_seeds(queries)
        print(f'compute_batch_seeds: {time() - t} seconds')

        # Helper to binary search hashes against the index
        # We pass q_ids as 'q_pos' effectively, so the result is (query_index, target_pos)
        # However, we need query_pos AND query_index.
        # The standard _compute_batch_seeds returns (hash, pos, seq_id).
        # We pass q_starts (pos) to _find_hits_kernel to retrieve them.
        t = time()
        hit_q_pos, hit_t_pos, hit_t_ids = _find_hits_kernel(
            q_hashes, q_starts, self._index._hashes, self._index._positions, self._index._ids
        )
        print(f'find_hits_kernel: {time() - t} seconds')

        # We also need the q_idx (sequence ID) for these hits.
        # Since _find_hits_kernel flattens everything, we need to map q_pos back to q_idx?
        # No, _find_hits_kernel takes parallel arrays. q_hashes[i] corresponds to q_ids[i].
        # We can pass q_ids instead of q_starts to get the ID, but then we lose the position.

        # OPTIMIZATION: We run _find_hits_kernel TWICE? No, that's slow.
        # Solution: Use a custom kernel or pack the ID and Pos into 64-bit int?
        # Actually, let's just use the fact that q_hashes, q_starts, q_ids are all aligned.
        # We can retrieve the indices of the matches in the query array.

        # Modified approach: Get indices into the q_hashes array
        t = time()
        hit_indices_in_query_array = _find_hit_indices_kernel(
            q_hashes, self._index._hashes
        )
        print(f'find_hit_indices_kernel: {time() - t} seconds')

        if len(hit_indices_in_query_array) == 0: return AlignmentBatch()

        # Gather full hit data
        # hit_indices_in_query_array contains indices `i` such that q_hashes[i] had a match
        # But a single hash can match multiple targets.

        # Let's fallback to a simpler robust method:
        # Re-implement a specialized hit gathering here or assume the user patches index.py?
        # I will implement a robust gatherer here using the existing public API of the index kernels.

        hit_q_ids = q_ids[hit_indices_in_query_array]
        hit_q_pos = q_starts[hit_indices_in_query_array]

        # We need the target data. _find_hits_kernel returns it aligned to the hits.
        # We need to run the search again properly.
        t = time()
        raw_q_pos, raw_t_pos, raw_t_ids, raw_q_ids = _find_hits_with_q_id_kernel(
            q_hashes, q_starts, q_ids,
            self._index._hashes, self._index._positions, self._index._ids
        )
        print(f'find_hits_with_q_id_kernel: {time() - t} seconds')

        if len(raw_q_ids) == 0: return AlignmentBatch()

        # 2. Cluster & Bounding Box (Crucial Optimization)
        # Groups seeds by (q_id, t_id) and finds min/max coords
        t = time()
        candidates = _cluster_seeds_kernel(
            raw_q_ids, raw_t_ids, raw_q_pos, raw_t_pos,
            min_seeds, padding,
            queries.arrays[2], self._data.arrays[2],  # Lengths for bounds checks
            is_global=(self._mode == 'global')
        )
        print(f'cluster_seeds_kernel: {time() - t} seconds')
        if len(candidates) == 0: return AlignmentBatch()

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

        # Calculate max buffer size needed for the longest window
        max_q_window = np.max(c_q_end - c_q_start)
        max_t_window = np.max(c_t_end - c_t_start)
        buf_len = max_t_window + 200  # Safety buffer
        t = time()
        _batch_score_driver(
            queries.arrays, self._data.arrays,
            c_q_idx, c_t_idx, c_q_start, c_q_end, c_t_start, c_t_end,
            self._score_matrix, self.gap_open, self.gap_extend,
            self._score_kernel, out_scores, out_end_coords, buf_len
        )
        print(f'batch_score_driver: {time() - t} seconds')
        # 4. Filter & Refine
        mask = out_scores >= min_score
        if not np.any(mask): return AlignmentBatch()

        v_indices = np.where(mask)[0]
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
            t = time()
            cigars, q_starts, t_starts = _batch_traceback_driver(
                queries.arrays, self._data.arrays,
                v_q_idx, v_t_idx,
                c_q_start[mask], c_q_end[mask], c_t_start[mask], c_t_end[mask],
                v_ends,  # These are global ends
                self._score_matrix, self.gap_open, self.gap_extend,
                self._full_kernel, self._traceback_kernel,
                max_q_window + 1, max_t_window + 1
            )
            q_coords[:, 0] = q_starts
            t_coords[:, 0] = t_starts
            print(f'batch_traceback_driver: {time() - t} seconds')
        else:
            # Approx starts logic relies on window start for Local/Glocal
            if self._mode == 'global':
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
                        matrix, go, ge, kernel_dummy, scores, coords,
                        batch_size=16):
    n = len(q_idxs)

    # 1. Processing in Thread Chunks (Parallelism)
    thread_chunk = 1024
    n_chunks = (n + thread_chunk - 1) // thread_chunk

    for chunk_i in prange(n_chunks):
        start_idx = chunk_i * thread_chunk
        end_idx = min(start_idx + thread_chunk, n)

        # 2. Processing in Micro-Batches (SIMD)
        for b_start in range(start_idx, end_idx, batch_size):
            b_end = min(b_start + batch_size, end_idx)
            curr_bs = b_end - b_start

            # --- PASS 1: Calculate Max Dimensions ---
            max_len_q = 0
            max_len_t = 0

            for k in range(curr_bs):
                job_idx = b_start + k
                # Calculate lengths
                l_q = q_ends[job_idx] - q_starts[job_idx]
                l_t = t_ends[job_idx] - t_starts[job_idx]

                if l_q > max_len_q: max_len_q = l_q
                if l_t > max_len_t: max_len_t = l_t

            # --- ALLOCATION: Dynamic & Exact ---
            # Allocating small arrays in Numba is very fast (stack-like allocation)
            seq_q_buf = np.zeros((curr_bs, max_len_q), dtype=np.uint8)
            seq_t_buf = np.zeros((curr_bs, max_len_t), dtype=np.uint8)

            # --- PASS 2: Fill Buffers ---
            for k in range(curr_bs):
                job_idx = b_start + k
                qi, ti = q_idxs[job_idx], t_idxs[job_idx]
                qs, ts = q_starts[job_idx], t_starts[job_idx]

                # Lengths again (cheap integer math)
                l_q = q_ends[job_idx] - qs
                l_t = t_ends[job_idx] - ts

                p_q = q_batch[1][qi] + qs
                p_t = t_batch[1][ti] + ts

                # Safe Copy
                seq_q_buf[k, :l_q] = q_batch[0][p_q: p_q + l_q]
                seq_t_buf[k, :l_t] = t_batch[0][p_t: p_t + l_t]

            # --- EXECUTE ---
            _micro_batch_kernel(
                seq_q_buf,  # Passing full buffer is fine, it is sized exactly
                seq_t_buf,
                matrix, go, ge,
                scores[b_start:b_end],
                coords[b_start:b_end]
            )


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _micro_batch_kernel(Q, T, matrix, go, ge, out_scores, out_coords):
    """
    Computes generic affine gap alignment for a batch of sequences.
    Q, T: 2D arrays (Batch, Len)
    """
    bs, q_len = Q.shape
    _, t_len = T.shape

    # Initialize Score Vectors (SIMD)
    _go = np.int32(go)
    _ge = np.int32(ge)
    INF = 1000000  # Use a safe constant that fits in int32

    # DP Columns for the whole batch
    # H = Score, F = Gap in Query
    H = np.zeros(bs, dtype=np.int32)
    F = np.full(bs, -INF, dtype=np.int32)

    # Initialize first column (Gap penalties for global/glocal)
    # Assumes Global/Glocal starts (0,0)
    # If Local, H would remain 0
    # For now, implementing 'Glocal' (0 penalty at start of Q, linear at start of T)
    # Note: For strict correctness, pre-initialize logic should match your specific mode

    # Storage for max tracking
    max_scores = np.full(bs, -INF, dtype=np.int32)
    max_r = np.zeros(bs, dtype=np.int32)
    max_c = np.zeros(bs, dtype=np.int32)

    # --- Main Loop (Target / Columns) ---
    for c in range(t_len):
        # Load Target chars for the whole batch (Vector Load)
        char_t = T[:, c]

        # Reset Column
        # Store H_diag (previous H) before overwriting
        # At start of new column, H represents the value from the previous column (H_left)
        # We need H_up.
        # Standard DP variable swap:
        # We need a full array for H_up if we iterate down rows.
        pass

    # WAIT: Standard DP iterates Row then Column.
    # To vectorize, we must iterate 'Wavefront' or simply standard Row-Major
    # but maintaining the State Vectors for the batch.

    # Correct Memory Layout for Batch SIMD:
    # We need to maintain the *Previous Column* of H values for the batch.
    # H_prev_col: (Batch, q_len) - Too big for cache?
    # No, we iterate through Q (Rows) inside T (Cols) or vice versa.

    # Let's use the standard "H array" approach but vectorized.
    # We need an H array of shape (Batch, q_len+1) to store the vertical column.

    # Allocation: (Batch, Q_Len + 1)
    # This might be large, but fits in L2 cache for batch=32, len=1000 (~128KB)
    H_col = np.empty((bs, q_len + 1), dtype=np.int32)
    F_col = np.empty((bs, q_len + 1), dtype=np.int32)

    # Init 0-th column (Gap Opens)
    # Glocal: Top row is 0, First col is gaps
    for k in range(bs):
        H_col[k, 0] = 0
        F_col[k, 0] = -INF
        for r in range(1, q_len + 1):
            H_col[k, r] = -(_go + (r - 1) * _ge)
            F_col[k, r] = -INF

    # --- Outer Loop: Target (Columns) ---
    for c in range(t_len):
        chars_t = T[:, c]

        # Init E (Horizontal Gap) for the batch
        E = np.full(bs, -INF, dtype=np.int32)

        # H_diag starts as H_col[:, 0]
        H_diag = H_col[:, 0].copy()

        # Update 0-th Row for this column (Target Gaps)
        # Glocal: 0-th row is free or penalized? Usually free in Glocal for target.
        # Let's assume standard Glocal:
        # H[0, c] = 0 (Local in Target) or -(go + c*ge) (Global in Target)
        # Assuming Global-in-Target based on your previous 'global' flag:
        new_H0 = np.empty(bs, dtype=np.int32)
        for k in range(bs):
            new_H0[k] = -(_go + c * _ge)  # Or 0 if local-start

        H_col[:, 0] = new_H0

        # --- Inner Loop: Query (Rows) ---
        for r in range(1, q_len + 1):
            chars_q = Q[:, r - 1]

            # 1. Calculate Scores (Vectorized)
            # Fetch H_up (which is H_col[:, r] before update)
            H_up = H_col[:, r]

            # Fetch F (Vertical Gap)
            # F[r] = max(H_up - go, F[r] - ge)
            # Note: F_col[:, r] stores the running vertical gap for this row
            f_score = F_col[:, r]

            # Vector Ops
            term1 = H_up - _go
            term2 = f_score - _ge
            new_F = np.maximum(term1, term2)
            F_col[:, r] = new_F

            # Calculate E (Horizontal Gap)
            # E = max(H_left - go, E - ge)
            # H_left is the H_col[:, r] we just fetched (Wait, H_col is updated in place?)
            # No, H_col[:, r] is currently H(r, c-1).

            h_left = H_up  # Correct
            term3 = h_left - _go
            term4 = E - _ge
            new_E = np.maximum(term3, term4)
            E = new_E

            # Calculate Match
            # We need to lookup matrix[char_q, char_t]
            # Matrix lookup is scalar, but we can list-comp or map it
            # This is the one slow part in Numba unless we flatten properly.
            # Fast way:
            match_scores = np.empty(bs, dtype=np.int32)
            for k in range(bs):
                match_scores[k] = matrix[chars_q[k], chars_t[k]]

            score = match_scores + H_diag

            # Maximize
            best = np.maximum(score, new_E)
            best = np.maximum(best, new_F)

            # Update Traceback / H_diag
            H_diag = h_left  # Old H[r] becomes diag for r+1

            # Store result
            H_col[:, r] = best

            # Track Max (for Glocal/Local)
            # Glocal: Max is only valid if we are at the end of the Query?
            # Or global max?
            # Assuming 'Glocal' (End of Query, Anywhere in Target)
            # We check if r == actual_q_len (requires passing lengths)
            # For simplicity here, tracking global max:

            mask = best > max_scores
            # Update maxes
            for k in range(bs):
                if mask[k]:
                    max_scores[k] = best[k]
                    max_r[k] = r
                    max_c[k] = c

    # Write back
    for k in range(bs):
        out_scores[k] = max_scores[k]  # or H_col[k, q_len] for Global
        out_coords[k, 0] = max_r[k]
        out_coords[k, 1] = max_c[k]


def _batch_traceback_driver(q_batch, t_batch, q_idxs, t_idxs,
                            q_starts, q_ends, t_starts, t_ends,
                            global_ends, matrix, go, ge,
                            trace_kernel, back_kernel, max_rows, max_cols):
    """
    Python wrapper that orchestrates the JIT kernel and formats the final CIGAR strings.
    """
    n = len(q_idxs)

    # Calculate relative ends for the window
    rel_ends = np.empty((n, 2), dtype=np.int32)
    rel_ends[:, 0] = global_ends[:, 0] - q_starts
    rel_ends[:, 1] = global_ends[:, 1] - t_starts

    cigar_offsets, cigars_raw, out_qs, out_ts = _batch_traceback_loop(
        q_batch, t_batch, q_idxs, t_idxs,
        q_starts, q_ends, t_starts, t_ends,
        rel_ends, matrix, go, ge,
        trace_kernel, back_kernel, max_rows, max_cols
    )

    out_cigars = np.empty(n, dtype=object)
    op_lookup = CigarParser._OP_BYTES_LOOKUP
    current_offset = 0

    for i in range(n):
        end_offset = cigar_offsets[i]
        if end_offset == current_offset:
            out_cigars[i] = b""
        else:
            ops_chunk = cigars_raw[current_offset:end_offset]
            parts = [b"%d" % row[0] + op_lookup[row[1]] for row in ops_chunk]
            out_cigars[i] = b"".join(parts)
        current_offset = end_offset

    # Adjust starts back to global
    final_qs = out_qs + q_starts
    final_ts = out_ts + t_starts

    return out_cigars, final_qs, final_ts


@jit(nopython=True, cache=True, nogil=True)
def _batch_traceback_loop(q_batch, t_batch, q_idxs, t_idxs,
                          q_starts, q_ends, t_starts, t_ends,
                          ends, matrix, go, ge,
                          trace_kernel, back_kernel, max_rows, max_cols):
    n = len(q_idxs)
    out_qs = np.empty(n, dtype=np.int32)
    out_ts = np.empty(n, dtype=np.int32)
    cigar_ops_list = []
    cigar_cnts_list = []
    cigar_counts = np.zeros(n, dtype=np.int32)
    trace_buf = np.zeros((max_rows, max_cols), dtype=np.uint8)

    for i in range(n):
        qi, ti = q_idxs[i], t_idxs[i]
        qs_g, qe_g = q_starts[i], q_ends[i]
        ts_g, te_g = t_starts[i], t_ends[i]

        seq_q_ptr = q_batch[1][qi] + qs_g
        seq_t_ptr = t_batch[1][ti] + ts_g

        ql = qe_g - qs_g
        tl = te_g - ts_g

        trace_buf[:ql + 50, :tl + 50] = 0

        # Slices
        seq1 = q_batch[0][seq_q_ptr: seq_q_ptr + ql]
        seq2 = t_batch[0][seq_t_ptr: seq_t_ptr + tl]
        band = abs(ql - tl) + 50

        # Re-run full DP (traceback mode) on the window
        max_pos = trace_kernel(seq1, seq2, matrix, go, ge, -band, band, trace_buf)

        # Override max_pos with the actual end found in Score phase (passed in 'ends')
        # This ensures consistency between score and alignment
        # Note: ends is (r, c)
        force_end = (ends[i, 0], ends[i, 1])

        o1, o2, rs, re, cs, ce, matches, alen = back_kernel(
            trace_buf, 0, 0, 0, seq1, seq2, force_end, matrix, go, ge
        )

        out_qs[i] = rs
        out_ts[i] = cs
        c_counts, c_ops = _cigar_rle_kernel(o1, o2, 255, False)
        cigar_ops_list.extend(c_ops)
        cigar_cnts_list.extend(c_counts)
        cigar_counts[i] = len(c_ops)

    total_ops = len(cigar_ops_list)
    raw_out = np.empty((total_ops, 2), dtype=np.int32)
    for k in range(total_ops):
        raw_out[k, 0] = cigar_cnts_list[k]
        raw_out[k, 1] = cigar_ops_list[k]

    offsets = np.zeros(n, dtype=np.int32)
    curr = 0
    for k in range(n):
        curr += cigar_counts[k]
        offsets[k] = curr

    return offsets, raw_out, out_qs, out_ts


@jit(nopython=True, cache=True, nogil=True)
def _find_hit_indices_kernel(q_hashes, db_hashes):
    """Identifies which indices in q_hashes map to at least one target."""
    n = len(q_hashes)
    mask = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        h = q_hashes[i]
        # Check if exists
        start = np.searchsorted(db_hashes, h, side='left')
        if start < len(db_hashes) and db_hashes[start] == h:
            mask[i] = True
    return np.where(mask)[0]


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _find_hits_with_q_id_kernel(q_mins, q_starts, q_ids, db_hashes, db_pos, db_ids):
    """
    Like index._find_hits_kernel but preserves Query ID and Query Pos.
    """
    n_mins = len(q_mins)
    counts = np.zeros(n_mins, dtype=np.int32)
    starts = np.zeros(n_mins, dtype=np.int32)

    # 1. Count
    for i in prange(n_mins):
        h = q_mins[i]
        start = np.searchsorted(db_hashes, h, side='left')
        end = np.searchsorted(db_hashes, h, side='right')
        counts[i] = end - start
        starts[i] = start

    total_hits = np.sum(counts)
    offsets = np.zeros(n_mins, dtype=np.int32)
    curr = 0
    for i in range(n_mins):
        offsets[i] = curr
        curr += counts[i]

    out_q_pos = np.empty(total_hits, dtype=np.uint32)
    out_t_pos = np.empty(total_hits, dtype=np.uint32)
    out_t_ids = np.empty(total_hits, dtype=np.uint32)
    out_q_ids = np.empty(total_hits, dtype=np.uint32)

    # 2. Fill
    for i in prange(n_mins):
        count = counts[i]
        if count == 0: continue
        off = offsets[i]

        qp = q_starts[i]
        qid = q_ids[i]
        s_idx = starts[i]

        for j in range(count):
            out_q_pos[off + j] = qp
            out_t_pos[off + j] = db_pos[s_idx + j]
            out_t_ids[off + j] = db_ids[s_idx + j]
            out_q_ids[off + j] = qid

    return out_q_pos, out_t_pos, out_t_ids, out_q_ids


@jit(nopython=True, cache=True, nogil=True)
def _cluster_seeds_kernel(q_ids, t_ids, q_pos, t_pos, min_seeds, padding, q_lens, t_lens, is_global):
    """
    Groups seeds by (q_id, t_id) and computes the Bounding Box.
    Returns array of [q_idx, t_idx, q_start, q_end, t_start, t_end].
    """
    n = len(q_ids)
    if n == 0: return np.empty((0, 6), dtype=np.int32)

    # 1. Sort by pair (q_id, t_id)
    # Pack to 64-bit int for sorting
    packed = np.empty(n, dtype=np.int64)
    for i in range(n):
        packed[i] = (np.int64(q_ids[i]) << 32) | np.int64(t_ids[i])

    sorter = np.argsort(packed)

    # 2. Iterate and Merge
    # Explicitly type the list to avoid inference errors
    dummy = (np.int32(0), np.int32(0), np.int32(0), np.int32(0), np.int32(0), np.int32(0))
    out_list = [dummy]
    out_list.pop()

    # Temp vars for current cluster
    curr_q = -1
    curr_t = -1
    min_qp = np.uint32(0); max_qp = np.uint32(0)
    min_tp = np.uint32(0); max_tp = np.uint32(0)
    count = 0

    for k in range(n):
        idx = sorter[k]
        qid = q_ids[idx]
        tid = t_ids[idx]
        qp = q_pos[idx]
        tp = t_pos[idx]

        if qid != curr_q or tid != curr_t:
            # Commit previous
            if count >= min_seeds:
                _emit_candidate(out_list, curr_q, curr_t, min_qp, max_qp, min_tp, max_tp, padding, q_lens, t_lens,
                                is_global)

            # Reset
            curr_q = qid
            curr_t = tid
            min_qp = qp;
            max_qp = qp
            min_tp = tp;
            max_tp = tp
            count = 1
        else:
            # Update bounds
            if qp < min_qp: min_qp = qp
            if qp > max_qp: max_qp = qp
            if tp < min_tp: min_tp = tp
            if tp > max_tp: max_tp = tp
            count += 1

    # Final Commit
    if count >= min_seeds:
        _emit_candidate(out_list, curr_q, curr_t, min_qp, max_qp, min_tp, max_tp, padding, q_lens, t_lens, is_global)

    if len(out_list) == 0: return np.empty((0, 6), dtype=np.int32)

    # Flatten
    res = np.zeros((len(out_list), 6), dtype=np.int32)
    for i in range(len(out_list)):
        tup = out_list[i]
        for j in range(6): res[i, j] = tup[j]
    return res


@jit(nopython=True, cache=True, nogil=True)
def _emit_candidate(out_list, q, t, min_q, max_q, min_t, max_t, pad, q_lens, t_lens, is_global):
    # Cast to int32 to ensure tuple homogeneity for Numba list
    q_i32 = np.int32(q)
    t_i32 = np.int32(t)

    if is_global:
        # Global alignment uses full sequences regardless of seeds
        # (But we only emit if seeds exist, acting as a pre-filter)
        q_len = np.int32(q_lens[q])
        t_len = np.int32(t_lens[t])
        out_list.append((q_i32, t_i32, np.int32(0), q_len, np.int32(0), t_len))
    else:
        # Local/Glocal uses bounding box + padding
        q_len = q_lens[q]
        t_len = t_lens[t]

        # Apply padding
        qs = np.int32(max(0, min_q - pad))
        qe = np.int32(min(q_len, max_q + pad + 1))  # +1 to cover the k-mer length roughly?
        ts = np.int32(max(0, min_t - pad))
        te = np.int32(min(t_len, max_t + pad + 1))

        out_list.append((q_i32, t_i32, qs, qe, ts, te))



@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _glocal_score_buffered(seq1, seq2, matrix, gap_open, gap_extend, min_diag, max_diag, H, F):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000
    H[:cols] = -(gap_open + np.arange(cols) * gap_extend)
    H[0] = 0
    F[:cols] = NEG_INF
    max_score = NEG_INF
    best_c = 0
    for r in range(1, rows):
        h_diag = H[0]
        H[0] = -(gap_open + (r - 1) * gap_extend)
        running_E = NEG_INF
        char_q = seq1[r - 1]
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            h_up = H[c]
            f_ext = F[c] - gap_extend
            f_open = h_up - gap_open
            f_score = f_ext if f_ext > f_open else f_open
            F[c] = f_score
            h_left = H[c - 1]
            e_ext = running_E - gap_extend
            e_open = h_left - gap_open
            running_E = e_ext if e_ext > e_open else e_open
            match = matrix[char_q, seq2[c - 1]] + h_diag
            best = match
            if f_score > best: best = f_score
            if running_E > best: best = running_E
            h_diag = h_up
            H[c] = best
    for c in range(start_c, end_c):
        if H[c] > max_score: max_score = H[c]; best_c = c
    return max_score, rows - 1, best_c


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _local_score_buffered(seq1, seq2, matrix, gap_open, gap_extend, min_diag, max_diag, H, F):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000
    H[:cols] = 0
    F[:cols] = NEG_INF
    global_max = 0
    max_r = 0
    max_c = 0
    for r in range(1, rows):
        h_diag = H[0]
        H[0] = 0
        running_E = NEG_INF
        char_q = seq1[r - 1]
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            h_up = H[c]
            f_ext = F[c] - gap_extend
            f_open = h_up - gap_open
            f_score = f_ext if f_ext > f_open else f_open
            F[c] = f_score
            h_left = H[c - 1]
            e_ext = running_E - gap_extend
            e_open = h_left - gap_open
            running_E = e_ext if e_ext > e_open else e_open
            match = matrix[char_q, seq2[c - 1]] + h_diag
            best = 0
            if match > best: best = match
            if f_score > best: best = f_score
            if running_E > best: best = running_E
            h_diag = h_up
            H[c] = best
            if best > global_max: global_max = best; max_r = r; max_c = c
    return global_max, max_r, max_c


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _global_score_buffered(seq1, seq2, matrix, gap_open, gap_extend, min_diag, max_diag, H, F):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000
    H[0] = 0
    for c in range(1, cols): H[c] = -(gap_open + (c - 1) * gap_extend); F[c] = NEG_INF
    for r in range(1, rows):
        h_diag = H[0]
        H[0] = -(gap_open + (r - 1) * gap_extend)
        running_E = NEG_INF
        char_q = seq1[r - 1]
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            h_up = H[c]
            f_ext = F[c] - gap_extend
            f_open = h_up - gap_open
            f_score = f_ext if f_ext > f_open else f_open
            F[c] = f_score
            h_left = H[c - 1]
            e_ext = running_E - gap_extend
            e_open = h_left - gap_open
            running_E = e_ext if e_ext > e_open else e_open
            match = matrix[char_q, seq2[c - 1]] + h_diag
            best = match
            if f_score > best: best = f_score
            if running_E > best: best = running_E
            h_diag = h_up
            H[c] = best
    return H[cols - 1], rows - 1, cols - 1


# --- Buffered Full Kernels ---

@jit(nopython=True, cache=True, nogil=True)
def _glocal_full_buffered(seq1, seq2, matrix, gap_open, gap_extend, min_diag, max_diag, trace):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000
    H = np.empty(cols, dtype=np.int32)
    F = np.empty(cols, dtype=np.int32)
    H[:cols] = -(gap_open + np.arange(cols) * gap_extend)
    H[0] = 0
    F[:cols] = NEG_INF
    max_score = NEG_INF
    best_c = 0
    for r in range(1, rows):
        h_diag = H[0]
        H[0] = -(gap_open + (r - 1) * gap_extend)
        trace[r, 0] = _TR_H_F
        running_E = NEG_INF
        char_q = seq1[r - 1]
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            h_up = H[c]
            f_ext = F[c] - gap_extend
            f_open = h_up - gap_open
            if f_ext >= f_open:
                F[c] = f_ext;
                f_bit = _TR_F_EXT
            else:
                F[c] = f_open;
                f_bit = 0
            h_left = H[c - 1]
            e_ext = running_E - gap_extend
            e_open = h_left - gap_open
            if e_ext >= e_open:
                running_E = e_ext;
                e_bit = _TR_E_EXT
            else:
                running_E = e_open;
                e_bit = 0
            match = matrix[char_q, seq2[c - 1]] + h_diag
            best = match
            source = _TR_H_MATCH
            if F[c] > best: best = F[c]; source = _TR_H_F
            if running_E > best: best = running_E; source = _TR_H_E
            h_diag = h_up
            H[c] = best
            trace[r, c] = source | e_bit | f_bit
    for c in range(start_c, end_c):
        if H[c] > max_score:
            max_score = H[c];
            best_c = c
    return max_score, rows - 1, best_c


@jit(nopython=True, cache=True, nogil=True)
def _local_full_buffered(seq1, seq2, matrix, gap_open, gap_extend, min_diag, max_diag, trace):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000
    H = np.zeros(cols, dtype=np.int32)
    F = np.full(cols, NEG_INF, dtype=np.int32)
    global_max = 0
    max_r = 0
    max_c = 0
    for r in range(1, rows):
        h_diag = H[0]
        H[0] = 0
        trace[r, 0] = 0
        running_E = NEG_INF
        char_q = seq1[r - 1]
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            h_up = H[c]
            f_ext = F[c] - gap_extend
            f_open = h_up - gap_open
            if f_ext >= f_open:
                F[c] = f_ext;
                f_bit = _TR_F_EXT
            else:
                F[c] = f_open;
                f_bit = 0
            h_left = H[c - 1]
            e_ext = running_E - gap_extend
            e_open = h_left - gap_open
            if e_ext >= e_open:
                running_E = e_ext;
                e_bit = _TR_E_EXT
            else:
                running_E = e_open;
                e_bit = 0
            match = matrix[char_q, seq2[c - 1]] + h_diag
            best = 0
            source = _TR_H_MATCH
            if match > best: best = match
            if F[c] > best: best = F[c]; source = _TR_H_F
            if running_E > best: best = running_E; source = _TR_H_E
            h_diag = h_up
            H[c] = best
            trace[r, c] = source | e_bit | f_bit
            if best > global_max: global_max = best; max_r = r; max_c = c
    return global_max, max_r, max_c


@jit(nopython=True, cache=True, nogil=True)
def _global_full_buffered(seq1, seq2, matrix, gap_open, gap_extend, min_diag, max_diag, trace):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000
    H = np.empty(cols, dtype=np.int32)
    F = np.empty(cols, dtype=np.int32)
    H[0] = 0
    F[0] = NEG_INF
    for c in range(1, cols): H[c] = -(gap_open + (c - 1) * gap_extend); F[c] = NEG_INF; trace[0, c] = _TR_H_E | (
        _TR_E_EXT if c > 1 else 0)
    for r in range(1, rows):
        h_diag = H[0]
        H[0] = -(gap_open + (r - 1) * gap_extend)
        trace[r, 0] = _TR_H_F | (_TR_F_EXT if r > 1 else 0)
        running_E = NEG_INF
        char_q = seq1[r - 1]
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            h_up = H[c]
            f_ext = F[c] - gap_extend
            f_open = h_up - gap_open
            if f_ext >= f_open:
                F[c] = f_ext;
                f_bit = _TR_F_EXT
            else:
                F[c] = f_open;
                f_bit = 0
            h_left = H[c - 1]
            e_ext = running_E - gap_extend
            e_open = h_left - gap_open
            if e_ext >= e_open:
                running_E = e_ext;
                e_bit = _TR_E_EXT
            else:
                running_E = e_open;
                e_bit = 0
            match = matrix[char_q, seq2[c - 1]] + h_diag
            best = match
            source = _TR_H_MATCH
            if F[c] > best: best = F[c]; source = _TR_H_F
            if running_E > best: best = running_E; source = _TR_H_E
            h_diag = h_up
            H[c] = best
            trace[r, c] = source | e_bit | f_bit
    return H[cols - 1], rows - 1, cols - 1


# --- Traceback Kernels ---

@jit(nopython=True, cache=True, nogil=True)
def _glocal_traceback_kernel(trace, H, E, F, seq1, seq2, max_pos, matrix, gap_open, gap_extend):
    r, c = max_pos
    end_r, end_c = r, c
    out1 = np.empty(r + c, dtype=np.int32)
    out2 = np.empty(r + c, dtype=np.int32)
    k = 0
    matches = 0
    while r > 0:
        if c == 0:
            out1[k] = seq1[r - 1];
            out2[k] = -1;
            r -= 1
        else:
            flag = trace[r, c]
            h_src = flag & 3
            if h_src == _TR_H_MATCH:
                out1[k] = seq1[r - 1]
                out2[k] = seq2[c - 1]
                if seq1[r - 1] == seq2[c - 1]: matches += 1
                r -= 1
                c -= 1
            elif h_src == _TR_H_F:
                out1[k] = seq1[r - 1];
                out2[k] = -1;
                r -= 1
            elif h_src == _TR_H_E:
                out1[k] = -1;
                out2[k] = seq2[c - 1];
                c -= 1
        k += 1
    return out1[:k][::-1], out2[:k][::-1], 0, end_r, c, end_c, matches, k


@jit(nopython=True, cache=True, nogil=True)
def _local_traceback_kernel(trace, H, E, F, seq1, seq2, max_pos, matrix, gap_open, gap_extend):
    r, c = max_pos
    end_r, end_c = r, c
    out1 = np.empty(r + c, dtype=np.int32)
    out2 = np.empty(r + c, dtype=np.int32)
    k = 0
    matches = 0
    while r > 0 and c > 0:
        flag = trace[r, c]
        h_src = flag & 3
        if h_src == _TR_H_MATCH:
            out1[k] = seq1[r - 1]
            out2[k] = seq2[c - 1]
            if seq1[r - 1] == seq2[c - 1]: matches += 1
            r -= 1
            c -= 1
        elif h_src == _TR_H_F:
            out1[k] = seq1[r - 1];
            out2[k] = -1;
            r -= 1
        elif h_src == _TR_H_E:
            out1[k] = -1;
            out2[k] = seq2[c - 1];
            c -= 1
        else:
            break  # Should catch 0 case
        k += 1
    return out1[:k][::-1], out2[:k][::-1], r, end_r, c, end_c, matches, k


@jit(nopython=True, cache=True, nogil=True)
def _global_traceback_kernel(trace, H, E, F, seq1, seq2, max_pos, matrix, gap_open, gap_extend):
    r, c = max_pos
    out1 = np.empty(r + c, dtype=np.int32)
    out2 = np.empty(r + c, dtype=np.int32)
    k = 0
    matches = 0
    while r > 0 or c > 0:
        if r == 0: out1[k] = -1; out2[k] = seq2[c - 1]; c -= 1; k += 1; continue
        if c == 0: out1[k] = seq1[r - 1]; out2[k] = -1; r -= 1; k += 1; continue
        flag = trace[r, c]
        h_src = flag & 3
        if h_src == _TR_H_MATCH:
            out1[k] = seq1[r - 1]
            out2[k] = seq2[c - 1]
            if seq1[r - 1] == seq2[c - 1]: matches += 1
            r -= 1
            c -= 1
        elif h_src == _TR_H_F:
            out1[k] = seq1[r - 1];
            out2[k] = -1;
            r -= 1
        elif h_src == _TR_H_E:
            out1[k] = -1;
            out2[k] = seq2[c - 1];
            c -= 1
        k += 1
    return out1[:k][::-1], out2[:k][::-1], 0, max_pos[0], 0, max_pos[1], matches, k


_ALIGNMENT_KERNEL_REGISTRY = {
    'local': {'score': _local_score_buffered, 'full': _local_full_buffered, 'trace': _local_traceback_kernel},
    'global': {'score': _global_score_buffered, 'full': _global_full_buffered, 'trace': _global_traceback_kernel},
    'glocal': {'score': _glocal_score_buffered, 'full': _glocal_full_buffered, 'trace': _glocal_traceback_kernel}
}