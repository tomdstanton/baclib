from abc import ABC, abstractmethod
import numpy as np

from baclib.core.seq import Seq, SeqBatch, Alphabet
from baclib.utils.resources import RESOURCES, jit

if 'numba' in RESOURCES.optional_packages:
    from numba import prange
else:
    prange = range


# --- Base Classes -----------------------------------------------------------------------------------------------------
class BaseIndex(ABC):
    """
    Abstract Base Class for all Sequence Indexes.
    Acts as the configuration context for hashing (k, alphabet, masking).

    Attributes:
        k (int): K-mer length.
        alphabet (Alphabet): The alphabet used.
        canonical (bool): Whether to use canonical k-mers (min of forward/reverse).
    """

    def __init__(self, k: int, alphabet: Alphabet, canonical: bool = True):
        self._k = k
        self._alphabet = alphabet
        self._canonical = canonical
        self._built = False

        # --- Hashing Configuration ---
        self._bps, self._mask, self._dtype = alphabet.masker(k)
        self._bps = np.uint8(self._bps)
        self._mask = self._dtype(self._mask)

        # Reverse Complement Table
        self._rc_table = None
        if canonical:
            if alphabet.complement is None:
                raise ValueError("Canonical indexing requires an Alphabet with a complement table.")
            self._rc_table = alphabet.complement

    @property
    def k(self) -> int: return self._k
    @property
    def alphabet(self) -> Alphabet: return self._alphabet
    @property
    def canonical(self) -> bool: return self._canonical
    @property
    def built(self) -> bool: return self._built
    @property
    def bps(self) -> int: return self._bps
    @property
    def mask(self) -> int: return self._mask
    @property
    def dtype(self) -> np.dtype: return self._dtype
    @property
    def rc_table(self) -> np.ndarray: return self._rc_table
    @abstractmethod
    def query(self, seq: Seq): ...
    @abstractmethod
    def query_batch(self, seq: SeqBatch): ...

    def _hash_scalar(self, seq: Seq) -> np.ndarray:
        """Helper: Hashes a single sequence. Used by MinHashSketch."""
        encoded = seq.encoded
        hashes = _rolling_hash_kernel(encoded, self._k, self._bps, self._mask, self._dtype)
        if self.canonical:
            hashes_rc = _rolling_hash_rc_kernel(
                encoded, self._k, self._bps, self._mask, self._dtype, self._rc_table
            )
            np.minimum(hashes, hashes_rc, out=hashes)
        return hashes


class SparseMapIndex(BaseIndex):
    """
    Parent class for Mapping/Seeding indexes (Minimizers & Syncmers).
    Manages the storage, sorting, and querying of sparse anchors.
    """

    def __init__(self, k: int, alphabet: Alphabet, canonical: bool = True):
        super().__init__(k, alphabet, canonical)
        self._n_seqs = 0

        # Use dtype from alphabet for hashes
        self._hashes = np.empty(0, dtype=self._dtype)
        self._positions = np.empty(0, dtype=np.uint32)
        self._ids = np.empty(0, dtype=np.uint32)

    def __len__(self) -> int:
        return self._n_seqs

    def build(self, batch: SeqBatch):
        """Builds the index from a single batch of sequences."""
        self._n_seqs = len(batch)
        hashes, pos, ids = self._compute_batch_seeds(batch)

        sorter = np.argsort(hashes)
        self._hashes = hashes[sorter]
        self._positions = pos[sorter]
        self._ids = ids[sorter]

        self._built = True

    def query(self, seq: Seq) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Queries the index with a single sequence.

        Args:
            seq: The query sequence.

        Returns:
            (query_pos, target_pos, target_id) arrays.
        """
        if not self._built: raise RuntimeError("Index not built")
        # Delegate to subclass to get anchors (now uses batch kernel for consistency)
        q_hashes, q_pos = self._get_anchors(seq)
        return _find_hits_kernel(q_hashes, q_pos, self._hashes, self._positions, self._ids)

    def query_batch(self, batch: SeqBatch) -> tuple[np.ndarray, np.ndarray]:
        """
        Queries the index with a batch of sequences.

        Args:
            batch: The query batch.

        Returns:
            (query_seq_idx, target_seq_idx) arrays.
        """
        if not self._built: raise RuntimeError("Index not built")
        # We only need hashes and sequence IDs for the query
        hashes, _, batch_ids = self._compute_batch_seeds(batch)
        
        # Pass batch_ids (query sequence indices) as 'q_pos' to the kernel
        # so the returned q_idxs correspond to sequence indices.
        q_idxs, _, t_ids = _find_hits_kernel(
            hashes, batch_ids, self._hashes, self._positions, self._ids
        )
        # Return query_seq_idx and target_seq_idx
        return q_idxs, t_ids

    @abstractmethod
    def _get_anchors(self, seq: Seq) -> tuple[np.ndarray, np.ndarray]: ...
    @abstractmethod
    def _compute_batch_seeds(self, batch: SeqBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  ...


# --- Implementations --------------------------------------------------------------------------------------------------
class MinimizerIndex(SparseMapIndex):
    """
    Selects anchors based on windowed minimums.

    Examples:
        >>> idx = MinimizerIndex(k=15, w=10)
        >>> idx.build(batch)
        >>> hits = idx.query(query_seq)
    """

    def __init__(self, k: int = 15, w: int = 10, alphabet: Alphabet = None):
        super().__init__(k, alphabet or Alphabet.dna(), canonical=True)
        self._w = w

    @property
    def w(self) -> int: return self._w

    def _get_anchors(self, seq: Seq):
        # reuse batch kernel with single item for max code reuse
        batch = SeqBatch([seq], seq.alphabet)
        h, p, _ = _batch_minimizer_kernel(
            *batch.arrays, self._k, self._w, self._bps, self._mask, self._dtype, self._rc_table
        )
        return h, p

    def _compute_batch_seeds(self, batch: SeqBatch):
        data, starts, lengths = batch.arrays
        n = len(batch)
        return _batch_minimizer_kernel(
            data, starts[:n], lengths[:n], self._k, self._w, self._bps, self._mask, self._dtype, self._rc_table
        )


class SyncmerIndex(SparseMapIndex):
    """
    Selects anchors based on internal structure (Open Syncmers).

    Examples:
        >>> idx = SyncmerIndex(k=15, s=5)
        >>> idx.build(batch)
    """

    def __init__(self, k: int = 15, s: int = 5, t: int = 0, alphabet: Alphabet = None):
        super().__init__(k, alphabet or Alphabet.dna(), canonical=True)
        if s >= k: raise ValueError("s must be smaller than k")
        if t > (k - s): raise ValueError("t is out of bounds")
        self._s = s
        self._t = t
        self._mask_s = self._dtype((1 << (int(self._bps) * (s - 1))) - 1)

    @property
    def s(self) -> int: return self._s
    @property
    def t(self) -> int: return self._t
    
    def _get_anchors(self, seq: Seq):
        batch = SeqBatch([seq], seq.alphabet)
        h, p, _ = _batch_syncmer_kernel(
            *batch.arrays, self._k, self._s, self._t, self._bps, self._mask, self._mask_s, self._dtype, self._rc_table
        )
        return h, p

    def _compute_batch_seeds(self, batch: SeqBatch):
        data, starts, lengths = batch.arrays
        n = len(batch)
        return _batch_syncmer_kernel(
            data, starts[:n], lengths[:n], self._k, self._s, self._t, self._bps, self._mask, self._mask_s, self._dtype, self._rc_table
        )


class MinHashSketch(BaseIndex):
    """
    Compression Index for Distance Estimation (Jaccard/ANI).
    Stores hashes in multiple memory blocks to allow instant updates.

    Examples:
        >>> sketch = MinHashSketch(k=21, size=1000)
        >>> sketch.build(batch)
        >>> dists = sketch.query(query_seq)
    """

    def __init__(self, k: int = 16, size: int = 1000, alphabet: Alphabet = None):
        super().__init__(k, alphabet or Alphabet.dna(), canonical=True)
        self._size = np.uint16(size)
        self._sketches = np.empty((0, self._size), dtype=self._dtype)

    @property
    def size(self): return self._size

    @property
    def sketches(self) -> np.ndarray:
        if not self._built: raise RuntimeError("Index not built")
        return self._sketches

    def build(self, batch: SeqBatch):
        """Builds the index from a single batch of sequences."""
        data, starts, lengths = batch.arrays
        n = len(batch)
        self._sketches = _batch_minhash_kernel(
            data, starts[:n], lengths[:n], self._k, self._size, self._bps, self._mask, self._dtype, self._rc_table
        )
        self._built = True

    def query(self, seq: Seq = None) -> np.ndarray:
        """
        Query the index.

        Args:
            seq: The query sequence. If None, computes all-vs-all distances.

        Returns:
            Array of Jaccard similarities.
        """
        if not self._built: raise RuntimeError("Index not built")

        # Case 1: All-vs-All (Expensive, so we merge first)
        if seq is None:
            return _all_vs_all_jaccard_kernel(self._sketches)

        # Case 2: One-vs-Many
        q_sketch = _compute_sketch_kernel(self._hash_scalar(seq), self._size)

        # Reshape (S,) -> (1, S) to use the many-vs-many kernel
        q_batch = q_sketch.reshape(1, self._size)
        scores = _many_vs_many_jaccard_kernel(q_batch, self._sketches)
        return scores[0]

    def query_batch(self, batch: SeqBatch) -> np.ndarray:
        """
        Batch query (Many-vs-Many).

        Args:
            batch: The query batch.

        Returns:
            Matrix of Jaccard similarities (M x N).
        """
        if not self._built: raise RuntimeError("Index not built")

        # 1. Sketch Query Batch
        data, starts, lengths = batch.arrays
        n = len(batch)
        q_sketches = _batch_minhash_kernel(
            data, starts[:n], lengths[:n], self._k, self._size, self._bps, self._mask, self._dtype, self._rc_table
        )

        # 2. Query against index
        return _many_vs_many_jaccard_kernel(q_sketches, self._sketches)


# --- Kernels ----------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _rolling_hash_kernel(int_seq, k, bits, mask, dtype):
    n = len(int_seq)
    if n < k: return np.empty(0, dtype=dtype)
    res = np.empty(n - k + 1, dtype=dtype)
    curr = dtype(0)
    for i in range(k):
        curr = (curr << bits) | int_seq[i]
    res[0] = curr
    for i in range(1, n - k + 1):
        curr = ((curr & mask) << bits) | int_seq[i + k - 1]
        res[i] = curr
    return res


@jit(nopython=True, cache=True, nogil=True)
def _rolling_hash_rc_kernel(int_seq, k, bits, mask, dtype, rc_table):
    n = len(int_seq)
    if n < k: return np.empty(0, dtype=dtype)
    res = np.empty(n - k + 1, dtype=dtype)
    curr = dtype(0)
    for i in range(k):
        val = rc_table[int_seq[n - 1 - i]]
        curr = (curr << bits) | val
    res[n - k] = curr
    for i in range(1, n - k + 1):
        val = rc_table[int_seq[n - k - i]]
        curr = ((curr & mask) << bits) | val
        res[n - k - i] = curr
    return res


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _find_hits_kernel(q_mins, q_pos, db_hashes, db_pos, db_ids):
    # [Kept same as provided - Standard Binary Search]
    n_mins = len(q_mins)
    counts = np.zeros(n_mins, dtype=np.int32)
    starts = np.zeros(n_mins, dtype=np.int32)
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
    out_q = np.empty(total_hits, dtype=np.uint32)
    out_t = np.empty(total_hits, dtype=np.uint32)
    out_id = np.empty(total_hits, dtype=np.uint32)
    for i in prange(n_mins):
        count = counts[i]
        if count == 0: continue
        off = offsets[i]
        qp = q_pos[i]
        s_idx = starts[i]
        for j in range(count):
            out_q[off + j] = qp
            out_t[off + j] = db_pos[s_idx + j]
            out_id[off + j] = db_ids[s_idx + j]
    return out_q, out_t, out_id


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_minimizer_kernel(data, starts, lengths, k, w, bits, mask, dtype, rc_table):
    n_seqs = len(starts)
    counts = np.zeros(n_seqs, dtype=np.int32)
    rc_high_shift = bits * (k - 1)

    # Pass 1: Count
    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        if l < k: continue
        seq_hashes = np.empty(l - k + 1, dtype=dtype)

        curr_f = dtype(0)
        curr_r = dtype(0)
        for j in range(k):
            val = data[s + j]
            curr_f = (curr_f << bits) | val
            curr_r = (curr_r >> bits) | (dtype(rc_table[val]) << rc_high_shift)
        seq_hashes[0] = curr_f if curr_f < curr_r else curr_r
        for j in range(1, l - k + 1):
            val = data[s + j + k - 1]
            val_rc = rc_table[val]
            curr_f = ((curr_f & mask) << bits) | val
            curr_r = (curr_r >> bits) | (dtype(val_rc) << rc_high_shift)
            seq_hashes[j] = curr_f if curr_f < curr_r else curr_r

        m_count = 0
        last_min_pos = -1
        for j in range(len(seq_hashes) - w + 1):
            current_min_idx = -1
            if last_min_pos >= j:
                if seq_hashes[j + w - 1] < seq_hashes[last_min_pos]:
                    current_min_idx = j + w - 1
                else:
                    current_min_idx = last_min_pos
            else:
                min_val = seq_hashes[j]
                min_idx = j
                for m in range(1, w):
                    if seq_hashes[j + m] < min_val:
                        min_val = seq_hashes[j + m]
                        min_idx = j + m
                current_min_idx = min_idx
            if current_min_idx != last_min_pos:
                m_count += 1
                last_min_pos = current_min_idx
        counts[i] = m_count

    # Offsets
    total_len = np.sum(counts)
    offsets = np.zeros(n_seqs, dtype=np.int32)
    curr = 0
    for i in range(n_seqs):
        offsets[i] = curr
        curr += counts[i]

    # Pass 2: Fill
    out_hashes = np.empty(total_len, dtype=dtype)
    out_pos = np.empty(total_len, dtype=np.uint32)
    out_ids = np.empty(total_len, dtype=np.uint32)

    for i in prange(n_seqs):
        start = starts[i]
        l = lengths[i]
        if l < k: continue
        off = offsets[i]
        seq_hashes = np.empty(l - k + 1, dtype=dtype)

        curr_f = dtype(0)
        curr_r = dtype(0)
        for j in range(k):
            val = data[start + j]
            curr_f = (curr_f << bits) | val
            curr_r = (curr_r >> bits) | (dtype(rc_table[val]) << rc_high_shift)
        seq_hashes[0] = curr_f if curr_f < curr_r else curr_r
        for j in range(1, l - k + 1):
            val = data[start + j + k - 1]
            val_rc = rc_table[val]
            curr_f = ((curr_f & mask) << bits) | val
            curr_r = (curr_r >> bits) | (dtype(val_rc) << rc_high_shift)
            seq_hashes[j] = curr_f if curr_f < curr_r else curr_r

        local_c = 0
        last_min_pos = -1
        for j in range(len(seq_hashes) - w + 1):
            current_min_idx = -1
            if last_min_pos >= j:
                if seq_hashes[j + w - 1] < seq_hashes[last_min_pos]:
                    current_min_idx = j + w - 1
                else:
                    current_min_idx = last_min_pos
            else:
                min_val = seq_hashes[j]
                min_idx = j
                for m in range(1, w):
                    if seq_hashes[j + m] < min_val:
                        min_val = seq_hashes[j + m]
                        min_idx = j + m
                current_min_idx = min_idx
            if current_min_idx != last_min_pos:
                out_hashes[off + local_c] = seq_hashes[current_min_idx]
                out_pos[off + local_c] = current_min_idx
                out_ids[off + local_c] = i
                local_c += 1
                last_min_pos = current_min_idx
    return out_hashes, out_pos, out_ids


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_syncmer_kernel(data, starts, lengths, k, s, t, bits, mask_k, mask_s, dtype, rc_table):
    n_seqs = len(starts)
    counts = np.zeros(n_seqs, dtype=np.int32)
    n_sub = k - s + 1
    rc_k_shift = bits * (k - 1)
    rc_s_shift = bits * (s - 1)

    # Pass 1: Count
    for i in prange(n_seqs):
        start = starts[i]
        l = lengths[i]
        if l < k: continue
        fk, rk = dtype(0), dtype(0)
        fs, rs = dtype(0), dtype(0)
        s_buf = np.zeros(n_sub, dtype=dtype)
        buf_idx = 0
        local_hits = 0
        for j in range(k - 1):
            val = data[start + j]
            val_rc = rc_table[val]
            fk = ((fk & mask_k) << bits) | val
            fs = ((fs & mask_s) << bits) | val
            rk = (rk >> bits) | (dtype(val_rc) << rc_k_shift)
            rs = (rs >> bits) | (dtype(val_rc) << rc_s_shift)
            can_s = fs if fs < rs else rs
            s_buf[buf_idx] = can_s
            buf_idx += 1
            if buf_idx >= n_sub: buf_idx = 0
        for j in range(k - 1, l):
            val = data[start + j]
            val_rc = rc_table[val]
            fk = ((fk & mask_k) << bits) | val
            fs = ((fs & mask_s) << bits) | val
            rk = (rk >> bits) | (dtype(val_rc) << rc_k_shift)
            rs = (rs >> bits) | (dtype(val_rc) << rc_s_shift)
            can_s = fs if fs < rs else rs
            s_buf[buf_idx] = can_s
            min_s = s_buf[0]
            min_idx = 0
            for m in range(1, n_sub):
                if s_buf[m] < min_s:
                    min_s = s_buf[m]
                    min_idx = m
            w_start = buf_idx + 1
            if w_start >= n_sub: w_start = 0
            if min_idx >= w_start:
                rel_pos = min_idx - w_start
            else:
                rel_pos = (min_idx + n_sub) - w_start
            if rel_pos == t: local_hits += 1
            buf_idx += 1
            if buf_idx >= n_sub: buf_idx = 0
        counts[i] = local_hits

    # Offsets
    total_len = np.sum(counts)
    offsets = np.zeros(n_seqs, dtype=np.int32)
    curr = 0
    for i in range(n_seqs):
        offsets[i] = curr
        curr += counts[i]
    out_hashes = np.empty(total_len, dtype=dtype)
    out_pos = np.empty(total_len, dtype=np.uint32)
    out_ids = np.empty(total_len, dtype=np.uint32)

    # Pass 2: Fill
    for i in prange(n_seqs):
        start = starts[i]
        l = lengths[i]
        if l < k: continue
        off = offsets[i]
        fk, rk = dtype(0), dtype(0)
        fs, rs = dtype(0), dtype(0)
        s_buf = np.zeros(n_sub, dtype=dtype)
        buf_idx = 0
        local_c = 0
        for j in range(k - 1):
            val = data[start + j]
            val_rc = rc_table[val]
            fk = ((fk & mask_k) << bits) | val
            fs = ((fs & mask_s) << bits) | val
            rk = (rk >> bits) | (dtype(val_rc) << rc_k_shift)
            rs = (rs >> bits) | (dtype(val_rc) << rc_s_shift)
            can_s = fs if fs < rs else rs
            s_buf[buf_idx] = can_s
            buf_idx += 1
            if buf_idx >= n_sub: buf_idx = 0
        for j in range(k - 1, l):
            val = data[start + j]
            val_rc = rc_table[val]
            fk = ((fk & mask_k) << bits) | val
            fs = ((fs & mask_s) << bits) | val
            rk = (rk >> bits) | (dtype(val_rc) << rc_k_shift)
            rs = (rs >> bits) | (dtype(val_rc) << rc_s_shift)
            can_s = fs if fs < rs else rs
            s_buf[buf_idx] = can_s
            min_s = s_buf[0]
            min_idx = 0
            for m in range(1, n_sub):
                if s_buf[m] < min_s:
                    min_s = s_buf[m]
                    min_idx = m
            w_start = buf_idx + 1
            if w_start >= n_sub: w_start = 0
            if min_idx >= w_start:
                rel_pos = min_idx - w_start
            else:
                rel_pos = (min_idx + n_sub) - w_start
            if rel_pos == t:
                out_hashes[off + local_c] = fk if fk < rk else rk
                out_pos[off + local_c] = j - k + 1
                out_ids[off + local_c] = i
                local_c += 1
            buf_idx += 1
            if buf_idx >= n_sub: buf_idx = 0
    return out_hashes, out_pos, out_ids


@jit(nopython=True, cache=True, nogil=True)
def _compute_sketch_kernel(hashes, size):
    # [Kept as is - Optimized selection]
    if len(hashes) > size * 10:
        limit = min(len(hashes), size * 2)
        hashes = np.partition(hashes, limit)[:limit]
    unique = np.unique(hashes)
    dtype = hashes.dtype
    fill = np.iinfo(dtype).max
    if len(unique) <= size:
        res = np.full(size, fill, dtype=dtype)
        res[:len(unique)] = unique
        return res
    return unique[:size]


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_minhash_kernel(data, starts, lengths, k, size, bits, mask, dtype, rc_table):
    # [Kept as is - Optimized Batch Sketching]
    n_seqs = len(starts)
    fill = np.iinfo(dtype).max
    out_matrix = np.full((n_seqs, size), fill, dtype=dtype)
    rc_high_shift = bits * (k - 1)
    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        if l < k: continue
        seq_hashes = np.empty(l - k + 1, dtype=dtype)
        curr_f = dtype(0)
        curr_r = dtype(0)
        for j in range(k):
            val = data[s + j]
            curr_f = (curr_f << bits) | val
            curr_r = (curr_r >> bits) | (dtype(rc_table[val]) << rc_high_shift)
        seq_hashes[0] = curr_f if curr_f < curr_r else curr_r
        for j in range(1, l - k + 1):
            val = data[s + j + k - 1]
            val_rc = rc_table[val]
            curr_f = ((curr_f & mask) << bits) | val
            curr_r = (curr_r >> bits) | (dtype(val_rc) << rc_high_shift)
            seq_hashes[j] = curr_f if curr_f < curr_r else curr_r

        limit = size * 2
        if len(seq_hashes) > limit * 2:
            candidates = np.partition(seq_hashes, limit)[:limit]
            candidates.sort()
            sorted_hashes = candidates
        else:
            seq_hashes.sort()
            sorted_hashes = seq_hashes

        count = 0
        if len(sorted_hashes) > 0:
            out_matrix[i, 0] = sorted_hashes[0]
            count = 1
            for j in range(1, len(sorted_hashes)):
                if count >= size: break
                if sorted_hashes[j] != sorted_hashes[j - 1]:
                    out_matrix[i, count] = sorted_hashes[j]
                    count += 1
    return out_matrix


# [ _batch_jaccard_kernel DELETED - REDUNDANT (Covered by Many-vs-Many) ]

@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _all_vs_all_jaccard_kernel(sketches):
    # [Kept as is - Optimized Symmetric N-vs-N]
    n = len(sketches)
    scores = np.eye(n, dtype=np.float32)
    dtype = sketches.dtype
    SENTINEL = np.iinfo(dtype).max
    eff_lens = np.empty(n, dtype=np.int32)
    for i in prange(n):
        s = sketches[i]
        l = len(s)
        if l > 0 and s[-1] != SENTINEL:
            eff_lens[i] = l
        else:
            count = 0
            for k in range(l):
                if s[k] == SENTINEL: break
                count += 1
            eff_lens[i] = count
    for i in prange(n):
        l1 = eff_lens[i]
        if l1 == 0: continue
        s1 = sketches[i]
        for j in range(i + 1, n):
            l2 = eff_lens[j]
            if l2 == 0: continue
            s2 = sketches[j]
            intersect = 0;
            idx1 = 0;
            idx2 = 0
            while idx1 < l1 and idx2 < l2:
                v1 = s1[idx1];
                v2 = s2[idx2]
                if v1 == v2:
                    intersect += 1; idx1 += 1; idx2 += 1
                elif v1 < v2:
                    idx1 += 1
                else:
                    idx2 += 1
            union = l1 + l2 - intersect
            if union > 0:
                val = intersect / union
                scores[i, j] = val
                scores[j, i] = val
            else:
                scores[i, j] = 0.0
                scores[j, i] = 0.0
    return scores


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _many_vs_many_jaccard_kernel(q_sketches, db_sketches):
    m = len(q_sketches)
    n = len(db_sketches)
    scores = np.empty((m, n), dtype=np.float32)
    dtype = q_sketches.dtype
    SENTINEL = np.iinfo(dtype).max
    db_lens = np.empty(n, dtype=np.int32)
    for i in range(n):
        s = db_sketches[i]
        l = len(s)
        if l > 0 and s[-1] != SENTINEL:
            db_lens[i] = l
        else:
            c = 0
            for k in range(l):
                if s[k] == SENTINEL: break
                c += 1
            db_lens[i] = c
    for i in prange(m):
        q_s = q_sketches[i]
        q_len = len(q_s)
        q_eff = 0
        if q_len > 0 and q_s[-1] != SENTINEL:
            q_eff = q_len
        else:
            for k in range(q_len):
                if q_s[k] == SENTINEL: break
                q_eff += 1
        if q_eff == 0:
            scores[i, :] = 0.0
            continue
        for j in range(n):
            db_eff = db_lens[j]
            if db_eff == 0: scores[i, j] = 0.0; continue
            db_s = db_sketches[j]
            intersect = 0;
            idx_q = 0;
            idx_db = 0
            while idx_q < q_eff and idx_db < db_eff:
                vq = q_s[idx_q];
                vdb = db_s[idx_db]
                if vq == vdb:
                    intersect += 1; idx_q += 1; idx_db += 1
                elif vq < vdb:
                    idx_q += 1
                else:
                    idx_db += 1
            union = q_eff + db_eff - intersect
            if union > 0:
                scores[i, j] = intersect / union
            else:
                scores[i, j] = 0.0
    return scores