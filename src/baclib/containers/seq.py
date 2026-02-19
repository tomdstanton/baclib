"""Core sequence containers with alphabet-aware encoding, hashing, and batch support."""
from binascii import hexlify
from hashlib import blake2b
from typing import Union, Iterable, Generator

import numpy as np

from baclib.core.interval import Interval
from baclib.containers import Batch, Batchable
from baclib.lib.resources import jit, RESOURCES
from baclib.lib.protocols import HasAlphabet

if RESOURCES.has_module('numba'):
    from numba import prange
else: 
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Seq(HasAlphabet, Batchable):
    """
    Immutable, alphabet-aware sequence container storing encoded integers (uint8).

    ``Seq`` objects should be created via ``Alphabet.seq()`` or ``Alphabet.random()``
    rather than directly, to ensure encoding consistency.

    Args:
        data: A numpy uint8 array of encoded symbol indices.
        alphabet: The ``Alphabet`` that owns this sequence.
        _validation_token: Internal token (must be the alphabet) to prevent
            direct construction.

    Examples:
        >>> seq = Alphabet.DNA.seq(b'ATGCGA')
        >>> len(seq)
        6
        >>> bytes(seq)
        b'ATGCGA'
        >>> seq[1:4]
        TGC
    """
    __slots__ = ('_data', '_alphabet', '_hash')
    def __init__(self, data: np.ndarray, alphabet: 'Alphabet', _validation_token: object = None):
        if _validation_token is not alphabet:
            raise PermissionError("Seq objects must be created via an Alphabet")
        self._alphabet = alphabet
        self._data = data
        self._hash = None
        self._data.flags.writeable = False  # Enforce immutability for hashing safety

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``SeqBatch`` class.
        """
        return SeqBatch

    @property
    def alphabet(self) -> 'Alphabet':
        """Returns the alphabet used for encoding/decoding.

        Returns:
            The owning ``Alphabet`` singleton.
        """
        return self._alphabet

    @property
    def encoded(self) -> np.ndarray:
        """Returns the underlying encoded integer array (zero-copy).

        Returns:
            A read-only ``uint8`` numpy array.
        """
        return self._data

    def __array__(self, dtype=None):
        """Allows the Seq to be treated as a numpy array."""
        return self._data.astype(dtype, copy=False) if dtype else self._data

    def __bytes__(self) -> bytes: return self._alphabet.decode(self._data)

    def tobytes(self) -> bytes:
        """Decodes the sequence to raw bytes.

        Returns:
            The decoded byte string.

        Examples:
            >>> seq.tobytes()
            b'ATGCGA'
        """
        return self.__bytes__()

    def __len__(self): return self._data.shape[0]
    def __str__(self): return self.__bytes__().decode('ascii')
    def __iter__(self): return iter(self._data)
    def __repr__(self):
        if len(self) <= 14: return str(self)
        # Optimization: Decode only the parts we show
        head = self._alphabet.decode(self._data[:7]).decode('ascii')
        tail = self._alphabet.decode(self._data[-7:]).decode('ascii')
        return f"{head}...{tail}"
    # Use numpy flip for reversal (Zero Copy view if possible)
    def __reversed__(self) -> 'Seq': return self._alphabet.seq_from(np.flip(self._data))

    def __contains__(self, item):
        # 1. Handle Integer (Raw code check)
        if isinstance(item, (int, np.integer)): return item in self._data
        # 2. Handle Subsequence (Seq, bytes, str)
        query = None
        if isinstance(item, Seq):
            if item.alphabet is not self._alphabet: return False
            query = item._data.tobytes()
        elif isinstance(item, (bytes, str)):
            if isinstance(item, str): item = item.encode('ascii')
            query = self._alphabet.encode(item).tobytes()
        # Use bytes substring search (fast C implementation) on raw encoded data
        if query is not None: return query in self._data.tobytes()
        return False

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, Seq): return False
        if self._alphabet is not other._alphabet: return False
        
        # Optimization: Fast hash check if available
        if self._hash is not None and other._hash is not None:
            if self._hash != other._hash: return False
            
        return np.array_equal(self._data, other._data)

    def __hash__(self):
        # Optimization: Hash the raw encoded integers directly
        if self._hash is None: self._hash = hash(memoryview(self._data))
        return self._hash

    def __add__(self, other: 'Seq') -> 'Seq':
        if self._alphabet is not other._alphabet:
            raise ValueError("Cannot concatenate sequences with different alphabets")
        # Fast Int concatenation
        return self._alphabet.seq_from(np.concatenate((self._data, other._data), axis=0))

    def __mul__(self, other: int) -> 'Seq':
        if not isinstance(other, int): return NotImplemented
        if other <= 0: return self._alphabet.empty_seq()
        return self._alphabet.seq_from(np.tile(self._data, other))

    def __rmul__(self, other: int) -> 'Seq': return self.__mul__(other)
    def __bool__(self): return len(self._data) > 0

    # Comparisons (Lexicographical)
    # We must decode to bytes because internal integer order (e.g. T=0, A=2)
    # might not match alphabetical order (A < T).
    def __lt__(self, other):
        if not isinstance(other, Seq): return NotImplemented
        return bytes(self) < bytes(other)

    def __le__(self, other):
        if not isinstance(other, Seq): return NotImplemented
        return bytes(self) <= bytes(other)

    def __gt__(self, other):
        if not isinstance(other, Seq): return NotImplemented
        return bytes(self) > bytes(other)

    def __ge__(self, other):
        if not isinstance(other, Seq): return NotImplemented
        return bytes(self) >= bytes(other)

    def __getitem__(self, item: Union[slice, int, Interval]) -> 'Seq':
        """Extracts a subsequence by index, slice, or ``Interval``.

        When an ``Interval`` with ``strand == -1`` is used, the result is
        automatically reverse-complemented.

        Args:
            item: An integer index, a Python slice, or an ``Interval``.

        Returns:
            A new ``Seq`` representing the subsequence.

        Examples:
            >>> seq = Alphabet.DNA.seq(b'ATGCGA')
            >>> seq[1:4]
            TGC
            >>> seq[Interval(1, 4, -1)]  # reverse complement
            GCA
        """
        # 1. Standard Slicing (Fastest)
        if isinstance(item, slice):
            # Numpy handles the slicing logic/views
            return self._alphabet.seq_from(self._data[item])

        # 2. Integer Access
        if isinstance(item, int): return self._alphabet.seq_from(self._data[item:item + 1])

        # 3. Interval Object
        # Only import Interval overhead if needed
        item = Interval.from_item(item, length=len(self))
        chunk_encoded = self._data[item.start:item.end]

        if item.strand == -1:
            # Resolve RC. If Alphabet supports int-based RC, usage is cleaner.
            # Optimization: Use cached RC table directly if available
            if self._alphabet.complement is not None:
                # [::-1] creates a view, fancy indexing creates a copy
                rc_data = self._alphabet.complement[chunk_encoded[::-1]]
                return self._alphabet.seq_from(rc_data)

            # For now, we assume we must decode->trans->encode or use a cached RC table
            # Ideally: return self._alphabet.core(encoded=self._alphabet.rc_code(chunk_encoded))
            return self._alphabet.reverse_complement(self._alphabet.seq_from(chunk_encoded))

        return self._alphabet.seq_from(chunk_encoded)

    def generate_id(self, digest_size: int = 8) -> bytes:
        """Generates a deterministic ID from the sequence content using BLAKE2b.

        Args:
            digest_size: Hash digest size in bytes (output is ``2 * digest_size``
                hex characters).

        Returns:
            The hex digest as bytes.

        Examples:
            >>> seq.generate_id()
            b'a1b2c3d4e5f6a7b8'
        """
        return hexlify(blake2b(self._data.tobytes(), digest_size=digest_size, usedforsecurity=False).digest())


class SeqBatch(Batch, HasAlphabet):
    """
    Flattened batch of sequences for Numba-accelerated parallel processing.

    Stores all encoded symbols in a single contiguous ``uint8`` array with
    per-sequence start/length metadata, enabling zero-copy slicing and
    parallel kernels.

    Args:
        data: Contiguous ``uint8`` array of all encoded symbols.
        starts: ``int32`` array of per-sequence start offsets into *data*.
        lengths: ``int32`` array of per-sequence lengths.
        alphabet: The shared ``Alphabet``.
        _validation_token: Internal token (must be the alphabet).

    Examples:
        >>> batch = Alphabet.DNA.batch_from([seq1, seq2, seq3])
        >>> len(batch)
        3
        >>> batch[0]
        ATGCGA
    """
    __slots__ = ('_alphabet', '_data', '_starts', '_lengths', '_count')
    def __init__(self, data: np.ndarray, starts: np.ndarray, lengths: np.ndarray, 
                 alphabet: 'Alphabet', _validation_token: object = None):
        if _validation_token is not alphabet:
            raise PermissionError("SeqBatch objects must be created via class methods or an Alphabet")
        self._data = data
        self._starts = starts
        self._lengths = lengths
        self._alphabet = alphabet
        self._count = len(starts)
        
        # Lock arrays for safety
        self._data.flags.writeable = False
        self._starts.flags.writeable = False
        self._lengths.flags.writeable = False

    @classmethod
    def build(cls, seqs: Iterable['Seq']) -> 'SeqBatch':
        """Creates a SeqBatch from an iterable of Seq objects.

        Infers the alphabet from the first sequence.

        Args:
            seqs: An iterable of ``Seq`` objects (must share the same alphabet).

        Returns:
            A new ``SeqBatch``.

        Raises:
            ValueError: If the iterable is empty (use ``Alphabet.empty_batch()``).

        Examples:
            >>> batch = SeqBatch.build([seq1, seq2])
            >>> len(batch)
            2
        """
        seqs_list = list(seqs)
        if not seqs_list:
             raise ValueError("Cannot create SeqBatch from empty sequence list. Use Alphabet.empty_batch() instead.")
        return seqs_list[0].alphabet.batch_from(seqs_list)

    @classmethod
    def zeros(cls, n: int) -> 'SeqBatch':
        """Not supported directly — use ``Alphabet.zeros_batch(n)`` instead.

        Raises:
            TypeError: Always.
        """
        raise TypeError("SeqBatch.zeros() requires an Alphabet. Use Alphabet.zeros_batch(n) instead.")

    @classmethod
    def empty(cls) -> 'SeqBatch':
        """Not supported directly — use ``Alphabet.empty_batch()`` instead.

        Raises:
            TypeError: Always.
        """
        raise TypeError("SeqBatch.empty() requires an Alphabet. Use Alphabet.empty_batch() instead.")

    @classmethod
    def concat(cls, batches: Iterable['SeqBatch']) -> 'SeqBatch':
        """Concatenates multiple SeqBatch objects into one.

        All batches must share the same alphabet.

        Args:
            batches: An iterable of ``SeqBatch`` objects.

        Returns:
            A single concatenated ``SeqBatch``.

        Raises:
            ValueError: If the list is empty or alphabets differ.

        Examples:
            >>> combined = SeqBatch.concat([batch_a, batch_b])
        """
        batches = list(batches)
        if not batches: raise ValueError("Cannot concatenate empty list of batches")
        
        # Validate alphabet consistency
        alphabet = batches[0].alphabet
        for b in batches[1:]:
            if b.alphabet != alphabet: raise ValueError("All batches must share the same alphabet")
            
        # Merge arrays
        data = np.concatenate([b._data for b in batches])
        lengths = np.concatenate([b._lengths for b in batches])
        
        starts = np.zeros(len(lengths), dtype=np.int32)
        if len(lengths) > 1: np.cumsum(lengths[:-1], out=starts[1:])
        return cls(data, starts, lengths, alphabet, _validation_token=alphabet)

    def __add__(self, other: 'SeqBatch') -> 'SeqBatch':
        """Concatenates two batches via the ``+`` operator."""
        return self.concat([self, other])

    # --- Numba Accessors ---
    # Properties to unpack into Numba function arguments: *batch.arrays
    @property
    def arrays(self):
        """Returns ``(data, starts, lengths)`` for Numba kernel unpacking.

        Returns:
            A 3-tuple of numpy arrays.
        """
        return self._data, self._starts, self._lengths

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``Seq`` class.
        """
        return Seq

    @property
    def alphabet(self) -> 'Alphabet':
        """Returns the shared alphabet.

        Returns:
            The ``Alphabet`` singleton.
        """
        return self._alphabet

    @property
    def encoded(self) -> np.ndarray:
        """Returns the flat encoded data array (zero-copy).

        Returns:
            A read-only ``uint8`` numpy array.
        """
        return self._data

    @property
    def starts(self) -> np.ndarray:
        """Returns the per-sequence start offsets.

        Returns:
            An ``int32`` numpy array.
        """
        return self._starts

    @property
    def lengths(self) -> np.ndarray:
        """Returns the per-sequence lengths.

        Returns:
            An ``int32`` numpy array.
        """
        return self._lengths

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by data, starts, and lengths arrays.
        """
        return self._data.nbytes + self._starts.nbytes + self._lengths.nbytes

    def copy(self) -> 'SeqBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``SeqBatch`` with copied arrays.
        """
        return self.__class__(self._data.copy(), self._starts.copy(), self._lengths.copy(), self._alphabet, _validation_token=self._alphabet)

    def empty(self) -> 'SeqBatch':
        """Returns an empty batch with the same alphabet.

        Returns:
            An empty ``SeqBatch``.
        """
        return self._alphabet.empty()

    def __repr__(self): return f"<SeqBatch: {len(self)} sequences>"

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, SeqBatch): return False
        if self._alphabet is not other._alphabet: return False
        if len(self) != len(other): return False
        return (np.array_equal(self._lengths, other._lengths) and
                np.array_equal(self._starts, other._starts) and
                np.array_equal(self._data, other._data))

    def __len__(self): return self._count
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0: idx += self._count
            if not 0 <= idx < self._count: raise IndexError("SeqBatch index out of range")
            start = self._starts[idx]
            length = self._lengths[idx]
            return self._alphabet.seq_from(self._data[start:start + length])
        elif isinstance(idx, (slice, np.ndarray, list)):
            # Optimized slicing: Gather arrays directly without creating Seq objects
            if isinstance(idx, slice):
                start, stop, step = idx.indices(self._count)
                if step == 1:
                    # Contiguous slice (Fastest)
                    new_count = max(0, stop - start)
                    if new_count == 0: return self._alphabet.empty()
                    
                    s_start = self._starts[start]
                    s_end = self._starts[stop] if stop < self._count else len(self._data)
                    
                    new_lengths = self._lengths[start:stop]
                    
                    # Check for physical contiguity to allow zero-copy slicing
                    # If deduplicated, s_end - s_start != sum(lengths)
                    if (s_end - s_start) == new_lengths.sum():
                        new_data = self._data[s_start:s_end]
                        new_starts = self._starts[start:stop] - s_start
                        return self._alphabet.new_batch(new_data, new_starts, new_lengths)
                
                # Non-contiguous slice -> Convert to array indices
                indices = np.arange(start, stop, step)
            else:
                # Array/List indices
                indices = np.asanyarray(idx)
                if indices.dtype == bool:
                    indices = np.flatnonzero(indices)
            
            return self._gather(indices)
            
        raise TypeError(f"Invalid index type: {type(idx)}")

    def __iter__(self) -> Generator[Seq, None, None]:
        for i in range(self._count):
            start = self._starts[i]
            length = self._lengths[i]
            yield self._alphabet.seq_from(self._data[start:start + length])

    def generate_id(self, digest_size: int = 8) -> bytes:
        """Generates a deterministic ID for the entire batch using BLAKE2b.

        Args:
            digest_size: Hash digest size in bytes.

        Returns:
            The hex digest as bytes.
        """
        return hexlify(blake2b(self._data.tobytes(), digest_size=digest_size, usedforsecurity=False).digest())

    def generate_ids(self, digest_size: int = 8) -> np.ndarray:
        """Generates a deterministic ID for each sequence in the batch.

        Args:
            digest_size: Hash digest size in bytes per ID.

        Returns:
            An object array of hex digest bytes, one per sequence.

        Examples:
            >>> ids = batch.generate_ids()
            >>> ids[0]
            b'a1b2c3d4e5f6a7b8'
        """
        ids = np.empty(self._count, dtype=object)
        for i in range(self._count):
            start = self._starts[i]
            length = self._lengths[i]
            ids[i] = hexlify(blake2b(self._data[start:start + length], digest_size=digest_size, usedforsecurity=False).digest())
        return ids

    def _gather(self, indices: np.ndarray) -> 'SeqBatch':
        """Internal method to gather sequences by index array.

        Args:
            indices: Integer array of sequence indices to gather.

        Returns:
            A new ``SeqBatch`` containing only the selected sequences.
        """
        if len(indices) == 0: return self._alphabet.empty()
        
        new_lengths = self._lengths[indices]
        total_len = new_lengths.sum()
        
        new_data = np.empty(total_len, dtype=np.uint8)
        new_starts = np.zeros(len(indices), dtype=np.int32)
        if len(indices) > 1:
            np.cumsum(new_lengths[:-1], out=new_starts[1:])
            
        _batch_gather_kernel(self._data, self._starts, self._lengths, indices, new_data, new_starts)
        return self._alphabet.new_batch(new_data, new_starts, new_lengths)


class CompressedSeq(HasAlphabet, Batchable):
    """
    A bit-packed sequence for reduced memory footprint.

    Stores symbols using fewer bits per base (e.g. 2 bits for DNA),
    achieving up to 4× compression over ``Seq``.

    Args:
        data: Packed ``uint8`` numpy array.
        length: Number of symbols (may differ from ``len(data)`` × packing ratio
            due to padding).
        alphabet: The owning ``Alphabet``.
        bits: Bits per symbol (e.g. 2 for DNA).
        _validation_token: Internal token (must be the alphabet).

    Examples:
        >>> cseq = Alphabet.DNA.compress(seq)
        >>> cseq.decompress() == seq
        True
    """
    __slots__ = ('_data', '_length', '_alphabet', '_bits')
    def __init__(self, data: np.ndarray, length: int, alphabet: 'Alphabet', bits: int, _validation_token: object = None):
        if _validation_token is not alphabet:
            raise PermissionError("CompressedSeq objects must be created via an Alphabet")
        self._data = data
        self._length = length
        self._alphabet = alphabet
        self._bits = bits

    def __len__(self): return self._length
    
    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``CompressedSeqBatch`` class.
        """
        return CompressedSeqBatch
    
    @property
    def alphabet(self) -> 'Alphabet':
        """Returns the alphabet used for encoding.

        Returns:
            The owning ``Alphabet`` singleton.
        """
        return self._alphabet

    def decompress(self) -> Seq:
        """Decompresses back to a standard ``Seq``.

        Returns:
            A full ``Seq`` object with the original encoded data.

        Examples:
            >>> cseq.decompress() == original_seq
            True
        """
        return self._alphabet.decompress(self)
    
    def __repr__(self):
        return f"<CompressedSeq: {self._length} bp, {self._bits} bits/sym>"


class CompressedSeqBatch(Batch, HasAlphabet):
    """
    A batch of bit-packed sequences, byte-aligned for efficient random access.

    Args:
        data: Packed ``uint8`` array of all compressed data.
        starts: ``int32`` offsets into *data* for each sequence.
        lengths: ``int32`` original (uncompressed) symbol counts.
        alphabet: The shared ``Alphabet``.
        bits: Bits per symbol.
        _validation_token: Internal token (must be the alphabet).

    Examples:
        >>> cbatch = Alphabet.DNA.compress_batch(batch)
        >>> cbatch.decompress() == batch
        True
    """
    __slots__ = ('_data', '_starts', '_lengths', '_alphabet', '_bits', '_count')
    def __init__(self, data: np.ndarray, starts: np.ndarray, lengths: np.ndarray, alphabet: 'Alphabet', bits: int,
                 _validation_token: object = None):
        if _validation_token is not alphabet:
            raise PermissionError("CompressedSeqBatch objects must be created via an Alphabet")
        self._data = data
        self._starts = starts
        self._lengths = lengths
        self._alphabet = alphabet
        self._bits = bits
        self._count = len(lengths)

    def __len__(self): return self._count

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``CompressedSeq`` class.
        """
        return CompressedSeq
    
    @property
    def alphabet(self) -> 'Alphabet':
        """Returns the shared alphabet.

        Returns:
            The ``Alphabet`` singleton.
        """
        return self._alphabet

    @classmethod
    def build(cls, components: Iterable[object]) -> 'Batch':
        """Not supported directly — use ``Alphabet.compress()``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Direct build not supported. Use Alphabet.compress()")

    @classmethod
    def concat(cls, batches: Iterable['CompressedSeqBatch']) -> 'CompressedSeqBatch':
        """Concatenates multiple CompressedSeqBatch objects into one.

        Args:
            batches: An iterable of ``CompressedSeqBatch`` objects.

        Returns:
            A single concatenated ``CompressedSeqBatch``.

        Raises:
            ValueError: If the list is empty.
        """
        batches = list(batches)
        if not batches: raise ValueError("Cannot concatenate empty list")
        first = batches[0]
        
        data = np.concatenate([b._data for b in batches])
        lengths = np.concatenate([b._lengths for b in batches])
        starts = np.zeros(len(lengths), dtype=np.int32) 
        
        if len(lengths) > 1: 
            per_byte = 8 // first._bits
            byte_lens = (lengths + per_byte - 1) // per_byte
            np.cumsum(byte_lens[:-1], out=starts[1:])
            
            np.cumsum(byte_lens[:-1], out=starts[1:])
            
        return cls(data, starts, lengths, first._alphabet, first._bits, _validation_token=first._alphabet)
    
    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by data, starts, and lengths arrays.
        """
        return self._data.nbytes + self._starts.nbytes + self._lengths.nbytes

    def copy(self) -> 'CompressedSeqBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``CompressedSeqBatch`` with copied arrays.
        """
        return self.__class__(self._data.copy(), self._starts.copy(), self._lengths.copy(), self._alphabet, self._bits, _validation_token=self._alphabet)

    @classmethod
    def empty(cls) -> 'CompressedSeqBatch':
        """Not supported directly — use ``Alphabet.empty_compressed()`` instead.

        Raises:
            TypeError: Always.
        """
        raise TypeError("CompressedSeqBatch.empty() requires an Alphabet. Use Alphabet.empty_compressed() instead.")

    @classmethod
    def zeros(cls, n: int) -> 'CompressedSeqBatch':
        """Not supported directly — use ``Alphabet.zeros_compressed(n)`` instead.

        Raises:
            TypeError: Always.
        """
        raise TypeError("CompressedSeqBatch.zeros() requires an Alphabet. Use Alphabet.zeros_compressed(n) instead.")

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            if item < 0: item += self._count
            if not 0 <= item < self._count: raise IndexError("Index out of range")
            start = self._starts[item]
            length = self._lengths[item]
            per_byte = 8 // self._bits
            byte_len = (length + per_byte - 1) // per_byte
            data = self._data[start : start + byte_len]
            return CompressedSeq(data, length, self._alphabet, self._bits, _validation_token=self._alphabet)
        
        if isinstance(item, slice):
            start, stop, step = item.indices(self._count)
            if step != 1: raise NotImplementedError("Batch slicing with step != 1 not supported")
            new_lengths = self._lengths[start:stop]
            p_start = self._starts[start]
            p_end = self._starts[stop] if stop < self._count else len(self._data)
            return CompressedSeqBatch(self._data[p_start:p_end], self._starts[start:stop] - p_start, new_lengths, self._alphabet, self._bits, _validation_token=self._alphabet)
            
        raise NotImplementedError("Slicing not implemented for CompressedSeqBatch")

    def decompress(self) -> SeqBatch:
        """Decompresses the batch back to a standard ``SeqBatch``.

        Returns:
            A ``SeqBatch`` with the original uncompressed symbol data.
        """
        return self._alphabet.decompress_batch(self)

    def __repr__(self): return f"<CompressedSeqBatch: {len(self)} sequences>"


# TODO: There should be no kernels in the containers module - move to core.alphabet or consider moving module
# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_gather_kernel(data, starts, lengths, indices, out_data, out_starts):
    n = len(indices)
    for i in prange(n):
        idx = indices[i]
        src_s = starts[idx]
        l = lengths[idx]
        dst_s = out_starts[i]
        out_data[dst_s : dst_s + l] = data[src_s : src_s + l]


# @jit(nopython=True, cache=True, nogil=True)
# def _sparse_reconstruct_kernel(
#         start: int, stop: int,
#         breakpoints: np.ndarray,
#         offsets: np.ndarray,
#         sources: np.ndarray,
#         ref_seq: np.ndarray,
#         mut_pool: np.ndarray
# ) -> np.ndarray:
#     """
#     Numba kernel to reconstruct a slice from the sparse index.
#     Performance: Zero Python overhead, copy-free reconstruction where possible.
#     """
#     length = stop - start
#     result = np.empty(length, dtype=np.uint8)
#
#     # Find the starting block index
#     # searchsorted returns the insertion point. We want the block *covering* start.
#     # So if breakpoints are [0, 10, 20] and start is 5:
#     # searchsorted(5, side='right') -> 1. block_idx = 0.
#     blk_idx = np.searchsorted(breakpoints, start, side='right') - 1
#     if blk_idx < 0: blk_idx = 0
#
#     current_virt = start
#     filled = 0
#     n_blocks = len(breakpoints)
#
#     while filled < length and blk_idx < n_blocks:
#         # Determine boundaries of the current block
#         blk_start = breakpoints[blk_idx]
#
#         # If this is the last block, the end is infinity (effectively)
#         if blk_idx < n_blocks - 1:
#             blk_end = breakpoints[blk_idx + 1]
#         else:
#             blk_end = 9223372036854775807
#
#         # Calculate overlap between [start, stop] and [blk_start, blk_end]
#         chunk_start = max(current_virt, blk_start)
#         chunk_end = min(stop, blk_end)
#         chunk_len = chunk_end - chunk_start
#
#         if chunk_len > 0:
#             offset = offsets[blk_idx]
#             phys_start = chunk_start + offset
#             phys_end = phys_start + chunk_len
#             source = sources[blk_idx]
#
#             if source == 0:  # Reference
#                 result[filled: filled + chunk_len] = ref_seq[phys_start: phys_end]
#             else:  # Mutation Pool
#                 result[filled: filled + chunk_len] = mut_pool[phys_start: phys_end]
#
#             filled += chunk_len
#             current_virt += chunk_len
#
#         blk_idx += 1
