"""
Module for representing biological sequences
"""
from binascii import hexlify
from hashlib import blake2b
from typing import Union, Iterable, Generator

import numpy as np

from baclib.core.interval import Interval
from baclib.utils.resources import jit, RESOURCES
from baclib.utils import Batch
from baclib.utils.protocols import HasAlphabet

if RESOURCES.has_module('numba'):
    from numba import prange
else: 
    prange = range


# Classes --------------------------------------------------------------------------------------------------------------
class Seq(HasAlphabet):
    """
    Sequence container optimized for high-throughput genomics.
    Holds ONLY encoded integers (uint8) to minimize memory usage.

    Note:
        Seq objects should be created via `Alphabet.seq()` or `Alphabet.random()`.
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
    def alphabet(self) -> 'Alphabet': return self._alphabet

    @property
    def encoded(self) -> np.ndarray:
        """Returns the underlying integer array (Zero Copy)."""
        return self._data

    def __array__(self, dtype=None):
        """Allows the Seq to be treated as a numpy array."""
        return self._data.astype(dtype, copy=False) if dtype else self._data

    def __bytes__(self) -> bytes: return self._alphabet.decode(self._data)
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
        """
        Gets a subsequence.

        Args:
            item: Index, slice, or Interval.

        Returns:
            A new Seq object representing the subsequence.
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
        """
        Generates a deterministic, fixed-length ID based on the sequence content.
        Uses BLAKE2b hashing. Output is a hex string (2 * digest_size chars).

        Args:
            digest_size: Size of the digest in bytes.

        Returns:
            The hex digest as bytes.
        """
        return hexlify(blake2b(self._data.tobytes(), digest_size=digest_size, usedforsecurity=False).digest())


class SeqBatch(Batch, HasAlphabet):
    """
    A 'Struct of Arrays' container that flattens a list of Seqs
    into contiguous memory for Numba parallel processing.
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
    def from_seqs(cls, seqs: Iterable['Seq']) -> 'SeqBatch':
        """
        Creates a SeqBatch from a list of Seq objects.
        Infers the alphabet from the first sequence.
        """
        seqs_list = list(seqs)
        if not seqs_list:
             raise ValueError("Cannot create SeqBatch from empty sequence list. Use Alphabet.empty_batch() instead.")
        return seqs_list[0].alphabet.batch_from(seqs_list)

    @classmethod
    def concat(cls, batches: Iterable['SeqBatch']) -> 'SeqBatch':
        """
        Concatenates multiple SeqBatches into a single batch.
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

    # --- Numba Accessors ---
    # Properties to unpack into Numba function arguments: *batch.arrays
    @property
    def arrays(self): return self._data, self._starts, self._lengths
    @property
    def alphabet(self): return self._alphabet
    @property
    def encoded(self): return self._data
    @property
    def starts(self): return self._starts
    @property
    def lengths(self): return self._lengths

    def empty(self) -> 'SeqBatch':
        return self._alphabet.empty_batch()

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
                    if new_count == 0: return self._alphabet.empty_batch()
                    
                    s_start = self._starts[start]
                    s_end = self._starts[stop] if stop < self._count else len(self._data)
                    
                    new_data = self._data[s_start:s_end]
                    new_lengths = self._lengths[start:stop]
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
        """
        Generates a deterministic, fixed-length ID based on the sequence content.
        Uses BLAKE2b hashing. Output is a hex string (2 * digest_size chars).
        """
        return hexlify(blake2b(self._data.tobytes(), digest_size=digest_size, usedforsecurity=False).digest())

    def _gather(self, indices: np.ndarray) -> 'SeqBatch':
        """Internal method to gather sequences by index."""
        if len(indices) == 0: return self._alphabet.empty_batch()
        
        new_lengths = self._lengths[indices]
        total_len = new_lengths.sum()
        
        new_data = np.empty(total_len, dtype=np.uint8)
        new_starts = np.zeros(len(indices), dtype=np.int32)
        if len(indices) > 1:
            np.cumsum(new_lengths[:-1], out=new_starts[1:])
            
        _batch_gather_kernel(self._data, self._starts, self._lengths, indices, new_data, new_starts)
        return self._alphabet.new_batch(new_data, new_starts, new_lengths)


class CompressedSeq(HasAlphabet):
    """
    A bit-packed sequence.
    Stores symbols in a compressed format (e.g. 2 bits per base for DNA).
    """
    __slots__ = ('_data', '_length', '_alphabet', '_bits')
    def __init__(self, data: np.ndarray, length: int, alphabet: 'Alphabet', bits: int):
        self._data = data
        self._length = length
        self._alphabet = alphabet
        self._bits = bits

    def __len__(self): return self._length
    
    def decompress(self) -> 'Seq':
        """Decompresses the sequence back to a standard Seq."""
        decoded = _unpack_seq_kernel(self._data, self._length, self._bits)
        return self._alphabet.seq_from(decoded)
    
    def __repr__(self):
        return f"<CompressedSeq: {self._length} bp, {self._bits} bits/sym>"


class CompressedSeqBatch(Batch, HasAlphabet):
    """
    A batch of bit-packed sequences.
    Each sequence is byte-aligned for efficient random access.
    """
    __slots__ = ('_data', '_starts', '_lengths', '_alphabet', '_bits', '_count')
    def __init__(self, data: np.ndarray, starts: np.ndarray, lengths: np.ndarray, alphabet: 'Alphabet', bits: int):
        self._data = data
        self._starts = starts
        self._lengths = lengths
        self._alphabet = alphabet
        self._bits = bits
        self._count = len(lengths)

    def __len__(self): return self._count
    
    def empty(self) -> 'CompressedSeqBatch':
        return CompressedSeqBatch(np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), self._alphabet, self._bits)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            if item < 0: item += self._count
            if not 0 <= item < self._count: raise IndexError("Index out of range")
            start = self._starts[item]
            length = self._lengths[item]
            # Calculate byte length
            per_byte = 8 // self._bits
            byte_len = (length + per_byte - 1) // per_byte
            data = self._data[start : start + byte_len]
            return CompressedSeq(data, length, self._alphabet, self._bits)
        
        if isinstance(item, slice):
            start, stop, step = item.indices(self._count)
            if step != 1: raise NotImplementedError("Batch slicing with step != 1 not supported")
            new_lengths = self._lengths[start:stop]
            p_start = self._starts[start]
            p_end = self._starts[stop] if stop < self._count else len(self._data)
            return CompressedSeqBatch(self._data[p_start:p_end], self._starts[start:stop] - p_start, new_lengths, self._alphabet, self._bits)
            
        raise NotImplementedError("Slicing not implemented for CompressedSeqBatch")

    def decompress(self) -> 'SeqBatch':
        """Decompresses the batch back to a standard SeqBatch."""
        total_len = self._lengths.sum()
        out_data = np.empty(total_len, dtype=np.uint8)
        out_starts = np.zeros(self._count, dtype=np.int32)
        if self._count > 1:
            np.cumsum(self._lengths[:-1], out=out_starts[1:])
            
        _unpack_batch_kernel(self._data, self._starts, self._lengths, out_data, out_starts, self._bits)
        return self._alphabet.new_batch(out_data, out_starts, self._lengths)

    def __repr__(self): return f"<CompressedSeqBatch: {len(self)} sequences>"


class SparseSeq(Seq):
    """
    A compressed sequence representation that stores edits relative to a reference.
    Effectively a linear coordinate projection (Compacted De Bruijn Graph path).
    """
    __slots__ = ('_reference', '_length', '_breakpoints', '_offsets', '_sources', '_mut_pool')

    def __init__(self, reference: Seq, mutations: Iterable):
        """
        Args:
            reference: The reference Seq.
            mutations: Iterable of Mutation objects (must have interval, ref_seq, alt_seq).
        """
        # Bypass Seq.__init__ validation for internal construction if needed,
        # but here we play nice by passing empty data and the correct alphabet.
        super().__init__(data=np.empty(0, dtype=self._DTYPE),
                         alphabet=reference.alphabet,
                         _validation_token=reference.alphabet)

        self._reference = reference
        self._build_index(mutations)

    @classmethod
    def _from_data(cls, reference: Seq, length, breakpoints, offsets, sources, mut_pool):
        """Internal factory for slicing/copying without re-building index."""
        obj = cls.__new__(cls)
        # Manually set Seq attributes since we skipped __init__
        obj._alphabet = reference.alphabet
        obj._data = np.empty(0, dtype=cls._DTYPE) # Dummy
        obj._hash = None
        
        obj._reference = reference
        obj._length = length
        obj._breakpoints = breakpoints
        obj._offsets = offsets
        obj._sources = sources
        obj._mut_pool = mut_pool
        return obj

    def _build_index(self, mutations: Iterable):
        reference = self._reference

        # 1. Sort mutations
        mutations = sorted(list(mutations), key=lambda x: x.interval.start)

        # 2. Pre-allocate flattened arrays
        capacity = len(mutations) * 2 + 1
        breakpoints = np.empty(capacity, dtype=np.int64)
        offsets = np.empty(capacity, dtype=np.int64)
        sources = np.empty(capacity, dtype=np.uint8)  # 0=Ref, 1=Pool

        mut_pool_parts = []
        mut_pool_cursor = 0
        virtual_pos = 0
        ref_pos = 0
        idx = 0

        for mut in mutations:
            # A. Ref Block (Sequence before mutation)
            ref_len_before = mut.interval.start - ref_pos
            if ref_len_before > 0:
                breakpoints[idx] = virtual_pos
                offsets[idx] = ref_pos - virtual_pos
                sources[idx] = 0
                idx += 1
                virtual_pos += ref_len_before
                ref_pos += ref_len_before

            # B. Mutation Block (Alt Sequence)
            if len(mut.alt_seq) > 0:
                breakpoints[idx] = virtual_pos
                encoded_alt = mut.alt_seq.encoded
                mut_pool_parts.append(encoded_alt)
                offsets[idx] = mut_pool_cursor - virtual_pos
                sources[idx] = 1
                idx += 1
                mut_len = len(encoded_alt)
                virtual_pos += mut_len
                mut_pool_cursor += mut_len

            ref_pos += len(mut.ref_seq)

        # C. Final Ref Block
        remaining_ref = len(reference) - ref_pos
        if remaining_ref > 0:
            breakpoints[idx] = virtual_pos
            offsets[idx] = ref_pos - virtual_pos
            sources[idx] = 0
            idx += 1
            virtual_pos += remaining_ref

        self._length = virtual_pos
        self._breakpoints = breakpoints[:idx]
        self._offsets = offsets[:idx]
        self._sources = sources[:idx]
        self._mut_pool = np.concatenate(mut_pool_parts) if mut_pool_parts else np.empty(0, dtype=self._DTYPE)

    def __len__(self): return self._length

    @property
    def encoded(self) -> np.ndarray:
        """Materializes the full sequence into a numpy array."""
        return _sparse_reconstruct_kernel(
            0, self._length,
            self._breakpoints, self._offsets, self._sources,
            self._reference.encoded, self._mut_pool
        )

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            if item < 0: item += self._length
            if not 0 <= item < self._length: raise IndexError("SparseSeq index out of range")
            arr = _sparse_reconstruct_kernel(
                item, item + 1,
                self._breakpoints, self._offsets, self._sources,
                self._reference.encoded, self._mut_pool
            )
            return self._alphabet.seq_from(arr)

        if isinstance(item, slice):
            start, stop, step = item.indices(self._length)
            if step != 1:
                # Complex slicing: materialize the range then slice
                r_start, r_stop = (start, stop) if step > 0 else (stop + 1, start + 1)
                arr = _sparse_reconstruct_kernel(
                    r_start, r_stop, self._breakpoints, self._offsets, self._sources,
                    self._reference.encoded, self._mut_pool
                )
                # Adjust slice relative to the reconstructed window
                return self._alphabet.seq_from(arr[::step])

            arr = _sparse_reconstruct_kernel(
                start, stop, self._breakpoints, self._offsets, self._sources,
                self._reference.encoded, self._mut_pool
            )
            return self._alphabet.seq_from(arr)

        if isinstance(item, Interval):
            # Interval logic (assumes step=1)
            item = Interval.from_item(item, length=self._length)
            arr = _sparse_reconstruct_kernel(
                item.start, item.end, self._breakpoints, self._offsets, self._sources,
                self._reference.encoded, self._mut_pool
            )
            seq = self._alphabet.seq_from(arr)
            if item.strand == -1:
                return self._alphabet.reverse_complement(seq)
            return seq

        raise TypeError(f"Invalid index type: {type(item)}")

    def densify(self) -> Seq:
        """Returns a standard dense Seq object."""
        return self._alphabet.seq_from(self.encoded)


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


@jit(nopython=True, cache=True, nogil=True)
def _sparse_reconstruct_kernel(
        start: int, stop: int,
        breakpoints: np.ndarray,
        offsets: np.ndarray,
        sources: np.ndarray,
        ref_seq: np.ndarray,
        mut_pool: np.ndarray
) -> np.ndarray:
    """
    Numba kernel to reconstruct a slice from the sparse index.
    Performance: Zero Python overhead, copy-free reconstruction where possible.
    """
    length = stop - start
    result = np.empty(length, dtype=np.uint8)

    # Find the starting block index
    # searchsorted returns the insertion point. We want the block *covering* start.
    # So if breakpoints are [0, 10, 20] and start is 5:
    # searchsorted(5, side='right') -> 1. block_idx = 0.
    blk_idx = np.searchsorted(breakpoints, start, side='right') - 1
    if blk_idx < 0: blk_idx = 0

    current_virt = start
    filled = 0
    n_blocks = len(breakpoints)

    while filled < length and blk_idx < n_blocks:
        # Determine boundaries of the current block
        blk_start = breakpoints[blk_idx]

        # If this is the last block, the end is infinity (effectively)
        if blk_idx < n_blocks - 1:
            blk_end = breakpoints[blk_idx + 1]
        else:
            blk_end = 9223372036854775807

        # Calculate overlap between [start, stop] and [blk_start, blk_end]
        chunk_start = max(current_virt, blk_start)
        chunk_end = min(stop, blk_end)
        chunk_len = chunk_end - chunk_start

        if chunk_len > 0:
            offset = offsets[blk_idx]
            phys_start = chunk_start + offset
            phys_end = phys_start + chunk_len
            source = sources[blk_idx]

            if source == 0:  # Reference
                result[filled: filled + chunk_len] = ref_seq[phys_start: phys_end]
            else:  # Mutation Pool
                result[filled: filled + chunk_len] = mut_pool[phys_start: phys_end]

            filled += chunk_len
            current_virt += chunk_len

        blk_idx += 1

    return result


@jit(nopython=True, cache=True, nogil=True)
def _unpack_seq_kernel(packed, length, bits):
    out = np.empty(length, dtype=np.uint8)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    
    for i in range(length):
        byte_idx = i // per_byte
        bit_offset = (per_byte - 1 - (i % per_byte)) * bits
        val = (packed[byte_idx] >> bit_offset) & mask
        out[i] = val
    return out


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _unpack_batch_kernel(packed, packed_starts, lengths, out_data, out_starts, bits):
    n_seqs = len(lengths)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    for i in prange(n_seqs):
        p_s = packed_starts[i]
        l = lengths[i]
        dst_s = out_starts[i]
        for j in range(l):
            byte_idx = j // per_byte
            bit_offset = (per_byte - 1 - (j % per_byte)) * bits
            out_data[dst_s + j] = (packed[p_s + byte_idx] >> bit_offset) & mask
