"""
Module for representing ASCII alphabets and biological sequences
"""
from hashlib import blake2b
from typing import Union, Iterable, Final, Generator, Literal

import numpy as np

from baclib.core.interval import Interval
from baclib.utils.resources import jit, RESOURCES

if 'numba' in RESOURCES.optional_packages: 
    from numba import prange
else: 
    prange = range


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class AlphabetError(Exception): pass
class TranslationError(AlphabetError): pass


# Classes --------------------------------------------------------------------------------------------------------------
class Alphabet:
    """
    A class to represent an alphabet of ASCII symbols
    """
    _EXTENSIONS = {'gbk': 'dna', 'genbank': 'dna', 'fasta': 'dna', 'fna': 'dna',
                   'ffn': 'dna', 'ffa': 'dna', 'faa': 'amino'}
    _CACHE = {}
    __slots__ = ('_data', '_lookup_table', '_complement', '_trans_table', '_delete_bytes')
    DTYPE: Final = np.uint8
    INVALID: Final = np.iinfo(DTYPE).max
    MAX_LEN: Final = INVALID + 1
    ENCODING: Final = 'ascii'

    def __init__(self, symbols: bytes, complement: bytes = None):
        if not symbols.isascii(): raise AlphabetError('Alphabet symbols must be a valid ASCII string')
        if len(symbols) > self.MAX_LEN:
            raise AlphabetError(f'Alphabet size cannot exceed {self.MAX_LEN} symbols ({self.DTYPE})')
        if len(set(symbols.upper())) != len(symbols): raise AlphabetError('Alphabet contains duplicate symbols')

        self._data: np.ndarray = np.frombuffer(symbols, dtype=self.DTYPE)

        # Build Lookup Table
        self._lookup_table = np.full(self.MAX_LEN, self.INVALID, dtype=self.DTYPE)
        indices = np.arange(len(symbols), dtype=self.DTYPE)
        self._lookup_table[np.frombuffer(symbols, dtype=self.DTYPE)] = indices
        self._lookup_table[np.frombuffer(symbols.lower(), dtype=self.DTYPE)] = indices

        # Build Translation Tables
        self._trans_table = self._lookup_table.tobytes()
        self._delete_bytes = np.where(self._lookup_table == self.INVALID)[0].astype(self.DTYPE).tobytes()

        self._complement = None
        if complement is not None:
            if len(complement) != len(symbols):
                raise AlphabetError("Complement must be the same length as symbols")
            comp_indices = self._lookup_table[np.frombuffer(complement, dtype=self.DTYPE)]
            if np.any(comp_indices == self.INVALID):
                raise AlphabetError("Complement contains symbols not in alphabet")
            self._complement = comp_indices

    def __len__(self): return len(self._data)

    def __contains__(self, item):
        # Fast O(1) lookup using the table
        try:
            if isinstance(item, (int, np.integer)):
                return self._lookup_table[item] != self.INVALID
            if isinstance(item, (str, bytes)):
                if len(item) != 1: return False
                val = ord(item) if isinstance(item, str) else item[0]
                return self._lookup_table[val] != self.INVALID
        except (IndexError, ValueError, TypeError):
            pass
        return False

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __array__(self, dtype=None):
        return self._data.astype(dtype, copy=False) if dtype else self._data

    def __repr__(self):
        return repr(self._data)

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, Alphabet): return False
        return np.array_equal(self._data, other._data)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._data.tobytes())

    @property
    def bits_per_symbol(self) -> int:
        return (len(self._data) - 1).bit_length()

    @property
    def complement(self):
        return self._complement

    def masker(self, k: int) -> tuple[int, int, np.dtype]:
        """Returns (bits_per_symbol, bit_mask, dtype) for a specific K."""
        bps = self.bits_per_symbol
        total_bits = k * bps
        if total_bits <= 32:
            dtype = np.uint32
        elif total_bits <= 64:
            dtype = np.uint64
        else:
            raise ValueError(f"K={k} is too large for 64-bit hashing with {self}")
        mask = (1 << (bps * (k - 1))) - 1
        return bps, mask, dtype

    @classmethod
    def dna(cls):
        if (cached := cls._CACHE.get('dna')) is None:  # Note: We can reuse the same RC table logic
            cls._CACHE['dna'] = (cached := Alphabet(b'TCAG', b'AGTC'))
        return cached

    @classmethod
    def amino(cls):
        if (cached := cls._CACHE.get('amino')) is None:
            cls._CACHE['amino'] = (cached := Alphabet(b'ACDEFGHIKLMNPQRSTVWY*'))
        return cached

    @classmethod
    def from_extension(cls, extension: str, *args, **kwargs):
        if alphabet := cls._EXTENSIONS.get(extension, None):
            return getattr(cls, alphabet)(*args, **kwargs)
        raise AlphabetError(f'Unknown extension "{extension}"')

    def encode(self, text: bytes) -> np.ndarray:
        """
        Zero-copy encoding from Byte String to Array.
        """
        return np.frombuffer(text.translate(self._trans_table, delete=self._delete_bytes), dtype=self.DTYPE)

    def decode(self, encoded: np.ndarray) -> bytes:
        return self._data[encoded].tobytes()

    def seq(self, seq: Union['Seq', str, bytes, np.ndarray]) -> 'Seq':
        """
        Factory method. The ONLY valid way to create a Seq.
        """
        # 1. Handle Pre-encoded (Optimization for internal use)
        if isinstance(seq, Seq):
            if seq.alphabet != self: raise AlphabetError(f'Sequence has a different alphabet "{seq.alphabet}"')
            return seq
        # 2. Handle Numpy Array (Assume it is uint8 text or encoded?)
        # Convention: If it's uint8 array passed as 'core', treat as encoded indices
        if isinstance(seq, np.ndarray): return Seq(seq, self, _validation_token=self)
        # 3. Handle Text/Bytes
        if isinstance(seq, str): seq = seq.encode(self.ENCODING)
        # Use the vectorized encoder
        data = self.encode(seq)
        return Seq(data, self, _validation_token=self)

    def random(self, rng: np.random.Generator = None, length: int = None, min_len: int = 5, max_len: int = 5000,
               weights=None) -> 'Seq':
        """Generates a random sequence from this alphabet and coerces it to a Seq object"""
        if rng is None: rng = RESOURCES.rng
        length = length or rng.integers(min_len, max_len)
        # Optimization: Generate encoded indices directly (avoiding bytes round-trip)
        n_sym = len(self._data)
        if weights is None:
            indices = rng.integers(0, n_sym, size=length, dtype=self.DTYPE)
        else:
            indices = rng.choice(n_sym, size=length, p=weights)
        return self.seq(indices.astype(self.DTYPE))

    def random_many(self, lengths: Iterable[int], rng: np.random.Generator = None, weights=None) -> Generator[
        'Seq', None, None]:
        """Generates multiple random sequences efficiently."""
        if rng is None: rng = RESOURCES.rng
        lengths = np.asanyarray(lengths, dtype=np.int64)
        if lengths.size == 0: return
        total_len = lengths.sum()
        n_sym = len(self._data)

        # Optimization: Generate encoded indices directly
        if weights is None:
            indices = rng.integers(0, n_sym, size=total_len, dtype=self.DTYPE)
        else:
            indices = rng.choice(n_sym, size=total_len, p=weights)

        full_arr = indices.astype(self.DTYPE)
        current = 0
        for l in lengths:
            yield self.seq(full_arr[current:current + l])
            current += l

    def reverse_complement(self, seq: 'Seq') -> 'Seq':
        """Reverse complements the sequence if the alphabet has a complement"""
        if self._complement is None: return seq
        # Use .encoded for direct numpy access (much faster than iterating reversed(seq))
        return self.seq(self._complement[seq.encoded[::-1]])


class GeneticCode:
    """Represents a genetic code table for translation"""
    _TABLES = {11: b'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'}
    _DNA = Alphabet.dna()
    _AMINO = Alphabet.amino()
    _CACHE = {}  # Store genetic code singletons here
    __slots__ = ('_data',)

    def __init__(self):
        # Verify the alphabet is compatible with the bitwise math
        # We need T=0, C=1, A=2, G=3
        expected = np.array([84, 67, 65, 71], dtype=Alphabet.DTYPE)  # ASCII for T, C, A, G
        if not np.array_equal(self._DNA._data, expected):
            raise AlphabetError("GeneticCode requires strict 'TCAG' alphabet ordering.")
        self._stop_val = self._AMINO._lookup_table[ord('*')]

    @classmethod
    def from_code(cls, code: int) -> 'GeneticCode':
        if (cached := cls._CACHE.get(code)) is None:
            if (table := cls._TABLES.get(code)) is None: raise NotImplementedError('Genetic code not implemented')
            cached = GeneticCode()
            # Optimization: Pre-encode the table to indices using lookup table directly
            cached._data = cls._AMINO._lookup_table[np.frombuffer(table, dtype=Alphabet.DTYPE)]
            cls._CACHE[code] = cached
        return cached

    def translate(self, seq: 'Seq', frame: Literal[0, 1, 2] = 0) -> 'Seq':
        # Use encoded sequence (0-3 integers) for faster lookup in small table
        # This avoids cache misses associated with the large 16MB lookup table
        n = len(seq)
        start = frame
        n_codons = (n - start) // 3
        if n_codons <= 0:
            if n < 3 - frame: raise TranslationError('Cannot translate sequence with less than 1 codon')
        translation = _translate_kernel(seq.encoded, self._data, start, n_codons, self._stop_val, Alphabet.DTYPE)
        return self._AMINO.seq(translation)

    def translate_all(self, seq: 'Seq') -> tuple['Seq', 'Seq', 'Seq']:
        """Translates all 3 forward reading frames efficiently."""
        if len(seq) < 3: raise TranslationError('Cannot translate sequence with less than 1 codon')

        # Optimization: Use Numba kernel to avoid allocating intermediate 'indices' and 'translation' arrays
        f0, f1, f2 = _translate_all_kernel(seq.encoded, self._data, Alphabet.DTYPE)
        return self._AMINO.seq(f0), self._AMINO.seq(f1), self._AMINO.seq(f2)

    def translate_batch(self, batch: 'SeqBatch', frame: int = 0) -> 'SeqBatch':
        """
        Translates an entire SeqBatch efficiently.
        """
        new_batch = SeqBatch([], self._AMINO)
        if len(batch) == 0: return new_batch
        data, starts, lengths = batch.arrays
        # 1. Calculate lengths (Pass 1)
        new_lengths = _batch_translate_len_kernel(
            data, starts, lengths, self._data, frame, self._stop_val
        )

        # 2. Calculate offsets
        new_count = len(lengths)
        new_starts = np.zeros(new_count, dtype=np.int32)
        if new_count > 0:
            np.cumsum(new_lengths[:-1], out=new_starts[1:])
            total_len = new_starts[-1] + new_lengths[-1]
        else:
            total_len = 0

        new_data = np.empty(total_len, dtype=Alphabet.DTYPE)

        # 3. Fill (Pass 2)
        if total_len > 0:
            _batch_translate_fill_kernel(
                data, starts, self._data, frame, new_data, new_starts, new_lengths
            )

        # 4. Construct Result
        new_data.flags.writeable = False
        new_starts.flags.writeable = False
        new_lengths.flags.writeable = False

        new_batch._data = new_data
        new_batch._starts = new_starts
        new_batch._lengths = new_lengths
        new_batch._count = new_count

        return new_batch


class Seq:
    """
    Sequence container optimized for high-throughput genomics.
    Holds ONLY encoded integers (uint8) to minimize memory usage.
    """
    __slots__ = ('_data', '_alphabet', '_hash')
    DTYPE: Final = np.uint8
    ENCODING: Final = 'ascii'

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
    def __str__(self): return self.__bytes__().decode(self.ENCODING)
    def __iter__(self): return iter(self._data)
    def __repr__(self):
        if len(self) <= 14: return str(self)
        # Optimization: Decode only the parts we show
        head = self._alphabet.decode(self._data[:7]).decode(self.ENCODING)
        tail = self._alphabet.decode(self._data[-7:]).decode(self.ENCODING)
        return f"{head}...{tail}"
    # Use numpy flip for reversal (Zero Copy view if possible)
    def __reversed__(self) -> 'Seq': return self._alphabet.seq(np.flip(self._data))

    def __contains__(self, item):
        # 1. Handle Integer (Raw code check)
        if isinstance(item, (int, np.integer)): return item in self._data
        # 2. Handle Subsequence (Seq, bytes, str)
        query = None
        if isinstance(item, Seq):
            if item.alphabet is not self.alphabet: return False
            query = item._data.tobytes()
        elif isinstance(item, (bytes, str)):
            if isinstance(item, str): item = item.encode(self.ENCODING)
            query = self.alphabet.encode(item).tobytes()
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
        if self._hash is None: self._hash = hash(self._data.tobytes())
        return self._hash

    def __add__(self, other: 'Seq') -> 'Seq':
        if self._alphabet is not other._alphabet:
            raise ValueError("Cannot concatenate sequences with different alphabets")
        # Fast Int concatenation
        return self._alphabet.seq(np.concatenate((self._data, other._data), axis=0))

    def __mul__(self, other: int) -> 'Seq':
        if not isinstance(other, int): return NotImplemented
        if other <= 0: return self._alphabet.seq(np.empty(0, dtype=self.DTYPE))
        return self._alphabet.seq(np.tile(self._data, other))

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
        # 1. Standard Slicing (Fastest)
        if isinstance(item, slice):
            # Numpy handles the slicing logic/views
            return self._alphabet.seq(self._data[item])

        # 2. Integer Access
        if isinstance(item, int): return self._alphabet.seq(self._data[item:item+1])

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
                return self._alphabet.seq(rc_data)

            # For now, we assume we must decode->trans->encode or use a cached RC table
            # Ideally: return self._alphabet.core(encoded=self._alphabet.rc_code(chunk_encoded))
            return self._alphabet.reverse_complement(self._alphabet.seq(chunk_encoded))

        return self._alphabet.seq(chunk_encoded)

    def generate_id(self, digest_size: int = 8) -> bytes:
        """
        Generates a deterministic, fixed-length ID based on the sequence content.
        Uses BLAKE2b hashing. Output is a hex string (2 * digest_size chars).
        """
        return blake2b(self._data.tobytes(), digest_size=digest_size, usedforsecurity=False).hexdigest().encode('ascii')


class SeqBatch:
    """
    A 'Struct of Arrays' container that flattens a list of Seqs
    into contiguous memory for Numba parallel processing.
    """
    __slots__ = ('_alphabet', '_data', '_starts', '_lengths', '_count')
    DTYPE = np.uint8
    def __init__(self, items: Iterable[Seq], alphabet: 'Alphabet' = None):
        """
        Optimized initialization that prevents memory spikes.
        """
        self._alphabet = None
        # 1. Handle Pre-sized Lists (Fast Path)
        if isinstance(items, (list, tuple)):
            self._count = len(items)
            if self._count == 0:
                self._setup_empty()
                return

            # First pass: Get lengths without creating new objects
            self._lengths = np.empty(self._count, dtype=np.int32)
            total_len = 0
            for i, s in enumerate(items):
                if self._alphabet is None: self._alphabet = s.alphabet
                elif self._alphabet != s.alphabet:
                    raise ValueError("Cannot concatenate sequences with different alphabets")
                l = len(s)
                self._lengths[i] = l
                total_len += l

            # Allocate exact memory once
            self._data = np.empty(total_len, dtype=self.DTYPE)

            # Starts
            self._starts = np.zeros(self._count, dtype=np.int32)
            if self._count > 1:
                np.cumsum(self._lengths[:-1], out=self._starts[1:])

            # Second pass: Fill data
            # This is much more memory efficient than np.concatenate([list...])
            curr = 0
            for s in items:
                l = len(s)
                self._data[curr:curr + l] = s.encoded
                curr += l

            # Lock arrays to ensure immutability and thread-safety
            self._data.flags.writeable = False
            self._starts.flags.writeable = False
            self._lengths.flags.writeable = False

        # 2. Handle Generic Iterators (Slower, requires consumption)
        else:
            # Fallback to list conversion if iterator
            # (We have to consume it anyway to know size)
            items_list = list(items)
            # Recursively call self
            self.__init__(items_list, alphabet)

    def _setup_empty(self):
        self._data = np.empty(0, dtype=self.DTYPE)
        self._starts = np.empty(0, dtype=np.int32)
        self._lengths = np.empty(0, dtype=np.int32)
        self._alphabet = Alphabet.dna()

        self._data.flags.writeable = False
        self._starts.flags.writeable = False
        self._lengths.flags.writeable = False

    # --- Numba Accessors ---
    # Properties to unpack into Numba function arguments: *batch.arrays
    @property
    def arrays(self): return self._data, self._starts, self._lengths
    @property
    def alphabet(self): return self._alphabet
    @property
    def encoded(self): return self._data
    def __len__(self): return self._count
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0: idx += self._count
            if not 0 <= idx < self._count: raise IndexError("SeqBatch index out of range")
            start = self._starts[idx]
            length = self._lengths[idx]
            return self._alphabet.seq(self._data[start:start + length])
        elif isinstance(idx, (slice, np.ndarray, list)):
            # Create a new SeqBatch from selected items
            selected_items = [self[i] for i in range(len(self)) if i in idx] # This is inefficient for large batches
            # TODO: Implement more efficient slicing for SeqBatch
            return SeqBatch(selected_items, self._alphabet)
        raise TypeError(f"Invalid index type: {type(idx)}")

    def __iter__(self) -> Generator[Seq, None, None]:
        # The alphabet property is available via the first sequence, or needs to be passed in init
        # For now, assume all sequences in a batch share the same alphabet.
        # This is a safe assumption given how BatchSeq is constructed.
        if self._count == 0: return
        for i in range(self._count):
            start = self._starts[i]
            length = self._lengths[i]
            yield self._alphabet.seq(self._data[start:start + length])

    def generate_id(self, digest_size: int = 8) -> bytes:
        """
        Generates a deterministic, fixed-length ID based on the sequence content.
        Uses BLAKE2b hashing. Output is a hex string (2 * digest_size chars).
        """
        return blake2b(self._data.tobytes(), digest_size=digest_size, usedforsecurity=False).hexdigest().encode('ascii')


# class SparseSeq(Seq):
#     __slots__ = ('reference', '_length', '_breakpoints', '_offsets', '_sources', '_mut_pool')
#
#     def __init__(self, reference: Seq, mutations: Iterable['Mutation']):
#         # Fix: Pass empty array to satisfy Seq.__init__ type checks
#         super().__init__(data=np.empty(0, dtype=self._DTYPE),
#                          alphabet=reference.alphabet,
#                          _validation_token=reference.alphabet)
#
#         self.reference = reference
#
#         # 1. Sort mutations
#         mutations = sorted(list(mutations), key=lambda x: x.interval.start)
#
#         # 2. Pre-allocate flattened arrays
#         # We estimate max blocks = 2 * mutations + 1
#         capacity = len(mutations) * 2 + 1
#         breakpoints = np.empty(capacity, dtype=np.int64)
#         offsets = np.empty(capacity, dtype=np.int64)
#         sources = np.empty(capacity, dtype=np.uint8)  # 0=Ref, 1=Pool
#
#         # Mutation Pool Builder
#         mut_pool_parts = []
#         mut_pool_cursor = 0
#
#         virtual_pos = 0
#         ref_pos = 0
#         idx = 0
#
#         for mut in mutations:
#             # A. Ref Block
#             ref_len_before = mut.interval.start - ref_pos
#             if ref_len_before > 0:
#                 breakpoints[idx] = virtual_pos
#                 # Logic: Real = Virtual + Offset  =>  Offset = Real - Virtual
#                 offsets[idx] = ref_pos - virtual_pos
#                 sources[idx] = 0
#                 idx += 1
#
#                 virtual_pos += ref_len_before
#                 ref_pos += ref_len_before
#
#             # B. Mutation Block
#             if len(mut.alt_seq) > 0:
#                 breakpoints[idx] = virtual_pos
#
#                 # Add to pool
#                 encoded_alt = mut.alt_seq.code
#                 mut_pool_parts.append(encoded_alt)
#
#                 # Calculate Offset into Pool
#                 # Real(Pool) = Virtual + Offset => Offset = Pool_Cursor - Virtual
#                 offsets[idx] = mut_pool_cursor - virtual_pos
#                 sources[idx] = 1
#                 idx += 1
#
#                 mut_len = len(encoded_alt)
#                 virtual_pos += mut_len
#                 mut_pool_cursor += mut_len
#
#             ref_pos += len(mut.ref_seq)
#
#         # C. Final Ref Block
#         remaining_ref = len(reference) - ref_pos
#         if remaining_ref > 0:
#             breakpoints[idx] = virtual_pos
#             offsets[idx] = ref_pos - virtual_pos
#             sources[idx] = 0
#             idx += 1
#             virtual_pos += remaining_ref
#
#         # Store
#         self._length = virtual_pos
#         self._breakpoints = breakpoints[:idx]
#         self._offsets = offsets[:idx]
#         self._sources = sources[:idx]
#
#         if mut_pool_parts: self._mut_pool = np.concatenate(mut_pool_parts)
#         else: self._mut_pool = np.empty(0, dtype=np.uint8)
#
#     def __len__(self): return self._length
#
#     @property
#     def code(self) -> np.ndarray:
#         # Fast full reconstruction
#         return _sparse_reconstruct_kernel(
#             0, self._length,
#             self._breakpoints, self._offsets, self._sources,
#             self.reference.code, self._mut_pool
#         )
#
#     @property
#     def symbols(self) -> bytes: return self._alphabet.decode(self.code)
#
#     def densify(self): pass


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _translate_kernel(encoded_seq, flat_table, start, n_codons, invalid, dtype):
    """
    Translates DNA -> Amino Acid using a flat lookup table and bitwise math.
    Assumes DNA encoding is 0=T, 1=C, 2=A, 3=G (2 bits).
    """
    res = np.empty(n_codons, dtype=dtype)
    for i in range(n_codons):
        # Calculate offset of current codon
        base = start + (i * 3)
        # Fetch the three 2-bit integers (0-3)
        b1 = encoded_seq[base]
        b2 = encoded_seq[base + 1]
        b3 = encoded_seq[base + 2]
        # Calculate 1D index: b1*16 + b2*4 + b3
        # Using bitwise ops is slightly faster/cleaner for powers of 2
        # b1 << 4  == b1 * 16
        # b2 << 2  == b2 * 4
        flat_idx = (b1 << 4) | (b2 << 2) | b3
        aa = flat_table[flat_idx]
        # Early stopping check
        if aa == invalid: return res[:i]
        res[i] = aa
    return res


@jit(nopython=True, cache=True, nogil=True)
def _translate_all_kernel(encoded_seq, flat_table, dtype):
    """
    Translates all 3 frames in a single pass.
    """
    n = len(encoded_seq)
    # Calculate sizes for each frame
    n0 = n // 3
    n1 = (n - 1) // 3
    n2 = (n - 2) // 3

    f0 = np.empty(n0, dtype=dtype)
    f1 = np.empty(n1, dtype=dtype)
    f2 = np.empty(n2, dtype=dtype)

    # We iterate up to n-2 to form codons
    for i in range(n - 2):
        # Calculate codon index
        flat_idx = (encoded_seq[i] << 4) | (encoded_seq[i + 1] << 2) | encoded_seq[i + 2]
        aa = flat_table[flat_idx]

        # Assign to appropriate frame based on index modulo
        rem = i % 3
        if rem == 0:
            f0[i // 3] = aa
        elif rem == 1:
            f1[i // 3] = aa
        else:
            f2[i // 3] = aa

    return f0, f1, f2


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_translate_len_kernel(data, starts, lengths, table, frame, invalid):
    n = len(lengths)
    out_lens = np.empty(n, dtype=np.int32)

    for i in prange(n):
        s = starts[i]
        l = lengths[i]

        n_codons = (l - frame) // 3
        if n_codons <= 0:
            out_lens[i] = 0
            continue

        # Scan for stop
        actual_len = n_codons
        for j in range(n_codons):
            base = s + frame + (j * 3)
            b1 = data[base]
            b2 = data[base + 1]
            b3 = data[base + 2]
            idx = (b1 << 4) | (b2 << 2) | b3
            if table[idx] == invalid:
                actual_len = j
                break
        out_lens[i] = actual_len
    return out_lens


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_translate_fill_kernel(data, starts, table, frame, out_data, out_starts, out_lengths):
    n = len(starts)
    for i in prange(n):
        s = starts[i]
        out_s = out_starts[i]
        l = out_lengths[i]

        for j in range(l):
            base = s + frame + (j * 3)
            b1 = data[base]
            b2 = data[base + 1]
            b3 = data[base + 2]
            idx = (b1 << 4) | (b2 << 2) | b3
            out_data[out_s + j] = table[idx]


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
#
#     current_virt = start
#     filled = 0
#
#     while filled < length:
#         # Determine boundaries of the current block
#         blk_start = breakpoints[blk_idx]
#
#         # If this is the last block, the end is infinity (effectively)
#         if blk_idx < len(breakpoints) - 1:
#             blk_end = breakpoints[blk_idx + 1]
#         else:
#             # Safe large number, or calculate exact total length if passed
#             blk_end = 9223372036854775807
#
#             # Calculate overlap between [start, stop] and [blk_start, blk_end]
#         # We only care about the part of the block *after* our current cursor
#         chunk_start = current_virt
#         chunk_end = blk_end
#         if chunk_end > stop: chunk_end = stop
#
#         chunk_len = chunk_end - chunk_start
#
#         if chunk_len > 0:
#             # Map Virtual -> Physical
#             # The magic: Offset works for BOTH Reference and Mutation Pool
#             offset = offsets[blk_idx]
#             phys_start = chunk_start + offset
#             phys_end = phys_start + chunk_len
#
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
#
#     return result
