"""
Module for representing ASCII alphabets and biological sequences
"""
from hashlib import blake2b
from typing import Union, Iterable, Final, Generator, Literal

import numpy as np

from baclib.core.interval import Interval, IntervalIndex
from baclib.utils.resources import jit, RESOURCES

if RESOURCES.has_module('numba'):
    from numba import prange
else: 
    prange = range


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class AlphabetError(Exception): pass
class TranslationError(AlphabetError): pass


# Classes --------------------------------------------------------------------------------------------------------------
class Alphabet:
    """
    A class to represent an alphabet of ASCII symbols.

    This class handles encoding/decoding of sequences and provides factory methods
    for creating `Seq` objects.

    Attributes:
        DTYPE: The numpy dtype used for internal storage (uint8).
        INVALID: The value used to represent invalid symbols.
        MAX_LEN: Maximum alphabet size.
        ENCODING: The encoding used for string conversion ('ascii').

    Examples:
        >>> dna = Alphabet.dna()
        >>> len(dna)
        4
        >>> 'A' in dna
        True
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
        """
        Initializes an Alphabet.

        Args:
            symbols: The symbols in the alphabet as bytes.
            complement: Optional complement symbols as bytes. Must be same length as symbols.

        Raises:
            AlphabetError: If symbols are not ASCII, too long, contain duplicates, or if complement is invalid.
        """
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
        """Returns the number of bits required to represent a symbol in this alphabet."""
        return (len(self._data) - 1).bit_length()

    @property
    def complement(self):
        """Returns the complement lookup table if available."""
        return self._complement

    def masker(self, k: int) -> tuple[int, int, np.dtype]:
        """
        Returns (bits_per_symbol, bit_mask, dtype) for a specific K.

        Args:
            k: The k-mer length.

        Returns:
            A tuple of (bits_per_symbol, bit_mask, dtype).

        Raises:
            ValueError: If K is too large for 64-bit hashing.
        """
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
        """
        Returns the standard DNA alphabet (TCAG).
        
        Returns:
            The DNA Alphabet singleton.
        """
        symbols = b'TCAG'
        if (cached := cls._CACHE.get(symbols)) is None:  # Note: We can reuse the same RC table logic
            cls._CACHE[symbols] = (cached := Alphabet(symbols, b'AGTC'))
        return cached

    @classmethod
    def amino(cls):
        """
        Returns the standard Amino Acid alphabet.
        
        Returns:
            The Amino Acid Alphabet singleton.
        """
        symbols = b'ACDEFGHIKLMNPQRSTVWY'
        if (cached := cls._CACHE.get(symbols)) is None: cls._CACHE[symbols] = (cached := Alphabet(symbols))
        return cached

    @classmethod
    def from_extension(cls, extension: str, *args, **kwargs):
        """
        Returns an Alphabet based on a file extension.

        Args:
            extension: The file extension (e.g., 'fasta', 'gbk').

        Returns:
            The corresponding Alphabet.

        Raises:
            AlphabetError: If the extension is unknown.
        """
        if alphabet := cls._EXTENSIONS.get(extension, None):
            return getattr(cls, alphabet)(*args, **kwargs)
        raise AlphabetError(f'Unknown extension "{extension}"')

    def encode(self, text: bytes) -> np.ndarray:
        """
        Zero-copy encoding from Byte String to Array.

        Args:
            text: The text to encode as bytes.

        Returns:
            A numpy array of encoded indices.
        """
        return np.frombuffer(text.translate(self._trans_table, delete=self._delete_bytes), dtype=self.DTYPE)

    def decode(self, encoded: np.ndarray) -> bytes:
        """
        Decodes an array of indices back to bytes.

        Args:
            encoded: The numpy array of indices.

        Returns:
            The decoded bytes.
        """
        return self._data[encoded].tobytes()

    def seq(self, seq: Union['Seq', str, bytes, np.ndarray]) -> 'Seq':
        """
        Factory method. The ONLY valid way to create a Seq.

        Args:
            seq: The sequence data. Can be a Seq, str, bytes, or numpy array.

        Returns:
            A Seq object.

        Raises:
            AlphabetError: If the sequence contains invalid symbols or has a different alphabet.

        Examples:
            >>> dna = Alphabet.dna()
            >>> s = dna.seq("ACGT")
            >>> str(s)
            'ACGT'
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

    def batch(self, data: np.ndarray, starts: np.ndarray, lengths: np.ndarray):
        return SeqBatch(data, starts, lengths, self)
    
    def random_seq(self, rng: np.random.Generator = None, length: int = None, min_len: int = 5, max_len: int = 5000,
                   weights=None) -> 'Seq':
        """
        Generates a random sequence from this alphabet and coerces it to a Seq object.

        Args:
            rng: Random number generator (optional).
            length: Exact length of sequence to generate.
            min_len: Minimum length if length is not specified.
            max_len: Maximum length if length is not specified.
            weights: Weights for each symbol (optional).

        Returns:
            A random Seq object.

        Examples:
            >>> dna = Alphabet.dna()
            >>> s = dna.random_seq(length=10)
            >>> len(s)
            10
        """
        if rng is None: rng = RESOURCES.rng
        length = length or rng.integers(min_len, max_len)
        # Optimization: Generate encoded indices directly (avoiding bytes round-trip)
        n_sym = len(self._data)
        if weights is None:
            indices = rng.integers(0, n_sym, size=length, dtype=self.DTYPE)
        else:
            indices = rng.choice(n_sym, size=length, p=weights)
        return self.seq(indices.astype(self.DTYPE))

    def random_batch(self, rng: np.random.Generator = None, n_seqs: int = None, min_seqs: int = 1,
                     max_seqs: int = 1000, length: int = None, min_len: int = 10, max_len: int = 5_000_000, 
                     weights=None) -> 'SeqBatch':
        """
        Generates a SeqBatch of random sequences efficiently.

        Args:
            rng: Random number generator (optional).
            n_seqs: Number of sequences to generate.
            min_seqs: Minimum number of sequences to generate.
            max_seqs: Maximum number of sequences to generate.
            length: Exact length of sequences to generate.
            min_len: Minimum length of sequences to generate.
            max_len: Maximum length of sequences to generate.
            weights: Weights for each symbol (optional).

        Returns:
            A SeqBatch containing the random sequences.
        """
        if rng is None: rng = RESOURCES.rng
        if n_seqs is None: n_seqs = int(rng.integers(min_seqs, max_seqs))

        if length is not None:
            if n_seqs > length:
                raise ValueError(f"Cannot partition length {length} into {n_seqs} sequences (min_len=1)")
            if n_seqs > 1:
                # Use choice without replacement to ensure distinct cuts (no zero-length seqs)
                cuts = np.sort(rng.choice(length - 1, size=n_seqs - 1, replace=False) + 1)
                bounds = np.concatenate(([0], cuts, [length]))
                lengths_arr = np.diff(bounds).astype(np.int32)
            else:
                lengths_arr = np.array([length], dtype=np.int32)
        else:
            lengths_arr = rng.integers(min_len, max_len, size=n_seqs, dtype=np.int32)
            np.maximum(1, lengths_arr, out=lengths_arr)
        
        if lengths_arr.size == 0:
            return SeqBatch(np.empty(0, dtype=self.DTYPE),
                            np.empty(0, dtype=np.int32),
                            np.empty(0, dtype=np.int32), self)

        total_len = lengths_arr.sum()
        n_sym = len(self._data)

        # Optimization: Generate encoded indices directly
        if weights is None:
            indices = rng.integers(0, n_sym, size=total_len, dtype=self.DTYPE)
        else:
            indices = rng.choice(n_sym, size=total_len, p=weights)

        starts = np.zeros(len(lengths_arr), dtype=np.int32)
        if len(lengths_arr) > 1:
            np.cumsum(lengths_arr[:-1], out=starts[1:])

        return SeqBatch(indices.astype(self.DTYPE, copy=False), starts, lengths_arr, self)

    def reverse_complement(self, seq: 'Seq') -> 'Seq':
        """
        Reverse complements the sequence if the alphabet has a complement.

        Args:
            seq: The input sequence.

        Returns:
            The reverse complemented sequence.

        Examples:
            >>> dna = Alphabet.dna()
            >>> s = dna.seq("ACGT")
            >>> rc = dna.reverse_complement(s)
            >>> str(rc)
            'ACGT'
        """
        if self._complement is None: return seq
        # Use .encoded for direct numpy access (much faster than iterating reversed(seq))
        return self.seq(self._complement[seq.encoded[::-1]])


class GeneticCode:
    """
    Represents a genetic code table for translation.
    """
    _DNA = Alphabet.dna()
    _AMINO = Alphabet.amino()
    _CACHE = {}  # Store genetic code singletons here
    _DTYPE = np.uint8
    __slots__ = ('_data', '_starts', '_stops')

    def __init__(self, table: bytes, starts: Iterable[bytes] = ()):
        # Optimization: Pre-encode the table to indices using lookup table directly
        self._data = self._AMINO._lookup_table[np.frombuffer(table, dtype=self._DTYPE)]
        # Populate stops and starts
        # We derive stops from the table string
        self._stops = np.frombuffer(table, dtype=self._DTYPE) == ord('*')
        # Populate starts
        self._starts = np.zeros(64, dtype=bool)
        
        valid_starts = [s for s in starts if len(s) == 3]
        if valid_starts:
            joined = b"".join(valid_starts)
            encoded = self._DNA.encode(joined)
            
            if len(encoded) == len(joined):
                # Fast path: Vectorized calculation (No invalid chars dropped)
                indices = (encoded[0::3] << 4) | (encoded[1::3] << 2) | encoded[2::3]
                self._starts[indices] = True
            else:  # Fallback: Process individually to handle invalid chars safely without frame shifts
                for s in valid_starts:
                    enc = self._DNA.encode(s)
                    if len(enc) == 3:
                        idx = (enc[0] << 4) | (enc[1] << 2) | enc[2]
                        self._starts[idx] = True

    @classmethod
    def bacterial(cls) -> 'GeneticCode':
        table = b'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
        if (cached := cls._CACHE.get(table)) is None:
            starts = [b'ATG', b'GTG', b'TTG', b'ATT', b'ATC', b'ATA']
            cls._CACHE[table] = (cached := GeneticCode(table, starts))
        return cached

    @property
    def starts(self) -> np.ndarray: return self._starts
    @property
    def stops(self) -> np.ndarray: return self._stops

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __array__(self, dtype=None):
        return self._data.astype(dtype, copy=False) if dtype else self._data

    def __repr__(self):
        return repr(self._data)

    def find_starts(self, seq: 'Seq') -> np.ndarray:
        """Returns indices of all start codons in the sequence (0-based)."""
        if seq.alphabet != self._DNA: raise ValueError("Sequence must use the DNA alphabet")
        return _find_codons_kernel(seq.encoded, self._starts)

    def find_stops(self, seq: 'Seq') -> np.ndarray:
        """Returns indices of all stop codons in the sequence (0-based)."""
        if seq.alphabet != self._DNA: raise ValueError("Sequence must use the DNA alphabet")
        return _find_codons_kernel(seq.encoded, self._stops)

    def find_orfs(self, seq: 'Seq', strand: Literal[0, 1, -1] = 0, min_len: int = 30, max_len: int = 3000, include_partials: bool = False) -> IntervalIndex:
        """
        Finds all Open Reading Frames (ORFs) in the sequence.
        Returns all in-frame start-stop pairs with no intervening stops.

        Args:
            seq: The input DNA sequence.
            strand: 1 (forward), -1 (reverse), or 0 (both).
            min_len: Minimum length in bases (inclusive).
            max_len: Maximum length in bases (inclusive).
            include_partials: If True, includes ORFs that are truncated at the sequence boundaries.

        Returns:
            An IntervalIndex of the found ORFs.
        """
        if seq.alphabet != self._DNA: raise ValueError("Sequence must use the DNA alphabet")
        starts, ends, strands = [], [], []
        # Forward Strand
        if strand >= 0:
            s, e = _find_orfs_kernel(seq.encoded, self._starts, self._stops, min_len, max_len, include_partials)
            if len(s) > 0:
                starts.append(np.array(s, dtype=np.int32))
                ends.append(np.array(e, dtype=np.int32))
                strands.append(np.ones(len(s), dtype=np.int32))
        # Reverse Strand
        if strand <= 0:
            rc_seq = seq.alphabet.reverse_complement(seq)
            s_rc, e_rc = _find_orfs_kernel(rc_seq.encoded, self._starts, self._stops, min_len, max_len, include_partials)
            if len(s_rc) > 0:
                s_rc = np.array(s_rc, dtype=np.int32)
                e_rc = np.array(e_rc, dtype=np.int32)
                L = len(seq)
                starts.append(L - e_rc)
                ends.append(L - s_rc)
                strands.append(np.full(len(s_rc), -1, dtype=np.int32))
        if not starts: return IntervalIndex()
        return IntervalIndex(np.concatenate(starts), np.concatenate(ends), np.concatenate(strands))

    def is_complete_cds(self, seq: 'Seq') -> bool:
        """Checks if the sequence starts with a start codon and ends with a stop codon."""
        if len(seq) < 3 or len(seq) % 3 != 0: return False
        return _check_cds_kernel(seq.encoded, self._starts, self._stops)

    def translate(self, seq: Union['Seq', 'SeqBatch'], frame: Literal[0, 1, 2] = 0, to_stop: bool = True) -> Union['Seq', 'SeqBatch']:
        """
        Translates a DNA sequence or SeqBatch to Amino Acids.

        Args:
            seq: The DNA sequence.
            frame: The reading frame (0, 1, or 2).
            to_stop: If True, translation terminates at the first stop codon.

        Returns:
            The translated protein sequence.

        Raises:
            TranslationError: If the sequence is too short.

        Examples:
            >>> dna = Alphabet.dna()
            >>> s = dna.seq("ATG")
            >>> gc = GeneticCode.from_code(11)
            >>> p = gc.translate(s)
            >>> str(p)
            'M'
        """
        if isinstance(seq, SeqBatch):
            return self._translate_batch(seq, frame, to_stop)

        # Use encoded sequence (0-3 integers) for faster lookup in small table
        # This avoids cache misses associated with the large 16MB lookup table
        n = len(seq)
        start = frame
        n_codons = (n - start) // 3
        if n_codons <= 0:
            if n < 3 - frame: raise TranslationError('Cannot translate sequence with less than 1 codon')
        translation = _translate_kernel(seq.encoded, self._data, self._stops, start, n_codons, to_stop, Alphabet.DTYPE)
        return self._AMINO.seq(translation)

    def _translate_batch(self, batch: 'SeqBatch', frame: int, to_stop: bool) -> 'SeqBatch':
        """
        Internal batch translation logic.

        Args:
            batch: The batch of sequences to translate.
            frame: The reading frame.
            to_stop: Whether to stop at stop codons.

        Returns:
            A new SeqBatch containing translated sequences.
        """
        if len(batch) == 0:
            return SeqBatch(np.empty(0, dtype=Alphabet.DTYPE), np.empty(0, dtype=np.int32),
                            np.empty(0, dtype=np.int32), self._AMINO)

        data, starts, lengths = batch.arrays
        # 1. Calculate lengths (Pass 1)
        new_lengths = _batch_translate_len_kernel(
            data, starts, lengths, self._data, self._stops, frame, to_stop
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

        return SeqBatch(new_data, new_starts, new_lengths, self._AMINO)


class Seq:
    """
    Sequence container optimized for high-throughput genomics.
    Holds ONLY encoded integers (uint8) to minimize memory usage.

    Note:
        Seq objects should be created via `Alphabet.seq()` or `Alphabet.random()`.

    Examples:
        >>> dna = Alphabet.dna()
        >>> s = dna.seq("ACGT")
        >>> len(s)
        4
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

        Args:
            digest_size: Size of the digest in bytes.

        Returns:
            The hex digest as bytes.
        """
        return blake2b(self._data.tobytes(), digest_size=digest_size, usedforsecurity=False).hexdigest().encode('ascii')


class SeqBatch:
    """
    A 'Struct of Arrays' container that flattens a list of Seqs
    into contiguous memory for Numba parallel processing.
    """
    __slots__ = ('_alphabet', '_data', '_starts', '_lengths', '_count')
    DTYPE = np.uint8
   
    def __init__(self, data: np.ndarray, starts: np.ndarray, lengths: np.ndarray, alphabet: Alphabet):
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
    def from_seqs(cls, seqs: Iterable[Seq], alphabet: Alphabet = None):
        """
        Optimized initialization that prevents memory spikes.

        Args:
            seqs: Iterable of Seq objects.
            alphabet: The alphabet of the sequences. If None, inferred from first sequence.
        """
        # Optimization: Fast path for existing SeqBatch (Clone)
        if isinstance(seqs, SeqBatch):
            if alphabet and alphabet != seqs.alphabet: raise ValueError("Alphabet mismatch")
            return cls(seqs._data.copy(), seqs._starts.copy(), seqs._lengths.copy(), seqs.alphabet)

        items = seqs if isinstance(seqs, (list, tuple)) else list(seqs)
        if not items:
            return cls(np.empty(0, dtype=cls.DTYPE), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), alphabet or Alphabet.dna())

        if alphabet is None: alphabet = items[0].alphabet

        count = len(items)
        lengths = np.empty(count, dtype=np.int32)

        # Pass 1: Lengths & Validation
        for i, s in enumerate(items):
            if s.alphabet != alphabet: raise ValueError("Mixed alphabets in SeqBatch")
            lengths[i] = len(s)

        # Pass 2: Fill Data (Optimized with C-level concatenation)
        if count > 0:
            if count == 1:
                # Zero-copy optimization for single sequence
                data = items[0].encoded
            else:
                data = np.concatenate([s.encoded for s in items])
        else:
            data = np.empty(0, dtype=cls.DTYPE)

        starts = np.zeros(count, dtype=np.int32)
        if count > 1:
            np.cumsum(lengths[:-1], out=starts[1:])

        return cls(data, starts, lengths, alphabet)

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
            # Optimized slicing: Gather arrays directly without creating Seq objects
            if isinstance(idx, slice):
                start, stop, step = idx.indices(self._count)
                if step == 1:
                    # Contiguous slice (Fastest)
                    new_count = max(0, stop - start)
                    if new_count == 0: return SeqBatch.from_seqs([], self._alphabet)
                    
                    s_start = self._starts[start]
                    s_end = self._starts[stop] if stop < self._count else len(self._data)
                    
                    new_data = self._data[s_start:s_end].copy()
                    new_lengths = self._lengths[start:stop].copy()
                    new_starts = self._starts[start:stop] - s_start
                    return SeqBatch(new_data, new_starts, new_lengths, self._alphabet)
                
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
        # The alphabet property is available via the first sequence, or needs to be passed in init
        # For now, assume all sequences in a batch share the same alphabet.
        # This is a safe assumption given how BatchSeq is constructed.
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

    def _gather(self, indices: np.ndarray) -> 'SeqBatch':
        """Internal method to gather sequences by index."""
        if len(indices) == 0: return SeqBatch.from_seqs([], self._alphabet)
        
        new_lengths = self._lengths[indices]
        total_len = new_lengths.sum()
        
        new_data = np.empty(total_len, dtype=self.DTYPE)
        new_starts = np.zeros(len(indices), dtype=np.int32)
        if len(indices) > 1:
            np.cumsum(new_lengths[:-1], out=new_starts[1:])
            
        _batch_gather_kernel(self._data, self._starts, self._lengths, indices, new_data, new_starts)
        return SeqBatch(new_data, new_starts, new_lengths, self._alphabet)


class SeqProperty:
    """
    Maps sequence symbols to numerical properties (e.g., Hydrophobicity).
    Calculates average scores for Seqs and SeqBatches.
    """
    __slots__ = ('_data', '_alphabet')
    _CACHE = {}
    def __init__(self, mapping: dict[Union[bytes, str], float], alphabet: Alphabet):
        """
        Args:
            mapping: Dictionary mapping characters (str) to values (float).
            alphabet: The Alphabet instance to align with.
        """
        self._alphabet = alphabet
        # Initialize with NaN so missing symbols are ignored in mean/sum
        self._data = np.full(len(alphabet), np.nan, dtype=np.float32)
        for char, val in mapping.items():
            if isinstance(char, str): char = char.encode('ascii')
            encoded = alphabet.encode(char)
            if len(encoded) > 0: self._data[encoded[0]] = val
        self._data.flags.writeable = False

    @classmethod
    def hydrophobicity(cls) -> 'SeqProperty':
        """
        Returns the Singleton instance for Kyte-Doolittle Hydrophobicity.
        Aligned to the standard Amino Alphabet.
        """
        if (cached := cls._CACHE.get('hydrophobicity')) is None:
            mapping = {
                b'I': 4.5, b'V': 4.2, b'L': 3.8, b'F': 2.8, b'C': 2.5, b'M': 1.9, b'A': 1.8, b'G': -0.4, b'T': -0.7,
                b'S': -0.8, b'W': -0.9, b'Y': -1.3, b'P': -1.6, b'H': -3.2, b'E': -3.5, b'Q': -3.5, b'D': -3.5,
                b'N': -3.5, b'K': -3.9, b'R': -4.5
            }
            cls._CACHE['hydrophobicity'] = (cached := SeqProperty(mapping, Alphabet.amino()))
        return cached
    
    @classmethod
    def GC(cls) -> 'SeqProperty':
        """
        Returns the Singleton instance for GC content
        """
        if (cached := cls._CACHE.get('GC')) is None:
            mapping = {b'A': 0, b'C': 1, b'G': 1, b'T': 0}
            cls._CACHE['GC'] = (cached := SeqProperty(mapping, Alphabet.dna()))
        return cached
    
    def _check_alphabet(self, seq: Union[Seq, SeqBatch]):
        if seq.alphabet != self._alphabet:
            raise ValueError(f"Seq alphabet {seq.alphabet} does not match Property alphabet {self._alphabet}")

    def map(self, seq: Union[Seq, SeqBatch]) -> np.ndarray:
        """
        Returns the property values across a sequence or batch.
        Result is a float32 array of the same shape as seq.encoded.
        """
        self._check_alphabet(seq)
        return self._data[seq.encoded]

    def sum(self, seq: Union[Seq, SeqBatch]) -> Union[float, np.ndarray]:
        """
        Calculates the sum of property values.
        Returns float for Seq, or ndarray[float] for SeqBatch.
        """
        self._check_alphabet(seq)
        if isinstance(seq, SeqBatch):
            data, starts, lengths = seq.arrays
            return _batch_property_sum_kernel(data, starts, lengths, self._data)
        return _property_sum_kernel(seq.encoded, self._data)

    def mean(self, seq: Union[Seq, SeqBatch]) -> Union[float, np.ndarray]:
        """
        Calculates the average property value.
        Returns float for Seq, or ndarray[float] for SeqBatch.
        """
        self._check_alphabet(seq)
        if isinstance(seq, SeqBatch):
            data, starts, lengths = seq.arrays
            return _batch_property_mean_kernel(data, starts, lengths, self._data)
        return _property_mean_kernel(seq.encoded, self._data)


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
def _translate_kernel(encoded_seq, flat_table, stops, start, n_codons, to_stop, dtype):
    """
    Translates DNA -> Amino Acid using a flat lookup table and bitwise math.
    Assumes DNA encoding is 0=T, 1=C, 2=A, 3=G (2 bits).
    """
    res = np.empty(n_codons, dtype=dtype)
    for i in range(n_codons):
        # Calculate offset of current codon
        base = start + (i * 3)
        flat_idx = _get_codon_index(encoded_seq, base)
        aa = flat_table[flat_idx]
        
        if stops[flat_idx] and to_stop:
            return res[:i]
            
        res[i] = aa
    return res


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_translate_len_kernel(data, starts, lengths, table, stops, frame, to_stop):
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
            idx = _get_codon_index(data, base)
            if stops[idx] and to_stop:
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
            idx = _get_codon_index(data, base)
            out_data[out_s + j] = table[idx]


@jit(nopython=True, cache=True, nogil=True)
def _find_codons_kernel(data, mask_table):
    n = len(data)
    if n < 3: return np.empty(0, dtype=np.int32)
    
    # Pass 1: Count
    count = 0
    for i in range(n - 2):
        idx = _get_codon_index(data, i)
        if mask_table[idx]:
            count += 1
            
    # Pass 2: Fill
    res = np.empty(count, dtype=np.int32)
    k = 0
    for i in range(n - 2):
        idx = _get_codon_index(data, i)
        if mask_table[idx]:
            res[k] = i
            k += 1
    return res


@jit(nopython=True, cache=True, nogil=True)
def _find_orfs_kernel(data, starts_table, stops_table, min_len, max_len, include_partials):
    n = len(data)
    out_starts = []
    out_ends = []

    # Iterate 3 frames
    for frame in range(3):
        open_starts = []
        for i in range(frame, n - 2, 3):
            idx = _get_codon_index(data, i)
            if stops_table[idx]:
                # Close ORFs
                end_pos = i + 3
                for s in open_starts:
                    length = end_pos - s
                    if length >= min_len:
                        if length <= max_len:
                            out_starts.append(s)
                            out_ends.append(end_pos)
                open_starts = []
            elif starts_table[idx]:
                open_starts.append(i)
            elif include_partials and i == frame:
                # 5' Partial: Start of frame is a valid start if requested
                open_starts.append(i)

        # 3' Partial: Close remaining at end of sequence
        if include_partials and len(open_starts) > 0:
            end_pos = n
            for s in open_starts:
                length = end_pos - s
                if min_len <= length <= max_len:
                    out_starts.append(s)
                    out_ends.append(end_pos)

    return out_starts, out_ends


@jit(nopython=True, cache=True, nogil=True)
def _check_cds_kernel(data, starts, stops):
    n = len(data)
    # Check start (first 3 bases)
    start_idx = _get_codon_index(data, 0)
    if not starts[start_idx]: return False
    
    # Check stop (last 3 bases)
    stop_idx = _get_codon_index(data, n - 3)
    if not stops[stop_idx]: return False
    
    return True


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_gather_kernel(data, starts, lengths, indices, out_data, out_starts):
    n = len(indices)
    for i in prange(n):
        idx = indices[i]
        src_s = starts[idx]
        l = lengths[idx]
        dst_s = out_starts[i]
        out_data[dst_s : dst_s + l] = data[src_s : src_s + l]


@jit(nopython=True, cache=True, nogil=True, inline='always')
def _get_codon_index(data, idx):
    """Helper to calculate 6-bit codon index from 3 bytes."""
    return (data[idx] << 4) | (data[idx + 1] << 2) | data[idx + 2]


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _property_mean_kernel(encoded_seq, prop_table):
    """
    Calculates average property value for a single sequence.
    Skips NaN entries in the table.
    """
    n = len(encoded_seq)
    if n == 0: return 0.0

    total = 0.0
    count = 0

    for i in prange(n):
        val = prop_table[encoded_seq[i]]
        if not np.isnan(val):
            total += val
            count += 1

    if count == 0: return 0.0
    return total / count


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_property_mean_kernel(data, starts, lengths, prop_table):
    """
    Parallel calculation of properties for a SeqBatch.
    """
    n_seqs = len(starts)
    results = np.zeros(n_seqs, dtype=np.float32)

    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]

        if l == 0:
            results[i] = 0.0
            continue

        total = 0.0
        count = 0

        for j in range(l):
            val = prop_table[data[s + j]]
            if not np.isnan(val):
                total += val
                count += 1

        if count > 0:
            results[i] = total / count
        else:
            results[i] = 0.0

    return results


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _property_sum_kernel(encoded_seq, prop_table):
    n = len(encoded_seq)
    total = 0.0
    for i in prange(n):
        val = prop_table[encoded_seq[i]]
        if not np.isnan(val):
            total += val
    return total


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_property_sum_kernel(data, starts, lengths, prop_table):
    n_seqs = len(starts)
    results = np.zeros(n_seqs, dtype=np.float32)
    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        total = 0.0
        for j in range(l):
            val = prop_table[data[s + j]]
            if not np.isnan(val):
                total += val
        results[i] = total
    return results

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
