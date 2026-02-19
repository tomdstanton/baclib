"""
Module for representing ASCII biological alphabets
"""
from typing import Union, Iterable, Final, Literal, ClassVar

import numpy as np

from baclib.core.interval import IntervalBatch
from baclib.containers.seq import Seq, SeqBatch, CompressedSeq, CompressedSeqBatch
from baclib.lib.resources import jit, RESOURCES

if RESOURCES.has_module('numba'):
    from numba import prange
else:
    prange = range


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class AlphabetError(Exception):
    """Raised when an alphabet is invalid or an operation is incompatible with the alphabet."""


class TranslationError(AlphabetError):
    """Raised when nucleotide-to-amino-acid translation fails (e.g. invalid codon)."""



# Classes --------------------------------------------------------------------------------------------------------------
class Alphabet:
    """
    A class to represent an alphabet of ASCII symbols.
    """
    __slots__ = ('_data', '_lookup_table', '_complement', '_trans_table', '_delete_bytes', '_decode_table')
    DTYPE: Final = np.uint8
    INVALID: Final = np.iinfo(DTYPE).max
    MAX_LEN: Final = INVALID + 1
    ENCODING: Final = 'ascii'

    DNA: ClassVar['Alphabet']
    RNA: ClassVar['Alphabet']
    AMINO: ClassVar['Alphabet']
    MURPHY_10: ClassVar['Alphabet']

    def __init__(self, symbols: bytes, complement: bytes = None, aliases: dict[bytes, bytes] = None):
        """
        Initializes an Alphabet.

        Args:
            symbols: The symbols in the alphabet as bytes.
            complement: Optional complement symbols as bytes. Must be same length as symbols.
            aliases: Optional mapping of invalid characters to valid ones (e.g. {b'N': b'A'}).

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

        # Apply Aliases (Map invalid chars to valid indices)
        if aliases:
            for src, dst in aliases.items():
                if len(src) != 1 or len(dst) != 1: raise AlphabetError("Aliases must be single bytes")

                # Resolve destination index
                dst_idx = self._lookup_table[ord(dst)]
                if dst_idx == self.INVALID: raise AlphabetError(f"Alias target {dst} not in alphabet")

                # Map source (both cases)
                self._lookup_table[ord(src)] = dst_idx
                self._lookup_table[ord(src.lower())] = dst_idx

        # Build Translation Tables
        self._trans_table = self._lookup_table.tobytes()
        self._delete_bytes = np.where(self._lookup_table == self.INVALID)[0].astype(self.DTYPE).tobytes()

        # Build Decode Table (for fast tobytes)
        decode_map = np.zeros(256, dtype=self.DTYPE)
        decode_map[:len(self._data)] = self._data
        self._decode_table = decode_map.tobytes()

        self._complement = None
        if complement is not None:
            if len(complement) != len(symbols):
                raise AlphabetError("Complement must be the same length as symbols")
            comp_indices = self._lookup_table[np.frombuffer(complement, dtype=self.DTYPE)]
            if np.any(comp_indices == self.INVALID):
                raise AlphabetError("Complement contains symbols not in alphabet")
            self._complement = comp_indices

    def __len__(self):
        return len(self._data)

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
        """Returns the number of bits required to represent a symbol in this alphabet.

        Returns:
            The number of bits (integer).
        """
        return (len(self._data) - 1).bit_length()

    @property
    def complement(self):
        """Returns the complement lookup table if available."""
        return self._complement

    @classmethod
    def detect(cls, text: bytes) -> 'Alphabet':
        """
        Detects the most likely alphabet for the given byte string.

        Candidates are checked in the following priority order:
        1. DNA
        2. RNA
        3. AMINO
        4. MURPHY_10

        Scoring is based on:
        1. Canonical Count: Number of characters strictly in the alphabet definition (case-insensitive).
        2. Valid Count: Number of characters valid in the alphabet (including aliases).

        Args:
            text: The input byte string (or ASCII string).

        Returns:
            The most likely Alphabet singleton.
        """
        if isinstance(text, str):
            text = text.encode(cls.ENCODING)

        data = np.frombuffer(text, dtype=cls.DTYPE)

        # Optimization: Scan data once to get character counts
        # This reduces complexity from O(K*N) to O(N + K*C) where K=num_alphabets, C=256
        counts = np.bincount(data, minlength=cls.MAX_LEN)

        # Candidate alphabets in priority order
        candidates = [cls.DNA, cls.RNA, cls.AMINO, cls.MURPHY_10]

        best_alpha = candidates[0]
        best_score = (-1, -1)

        for alpha in candidates:
            # 1. Canonical Score
            # Create mask for canonical symbols (both cases)
            is_canonical = np.zeros(cls.MAX_LEN, dtype=bool)
            is_canonical[alpha._data] = True

            # Handle lowercase
            lower_indices = np.frombuffer(alpha._data.tobytes().lower(), dtype=cls.DTYPE)
            is_canonical[lower_indices] = True

            # Sum counts of canonical characters
            n_canonical = counts[is_canonical].sum()

            # 2. Valid Score
            # Use lookup table - anything not INVALID is valid
            is_valid = (alpha._lookup_table != cls.INVALID)
            n_valid = counts[is_valid].sum()

            score = (n_canonical, n_valid)

            if score > best_score:
                best_score = score
                best_alpha = alpha

        return best_alpha

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
        """Decodes an array of indices back to bytes.

        Args:
            encoded: The numpy array of indices (uint8).

        Returns:
            The decoded bytes string.
        """
        # Ensure uint8 for byte-wise translation
        if encoded.dtype != self.DTYPE:
            encoded = encoded.astype(self.DTYPE, copy=False)
        return encoded.tobytes().translate(self._decode_table)

    def entropy(self, seq: Union['Seq', 'SeqBatch']) -> Union[float, np.ndarray]:
        """
        Calculates the Shannon entropy of a sequence or batch.
        H = -sum(p_i * log2(p_i))

        Args:
            seq: A Seq or SeqBatch object.

        Returns:
            The entropy in bits (float for Seq, ndarray for SeqBatch).
        """
        if seq.alphabet != self:
            raise ValueError(f"Seq alphabet {seq.alphabet} does not match Alphabet {self}")

        if isinstance(seq, SeqBatch):
            data, starts, lengths = seq.arrays
            return _batch_entropy_kernel(data, starts, lengths, len(self))

        # Single Seq
        if len(seq) == 0: return 0.0
        counts = np.bincount(seq.encoded, minlength=len(self))
        # Filter zero counts to avoid log(0)
        counts = counts[counts > 0]
        probs = counts / len(seq)
        return -np.sum(probs * np.log2(probs))

    def compress(self, seq: Union['Seq', 'SeqBatch']) -> Union['CompressedSeq', 'CompressedSeqBatch']:
        """
        Compresses a Seq or SeqBatch using bit-packing.
        Only supports alphabets with <= 4 bits per symbol (e.g. DNA, RNA).

        Args:
            seq: The sequence or batch to compress.

        Returns:
            A CompressedSeq or CompressedSeqBatch.
        """
        bits = max(1, self.bits_per_symbol)
        if bits > 4:
            raise ValueError(f"Compression not supported for alphabets with > 4 bits per symbol (current: {bits})")

        if isinstance(seq, Seq):
            packed = _pack_seq_kernel(seq.encoded, len(seq), bits)
            return self.new_compressed_seq(packed, len(seq), bits)

        if isinstance(seq, SeqBatch):
            data, starts, lengths = seq.arrays
            per_byte = 8 // bits
            byte_lengths = (lengths + per_byte - 1) // per_byte
            total_bytes = byte_lengths.sum()

            packed_data = np.zeros(total_bytes, dtype=np.uint8)
            packed_starts = np.zeros(len(lengths), dtype=np.int32)
            if len(lengths) > 1:
                np.cumsum(byte_lengths[:-1], out=packed_starts[1:])

            _pack_batch_fill_kernel(data, starts, lengths, packed_data, packed_starts, bits)
            return self.new_compressed_batch(packed_data, packed_starts, lengths, bits)

        raise TypeError(f"Cannot compress {type(seq)}")

    def decompress(self, compressed: 'CompressedSeq') -> 'Seq':
        """Decompresses a CompressedSeq back to a standard Seq."""
        decoded = _unpack_seq_kernel(compressed._data, compressed._length, compressed._bits)
        return self.seq_from(decoded)

    def decompress_batch(self, batch: 'CompressedSeqBatch') -> 'SeqBatch':
        """Decompresses a CompressedSeqBatch back to a standard SeqBatch."""
        total_len = batch._lengths.sum()
        out_data = np.empty(total_len, dtype=np.uint8)
        out_starts = np.zeros(len(batch), dtype=np.int32)
        if len(batch) > 1:
            np.cumsum(batch._lengths[:-1], out=out_starts[1:])
            
        _unpack_batch_kernel(batch._data, batch._starts, batch._lengths, out_data, out_starts, batch._bits)
        return self.new_batch(out_data, out_starts, batch._lengths)

    def new_seq(self, data: np.ndarray) -> 'Seq':
        """
        Factory method. The ONLY valid way to create a Seq.
        """
        return Seq(data, self, _validation_token=self)

    def seq_from(self, data: Union['Seq', str, bytes, np.ndarray]) -> 'Seq':
        """Creates a Seq object from various input types, ensuring correct encoding.

        Args:
            data: The input data. Can be a ``Seq``, string, bytes, or numpy array.

        Returns:
            A new ``Seq`` object with this alphabet.

        Raises:
            AlphabetError: If the input data contains symbols not in the alphabet.
        """
        # 1. Handle Pre-encoded (Optimization for internal use)
        if isinstance(data, Seq):
            if data.alphabet != self: raise AlphabetError(f'Sequence has a different alphabet "{data.alphabet}"')
            return data
        # 2. Handle Numpy Array (Assume it is uint8 text or encoded?)
        # Convention: If it's uint8 array passed as 'core', treat as encoded indices
        if isinstance(data, np.ndarray): return self.new_seq(data)
        # 3. Handle Text/Bytes
        if isinstance(data, str): data = data.encode(self.ENCODING)
        # Use the vectorized encoder
        return self.new_seq(self.encode(data))

    def empty_seq(self) -> 'Seq':
        """Returns an empty sequence with this alphabet.

        Returns:
            An empty ``Seq``.
        """
        return self.new_seq(np.empty(0, dtype=self.DTYPE))

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
        return self.seq_from(indices.astype(self.DTYPE))

    def new_batch(self, data: np.ndarray, starts: np.ndarray, lengths: np.ndarray):
        """
        Factory method. The ONLY valid way to create a SeqBatch.
        """
        return SeqBatch(data, starts, lengths, self, _validation_token=self)

    def batch_from(self, data: Iterable['Seq'], deduplicate: bool = False) -> 'SeqBatch':
        """Creates a SeqBatch from an iterable of sequences.

        Args:
            data: An iterable of ``Seq`` objects (must have this alphabet).
            deduplicate: If ``True``, deduplicates identical sequences to save memory.

        Returns:
            A new ``SeqBatch``.

        Raises:
            AlphabetError: If any sequence has a different alphabet.
        """
        # Optimization: Fast path for existing SeqBatch (Clone)
        if isinstance(data, SeqBatch):
            if data.alphabet != self: raise AlphabetError(
                "Can only create a batch from batches with the same alphabet.")
            return self.new_batch(data.encoded.copy(), data.starts.copy(), data.lengths.copy())

        items = data if isinstance(data, (list, tuple)) else list(data)
        if not items: return self.empty_batch()

        # Pass 1: Lengths & Validation
        count = len(items)
        lengths = np.empty(count, dtype=np.int32)
        for i, s in enumerate(items):
            lengths[i] = len(s)
            if s.alphabet != self: raise AlphabetError("Can only create a batch from sequences with the same alphabet.")

        # Pass 2: Fill Data (Optimized with C-level concatenation)
        starts = np.zeros(count, dtype=np.int32)
        
        if deduplicate and count > 1:
            # Deduplication Logic
            unique_map = {} # Seq -> (offset, length)
            unique_parts = []
            current_offset = 0
            
            for i, s in enumerate(items):
                if s in unique_map:
                    offset, _ = unique_map[s]
                    starts[i] = offset
                else:
                    starts[i] = current_offset
                    unique_map[s] = (current_offset, lengths[i])
                    unique_parts.append(s.encoded)
                    current_offset += lengths[i]
            
            data = np.concatenate(unique_parts) if unique_parts else np.empty(0, dtype=self.DTYPE)
        else:
            # Standard Contiguous Logic
            if count > 0:
                data = items[0].encoded if count == 1 else np.concatenate([s.encoded for s in items])
            else:
                data = np.empty(0, dtype=self.DTYPE)
            
            if count > 1: np.cumsum(lengths[:-1], out=starts[1:])
            
        return self.new_batch(data, starts, lengths)

    def empty_batch(self) -> 'SeqBatch':
        """Returns an empty sequence batch with this alphabet.

        Returns:
            An empty ``SeqBatch``.
        """
        return self.new_batch(
            np.empty(0, dtype=self.DTYPE), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))

    def zeros_batch(self, n: int) -> 'SeqBatch':
        """Returns a SeqBatch of n zero-length sequences."""
        return self.new_batch(
            np.empty(0, dtype=self.DTYPE), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32))

    def new_compressed_seq(self, data: np.ndarray, length: int, bits: int) -> 'CompressedSeq':
        """
        Factory method. The ONLY valid way to create a CompressedSeq.
        """
        return CompressedSeq(data, length, self, bits, _validation_token=self)

    def new_compressed_batch(self, data: np.ndarray, starts: np.ndarray, lengths: np.ndarray, bits: int) -> 'CompressedSeqBatch':
        """
        Factory method. The ONLY valid way to create a CompressedSeqBatch.
        """
        return CompressedSeqBatch(data, starts, lengths, self, bits, _validation_token=self)

    def empty_compressed(self, bits: int = 2) -> 'CompressedSeqBatch':
        """Returns an empty CompressedSeqBatch."""
        return self.new_compressed_batch(
            np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), bits
        )

    def zeros_compressed(self, n: int, bits: int = 2) -> 'CompressedSeqBatch':
        """Returns a CompressedSeqBatch of n zero-length sequences."""
        return self.new_compressed_batch(
            np.empty(0, dtype=np.uint8),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            bits
        )


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

        if lengths_arr.size == 0: return self.empty_batch()

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

        return self.new_batch(indices.astype(self.DTYPE, copy=False), starts, lengths_arr)

    def reverse_complement(self, seq: 'Seq') -> 'Seq':
        """Returns the reverse complement of the sequence.

        Args:
            seq: The input sequence.

        Returns:
            A new ``Seq`` object (reverse complemented). Returns the input sequence unchanged if no complement is defined.
        """
        if self._complement is None: return seq
        # Use .encoded for direct numpy access (much faster than iterating reversed(seq))
        return self.seq_from(self._complement[seq.encoded[::-1]])



# Initialize Standard Alphabets
Alphabet.DNA = Alphabet(b'TCAG', b'AGTC', aliases={b'N': b'A', b'U': b'T'})
Alphabet.RNA = Alphabet(b'UCAG', b'AGUC', aliases={b'N': b'A', b'T': b'U'})
Alphabet.AMINO = Alphabet(b'ACDEFGHIKLMNPQRSTVWY',
                          aliases={b'X': b'A', b'B': b'D', b'Z': b'E', b'J': b'L', b'U': b'C', b'O': b'K'})
Alphabet.MURPHY_10 = Alphabet(b'LCAGSPFEKH')


class AlphabetProperty:
    """
    Maps sequence symbols to numerical properties (e.g., Hydrophobicity).
    Calculates average scores for Seqs and SeqBatches.
    """
    __slots__ = ('_data', '_alphabet', '_default')
    HYDROPHOBICITY: ClassVar['AlphabetProperty']
    GC: ClassVar['AlphabetProperty']

    def __init__(self, alphabet: 'Alphabet', mapping: dict[Union[bytes, str], float], default: float = np.nan):
        """
        Args:
            mapping: Dictionary mapping characters (str) to values (float).
            alphabet: The Alphabet instance to align with.
            default: Value to assign to symbols not in the mapping (default: NaN).
        """
        self._alphabet = alphabet
        self._default = default
        self._data = np.full(len(alphabet), default, dtype=np.float32)
        for char, val in mapping.items():
            if isinstance(char, str): char = char.encode(Alphabet.ENCODING)
            encoded = alphabet.encode(char)
            if len(encoded) > 0: self._data[encoded[0]] = val
        self._data.flags.writeable = False

    def encode(self, seq: Union[Seq, SeqBatch]) -> np.ndarray:
        """Encodes a sequence or batch into property values.

        Args:
            seq: The sequence or batch to encode.

        Returns:
            A numpy array of float values.

        Raises:
            ValueError: If the sequence alphabet does not match.
        """
        if seq.alphabet != self._alphabet:
            raise ValueError(f"Seq alphabet {seq.alphabet} does not match alphabet {self._alphabet}")
        return self._data[seq.encoded]

    def score(self, seq: Union[Seq, SeqBatch], aggregator: str = 'mean') -> Union[float, np.ndarray]:
        """
        Calculates the aggregate score (mean/sum) for the sequence(s).
        Ignores NaN values (unmapped symbols).

        Args:
            seq: Seq or SeqBatch.
            aggregator: 'mean' or 'sum'.

        Returns:
            Float score (for Seq) or Array of scores (for SeqBatch).
        """
        mapped = self.encode(seq)
        if isinstance(seq, SeqBatch):
            return _batch_score_kernel(mapped, seq.starts, seq.lengths, aggregator == 'mean')
        else:
            if aggregator == 'mean': return np.nanmean(mapped)
            if aggregator == 'sum': return np.nansum(mapped)
            raise ValueError(f"Unknown aggregator: {aggregator}")

AlphabetProperty.HYDROPHOBICITY = AlphabetProperty(Alphabet.AMINO, {
    b'I': 4.5, b'V': 4.2, b'L': 3.8, b'F': 2.8, b'C': 2.5, b'M': 1.9, b'A': 1.8, b'G': -0.4, b'T': -0.7,
    b'S': -0.8, b'W': -0.9, b'Y': -1.3, b'P': -1.6, b'H': -3.2, b'E': -3.5, b'Q': -3.5, b'D': -3.5,
    b'N': -3.5, b'K': -3.9, b'R': -4.5})
AlphabetProperty.GC = AlphabetProperty(Alphabet.DNA, {b'A': 0, b'C': 1, b'G': 1, b'T': 0})


class AlphabetConverter:
    """
    Converts sequences from one Alphabet to another.
    """
    __slots__ = ('_source', '_target', '_table', '_rev_table')
    TRANSCRIBE: ClassVar['AlphabetConverter']
    TO_MURPHY_10: ClassVar['AlphabetConverter']

    def __init__(self, source: Alphabet, target: Alphabet, mapping: dict[bytes, bytes] = None, default: bytes = None):
        """Initializes a converter between two alphabets.

        Args:
            source: Source alphabet.
            target: Target alphabet.
            mapping: Optional dictionary for custom symbol mapping.
            default: Optional default value for unmapped symbols.
        """
        self._source = source
        self._target = target
        self._table = self._build_table(source, target, mapping, default)

        # Build Reverse Table (target -> source)
        rev_mapping = {}
        if mapping:
            for k, v in mapping.items():
                # Heuristic to invert mapping:
                # 1. If v is not in source, we must map it back (e.g. U->T if U not in DNA).
                # 2. If v is in mapping (as a key), it means v was remapped in forward (e.g. Swap A->C, C->A),
                #    so we should map it back (C->A).
                # 3. Otherwise (v is in source and not remapped), we assume v->v identity is preferred
                #    (e.g. V->L, L->L identity preferred over L->V).
                if v not in source or v in mapping: rev_mapping[v] = k

        self._rev_table = self._build_table(target, source, rev_mapping, None)

    @staticmethod
    def _build_table(source, target, mapping, default):
        final_mapping = {}
        for i in range(len(source)):
            sym = source.decode(np.array([i], dtype=source.DTYPE))
            if sym in target:
                final_mapping[sym] = sym

        if mapping:
            final_mapping.update(mapping)

        table = np.full(len(source), target.INVALID, dtype=target.DTYPE)

        if default is not None:
            def_encoded = target.encode(default)
            if len(def_encoded) > 0:
                table[:] = def_encoded[0]

        for src, dst in final_mapping.items():
            s_idx = source.encode(src)
            d_idx = target.encode(dst)
            if len(s_idx) > 0 and len(d_idx) > 0:
                table[s_idx[0]] = d_idx[0]

        table.flags.writeable = False
        return table

    def convert(self, seq: Union[Seq, SeqBatch]) -> Union[Seq, SeqBatch]:
        """Converts a sequence or batch to the target alphabet.

        Args:
            seq: Input sequence or batch.

        Returns:
            The converted sequence or batch.

        Raises:
            ValueError: If the input alphabet does not match source or target (for reverse conversion).
            ValueError: If conversion results in invalid symbols.
        """
        if seq.alphabet == self._source:
            table = self._table
            target_alpha = self._target
        elif seq.alphabet == self._target:
            table = self._rev_table
            target_alpha = self._source
        else:
            raise ValueError(f"Input sequence alphabet {seq.alphabet} does not match converter source {self._source} or target {self._target}")

        new_data = table[seq.encoded]

        if np.any(new_data == target_alpha.INVALID):
             raise ValueError("Conversion resulted in invalid symbols (missing mapping for some characters)")

        if isinstance(seq, SeqBatch):
            return target_alpha.new_batch(new_data, seq.starts.copy(), seq.lengths.copy())
        return target_alpha.new_seq(new_data)

AlphabetConverter.TRANSCRIBE = AlphabetConverter(Alphabet.DNA, Alphabet.RNA, mapping={b'T': b'U'})
AlphabetConverter.TO_MURPHY_10 = AlphabetConverter(Alphabet.AMINO, Alphabet.MURPHY_10, mapping={
    b'V': b'L', b'I': b'L', b'M': b'L',
    b'T': b'S',
    b'Y': b'F', b'W': b'F',
    b'D': b'E', b'N': b'E', b'Q': b'E',
    b'R': b'K'
})


class GeneticCode:
    """
    Represents a genetic code table for translation.
    """
    __slots__ = ('_data', '_starts', '_stops')
    _DNA = Alphabet.DNA
    _AMINO = Alphabet.AMINO
    BACTERIA: ClassVar['GeneticCode']

    def __init__(self, table: bytes, starts: Iterable[bytes] = ()):
        """Initializes a genetic code.

        Args:
            table: 64-byte ASCII string representing the translation table.
            starts: Iterable of start codons (e.g. ``[b'ATG', b'GTG']``).
        """
        # Optimization: Pre-encode the table to indices using lookup table directly
        self._data = self._AMINO._lookup_table[np.frombuffer(table, dtype=Alphabet.DTYPE)]
        # Populate stops and starts
        # We derive stops from the table string
        self._stops = np.frombuffer(table, dtype=Alphabet.DTYPE) == ord('*')
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

    @property
    def starts(self) -> np.ndarray:
        """Boolean array indicating valid start codons (size 64)."""
        return self._starts

    @property
    def stops(self) -> np.ndarray:
        """Boolean array indicating stop codons (size 64)."""
        return self._stops

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

    def find_orfs(self, seq: 'Seq', strand: Literal[0, 1, -1] = 0, min_len: int = 30, max_len: int = 3000,
                  include_partials: bool = False) -> IntervalBatch:
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
            An IntervalBatch of the found ORFs.
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
            s_rc, e_rc = _find_orfs_kernel(rc_seq.encoded, self._starts, self._stops, min_len, max_len,
                                           include_partials)
            if len(s_rc) > 0:
                s_rc = np.array(s_rc, dtype=np.int32)
                e_rc = np.array(e_rc, dtype=np.int32)
                L = len(seq)
                starts.append(L - e_rc)
                ends.append(L - s_rc)
                strands.append(np.full(len(s_rc), -1, dtype=np.int32))
        if not starts: return IntervalBatch()
        return IntervalBatch(np.concatenate(starts), np.concatenate(ends), np.concatenate(strands))

    def is_complete_cds(self, seq: 'Seq') -> bool:
        """Checks if the sequence starts with a start codon and ends with a stop codon."""
        if len(seq) < 3 or len(seq) % 3 != 0: return False
        return _check_cds_kernel(seq.encoded, self._starts, self._stops)

    def translate(self, seq: Union['Seq', 'SeqBatch'], frame: Literal[0, 1, 2] = 0, to_stop: bool = True) -> Union[
        'Seq', 'SeqBatch']:
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
        return self._AMINO.seq_from(translation)

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
        if len(batch) == 0: return self._AMINO.empty_batch()

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

        return self._AMINO.new_batch(new_data, new_starts, new_lengths)


GeneticCode.BACTERIA = GeneticCode(b'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
                                   (b'ATG', b'GTG', b'TTG', b'ATT', b'ATC', b'ATA'))


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True, inline='always')
def _get_codon_index(data, idx):
    """Helper to calculate 6-bit codon index from 3 bytes."""
    return (data[idx] << 4) | (data[idx + 1] << 2) | data[idx + 2]


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
def _batch_entropy_kernel(data, starts, lengths, alphabet_size):
    n_seqs = len(starts)
    results = np.zeros(n_seqs, dtype=np.float32)

    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        if l == 0:
            results[i] = 0.0
            continue

        # Allocate counts on stack/L1 (alphabet_size is small, e.g. 4 or 20)
        counts = np.zeros(alphabet_size, dtype=np.int32)
        for j in range(l):
            val = data[s + j]
            counts[val] += 1

        entropy = 0.0
        for k in range(alphabet_size):
            c = counts[k]
            if c > 0:
                p = c / l
                entropy -= p * np.log2(p)

        results[i] = entropy

    return results


@jit(nopython=True, cache=True, nogil=True)
def _pack_seq_kernel(data, length, bits):
    per_byte = 8 // bits
    n_bytes = (length + per_byte - 1) // per_byte
    out = np.zeros(n_bytes, dtype=np.uint8)

    for i in range(length):
        byte_idx = i // per_byte
        bit_offset = (per_byte - 1 - (i % per_byte)) * bits
        val = data[i]
        out[byte_idx] |= (val << bit_offset)
    return out


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _pack_batch_fill_kernel(data, starts, lengths, out_data, out_starts, bits):
    n_seqs = len(starts)
    per_byte = 8 // bits
    for i in prange(n_seqs):
        s = starts[i]
        l = lengths[i]
        out_s = out_starts[i]
        for j in range(l):
            byte_idx = j // per_byte
            bit_offset = (per_byte - 1 - (j % per_byte)) * bits
            out_data[out_s + byte_idx] |= (data[s + j] << bit_offset)


@jit(nopython=True, cache=True, nogil=True, parallel=True)
def _batch_score_kernel(data, starts, lengths, is_mean):
    n = len(starts)
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        s = starts[i]
        l = lengths[i]
        if l == 0:
            out[i] = np.nan
            continue
        
        total = 0.0
        count = 0
        for j in range(l):
            val = data[s + j]
            if not np.isnan(val):
                total += val
                count += 1
        
        if count == 0:
            out[i] = np.nan
        elif is_mean:
            out[i] = total / count
        else:
            out[i] = total
    return out


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
