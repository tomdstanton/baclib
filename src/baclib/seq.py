"""
Module containing data structures for representing bacterial sequence components, similar to Biopython.
"""
from copy import deepcopy
from operator import attrgetter
from bisect import bisect_left, bisect_right
from typing import Literal, Iterator, Union, Any, Generator, Callable, Iterable
from random import Random
from hashlib import new
from warnings import warn
from collections.abc import MutableSequence
from re import compile as regex

import numpy as np

from . import RESOURCES, BaclibWarning, jit

# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class AlphabetError(Exception): pass
class SeqError(Exception): pass
class IntervalError(Exception): pass
class TranslationError(AlphabetError): pass
class TranslationWarning(BaclibWarning): pass


# Classes --------------------------------------------------------------------------------------------------------------
class CigarParser:
    """Parses CIGAR strings into operations and lengths."""
    _OPS_REGEX = regex(r'(?P<n>[0-9]+)(?P<operation>[MIDNSHP=X])')
    _CONSUMES_QUERY = frozenset({"M", "I", "S", "=", "X"})
    _CONSUMES_TARGET = frozenset({"M", "D", "N", "=", "X"})
    _CONSUMES_BOTH = frozenset({"M", "=", "X"})

    def parse(self, cigar: str) -> Generator[tuple[str, int, int, int, int], None, None]:
        """
        Parses a CIGAR string.

        Args:
            cigar: The CIGAR string (e.g., "10M2D5M").

        Yields:
            Tuples containing (operation, length, query_consumed, target_consumed, alignment_length).
        """
        q_len, t_len, aln_len = 0, 0, 0
        for match in self._OPS_REGEX.finditer(cigar):
            op, n = match['operation'], int(match['n'])
            if op in self._CONSUMES_QUERY: q_len += n
            if op in self._CONSUMES_TARGET: t_len += n
            if op in self._CONSUMES_QUERY or op in self._CONSUMES_TARGET: aln_len += n
            yield op, n, q_len, t_len, aln_len


class GeneticCode:
    _TABLES = {
        11: ('FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
             {'ATG', 'GTG', 'TTG', 'ATT', 'ATC', 'CTG'}, {'TAA', 'TAG', 'TGA'})
    }
    _BASES = 'TCAG'
    __slots__ = ('start_codons', 'stop_codons', 'codons')
    def __init__(self, code: int):
        if (table := self._TABLES.get(code)) is None: raise NotImplementedError('Genetic code not implemented')
        self.start_codons = {'ATG', 'GTG', 'TTG', 'ATT', 'ATC', 'CTG'}
        self.stop_codons = {'TAA', 'TAG', 'TGA'}
        self.codons = dict(zip((a + b + c for a in self._BASES for b in self._BASES for c in self._BASES), table[0]))


class Alphabet:
    """
    A class to represent an alphabet of symbols.
    """
    _EXTENSIONS = {'gbk': 'dna', 'genbank': 'dna', 'fasta': 'dna', 'fna': 'dna', 'ffn': 'dna', 'ffa': 'dna',
                   'faa': 'amino'}  # Map fasta file extensions to alphabets
    _CLEANER = str.maketrans('', '', '0123456789 \t\n')  # Simple, fast, sequence cleaner
    _CACHE = {}  # Store singletons here
    __slots__ = ('symbols', 'set', 'hash_algorithm', 'complement', '_processing_funcs')
    def __init__(self, symbols: str, complement: str = None, hash_algorithm: str = 'sha1'):
        self.symbols = symbols
        self.set: frozenset[str] = frozenset(symbols)
        if len(self.set) < len(self.symbols):
            raise AlphabetError(f'Alphabet symbols "{symbols}" are not unique')
        self.hash_algorithm = hash_algorithm
        self.complement = str.maketrans(complement, self.symbols) if complement else None
        self._processing_funcs: list[Callable] = [lambda i: i.translate(self._CLEANER).upper()]

    def __len__(self): return len(self.symbols)
    def __repr__(self): return self.symbols
    def __iter__(self): return iter(self.symbols)
    def __getitem__(self, item): return self.symbols[item]
    def __hash__(self): return hash((self.set, type(self)))
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.symbols == other.symbols

    @classmethod
    def dna(cls, *args, **kwargs):
        if (cached := cls._CACHE.get('dna')) is None:
            cls._CACHE['dna'] = (cached := NucleotideAlphabet(*args, **kwargs))
        return cached

    @classmethod
    def amino(cls, *args, **kwargs):
        if (cached := cls._CACHE.get('amino')) is None:
            cls._CACHE['amino'] = (cached := AminoAcidAlphabet(*args, **kwargs))
        return cached

    @classmethod
    def from_extension(cls, extension: str, *args, **kwargs):
        if alphabet := cls._EXTENSIONS.get(extension, None): return getattr(cls, alphabet)(*args, **kwargs)
        raise AlphabetError(f'Unknown extension "{extension}"')

    def hexdigest(self, *sequences: str) -> str:
        """
        Hashes sequences using a hash algorithm, used for generating locus tags
        :param sequences: Seq objects to hash
        :return: Hash string
        """
        return new(self.hash_algorithm, ''.join(sequences).encode(), usedforsecurity=False).hexdigest()

    def process_string(self, seq: str) -> str:
        for func in self._processing_funcs: seq = func(seq)
        return seq

    def seq(self, seq: str, process_string: bool = False, check_alphabet: bool = False) -> '_Seq':
        """Processes a string and returns a Seq object attached to an instance of this alphabet"""
        if process_string: seq = self.process_string(seq)
        if check_alphabet:
            if (sym := set(seq)) >= self.set:
                raise AlphabetError(f"Sequence contains more symbols ({sym}) than the alphabet ({self})")
        return _Seq(seq, self)

    def generate_seq(self, rng: Random = None, length: int = None, min_len: int = 5, max_len: int = 5000, weights=None) -> str:
        if rng is None: rng = RESOURCES.rng
        return ''.join(rng.choices(self.symbols, k=length or rng.randint(min_len, max_len), weights=weights))

    def reverse_complement(self, seq: str) -> str:
        """Reverse complements the sequence if the alphabet has a complement"""
        return seq.translate(self.complement)[::-1] if self.complement else seq


class AminoAcidAlphabet(Alphabet):
    """
    Child-class of Alphabet to represent the IUPAC unambiguous Amino Acid symbols ACDEFGHIKLMNPQRSTVWY,
    and extra methods unique to Amino Acids.
    """

    def __init__(self, hash_algorithm: str = 'md5', stop_symbol: str = '*'):
        super().__init__('ACDEFGHIKLMNPQRSTVWY', None, hash_algorithm)
        self.stop_symbol = stop_symbol
        self._processing_funcs.append(lambda i: i.removesuffix('*'))


class NucleotideAlphabet(Alphabet):
    """
    Child-class of Alphabet to represent the IUPAC unambiguous DNA symbols TCAG, and extra methods unique to DNA.
    """
    _CACHE = {}  # Cache for genetic code
    def __init__(self, hash_algorithm: str = 'sha1', genetic_code: int = 11):
        super().__init__('TCAG', 'AGTC', hash_algorithm)
        if (cached := self._CACHE.get(genetic_code)) is None:
            self._CACHE[genetic_code] = (cached := GeneticCode(genetic_code))
        self.genetic_code: GeneticCode = cached
        self._processing_funcs.append(lambda i: i.replace('N', ''))

    def translate(self, seq: '_Seq', to: AminoAcidAlphabet = None, to_stop: bool = True, frame: Literal[0, 1, 2] = 0,
                  gap_character: str = None) -> '_Seq':
        if not isinstance(seq, _Seq): raise TranslationError(f"Seq must be of type {_Seq}, not {type(seq)}")
        if seq.alphabet != self: raise AlphabetError(f"Seq alphabet is {seq.alphabet}, not {self}")
        if len(seq) < (3 - frame): raise TranslationError(f'Cannot translate sequence of length {len(seq)}')
        if to is None: to = AminoAcidAlphabet()
        protein = []
        seq = str(seq)[frame:]
        for i in range(0, len(seq), 3):  # Iterate over seq codons (chunks of 3)
            if len(codon := seq[i:i + 3]) < 3: break  # We can't translate chunks less than 3 (not codons) so break here
            if gap_character and gap_character in codon: protein.append(gap_character)
            else:
                residue = self.genetic_code.codons.get(codon, to.stop_symbol)
                if to_stop and residue == to.stop_symbol: break  # Break if to_stop == True
                protein.append(residue)
        if (protein := ''.join(protein)) == '':
            warn(f'Translated to an emtpy sequence: {seq}', TranslationWarning)
        return to.seq(protein)

    def is_CDS(self, seq: str) -> bool:
        return seq[:3] in self.genetic_code.start_codons and seq[-3:] in self.genetic_code.stop_codons


class Interval:
    __slots__ = ('start', 'end', 'strand')
    _AMBIGUOUS_SYMBOLS = {0, '0', '.', '?', None}
    _SENSE_SYMBOLS = {'+', '1', 1}
    _ANTISENSE_SYMBOLS = {'-', '-1', -1}
    def __init__(self, start: int, end: int, strand: Any = None):
        self.start: int = start
        self.end: int = end
        self.strand: Literal[1, -1, 0] = self._parse_strand(strand)

    @classmethod
    def from_slice(cls, s: slice) -> 'Interval':
        if s.start >= s.stop: return cls(s.stop, s.start, -1)
        return cls(s.start, s.stop, 1)

    def __hash__(self): return hash((self.start, self.end, self.strand))
    def __repr__(self): return f"{self.start}:{self.end}({self.decode_sense()})"
    def __len__(self): return self.end - self.start
    def __iter__(self): return iter((self.start, self.end, self.strand))

    def __contains__(self, item: Union['Interval', 'Feature', int, float, slice]):
        item = _coerce_interval(item)
        return self.start <= item.start and self.end >= item.end

    def __add__(self, other: Union['Interval', 'Feature']) -> 'Interval':
        other = _coerce_interval(other)
        return Interval(min(self.start, other.start), max(self.end, other.end), self.strand)

    def __radd__(self, other: Union['Interval', 'Feature']) -> 'Interval':
        other = _coerce_interval(other)
        return other.__add__(self)

    def __iadd__(self, other: Union['Interval', 'Feature']):
        other = _coerce_interval(other)
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        return self

    def __delitem__(self, item: Union[slice, int, 'Interval', 'Feature']):
        item = _coerce_interval(item)
        self.start = max(self.start, item.start)
        self.end = min(self.end, item.end)

    def _parse_strand(self, symbol: Any) -> Literal[1, -1, 0]:
        return 0 if symbol in self._AMBIGUOUS_SYMBOLS else 1 if symbol in self._SENSE_SYMBOLS else -1

    def decode_sense(self) -> Literal['+', '-', '.']: return {1: '+', -1: '-', 0: '.'}[self.strand]

    def overlap(self, other: Union['Interval', 'Feature']) -> int:
        if isinstance(other, Feature): other = other.interval
        if isinstance(other, Interval): return max(0, min(self.end, other.end) - max(self.start, other.start))
        else: raise TypeError(other)

    def shift(self, by: int): return Interval(self.start + by, self.end + by, self.strand)

    def reverse_complement(self, parent_length: int) -> 'Interval':
        return Interval(*_rc_interval(self.start, self.end, parent_length), self.strand * -1)

    @classmethod
    def random(cls, rng: Random = None, length: int = None, min_len: int = 1, max_len: int = 10000,
               min_start: int = 0, max_start: int = 1000000):
        if rng is None: rng = RESOURCES.rng
        if not length: length = rng.randint(min_len, max_len)
        start = rng.randint(min_start, max_start - length)
        return cls(start, start + length, not not rng.getrandbits(1))


class _Seq:
    # Using ABC "locks" the init so Seqs can only be created by via alphabets and methods - consider this for future
    # rather than making _Seq private.
    __slots__ = ('_seq', 'alphabet')
    def __init__(self, seq: str, alphabet: Alphabet):
        self._seq: str = seq
        self.alphabet: Alphabet = alphabet

    def __repr__(self): return f"{self._seq if len(self._seq) < 13 else f'{self._seq[:5]}...{self._seq[-5:]}'}"
    def __str__(self): return self._seq
    def __bytes__(self): return self._seq.encode()
    def __iter__(self): return iter(self._seq)
    def __len__(self): return len(self._seq)
    def __hash__(self) -> int: return hash(self._seq)
    def __reversed__(self) -> '_Seq': return self.alphabet.seq(self._seq[::-1])

    def __eq__(self, other):
        if isinstance(other, _Seq): return self._seq == other._seq
        elif isinstance(other, str): return self._seq == other.upper()
        return False

    def __contains__(self, item):
        if isinstance(item, _Seq): return str(item) in self._seq
        elif isinstance(item, str): return item.upper() in self._seq
        return False

    def __add__(self, other: Union[str, '_Seq']) -> '_Seq':
        if isinstance(other, _Seq):
            if self.alphabet != other.alphabet: raise SeqError('Both sequences need to be of the same alphabet')
            other = str(other)
        return self.alphabet.seq(self._seq + other)

    def __getitem__(self, item: Union[slice, int, Interval, 'Feature']) -> '_Seq':
        item = _coerce_interval(item)
        chunk = self._seq[item.start:item.end]
        # Only RC if strictly negative. Treat 0 (unknown) as forward.
        if item.strand == -1: chunk = self.alphabet.reverse_complement(chunk)
        # Returns forward sequence for Strand 1 AND Strand 0
        return self.alphabet.seq(chunk)

    def __delitem__(self, item):
        item = _coerce_interval(item)
        self._seq = self._seq[:item.start] + self._seq[item.end:]

    def hexdigest(self): return self.alphabet.hexdigest(self._seq)


class Qualifier:
    """
    Class to represent a qualifier as a key and value

    Attributes:
        key: str
        value: Any
    """
    def __init__(self, key: str, value: Any = None):
        self.key = key
        self.value = value if value != '' or value is not None else ''  # We still want zeroes

    def __repr__(self): return f"{self.key}={self.value}" if self.value != '' or self.value is not None else self.key

    def __iter__(self) -> Iterator[tuple[str, Any]]: return iter((self.key, self.value))

    def __eq__(self, other):
        return self.key == other.key and self.value == other.value if isinstance(other, Qualifier) else False


class Record:
    """
    Represents a sequence record.
    """
    __slots__ = ('seq', 'id', 'description', 'qualifiers', '_features', '_interval_index', '_dirty_index')

    def __init__(self, seq: _Seq, id_: str = None, desc: str = None, qualifiers: list[Qualifier] = None,
                 features: list['Feature'] = None):
        self.seq: _Seq = seq
        self.id: str = id_ or self.seq.hexdigest()
        self.description: str = desc or ''
        self.qualifiers: list[Qualifier] = qualifiers or []
        # INITIALIZE FeatureList wrapper
        self._dirty_index = True
        self._interval_index = None
        # We assign to private _features via the wrapper
        self._features = FeatureList(self, features)

    def __repr__(self) -> str: return f"{self.id} {self.seq.__repr__()}"
    def __str__(self) -> str: return self.id
    def __len__(self) -> int: return len(self.seq)
    def __hash__(self) -> int: return hash(self.id)
    def __iter__(self) -> Iterator['Feature']: return iter(self.features)
    def __eq__(self, other) -> bool: return self.id == other.id if isinstance(other, Record) else False

    @property
    def features(self) -> 'FeatureList': return self._features

    @features.setter
    def features(self, value):
        # If user does record.features = [...], we need to re-wrap it
        self._features = FeatureList(self, value)
        self._dirty_index = True

    @property
    def interval_index(self) -> 'IntervalIndex':
        if self._dirty_index or self._interval_index is None:
            self._interval_index = IntervalIndex(*self.features)
            self._dirty_index = False
        return self._interval_index

    def add_features(self, *features: 'Feature'):
        # Just delegate to the list extension, which flags dirty automatically
        self.features.extend(features)
        self.features.sort(key=attrgetter('interval.start'))

    def __getitem__(self, item) -> 'Record':
        item = _coerce_interval(item)
        new_record = Record(self.seq[item], f"{self.id}_{item.start}-{item.end}")
        # Use binary search to find the range of features that could possibly be in the slice
        for i in range(bisect_left(self.features, item.start, key=attrgetter('interval.start')), len(self.features)):
            if (feature := self.features[i]).interval.start >= item.end: break # Past the slice
            if feature.interval in item: new_record.features.append(feature.shift(-item.start))
        return new_record

    def get_overlapping(self, start: int, end: int, max_feature_size: int = 1_000_000) -> Generator['Feature', None, None]:
        """
        Yields features overlapping [start, end).
        Catches features starting before 'start' that extend into the region.
        """
        # 1. Start searching from the right (based on Query End)
        idx = bisect_right(self.features, end, key=attrgetter('interval.start'))
        for i in range(idx - 1, -1, -1):
            feature = self.features[i]
            # If the feature ends before our query starts, we are done*
            if feature.interval.end <= start:
                # *Heuristic optimization:
                # If features are sorted by start, a feature ending way before
                # 'start' usually means we passed the cluster.
                # Remove this break if you have massive nested features (like whole-genome operons).
                if feature.interval.end < start - max_feature_size: break
                continue

            yield feature


    def __add__(self, other: 'Record') -> 'Record':
        if isinstance(other, Record):
            new = Record(self.seq + other.seq, f"{self.id}_{other.id}")
            merged_feature = None
            features = []
            self_last_feature = self.features[-1] if self.features and self.features[-1].interval.end == len(self) else None
            other_first_feature = other.features[0] if other.features and other.features[0].interval.start == 0 else None
            # If both boundary features exist and are of the same kind, merge them.
            if self_last_feature and other_first_feature and self_last_feature.kind == other_first_feature.kind:
                features.append(
                    merged_feature := Feature(
                        self_last_feature.interval + other_first_feature.interval.shift(len(self)),
                        self_last_feature.kind,
                        deepcopy(self_last_feature.qualifiers)
                    )  # Exclude the original boundary features from being added separately.
                )
            for feature in (self.features[:-1] if merged_feature else self.features):
                features.append(deepcopy(feature))  # Deep copy feature for full independence and Add to new record's features

            for feature in (other.features[1:] if merged_feature else other.features):  # Other feature intervals need to be updated
                features.append(feature.shift(len(self)))  # Creates a new feature and interval

            new.add_features(*features)
            return new

        else: raise TypeError(other)

    def __radd__(self, other: 'Record') -> 'Record': return other.__add__(self)

    def __iadd__(self, other: 'Record'):
        if isinstance(other, Record):
            self.id = f"{self.id}_{other.id}"  # Update the ref ID
            self.features += (i.shift(len(self)) for i in other.features)  # Add shifted features from other
            self.features.sort(key=attrgetter('interval.start'))  # Sort features
            self.seq += other.seq  # Now we can update the sequence
            return self
        else: raise TypeError(other)

    def __delitem__(self, key: Union[slice, int]):
        """Deletes a slice from the record, adjusting features accordingly."""
        if not isinstance(key, slice):
            raise TypeError(f"Deletion from a Record is only supported for slices, not {type(key)}")
        start, stop, step = key.indices(len(self))
        if step != 1: raise ValueError("Deletion with a step is not supported.")
        slice_len = stop - start
        if slice_len <= 0: return  # Nothing to delete
        new_features = []
        # Find the index of the first feature that could be affected (starts before the slice ends)
        start_idx = bisect_right(self.features, start, key=attrgetter('interval.start'))
        # Add all features that are entirely before the slice
        new_features.extend(self.features[:start_idx])
        # Iterate only over features that could be affected by the deletion
        for feature in self.features[start_idx:]:
            f_start, f_end = feature.interval.start, feature.interval.end
            # This feature is entirely after the deleted slice, shift it and add it.
            # All subsequent features will also be after.
            if f_start >= stop: new_features.append(feature.shift(-slice_len))
            # This feature is partially or fully overlapped
            else:
                # Truncate if it overlaps the start of the slice
                if f_start < start < f_end: feature.interval.end = start
                # Truncate and shift if it overlaps the end of the slice
                if f_start < stop < f_end: feature.interval.start = stop - slice_len
                # If a feature was modified (i.e., not fully deleted), add its modified version.
                if feature.interval.end > feature.interval.start: new_features.append(feature)
                # Features fully contained within the slice are implicitly dropped.

        self.features = new_features
        self.seq = self.seq[:start] + self.seq[stop:]
        self._dirty_index = True  # Mark dirty

    def shred(self, rng: Random = None, n_breaks: int = None, break_points: list[int] = None
              ) -> Generator['Record', None, None]:
        """
        Shreds the record into smaller records at the specified break points.

        :param rng: Random number generator
        :param n_breaks: The number of breaks to make in the record. If not provided, a random number of breaks will be
            made between 1 and half the length of the record.
        :param break_points: A list of break points to use. If not provided, random break points will be generated.
        :return: A generator of smaller records
        """
        if rng is None: rng = RESOURCES.rng
        if not n_breaks: n_breaks = rng.randint(1, len(self) // 2)
        if not break_points: break_points = sorted([rng.randint(0, len(self)) for _ in range(n_breaks)])
        previous_end = 0
        for break_point in break_points:
            yield self[previous_end:break_point]
            previous_end = break_point
        yield self[previous_end:]

    def insert(self, other: 'Record', at: int, replace: bool = True) -> 'Record':
        """
        Inserts another record into this record at the specified position.

        :param other: The record to insert.
        :param at: The position to insert the other record at.
        :param replace: Whether to replace the existing sequence at the insertion point with the inserted sequence.
            If False, the inserted sequence will be inserted without removing any existing sequence.
        :return: A new Record instance with the other record inserted.
        """
        if not 0 < at < len(self): raise IndexError(f'Cannot insert at {at}, must be between 0 and {len(self)}')
        else: return self[:at] + other + self[at if not replace else at + len(other):]

    def get_qualifier(self, key: str, default: Any = None):
        return next((v for k, v in self.qualifiers if k == key), default)


class Feature:
    __slots__ = ('interval', 'kind', 'qualifiers')
    def __init__(self, interval: Interval, kind: str = 'misc_feature', qualifiers: list[Qualifier] = None):
        self.interval = interval
        self.kind = kind
        self.qualifiers = qualifiers or []

    def __len__(self) -> int: return len(self.interval)
    def __repr__(self): return f"{self.kind}({self.interval})"
    def __iter__(self): return self.interval.__iter__()
    def __contains__(self, item) -> bool: return self.interval.__contains__(item)
    def __eq__(self, other):
        if isinstance(other, self.__class__): return self.interval == other.interval and self.kind == other.kind
        return False

    def __getitem__(self, item):
        for q in self.qualifiers:
            if q.key == item: return q.value
        return None

    def __setitem__(self, key: str, value: Any):
        """
        Syntactic sugar: entity['gene'] = 'blaKPC'
        Replaces the FIRST occurrence of key, or appends if new.
        This helps 'restrict redundant qualifiers' by defaulting to replacement.
        """
        for q in self.qualifiers:
            if q.key == key:
                q.value = value
                return
        self.qualifiers.append(Qualifier(key, value))

    def __delitem__(self, item: Union[slice, int, Interval, 'Feature']):
        item = _coerce_interval(item)
        del self.interval[item]
        self.qualifiers = []

    def overlap(self, other) -> int: return self.interval.overlap(other)

    def shift(self, by: int) -> 'Feature': return Feature(self.interval.shift(by), self.kind, deepcopy(self.qualifiers))

    def reverse_complement(self, parent_length: int) -> 'Feature':
        return Feature(self.interval.reverse_complement(parent_length), self.kind, deepcopy(self.qualifiers))

    def get_all(self, key: str) -> list[Any]: return [q.value for q in self.qualifiers if q.key == key]

    def add_qualifier(self, key: str, value: Any = True): self.qualifiers.append(Qualifier(key, value))


class FeatureList(MutableSequence):
    """A list that notifies a parent Record when modified."""
    __slots__ = ('_data', '_parent')

    def __init__(self, parent: 'Record', features: Iterable['Feature'] = None):
        self._data: list[Feature] = list(features) if features else []
        self._parent = parent

    def __getitem__(self, index): return self._data[index]
    def __repr__(self): return repr(self._data)
    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)

    def _flag_dirty(self):
        if self._parent: self._parent._dirty_index = True

    def __setitem__(self, index, value):
        self._data[index] = value
        self._flag_dirty()

    def __delitem__(self, index):
        del self._data[index]
        self._flag_dirty()

    def insert(self, index, value):
        self._data.insert(index, value)
        self._flag_dirty()

    def sort(self, key=None, reverse=False):
        self._data.sort(key=key, reverse=reverse)
        self._flag_dirty()


class Mutation(Feature):
    """
    Represents a specific change in sequence: SNP, Insertion, or Deletion.
    Inherits from Feature to allow spatial indexing and overlap detection.
    """
    __slots__ = ('ref_seq', 'alt_seq', 'aa_ref', 'aa_alt', 'aa_pos', 'effect')
    _CIGAR_PARSER = CigarParser()
    _CACHE = {}  # For storing genetic code instances
    def __init__(self, interval: Interval, ref_seq: str, alt_seq: str,
                 aa_ref: str = None, aa_alt: str = None, aa_pos: int = None,
                 effect: str = None, qualifiers: list[Qualifier] = None):
        super().__init__(interval, kind='mutation', qualifiers=qualifiers)
        self.ref_seq = ref_seq
        self.alt_seq = alt_seq

        # Protein level data (optional/computed later)
        self.aa_ref = aa_ref
        self.aa_alt = aa_alt
        self.aa_pos = aa_pos
        self.effect = effect  # e.g., 'synonymous', 'missense', 'frameshift'

    def __repr__(self):
        # VCF-style notation: Pos Ref>Alt
        dna_str = f"{self.interval.start + 1}:{self.ref_seq}>{self.alt_seq}"
        if self.aa_ref and self.aa_alt:
            return f"<Mutation {dna_str} ({self.aa_ref}{self.aa_pos}{self.aa_alt})>"
        return f"<Mutation {dna_str}>"

    @property
    def is_snp(self): return len(self.ref_seq) == 1 and len(self.alt_seq) == 1

    @property
    def is_indel(self): return len(self.ref_seq) != len(self.alt_seq)

    def predict_effect(self, cds_record: Record, genetic_code: int = 11):
        """
        Calculates the amino acid effect of this mutation given the overlapping CDS.
        """
        # 1. Coordinate Math: Where is this mutation relative to the CDS start?
        # (Assumes CDS is on Forward strand for simplicity - you'd add RC logic here)
        cds_start = cds_record.features[0].interval.start  # Simplified

        rel_start = self.interval.start - cds_start
        if rel_start < 0: return  # Upstream

        # 2. Frame check
        frame = rel_start % 3
        codon_start = rel_start - frame

        # 3. Frameshift Detection
        if self.is_indel and abs(len(self.ref_seq) - len(self.alt_seq)) % 3 != 0:
            self.effect = 'frameshift'
            self.aa_pos = (rel_start // 3) + 1
            return

        # 4. Extract Codons
        # We need the full codon from the reference to translate
        # Slice: [codon_start : codon_start + 3] relative to CDS
        # Note: We need to grab this from the ACTUAL sequence
        # Let's assume 'cds_record' is the sliced CDS record

        if codon_start + 3 > len(cds_record): return

        # Original Codon
        ref_codon = str(cds_record.seq[codon_start: codon_start + 3])

        # Mutated Codon
        # We splice the 'alt_sq' into the codon
        # This logic is complex for indels/spanning boundaries,
        # so here is the SNP version:
        if self.is_snp:
            mut_codon_list = list(ref_codon)
            mut_codon_list[frame] = self.alt_seq
            alt_codon = "".join(mut_codon_list)
        else:
            # For in-frame indels, it's safer to mark as 'indel' without reconstructing
            self.effect = 'in_frame_indel'
            self.aa_pos = (rel_start // 3) + 1
            return

        # 5. Translation
        if (cached := self._CACHE.get(genetic_code)) is None:
            self._CACHE[genetic_code] = (cached := GeneticCode(genetic_code))
        genetic_code = cached

        aa_ref = genetic_code.codons.get(ref_codon, 'X')
        aa_alt = genetic_code.codons.get(alt_codon, 'X')

        self.aa_ref = aa_ref
        self.aa_alt = aa_alt
        self.aa_pos = (rel_start // 3) + 1

        # 6. Classification
        if aa_ref == aa_alt: self.effect = 'synonymous'
        elif aa_alt == '*': self.effect = 'nonsense'  # Early stop
        elif aa_ref == '*': self.effect = 'stop_lost'
        else: self.effect = 'missense'

    @classmethod
    def from_alignment(cls, alignment: 'Alignment', query_seq: str, target_seq: str) -> Generator[
        'Mutation', None, None]:
        """
        Generates Mutations using the CigarParser.
        """
        # Track current positions in the sequences
        # Note: 'q_len' etc from parser are CUMULATIVE lengths, so we need to track
        # the *start* of the current block manually or subtract n.

        curr_t_pos = alignment.interval.start
        curr_q_pos = alignment.query_interval.start

        # Iterate over operations yielded by parser
        for op, n, cum_q, cum_t, cum_aln in cls._CIGAR_PARSER.parse(alignment.cigar):

            if op in cls._CIGAR_PARSER._CONSUMES_BOTH:
                # Match/Mismatch: scan base by base
                # Only iterate if we suspect mismatches (X) or generic match (M)
                # If '=' is used strictly, we could skip this block for speed.

                # Slicing is expensive in tight loops, so we only slice the block
                t_block = target_seq[curr_t_pos: curr_t_pos + n]
                q_block = query_seq[curr_q_pos: curr_q_pos + n]

                for i in range(n):
                    if t_block[i] != q_block[i]:
                        yield cls(
                            Interval(curr_t_pos + i, curr_t_pos + i + 1, 1),
                            ref_seq=t_block[i],
                            alt_seq=q_block[i]
                        )

                curr_t_pos += n
                curr_q_pos += n

            elif op == 'I':
                # Insertion in Query (Gap in Target)
                # We define the mutation at the anchor point on the Target
                inserted_bases = query_seq[curr_q_pos: curr_q_pos + n]
                yield cls(
                    Interval(curr_t_pos, curr_t_pos, 1),  # 0-width interval
                    ref_seq="",
                    alt_seq=inserted_bases
                )
                curr_q_pos += n

            elif op == 'D':
                # Deletion in Query (Gap in Query, Bases in Target)
                deleted_bases = target_seq[curr_t_pos: curr_t_pos + n]
                yield cls(
                    Interval(curr_t_pos, curr_t_pos + n, 1),
                    ref_seq=deleted_bases,
                    alt_seq=""
                )
                curr_t_pos += n

            elif op == 'S':
                # Soft clipping consumes Query but not Target
                curr_q_pos += n

            elif op == 'N':
                # Skipped region (usually introns) consumes Target but not Query
                curr_t_pos += n

            # H (Hard clip) and P (Padding) consume neither sequence locally
            # so curr_t_pos / curr_q_pos do not change.


class IntervalIndex:
    __slots__ = ('_data',)
    _DTYPE = np.int32  # Consider int64 for interval coordinates to future-proof against large assemblies.
    def __init__(self, *intervals: Union[slice, int, Interval, 'Feature']):
        if not intervals: self._data = np.empty((0, 3), dtype=self._DTYPE)
        else:
            # Fast path for Feature objects (saves attribute lookup overhead in loop)
            if hasattr(intervals[0], 'interval'):  # Extract directly
                flat_iter = (val for obj in intervals for val in obj.interval)
            else:  # Fallback to safe coercion
                flat_iter = (val for i in intervals for val in _coerce_interval(i))
            # We know the size, passing 'count' to fromiter allows pre-allocation
            self._data = np.fromiter(flat_iter, dtype=self._DTYPE, count=len(intervals) * 3).reshape(-1, 3)
            # CRITICAL: Always sort on creation to enable fast kernels
            self.sort()

    def __len__(self): return len(self._data)
    def __iter__(self): return self._generate_intervals(self.starts, self.ends, self.strands)

    def span(self): return self._data.max() - self._data.min()

    @property
    def starts(self): return self._data[:, 0]

    @property
    def ends(self): return self._data[:, 1]

    @property
    def strands(self): return self._data[:, 2]

    @staticmethod
    def _generate_intervals(s, e, st): yield from (Interval(*i) for i in zip(s, e, st))

    def subtract(self, other: 'IntervalIndex', stranded: bool = False) -> 'IntervalIndex':
        if len(other) == 0: return deepcopy(self)

        # Merge 'other' to simplify subtraction logic (A - B_merged)
        merged_b = IntervalIndex(*other.merge())

        # Sort self to ensure linear processing
        # (Assuming self is usually sorted, but safe to enforce or rely on existing sort)
        if len(self) > 1 and self._data[0, 0] > self._data[-1, 0]:
            self.sort()

        a_s, a_e, a_st = self.starts, self.ends, self.strands
        b_s, b_e, b_st = merged_b.starts, merged_b.ends, merged_b.strands

        # Run the kernel
        out_s, out_e, out_st = _subtract_kernel(a_s, a_e, a_st, b_s, b_e, b_st, stranded)

        new = IntervalIndex()
        if len(out_s) > 0:
            new._data = np.column_stack((out_s, out_e, out_st)).astype(self._DTYPE)
        return new

    def sort(self):
        # Sort by Start (Primary) and End (Secondary)
        self._data = self._data[np.lexsort((self._data[:, 1], self._data[:, 0]))]

    def copy(self):
        new = IntervalIndex()
        new._data = self._data.copy()
        return new

    def intersect(self, other: 'IntervalIndex', stranded: bool = False) -> 'IntervalIndex':
        if len(self) == 0 or len(other) == 0: return IntervalIndex()

        # Ensure sorting (should be guaranteed by init/add, but safety first)
        # We assume 'other' is the database (b)

        b_starts, b_ends, b_strands = other.starts, other.ends, other.strands
        a_starts, a_ends, a_strands = self.starts, self.ends, self.strands

        # Run Numba Kernel
        out_s, out_e, out_st = _intersect_kernel(a_starts, a_ends, a_strands,b_starts, b_ends, b_strands,stranded)

        new = IntervalIndex()
        if len(out_s) > 0: new._data = np.column_stack((out_s, out_e, out_st)).astype(self._DTYPE)
        return new

    def pad(self, upstream: int, downstream: int = None) -> 'IntervalIndex':
        """
        Expands intervals. Respects strand if present.
        If downstream is None, applies symmetric padding (upstream) to both sides.
        """
        if downstream is None: downstream = upstream
        if len(self) == 0: return self

        new = self.copy()
        s, e, st = new.starts, new.ends, new.strands

        # Logic:
        # Strand 1 (+): Start - Up, End + Down
        # Strand -1 (-): Start - Down, End + Up
        # Strand 0: Start - Up, End + Up (Symmetric)

        # 1. Forward/Unstranded
        mask_fwd = st >= 0
        s[mask_fwd] -= upstream
        e[mask_fwd] += downstream

        # 2. Reverse
        mask_rev = st < 0
        s[mask_rev] -= downstream
        e[mask_rev] += upstream

        # 3. Clip negative values to 0
        np.maximum(new._data[:, 0], 0, out=new._data[:, 0])

        # 4. Re-sort because padding might overlap/reorder small intervals
        new.sort()
        return new.merge()  # Usually you want to merge after padding

    def complement(self, length: int) -> 'IntervalIndex':
        """
        Returns the 'gaps' in the index up to 'length'.
        Essential for finding intergenic regions or uncovered regions.
        """
        if len(self) == 0:
            # Whole genome is empty
            return IntervalIndex(Interval(0, length))

        merged = self.merge()
        s, e = merged.starts, merged.ends

        gap_starts = []
        gap_ends = []

        # Gap before first interval
        if s[0] > 0:
            gap_starts.append(0)
            gap_ends.append(s[0])

        # Gaps between intervals
        # Gap Start = Previous End
        # Gap End = Current Start
        mid_gaps_mask = s[1:] > e[:-1]
        if np.any(mid_gaps_mask):
            gap_starts.extend(e[:-1][mid_gaps_mask])
            gap_ends.extend(s[1:][mid_gaps_mask])

        # Gap after last interval
        if e[-1] < length:
            gap_starts.append(e[-1])
            gap_ends.append(length)

        if not gap_starts:
            return IntervalIndex()

        new = IntervalIndex()
        # Gaps are usually unstranded (0)
        strands = np.zeros(len(gap_starts), dtype=self._DTYPE)
        new._data = np.column_stack((gap_starts, gap_ends, strands)).astype(self._DTYPE)
        return new

    def coverage(self) -> int:
        """Returns total unique bases covered."""
        merged = self.merge()
        return np.sum(merged.ends - merged.starts)

    def merge(self, tolerance: int = 0) -> 'IntervalIndex':
        # (Just ensure you update it to use self._data directly if you change internals)
        if len(self._data) == 0: return self
        sorted_data = self._data  # Already sorted by sort()
        starts, ends, strands = sorted_data[:, 0], sorted_data[:, 1], sorted_data[:, 2]

        max_ends = np.maximum.accumulate(ends)
        gaps = starts[1:] > (max_ends[:-1] + tolerance)
        split_indices = np.flatnonzero(gaps) + 1
        reduce_indices = np.concatenate(([0], split_indices))

        merged_starts = starts[reduce_indices]
        merged_ends = np.maximum.reduceat(ends, reduce_indices)

        # Simple strand logic for speed (take first).
        # Use your consensus logic if strictly needed, but this is 10x faster.
        merged_strands = strands[reduce_indices]

        new = IntervalIndex()
        new._data = np.column_stack((merged_starts, merged_ends, merged_strands)).astype(self._DTYPE)
        return new


class MotifFinder:
    """
    A generic class for finding sequence motifs using Regular Expressions and
    associating them with nearby features in a Record.
    """
    _ALPHABET = Alphabet.dna()
    def __init__(
            self, name: str,
            pattern: str, description:
            str = None,
            dist: int = 100,
            direction: Literal['upstream', 'downstream', 'both'] = 'both',
            strands: Literal[0, 1, -1] = 0,
            same_strand: bool = True,
            target_kinds: set[str] = frozenset({'CDS'})
    ):
        self.name = name
        self.pattern = regex(pattern)  # Compiles the regex
        self.description = description
        self.dist: int = dist
        self.direction: Literal['upstream', 'downstream', 'both'] = direction
        self.strands: Literal[0, 1, -1] = strands
        self.same_strand: bool = same_strand
        self.target_kinds: set[str] = target_kinds

    def _find_motifs(self, record: Record) -> Generator[Feature, None, None]:
        seq_len = len(record)

        # 1. Forward Strand Search
        if self.strands in (0, 1):
            for m in self.pattern.finditer(str(record.seq)):
                # Create the main feature
                feat = Feature(Interval(m.start(), m.end(), 1), kind=self.name)
                # Extract capture groups into qualifiers
                self._extract_groups(m, feat)
                yield feat

        # 2. Reverse Strand Search
        if self.strands in (0, -1):
            # We search on the Reverse Complement, but must map coordinates back to Forward
            rc_seq = self._ALPHABET.reverse_complement(str(record.seq))
            for m in self.pattern.finditer(rc_seq):
                # Transform RC coordinates to Forward coordinates
                # RC start index i corresponds to Forward index L - 1 - i
                # Since intervals are half-open:
                # RC [start, end) -> Forward [L - end, L - start)
                feat = Feature(Interval(*_rc_interval(m.start(), m.end(), seq_len), -1), kind=self.name)
                self._extract_groups(m, feat)
                yield feat

    @staticmethod
    def _extract_groups(match_obj, feature: Feature):
        """Helper to push regex capture groups into feature qualifiers."""
        for name, text in match_obj.groupdict().items():
            if text:
                # Capture the interval of the sub-group relative to the parent sequence
                # Note: Logic gets hairy for RC subgroups, so we store the TEXT and
                # relative location could be calculated if strictly needed.
                feature.add_qualifier(name, text)

    def find(self, record: Record) -> Generator[tuple[Feature, Feature], None, None]:
        for motif in self._find_motifs(record):
            # 1. Define Search Window based on direction relative to Motif Strand
            # Logic: If Motif is (+), Downstream is (End, End+Dist).
            #        If Motif is (-), Downstream is (Start-Dist, Start).

            m_start, m_end, m_strand = motif.interval.start, motif.interval.end, motif.interval.strand
            search_start, search_end = 0, 0
            if self.direction == 'both':
                search_start = max(0, m_start - self.dist)
                search_end = min(len(record), m_end + self.dist)
            else:
                is_downstream = (self.direction == 'downstream')
                # XOR logic: If Strand is (-) and we want Downstream, we look mathematically "left" (decreasing coords)
                look_forward = (m_strand == 1) == is_downstream

                if look_forward:
                    search_start = m_end
                    search_end = min(len(record), m_end + self.dist)
                else:
                    search_start = max(0, m_start - self.dist)
                    search_end = m_start

            # 2. Use Record's optimized binary search to find overlaps in the window
            if search_start >= search_end: continue
            for target in record.get_overlapping(search_start, search_end):
                if target is motif: continue  # Don't find self
                # 3. Filter by Strand
                if self.same_strand and target.interval.strand != m_strand: continue
                # 4. Filter by Type (optional, usually we want CDS or gene)
                if target.kind not in self.target_kinds: continue
                yield motif, target


class PromoterFinder(MotifFinder):
    """
    Specialized MotifFinder for Sigma70 Bacterial Promoters.
    Matches the -35 and -10 boxes with variable spacing (15-21bp).
    """
    # -35 box (TTGACA approx) ... 15-21bp spacer ... -10 box (TATAAT approx)
    # The regex allows for some degeneracy common in bacteria
    _SIGMA70 = r"(?P<minus_35>TT[GCA][ATGC]{2}[A]).{15,21}(?P<minus_10>TA[ATGC]{2}A[AT])"
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="promoter",
            pattern=self._SIGMA70,
            description="Sigma70-like promoter prediction",
            *args, **kwargs
        )


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _intersect_kernel(s_a, e_a, st_a, s_b, e_b, st_b, stranded):
    """
    Intersection using a 'Zipper' approach (or Merge Join approach).
    Since both lists are sorted by Start, we can iterate efficiently.
    Complexity: O(N + M) for typical genomic data (roughly linear).
    """
    n_a = len(s_a)
    n_b = len(s_b)

    # Heuristic: maximum output size approx n_a (if 1-to-1) or larger
    # We use a dynamic list for results
    out_s = []
    out_e = []
    out_st = []

    b_idx = 0

    # Iterate over Query (A)
    for i in range(n_a):
        curr_s_a = s_a[i]
        curr_e_a = e_a[i]
        curr_st_a = st_a[i]

        # 1. Advance B pointer to the first potential overlap
        # We need B.end > A.start
        # Since B is sorted by START, we can't just skip blindly,
        # but generally B.start increases.
        # Optimization: We only advance b_idx if b_starts[b_idx] is clearly too far left
        # AND its end is also too far left.

        # Simple sweep: check B intervals starting from where we left off
        # Reset temp pointer
        temp_b = b_idx

        while temp_b < n_b:
            curr_s_b = s_b[temp_b]

            # Optimization: If B starts after A ends, no further B can overlap this A
            # (Because B is sorted by start)
            if curr_s_b >= curr_e_a:
                break

            curr_e_b = e_b[temp_b]

            # If B ends before A starts, it's irrelevant.
            # AND if this is the main pointer 'b_idx', we can permanently advance it
            # because this B will never overlap any FUTURE A (since As are also sorted).
            if curr_e_b <= curr_s_a:
                if temp_b == b_idx:
                    b_idx += 1
                temp_b += 1
                continue

            # Overlap found!
            # Check strand
            curr_st_b = st_b[temp_b]
            if not stranded or (curr_st_a == curr_st_b):
                # Calc intersection
                new_s = max(curr_s_a, curr_s_b)
                new_e = min(curr_e_a, curr_e_b)

                out_s.append(new_s)
                out_e.append(new_e)
                # Inherit A's strand
                out_st.append(curr_st_a)

            temp_b += 1

    return np.array(out_s), np.array(out_e), np.array(out_st)


@jit(nopython=True, cache=True, nogil=True)
def _subtract_kernel(s_a, e_a, st_a, s_b, e_b, st_b, stranded):
    n_a, n_b = len(s_a), len(s_b)
    out_s, out_e, out_st = [], [], []

    # Pointer for B
    b_idx = 0

    for i in range(n_a):
        curr_s = s_a[i]
        end_a = e_a[i]
        strand_a = st_a[i]

        # 1. Advance B pointer to the first relevant interval
        # We need a B that ends AFTER our current start
        while b_idx < n_b and e_b[b_idx] <= curr_s:
            b_idx += 1

        # 2. Iterate through overlapping B intervals
        # We use a temp pointer because one B might overlap multiple As
        temp_b = b_idx

        while temp_b < n_b and s_b[temp_b] < end_a:
            b_start = s_b[temp_b]
            b_end = e_b[temp_b]
            b_strand = st_b[temp_b]

            # Check strand logic if required
            if stranded and (b_strand != strand_a):
                temp_b += 1
                continue

            # If B starts after our current cursor, we have a clear chunk of A to keep
            if b_start > curr_s:
                out_s.append(curr_s)
                out_e.append(b_start)
                out_st.append(strand_a)

            # Advance our cursor to the end of the subtraction (punch the hole)
            curr_s = max(curr_s, b_end)

            # If we've consumed all of A, stop checking Bs
            if curr_s >= end_a:
                break

            temp_b += 1

        # 3. If there is leftover A after all overlaps
        if curr_s < end_a:
            out_s.append(curr_s)
            out_e.append(end_a)
            out_st.append(strand_a)

    return np.array(out_s), np.array(out_e), np.array(out_st)


# Functions ------------------------------------------------------------------------------------------------------------
def _coerce_interval(item: Union[slice, int, Interval, Feature]) -> Interval:
    if hasattr(item, 'interval'): return item.interval
    if isinstance(item, Interval): return item
    # elif isinstance(item, slice):
    #     start, end, _ = item.indices(len(self._seq))
    # elif isinstance(item, int):
    #     if item < 0:
    #         item += len(self._seq)
    #     start, end = item, item + 1
    if isinstance(item, slice): return Interval.from_slice(item)
    if isinstance(item, int): return Interval.from_slice(slice(item, item))
    raise TypeError(f"Cannot coerce {type(item)} to Interval")


def _rc_interval(start, end, parent_length): return parent_length - end, parent_length - start

