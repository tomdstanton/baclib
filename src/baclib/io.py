"""
Module for parsing and managing bacterial sequence files and data.
"""
from abc import ABC, abstractmethod
from re import compile as regex
from io import IOBase, TextIOWrapper, BufferedWriter, BufferedReader
from typing import Union, Generator, IO, Literal, TextIO, get_args, Optional, List
from itertools import chain
from pathlib import Path
from sys import stdout, stdin
from importlib import import_module

from . import BaclibWarning
from .graph import Edge
from .seq import Record, Feature, Alphabet, Qualifier, Interval, CigarParser
from .alignment import Alignment


# Exceptions and Warnings -----------------------------------------------------------------------------------------
class SeqFileError(Exception):
    """Exception raised for errors in sequence file processing."""
    pass


class ParserError(Exception):
    """Exception raised for errors during parsing."""
    pass


class SeqFileWarning(BaclibWarning):
    """Warning category for sequence file issues."""
    pass


# Helpers --------------------------------------------------------------------------------------------------------------
class PeekableHandle:
    """
    A wrapper around a TextIO stream that allows peeking at the beginning of the
    content without consuming it. Used by SeqFile to sniff formats.
    """
    __slots__ = ('_stream', '_peek_buffer', '_buffer_pos', '_buffer_len')

    def __init__(self, stream: TextIO, max_peek: int = 4096):
        """
        Initializes the PeekableHandle.

        Args:
            stream: The underlying text stream.
            max_peek: Maximum number of characters to buffer for peeking.
        """
        self._stream = stream
        # Read characters (not bytes) from the TextIO stream
        self._peek_buffer = stream.read(max_peek)
        self._buffer_pos = 0
        self._buffer_len = len(self._peek_buffer)

    def peek(self, size: int = -1) -> str:
        """
        Returns content from the buffer without advancing the stream position.

        Args:
            size: Number of characters to peek. If -1, returns the entire buffer.

        Returns:
            The peeked string.
        """
        if size == -1 or size > self._buffer_len: return self._peek_buffer
        return self._peek_buffer[:size]

    def read(self, size: int = -1) -> str:
        """
        Reads from the stream, consuming the buffer first if available.

        Args:
            size: Number of characters to read. If -1, reads until EOF.

        Returns:
            The read string.
        """
        # 1. Buffer exhausted
        if self._buffer_pos >= self._buffer_len:
            return self._stream.read(size)

        # 2. Read all (rest of buffer + stream)
        if size == -1:
            chunk = self._peek_buffer[self._buffer_pos:]
            self._buffer_pos = self._buffer_len
            return chunk + self._stream.read()

        # 3. Read partial
        available = self._buffer_len - self._buffer_pos
        if size <= available:
            chunk = self._peek_buffer[self._buffer_pos: self._buffer_pos + size]
            self._buffer_pos += size
            return chunk
        else:
            chunk = self._peek_buffer[self._buffer_pos:]
            self._buffer_pos = self._buffer_len
            return chunk + self._stream.read(size - available)

    def __iter__(self):
        """
        Iterates over lines in the stream, handling the buffer seamlessly.

        Yields:
            Lines from the stream.
        """
        # 1. Yield lines from the buffer
        if self._buffer_pos < self._buffer_len:
            fragment = self._peek_buffer[self._buffer_pos:]
            self._buffer_pos = self._buffer_len

            # We use splitlines(keepends=True) to handle line boundaries safely
            lines = fragment.splitlines(keepends=True)

            # If the last segment doesn't end with a newline, it's incomplete.
            # We must stitch it with the next line from the stream.
            for i, line in enumerate(lines):
                if i == len(lines) - 1 and not line.endswith('\n'):
                    # Stitch with next line from stream
                    yield line + self._stream.readline()
                else:
                    yield line

        # 2. Yield rest from stream
        yield from self._stream

    def close(self):
        """Closes the underlying stream if possible."""
        if hasattr(self._stream, 'close'): self._stream.close()


class Xopen:
    """
    Handles the Physical Layer: Compression and File System.
    """
    _MAGIC = {
        b'\x1f\x8b': 'gzip',
        b'\x42\x5a': 'bz2',
        b'\xfd7zXZ\x00': 'lzma',
        b'\x28\xb5\x2f\xfd': 'zstandard'
    }
    _EXT_TO_PKG = {'gz': 'gzip', 'bz2': 'bz2', 'xz': 'lzma', 'zst': 'zstandard'}
    _MIN_N_BYTES = max(len(i) for i in _MAGIC.keys())
    _OPEN_FUNCS = {}

    def __init__(self, file: Union[str, Path, IO], mode: str = 'r', encoding: str = 'utf-8'):
        """
        Initializes the Xopen context manager.

        Args:
            file: File path (str or Path) or an existing file object.
            mode: File opening mode (e.g., 'r', 'w').
            encoding: Text encoding to use.
        """
        self.file = file
        self.mode = mode
        self.encoding = encoding
        self._handle: Optional[IO] = None
        self._close_on_exit = False

    def __enter__(self) -> TextIO:
        """
        Opens the file and returns the file handle.

        Returns:
            The opened file handle.
        """
        self._handle = self._open()
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the file handle if it was opened by this instance.
        """
        if self._close_on_exit and self._handle: self._handle.close()

    def _get_opener(self, pkg_name: str):
        """
        Retrieves the open function for a compression package, importing it if necessary.

        Args:
            pkg_name: Name of the compression package (e.g., 'gzip').

        Returns:
            The open function from the package.

        Raises:
            SeqFileError: If the module cannot be imported.
        """
        if pkg_name not in self._OPEN_FUNCS:
            try:
                mod = import_module(pkg_name)
                self._OPEN_FUNCS[pkg_name] = mod.open
            except ImportError:
                raise SeqFileError(f"Compression module '{pkg_name}' not installed.")
        return self._OPEN_FUNCS[pkg_name]

    def _open(self) -> TextIO:
        """
        Internal method to open the file based on type and compression.

        Returns:
            The opened text file handle.
        """
        if isinstance(self.file, IOBase):
            if 't' in self.mode and isinstance(self.file, (BufferedWriter, BufferedReader)):
                return TextIOWrapper(self.file, encoding=self.encoding)
            return self.file

        if str(self.file) in {'-', 'stdin'} and 'r' in self.mode: return stdin
        if str(self.file) in {'-', 'stdout'} and 'w' in self.mode: return stdout

        path = Path(self.file)
        self._close_on_exit = True

        if 'w' in self.mode or 'a' in self.mode:
            ext = path.suffix.lower().lstrip('.')
            if pkg := self._EXT_TO_PKG.get(ext):
                return self._get_opener(pkg)(path, mode=self.mode + 't', encoding=self.encoding)
            return open(path, mode=self.mode, encoding=self.encoding)

        magic_pkg = self._detect_magic(path)
        if magic_pkg:
            return self._get_opener(magic_pkg)(path, mode='rt', encoding=self.encoding)

        return open(path, mode='rt', encoding=self.encoding)

    def _detect_magic(self, path: Path) -> Optional[str]:
        """
        Detects compression format based on file magic bytes.

        Args:
            path: Path to the file.

        Returns:
            The name of the compression package or None if not detected.
        """
        if not path.exists(): return None
        try:
            with open(path, 'rb') as f:
                start = f.read(self._MIN_N_BYTES)
            for magic, pkg in self._MAGIC.items():
                if start.startswith(magic): return pkg
        except OSError: pass
        return None


class QualifierParser:
    """Parses qualifier strings from various file formats."""
    _TYPE_MAP = {'f': float, 'i': int, 'Z': str, 'A': str, 'H': str, 'B': str}
    _ABBREVIATIONS = {'dp': 'depth', 'ln': 'length', 'kc': 'kmer count'}

    def parse_gff_attributes(self, items: str) -> list[Qualifier]:
        """
        Parses GFF3 attribute strings.

        Args:
            items: The attribute string (e.g., "ID=gene1;Name=foo").

        Returns:
            A list of Qualifier objects.
        """
        if not items or items == '.': return []
        quals = []
        for item in items.split(';'):
            if not item: continue
            key, _, val = item.partition('=')
            if not key: continue
            val = (val.replace('%3B', ';').replace('%3D', '=')
                   .replace('%26', '&').replace('%2C', ','))
            quals.append(Qualifier(key, val))
        return quals

    def parse_tags(self, items: list[str]) -> list[Qualifier]:
        """
        Parses SAM/PAF style tags.

        Args:
            items: List of tag strings (e.g., ["NM:i:0", "AS:f:100"]).

        Returns:
            A list of Qualifier objects.
        """
        quals = []
        for item in items:
            if len(item) < 5: continue
            parts = item.split(':', 2)
            if len(parts) != 3: continue
            tag, typ, val = parts
            tag = self._ABBREVIATIONS.get(tag.lower(), tag)
            converter = self._TYPE_MAP.get(typ, str)
            try: quals.append(Qualifier(tag, converter(val)))
            except ValueError: continue
        return quals


# Base Readers ---------------------------------------------------------------------------------------------------------
class BaseReader(ABC):
    """Abstract base class for sequence file readers."""
    _QUALIFIER_PARSER = QualifierParser()
    _CIGAR_PARSER = CigarParser()
    _DEFAULT_ALPHABET = Alphabet.dna()

    def __init__(self, handle: TextIO, **kwargs):
        """
        Initializes the reader.

        Args:
            handle: The open file handle to read from.
            **kwargs: Additional arguments.
        """
        self._handle = handle

    @abstractmethod
    def __iter__(self) -> Generator[Union[Record, Feature, Edge, Alignment], None, None]:
        """
        Iterates over records in the file.

        Yields:
            Parsed objects (Record, Feature, Edge, or Alignment).
        """
        pass


class TabularReader(BaseReader):
    """Base class for readers of tabular formats (GFF, BED, PAF)."""
    _delim = '\t'
    _min_cols: int = 1

    def __iter__(self) -> Generator[Union[Feature, Alignment], None, None]:
        """
        Iterates over lines, parsing valid rows.

        Yields:
            Parsed Feature or Alignment objects.
        """
        delim = self._delim
        min_cols = self._min_cols
        parse = self.parse_row
        for line in self._handle:
            if line.startswith('#'): continue
            line = line.rstrip()
            if not line: continue
            parts = line.split(delim)
            if len(parts) < min_cols: continue
            yield parse(parts)

    @abstractmethod
    def parse_row(self, parts: list[str]) -> Union[Feature, Alignment]:
        """
        Parses a single row split by delimiter.

        Args:
            parts: List of column strings.

        Returns:
            A Feature or Alignment object.
        """
        pass


# Concrete Readers -----------------------------------------------------------------------------------------------------
class FastaReader(BaseReader):
    """Reader for FASTA format files."""
    _TOPOLOGY_REGEX = regex(r'(?i)(\bcircular\b|\bcircular\s*=\s*true\b)')
    _COPY_NUMBER_REGEX = regex(r'depth=(\d+\.\d+)')

    def __init__(self, handle: TextIO, alphabet: Alphabet = None, **kwargs):
        """
        Initializes the FastaReader.

        Args:
            handle: Open file handle.
            alphabet: Alphabet to use for sequences (default: DNA).
            **kwargs: Additional arguments.
        """
        super().__init__(handle)
        self.alphabet = alphabet or self._DEFAULT_ALPHABET

    def __iter__(self) -> Generator[Record, None, None]:
        """
        Iterates over FASTA records.

        Yields:
            Record objects.
        """
        process = self.alphabet.process_string
        buffer = []
        header = ''
        for line in chain(self._handle, ['>']):
            line = line.rstrip()
            if not line: continue
            if line.startswith('>'):
                if header:
                    name, _, desc = header.partition(' ')
                    yield Record(
                        self.alphabet.seq("".join(buffer)),
                        name, desc,
                        qualifiers=[
                            Qualifier('topology', 'circular' if self._TOPOLOGY_REGEX.search(desc) else 'linear'),
                            Qualifier('depth', float(m[1]) if (m := self._COPY_NUMBER_REGEX.search(desc)) else 1)
                        ]
                    )
                    buffer.clear()
                header = line[1:]
            else:
                buffer.append(process(line))


class FastqReader(BaseReader):
    """Reader for FASTQ format files."""

    def __init__(self, handle: TextIO, alphabet: Alphabet = None, **kwargs):
        """
        Initializes the FastqReader.

        Args:
            handle: Open file handle.
            alphabet: Alphabet to use for sequences (default: DNA).
            **kwargs: Additional arguments.
        """
        super().__init__(handle)
        self.alphabet = alphabet or self._DEFAULT_ALPHABET

    def __iter__(self) -> Generator[Record, None, None]:
        """
        Iterates over FASTQ records.

        Yields:
            Record objects.
        """
        process = self.alphabet.process_string
        lines = iter(self._handle)
        while True:
            try:
                line1 = next(lines).rstrip()
                while not line1: line1 = next(lines).rstrip()
            except StopIteration: break
            if not line1.startswith('@'): raise ParserError(f"Invalid FASTQ header: {line1}")
            try:
                line2 = next(lines).rstrip()
                line3 = next(lines).rstrip()
                line4 = next(lines).rstrip()
            except StopIteration: raise ParserError("Truncated FASTQ record")
            name, _, desc = line1[1:].partition(' ')
            yield Record(
                self.alphabet.seq(process(line2)),
                name, desc,
                qualifiers=[Qualifier('quality', line4)]
            )


class GenbankReader(BaseReader):
    """
    High-performance State Machine Genbank Reader.
    Robust to fuzzy coordinates (<1..>100) and loose indentation.
    """
    # Relaxed regex to handle <, > and partial entries
    _INTERVAL_REGEX = regex(r'(?P<partial_start><)?(?P<start>[0-9]+)\.\.(?P<partial_end>>)?(?P<end>[0-9]+)')
    _TOPOLOGY_REGEX = regex(r'(?i)(\bcircular\b|\bcircular\s*=\s*true\b)')
    _SUPPORTED_KINDS = frozenset({'CDS', 'source'})
    _STATE_HEADER = 0
    _STATE_FEATURES = 1
    _STATE_ORIGIN = 2

    def __init__(self, handle: TextIO, alphabet: Alphabet = None, **kwargs):
        """
        Initializes the GenbankReader.

        Args:
            handle: Open file handle.
            alphabet: Alphabet to use for sequences (default: DNA).
            **kwargs: Additional arguments.
        """
        super().__init__(handle)
        self.alphabet = alphabet or self._DEFAULT_ALPHABET
        self._base = 1

    def __iter__(self) -> Generator[Record, None, None]:
        """
        Iterates over Genbank records.

        Yields:
            Record objects.
        """
        state = self._STATE_HEADER
        record: Optional[Record] = None
        current_feature_lines: List[str] = []
        seq_buffer: List[str] = []

        def process_pending_feature():
            if current_feature_lines and record is not None:
                self._add_feature(record, current_feature_lines)
                current_feature_lines.clear()

        for line in self._handle:
            # Check for end of record
            if line.startswith('//'):
                if record is not None:
                    process_pending_feature()
                    if seq_buffer:
                        record.seq = self.alphabet.seq("".join(seq_buffer), process_string=True)
                    yield record
                record = None
                state = self._STATE_HEADER
                current_feature_lines = []
                seq_buffer = []
                continue

            if state == self._STATE_HEADER:
                if line.startswith('LOCUS'):
                    parts = line.split()
                    name = parts[1] if len(parts) > 1 else 'unknown'
                    record = Record(
                        self.alphabet.seq(''),
                        name,
                        qualifiers=[Qualifier('topology', 'circular' if self._TOPOLOGY_REGEX.search(line) else 'linear')]
                    )
                elif line.startswith('DEFINITION') and record is not None:
                    record.description = line[12:].strip()
                elif line.startswith('FEATURES'):
                    state = self._STATE_FEATURES

            elif state == self._STATE_FEATURES:
                if line.startswith('ORIGIN'):
                    process_pending_feature()
                    state = self._STATE_ORIGIN
                else:
                    if not line.strip(): continue
                    # Key detection: Indented less than qualifiers (usually col 5 vs 21)
                    # We accept anything starting at col 0-10 that isn't a qualifier (/)
                    if len(line) > 2 and line[0:10].strip() and not line.strip().startswith('/'):
                        process_pending_feature()
                        current_feature_lines.append(line)
                    elif current_feature_lines:
                        current_feature_lines.append(line)

            elif state == self._STATE_ORIGIN:
                seq_buffer.append(line)

        # Handle last record
        if record is not None:
            process_pending_feature()
            if seq_buffer: record.seq = self.alphabet.seq("".join(seq_buffer), process_string=True)
            yield record

    def _add_feature(self, record: Record, lines: list[str]):
        """
        Parses feature lines and adds the feature to the record.

        Args:
            record: The Record object to update.
            lines: List of lines belonging to the feature.
        """
        header_line = lines[0].strip()

        if not header_line: return

        parts = header_line.split(maxsplit=1)
        kind = parts[0]
        if kind not in self._SUPPORTED_KINDS: return

        loc_str = parts[1] if len(parts) > 1 else ""
        qual_start_index = 1

        # Accumulate location string
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith('/'): break
            loc_str += stripped
            qual_start_index += 1

        # Try to parse intervals
        strand = -1 if 'complement' in loc_str else 1
        intervals = []
        for m in self._INTERVAL_REGEX.finditer(loc_str):
            intervals.append(Interval(int(m.group('start')) - self._base, int(m.group('end')), strand))

        # IMPORTANT: Even if intervals are empty (failed parse), we proceed if it's 'source'
        # so we don't lose the Organism/Strain qualifiers.
        if not intervals:
            if kind == 'source':
                # Create a dummy interval for source so we can capture qualifiers
                intervals = [Interval(0, 0, 1)]
            else:
                return

        feat = Feature(intervals[0], kind)
        current_qual = []
        for line in lines[qual_start_index:]:
            stripped = line.strip()
            if stripped.startswith('/'):
                if current_qual: self._parse_and_add_qual(feat, current_qual)
                current_qual = [stripped]
            else:
                current_qual.append(stripped)
        if current_qual: self._parse_and_add_qual(feat, current_qual)

        if kind == 'source':
            record.qualifiers.extend(feat.qualifiers)
        else:
            record.features.append(feat)

    def _parse_and_add_qual(self, feat: Feature, lines: list[str]):
        """
        Parses qualifier lines and adds them to the feature.

        Args:
            feat: The Feature object to update.
            lines: List of lines belonging to the qualifier.
        """
        full_text = "".join(lines)
        key, sep, val = full_text[1:].partition('=')
        if not sep:
            feat.add_qualifier(key, True)
        else:
            val = val.strip()
            if val.startswith('"') and val.endswith('"'): val = val[1:-1]
            feat.add_qualifier(key, val)


class GffReader(TabularReader):
    """Reader for GFF3 format files."""
    _min_cols = 9

    def parse_row(self, parts: list[str]) -> Feature:
        """
        Parses a GFF3 row.

        Args:
            parts: List of column strings.

        Returns:
            A Feature object.
        """
        start, end = int(parts[3]) - 1, int(parts[4])
        quals = self._QUALIFIER_PARSER.parse_gff_attributes(parts[8])
        quals.append(Qualifier('source', parts[0]))
        if parts[1] != '.': quals.append(Qualifier('tool', parts[1]))
        if parts[5] != '.': quals.append(Qualifier('score', float(parts[5])))
        if parts[7] != '.': quals.append(Qualifier('phase', int(parts[7])))
        return Feature(Interval(start, end, parts[6]), parts[2], qualifiers=quals)


class BedReader(TabularReader):
    """Reader for BED format files."""
    _min_cols = 3

    def parse_row(self, parts: list[str]) -> Feature:
        """
        Parses a BED row.

        Args:
            parts: List of column strings.

        Returns:
            A Feature object.
        """
        start, end = int(parts[1]), int(parts[2])
        n_cols = len(parts)
        kind = parts[3] if n_cols > 3 else 'feature'
        score = float(parts[4]) if n_cols > 4 and parts[4] != '.' else 0.0
        strand = parts[5] if n_cols > 5 else '.'
        quals = []
        quals.append(Qualifier('source', parts[0]))
        if score: quals.append(Qualifier('score', score))
        if n_cols > 9: quals.append(Qualifier('blocks', ','.join(parts[9:])))
        return Feature(Interval(start, end, strand), kind, qualifiers=quals)


class PafReader(TabularReader):
    """Reader for PAF (Pairwise mApping Format) files."""
    _min_cols = 12

    def parse_row(self, parts: list[str]) -> Alignment:
        """
        Parses a PAF row.

        Args:
            parts: List of column strings.

        Returns:
            An Alignment object.
        """
        q_len, t_len, block_len = int(parts[1]), int(parts[6]), int(parts[10])
        n_matches = int(parts[9])
        cigar, score, quals = None, None, []
        for qual in self._QUALIFIER_PARSER.parse_tags(parts[12:]):
            if qual.key == 'cg': cigar = qual.value
            elif qual.key == 'AS': score = qual.value
            else: quals.append(qual)

        cigar = next((q.value for q in quals if q.key == 'cg'), None)
        return Alignment(
            query=parts[0], query_interval=Interval(int(parts[2]), int(parts[3])),
            target=parts[5], interval=Interval(int(parts[7]), int(parts[8]), parts[4]),
            query_length=q_len, target_length=t_len, length=block_len, score=score,
            cigar=cigar, n_matches=n_matches, quality=int(parts[11]), qualifiers=quals
        )


class GfaReader(BaseReader):
    """Reader for GFA (Graphical Fragment Assembly) files."""

    def __init__(self, handle: TextIO, alphabet: Alphabet = None, min_seq_length: int = 1, **kwargs):
        """
        Initializes the GfaReader.

        Args:
            handle: Open file handle.
            alphabet: Alphabet to use for sequences.
            min_seq_length: Minimum sequence length to include.
            **kwargs: Additional arguments.
        """
        super().__init__(handle)
        self.alphabet = alphabet or self._DEFAULT_ALPHABET
        self.min_seq_length = min_seq_length

    def __iter__(self) -> Generator[Union[Record, Edge], None, None]:
        """
        Iterates over GFA lines (Segments as Records, Links as Edges).

        Yields:
            Record or Edge objects.
        """
        for line in self._handle:
            if line.startswith('S\t'):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                name, seq = parts[1], parts[2]
                if len(seq) >= self.min_seq_length:
                    yield Record(
                        self.alphabet.seq(seq if seq != '*' else '', process_string=True),
                        name, qualifiers=self._QUALIFIER_PARSER.parse_tags(parts[3:] if len(parts) > 3 else [])
                    )
            elif line.startswith('L\t'):
                parts = line.strip().split('\t')
                if len(parts) < 6: continue
                yield Edge(parts[1], parts[3], {
                    'u_strand': parts[2], 'v_strand': parts[4],
                    'overlap': next((n for op, n, *_, in self._CIGAR_PARSER.parse(parts[5]) if op == 'M'), 0)
                })


class SeqFile:
    """
    Main interface for reading sequence files. Automatically detects format and compression.
    """
    _SUPPORTED_FORMATS = Literal['fasta', 'gfa', 'genbank', 'fastq', 'gff', 'bed', 'paf']
    _READERS = dict(zip(get_args(_SUPPORTED_FORMATS),
                        (FastaReader, GfaReader, GenbankReader, FastqReader, GffReader, BedReader, PafReader)))
    _FORMAT_SNIFFERS = [
        ('fasta', lambda s: s.startswith('>')),
        ('fastq', lambda s: s.startswith('@')),
        ('genbank', lambda s: 'LOCUS' in s[:100]),
        ('gfa', lambda s: s.startswith('S\t')),
        ('gff', lambda s: s.startswith('##gff')),
        ('paf', lambda s: s.count('\t') > 11),
        ('bed', lambda s: s.count('\t') >= 2)
    ]

    def __init__(self, file: Union[str, Path, IO], format_: _SUPPORTED_FORMATS = None, alphabet: Alphabet = None):
        """
        Initializes the SeqFile reader.

        Args:
            file: File path or object.
            format_: Specific format to use (optional).
            alphabet: Alphabet to use for sequences (optional).
        """
        self._xopen = Xopen(file, mode='r')
        self._handle = None
        self.format = format_
        self.alphabet = alphabet
        self._reader = None

    def __iter__(self) -> Generator[Union[Record, Feature, Edge, Alignment], None, None]:
        """
        Iterates over the content of the file.

        Yields:
            Parsed objects depending on the format.
        """
        self._handle = self._xopen.__enter__()
        peekable_handle = PeekableHandle(self._handle)

        if self.format is None:
            self.format = self._sniff_format(peekable_handle)

        if self.format not in self._READERS:
            raise SeqFileError(f"Cannot parse file: format '{self.format}' is unknown or unsupported.")

        self._reader = self._READERS[self.format](peekable_handle, alphabet=self.alphabet)
        yield from self._reader

    def _sniff_format(self, handle: PeekableHandle) -> str:
        """
        Detects the file format by peeking at the content.

        Args:
            handle: The peekable file handle.

        Returns:
            The detected format string.

        Raises:
            ValueError: If format cannot be determined.
        """
        peek_window = handle.peek(1024)
        for fmt, check in self._FORMAT_SNIFFERS:
            if check(peek_window): return fmt
        raise ValueError("Could not determine format from stream content.")

    def close(self):
        """Closes the file."""
        self._xopen.__exit__(None, None, None)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Writers --------------------------------------------------------------------------------------------------------------
class BaseWriter(ABC):
    """Abstract base class for sequence file writers."""

    def __init__(self, file: Union[str, Path, IO], mode: str = 'w', compression: str = 'guess', **kwargs):
        """
        Initializes the writer.

        Args:
            file: File path or object.
            mode: File mode ('w' or 'a').
            compression: Compression method ('guess', 'gzip', etc.).
            **kwargs: Additional arguments passed to Xopen.
        """
        self._xopen = Xopen(file, mode=mode, **kwargs)
        self._handle = None

    def __enter__(self):
        """Opens the file and writes the header."""
        self._handle = self._xopen.__enter__()
        self.write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the file."""
        self._xopen.__exit__(exc_type, exc_val, exc_tb)

    def write(self, *records: Union[Record, Feature]):
        """
        Writes multiple records or features.

        Args:
            *records: Variable number of Record or Feature objects.
        """
        for item in records: self.write_one(item)

    @abstractmethod
    def write_one(self, item: Union[Record, Feature]):
        """
        Writes a single item.

        Args:
            item: Record or Feature to write.
        """
        pass

    def write_header(self):
        """Writes the file header if applicable."""
        pass


class FastaWriter(BaseWriter):
    """Writer for FASTA format files."""

    def __init__(self, file: Union[str, Path, IO], width: int = 60, **kwargs):
        """
        Initializes the FastaWriter.

        Args:
            file: File path or object.
            width: Line width for sequence wrapping (0 for no wrapping).
            **kwargs: Additional arguments.
        """
        super().__init__(file, **kwargs)
        self.width = width

    def write_one(self, record: Record):
        """
        Writes a single FASTA record.

        Args:
            record: The Record object to write.
        """
        if not isinstance(record, Record): raise TypeError("FastaWriter expects Record objects")
        self._handle.write(f">{record.id} {record.description}\n")
        seq_str = str(record.seq)
        if self.width > 0:
            for i in range(0, len(seq_str), self.width):
                self._handle.write(seq_str[i:i + self.width] + "\n")
        else:
            self._handle.write(seq_str + "\n")


class GfaWriter(BaseWriter):
    """Writer for GFA format files."""

    def write_one(self, item: Union[Record, Edge]):
        """
        Writes a Record (Segment) or Edge (Link).

        Args:
            item: Record or Edge object.
        """
        if isinstance(item, Record):
            self._handle.write(format(item, 'gfa'))
        elif isinstance(item, Edge):
            cigar = item.data.get('cigar', '0M')
            self._handle.write(f"L\t{item.u}\t{item.data.get('u_strand', '+')}\t"
                               f"{item.v}\t{item.data.get('v_strand', '+')}\t{cigar}\n")


class GffWriter(BaseWriter):
    """Writer for GFF3 format files."""

    def write_header(self):
        """Writes the GFF3 header."""
        self._handle.write("##gff-version 3\n")

    def write_one(self, record: Record):
        """
        Writes a Record and its features in GFF3 format.

        Args:
            record: The Record object.
        """
        if not isinstance(record, Record): raise TypeError("GffWriter expects Record objects")
        self._handle.write(f"##sequence-region {record.id} 1 {len(record)}\n")
        for feature in record.features: self._write_feature(record.id, feature)

    def _write_feature(self, seq_id: str, feature: Feature):
        """
        Writes a single feature.

        Args:
            seq_id: ID of the sequence containing the feature.
            feature: The Feature object.
        """
        source = feature['source'] or 'baclib'
        start = feature.interval.start + 1
        end = feature.interval.end
        score = feature['score'] or '.'
        strand = feature.interval.decode_sense()
        phase = feature['phase'] or '.'

        attr_strings = []
        if val := feature['ID']: attr_strings.append(f"ID={val}")
        if val := feature['Name']: attr_strings.append(f"Name={val}")
        for q in feature.qualifiers:
            if q.key in {'source', 'score', 'phase', 'ID', 'Name'}: continue
            safe_val = str(q.value).replace(';', '%3B').replace('=', '%3D').replace('&', '%26')
            attr_strings.append(f"{q.key}={safe_val}")
        attr_block = ";".join(attr_strings) if attr_strings else "."

        self._handle.write(f"{seq_id}\t{source}\t{feature.kind}\t{start}\t{end}\t"
                           f"{score}\t{strand}\t{phase}\t{attr_block}\n")


class BedWriter(BaseWriter):
    """Writer for BED format files."""

    def write_one(self, record: Record):
        """
        Writes features of a Record in BED format.

        Args:
            record: The Record object.
        """
        if not isinstance(record, Record): raise TypeError("BedWriter expects Record objects")
        for feature in record.features:
            name = feature['Name'] or feature['ID'] or feature['gene'] or feature.kind
            score = feature['score'] or 0
            strand = feature.interval.decode_sense()
            self._handle.write(f"{record.id}\t{feature.interval.start}\t{feature.interval.end}\t"
                               f"{name}\t{score}\t{strand}\n")
