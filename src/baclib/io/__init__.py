"""
Module for parsing and managing bacterial sequence files and data.
"""
from abc import ABC, abstractmethod
from typing import Union, Generator, BinaryIO, Callable
from pathlib import Path
from urllib.parse import unquote_to_bytes

from baclib.core.seq import Alphabet
from baclib.containers.graph import Edge, EdgeBatch
from baclib.containers.record import Record, Feature, RecordBatch, FeatureBatch
from baclib.align.alignment import Alignment, AlignmentBatch
from baclib.utils import Xopen, PeekableHandle


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class ParserError(Exception): pass
class SeqFileError(Exception):
    """Exception raised for errors in sequence file processing."""
    pass


# Classes --------------------------------------------------------------------------------------------------------------
class QualifierParser:
    """
    Parses qualifier strings from various file formats.
    """
    _TYPE_MAP = {b'f': float, b'i': int, b'Z': bytes, b'A': bytes, b'H': bytes, b'B': bytes}
    _ABBREVIATIONS = {b'dp': b'depth', b'ln': b'length', b'kc': b'kmer count'}
    # _TOPOLOGY_REGEX = regex(rb'(?i)(\bcircular\b|\bcircular\s*=\s*true\b)')
    # _COPY_NUMBER_REGEX = regex(rb'depth=(\d+\.\d+)')

    def parse_gff_attributes(self, items: bytes) -> list[tuple]:
        """
        Parses GFF3 attribute strings.

        Args:
            items: The attribute string (e.g., "ID=gene1;Name=foo").

        Returns:
            A list of Qualifier objects.
        """
        if not items or items == b'.': return []
        quals = []
        for item in items.split(b';'):
            if not item: continue
            key, _, val = item.partition(b'=')
            if not key: continue
            val = unquote_to_bytes(val)
            quals.append((key, val))
        return quals

    def parse_tags(self, items: list[bytes]) -> list[tuple]:
        """
        Parses SAM/PAF style tags.

        Args:
            items: List of tag bytes (e.g., ["NM:i:0", "AS:f:100"]).

        Returns:
            A list of Qualifier objects.
        """
        quals = []
        for item in items:
            if len(item) < 5: continue
            parts = item.split(b':', 2)
            if len(parts) != 3: continue
            tag, typ, val = parts
            tag = self._ABBREVIATIONS.get(tag.lower(), tag)
            converter = self._TYPE_MAP.get(typ, bytes)
            try: quals.append((tag, converter(val)))
            except ValueError: continue
        return quals

    # def parse_header(self, header: bytes) -> Generator[tuple, None, None]:
    #     (b'topology', b'circular' if self._TOPOLOGY_REGEX.search(desc) else b'linear'),
    #     (b'depth', float(m[1]) if (m := self._COPY_NUMBER_REGEX.search(desc)) else 1.0)


class BaseReader(ABC):
    """Abstract base class for sequence file readers."""
    _QUALIFIER_PARSER = QualifierParser()
    _DEFAULT_ALPHABET = Alphabet.dna()
    _CHUNK_SIZE = 65536
    __slots__ = ('_handle',)
    def __init__(self, handle: BinaryIO, **kwargs):
        """
        Initializes the reader.

        Args:
            handle: The open file handle to read from.
            **kwargs: Additional arguments.
        """
        self._handle = handle

    @abstractmethod
    def __iter__(self) -> Generator: ...
    
    @classmethod
    @abstractmethod
    def sniff(cls, s: bytes) -> bool: ...

    def batches(self, size: int = 1024) -> Generator[RecordBatch, None, None]:
        """
        Yields records in batches for efficiency.

        Args:
            size: Number of records per batch.

        Yields:
            RecordBatch objects.
        """
        batch_records = []
        for record in self:
            batch_records.append(record)
            if len(batch_records) >= size:
                yield self._make_batch(batch_records)
                batch_records = []
        if batch_records:
            yield self._make_batch(batch_records)

    def _make_batch(self, records: list) -> RecordBatch:
        return RecordBatch(records)


class BaseWriter(ABC):
    """
    Abstract base class for sequence file writers.

    Examples:
        >>> with FastaWriter("output.fasta") as w:
        ...     w.write(record1, record2)
    """
    __slots__ = ('_opener', '_handle',)
    def __init__(self, file: Union[str, Path, BinaryIO], mode: str = 'wb', compression: str = 'guess', **kwargs):
        """
        Initializes the writer.

        Args:
            file: File path or object.
            mode: File mode ('wb' or 'ab').
            compression: Compression method ('guess', 'gzip', etc.).
            **kwargs: Additional arguments passed to Xopen.
        """
        self._opener = Xopen(file, mode=mode, **kwargs)
        self._handle = None

    def __enter__(self):
        """Opens the file and writes the header."""
        self._handle = self._opener.__enter__()
        self.write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the file."""
        self._opener.__exit__(exc_type, exc_val, exc_tb)

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


class SeqFile:
    """
    Main interface for reading sequence files.

    Automatically detects file format (e.g., FASTA, GenBank, GFF) and
    compression (e.g., gzip, bzip2), providing a single, unified way to
    iterate over sequence data.

    Examples:
        >>> # Reading a FASTA file
        >>> with SeqFile("genome.fasta") as f:
        ...     for record in f:
        ...         print(record.id)
    """
    _REGISTRY = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable) -> Callable:
            cls._REGISTRY[name] = func
            return func
        return decorator

    __slots__ = ('_opener', '_handle', '_format', '_reader', '_reader_kwargs')
    def __init__(self, file: Union[str, Path, BinaryIO], format_: str = None, **reader_kwargs):
        """
        Initializes the SeqFile reader.

        Args:
            file: File path or object.
            format_: Specific format to use (optional).
            alphabet: Alphabet to use for sequences (optional).
        """
        self._opener = Xopen(file, mode='rb')
        self._handle = None
        self._format = format_
        self._reader = None
        self._reader_kwargs = reader_kwargs

    def __iter__(self) -> Generator[Union[Record, Feature, Edge, Alignment], None, None]:
        """
        Iterates over the content of the file.

        Yields:
            Parsed objects depending on the format.
        """
        # Auto-open if not used as a context manager
        close_on_exit = False
        if self._handle is None:
            self._handle = self._opener.__enter__()
            close_on_exit = True

        handle_to_use = self._handle

        try:
            if self._format is None:
                # Only wrap in PeekableHandle if we actually need to sniff
                peekable_handle = PeekableHandle(self._handle)
                self._format = self._sniff_format(peekable_handle)
                handle_to_use = peekable_handle

            if self._format not in self._REGISTRY:
                raise SeqFileError(f"Cannot parse file: format '{self._format}' is unknown or unsupported.")

            self._reader = self._REGISTRY[self._format](handle_to_use, **self._reader_kwargs)
            yield from self._reader
        finally:
            if close_on_exit:
                self.close()

    def batches(self, size: int = 1024) -> Generator[Union[RecordBatch, FeatureBatch, AlignmentBatch, EdgeBatch], None, None]:
        """
        Iterates over the content of the file in batches.

        Args:
            size: Number of records per batch.

        Yields:
            Batch objects depending on the format.
        """
        # Auto-open if not used as a context manager
        close_on_exit = False
        if self._handle is None:
            self._handle = self._opener.__enter__()
            close_on_exit = True

        handle_to_use = self._handle

        try:
            if self._format is None:
                # Only wrap in PeekableHandle if we actually need to sniff
                peekable_handle = PeekableHandle(self._handle)
                self._format = self._sniff_format(peekable_handle)
                handle_to_use = peekable_handle

            if self._format not in self._REGISTRY:
                raise SeqFileError(f"Cannot parse file: format '{self._format}' is unknown or unsupported.")

            self._reader = self._REGISTRY[self._format](handle_to_use, **self._reader_kwargs)
            yield from self._reader.batches(size)
        finally:
            if close_on_exit:
                self.close()

    def _sniff_format(self, handle: PeekableHandle) -> BaseReader:
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
        for fmt, reader in self._REGISTRY.items():
            if reader.sniff(peek_window): return fmt
        raise ValueError("Could not determine format from stream content.")

    def close(self):
        """Closes the file."""
        self._opener.__exit__(None, None, None)

    def __enter__(self):
        """Context manager entry."""
        self._handle = self._opener.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Import submodules to populate the registry
from baclib.io import seq, tabular, genbank, motif
