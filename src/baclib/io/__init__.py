"""
Module for parsing and managing bacterial sequence files and data.
"""
from abc import ABC, abstractmethod
from typing import Union, Generator, BinaryIO, Callable, LiteralString
from pathlib import Path
from urllib.parse import unquote_to_bytes

import numpy as np

from baclib.core.seq import Alphabet, SeqBatch
from baclib.containers.graph import Edge, EdgeBatch
from baclib.containers.record import Record, Feature, RecordBatch, FeatureBatch
from baclib.containers.alignment import Alignment, AlignmentBatch
from baclib.utils import Xopen, PeekableHandle, ThreadedChunkReader, ThreadedChunkWriter, Batch


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class ParserError(Exception): pass
class SeqFileError(Exception):
    """Exception raised for errors in sequence file processing."""
    pass


# Classes --------------------------------------------------------------------------------------------------------------
class Qualifier:
    """
    Parses qualifier strings from various file formats.
    """
    _TYPE_MAP = {b'f': float, b'i': int}
    _ABBREVIATIONS = {b'dp': b'depth', b'ln': b'length', b'kc': b'kmer count'}
    # _TOPOLOGY_REGEX = regex(rb'(?i)(\bcircular\b|\bcircular\s*=\s*true\b)')
    # _COPY_NUMBER_REGEX = regex(rb'depth=(\d+\.\d+)')

    @staticmethod
    def parse_gff_attributes(items: bytes) -> list[tuple]:
        """
        Parses GFF3 attribute strings.

        Args:
            items: The attribute string (e.g., "ID=gene1;Name=foo").

        Returns:
            A list of Qualifier objects.
        """
        if not items or not b'=' in items: return []
        quals = []
        for item in items.split(b';'):
            if not item: continue
            key, _, val = item.partition(b'=')
            if not key: continue
            val = unquote_to_bytes(val)
            quals.append((key, val))
        return quals

    @classmethod
    def parse_tags(cls, items: list[bytes]) -> list[tuple]:
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
            tag = cls._ABBREVIATIONS.get(tag.lower(), tag)
            converter = cls._TYPE_MAP.get(typ, bytes)
            try: quals.append((tag, converter(val)))
            except ValueError: continue
        return quals

    # def parse_header(self, header: bytes) -> Generator[tuple, None, None]:
    #     (b'topology', b'circular' if self._TOPOLOGY_REGEX.search(desc) else b'linear'),
    #     (b'depth', float(m[1]) if (m := self._COPY_NUMBER_REGEX.search(desc)) else 1.0)


class BaseReader(ABC):
    """Abstract base class for sequence file readers."""
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

    def read_chunks(self, chunk_size: int = None) -> Generator[bytes, None, None]:
        """
        Yields chunks of data from the file handle.
        Uses a background thread for prefetching.
        """
        if chunk_size is None: chunk_size = self._CHUNK_SIZE
        return ThreadedChunkReader(self._handle, chunk_size)

    def batches(self, size: int = 1024) -> Generator[Batch, None, None]:
        """
        Yields records in batches for efficiency.

        Args:
            size: Number of records per batch.

        Yields:
            Batch objects.
        """
        batch_records = []
        for record in self:
            batch_records.append(record)
            if len(batch_records) >= size:
                yield self._make_batch(batch_records)
                batch_records = []
        if batch_records:
            yield self._make_batch(batch_records)

    def _make_batch(self, records: list) -> Batch:
        return RecordBatch(records)

    def _build_seq_batch(self, seq_bytes_list: list[bytes], alphabet: Alphabet) -> SeqBatch:
        """Helper to efficiently build a SeqBatch from a list of sequence bytes."""
        if not seq_bytes_list:
             return alphabet.new_batch(np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
             
        # 1. Encode individually to ensure lengths match the encoded data
        # (Alphabet.encode drops invalid chars like newlines, so raw length != encoded length)
        encoded_list = [alphabet.encode(s) for s in seq_bytes_list]

        # 2. Build Metadata
        n = len(encoded_list)
        lengths = np.array([len(s) for s in encoded_list], dtype=np.int32)
        starts = np.zeros(n, dtype=np.int32)
        if n > 1:
            np.cumsum(lengths[:-1], out=starts[1:])
            
        encoded_data = np.concatenate(encoded_list) if n > 0 else np.empty(0, dtype=np.uint8)

        return alphabet.new_batch(encoded_data, starts, lengths)

class BaseWriter(ABC):
    """
    Abstract base class for sequence file writers.

    Examples:
        >>> with FastaWriter("output.fasta") as w:
        ...     w.write(record1, record2)
    """
    __slots__ = ('_opener', '_handle', '_threaded', '_writer_wrapper', '_real_handle')
    def __init__(self, file: Union[str, Path, BinaryIO], mode: str = 'wb', compression: str = 'guess', 
                 threaded: bool = True, **kwargs):
        """
        Initializes the writer.

        Args:
            file: File path or object.
            mode: File mode ('wb' or 'ab').
            compression: Compression method ('guess', 'gzip', etc.).
            threaded: If True, uses a background thread for writing/compression.
            **kwargs: Additional arguments passed to Xopen.
        """
        self._opener = Xopen(file, mode=mode, **kwargs)
        self._handle = None
        self._threaded = threaded
        self._writer_wrapper = None
        self._real_handle = None

    def __enter__(self):
        """Opens the file and writes the header."""
        self._handle = self._opener.__enter__()
        if self._threaded:
            self._writer_wrapper = ThreadedChunkWriter(self._handle)
            self._real_handle = self._handle
            self._handle = self._writer_wrapper
        self.write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the file."""
        if self._writer_wrapper:
            self._writer_wrapper.close()
            self._handle = self._real_handle
        self._opener.__exit__(exc_type, exc_val, exc_tb)

    def write(self, *items: Union[Record, Feature, Batch, list, tuple]):
        """
        Writes multiple records or features.
        Automatically unpacks Batches and lists.

        Args:
            *items: Variable number of Record, Feature, Batch, or list objects.
        """
        for item in items:
            if isinstance(item, Batch):
                self.write_batch(item)
            elif isinstance(item, (list, tuple)):
                for sub_item in item: self.write_one(sub_item)
            else:
                self.write_one(item)

    def write_batch(self, batch: Batch):
        """Writes a batch of items. Subclasses can override for optimization."""
        for item in batch:
            self.write_one(item)

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


class SeqFileFormat(str, Enum):
    """Supported sequence file formats."""
    FASTA = 'fasta'
    FASTQ = 'fastq'
    GENBANK = 'genbank'
    GFA = 'gfa'
    GFF = 'gff'
    BED = 'bed'
    PAF = 'paf'
    MEME = 'meme'
    TRANSFAC = 'transfac'


class FormatSpec:
    """Metadata for a supported file format."""
    __slots__ = ('format', 'extensions', 'alphabets', 'reader', 'writer')

    def __init__(self, fmt: SeqFileFormat, extensions: list[str] = None, alphabets: dict[str, 'Alphabet'] = None):
        self.format = fmt
        self.extensions = extensions or []
        self.alphabets = alphabets or {}
        self.reader: Type[BaseReader] = None
        self.writer: Type[BaseWriter] = None


class SeqFile:
    """
    Main interface for reading sequence files.

    Automatically detects file format (e.g., FASTA, GenBank, GFF) and
    compression (e.g., gzip, bzip2), providing a single, unified way to
    iterate over sequence data.
    """
    _REGISTRY = {}

    @classmethod
    def register(cls, fmt: SeqFileFormat, extensions: list[str] = None, alphabets: dict[str, 'Alphabet'] = None):
        """
        Decorator to register a Reader or Writer for a specific format.
        """

        def decorator(func: Callable) -> Callable:
            if fmt not in cls._REGISTRY:
                cls._REGISTRY[fmt] = FormatSpec(fmt)

            spec = cls._REGISTRY[fmt]
            if extensions:
                for ext in extensions:
                    if ext not in spec.extensions: spec.extensions.append(ext)
            if alphabets:
                spec.alphabets.update(alphabets)

            if issubclass(func, BaseReader):
                spec.reader = func
            elif issubclass(func, BaseWriter):
                spec.writer = func
            return func

        return decorator

    __slots__ = ('_opener', '_handle', '_format', '_reader', '_reader_kwargs')

    def __init__(self, file: Union[str, Path, BinaryIO], format_: Union[str, SeqFileFormat] = None,
                 alphabet: Alphabet = None, **reader_kwargs):
        """
        Initializes the SeqFile reader.
        """
        self._opener = Xopen(file, mode='rb')
        self._handle = None

        self._format = SeqFileFormat(format_) if format_ else None
        self._reader = None
        self._reader_kwargs = reader_kwargs

        # Auto-detect Alphabet from extension if not provided
        if alphabet is None and isinstance(file, (str, Path)) and self._format:
            spec = self._REGISTRY.get(self._format)
            if spec:
                path_str = str(file)
                for ext, alpha in spec.alphabets.items():
                    if path_str.endswith(ext):
                        alphabet = alpha
                        break

        if alphabet:
            self._reader_kwargs['alphabet'] = alphabet

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
                # Optimization: Use native peek if available to avoid PeekableHandle overhead
                if hasattr(self._handle, 'peek'):
                    self._format = self._sniff_format(self._handle)
                else:
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

    def batches(self, size: int = 1024) -> Generator[Batch, None, None]:
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
                # Optimization: Use native peek if available
                if hasattr(self._handle, 'peek'):
                    self._format = self._sniff_format(self._handle)
                else:
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

    def _sniff_format(self, handle: Union[PeekableHandle, BinaryIO]) -> BaseReader:
        """
        Detects the file format by peeking at the content.

        Args:
            handle: The peekable file handle.

        Returns:
            The detected format string.

        Raises:
            ValueError: If format cannot be determined.
        """
        peek_window = handle.peek(1024)[:1024]
        for fmt, reader in self._REGISTRY.items():
            if reader.sniff(peek_window): return fmt
        raise ValueError("Could not determine format from stream content.")

    def close(self):
        """Closes the file."""
        self._opener.__exit__(None, None, None)
        self._handle = None

    def __enter__(self):
        """Context manager entry."""
        self._handle = self._opener.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Import submodules to populate the registry
from baclib.io import seq, tabular, genbank, motif
