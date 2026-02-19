"""
Module for parsing and managing bacterial sequence files and data.
"""
from abc import ABC, abstractmethod
from typing import Union, Generator, BinaryIO, Callable, Type
from enum import Enum
from pathlib import Path
from urllib.parse import unquote_to_bytes

import numpy as np

from baclib.core.alphabet import Alphabet
from baclib.containers.seq import SeqBatch
from baclib.containers.graph import Edge
from baclib.containers.record import Record, Feature, RecordBatch
from baclib.containers.alignment import Alignment
from baclib.containers import Batch
from baclib.lib.io import Xopen, PeekableHandle, ThreadedChunkReader, ThreadedChunkWriter


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class SeqIOError(IOError):
    """Base class for sequence I/O errors."""

class TruncatedFileError(SeqIOError):
    """Raised when a file appears to be truncated."""

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


class BaseReader(ABC):
    """Abstract base class for sequence file readers."""
    _CHUNK_SIZE = 65536
    __slots__ = ('_handle', '_iterator')
    def __init__(self, handle: BinaryIO, **kwargs):
        """
        Initializes the reader.

        Args:
            handle: The open file handle to read from.
            **kwargs: Additional arguments.
        """
        self._handle = handle
        self._iterator = None
    
    @classmethod
    @abstractmethod
    def sniff(cls, s: bytes) -> bool: ...
    @abstractmethod
    def __iter__(self) -> Generator: ...
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def __next__(self):
        if self._iterator is None:
            self._iterator = self.__iter__()
        return next(self._iterator)

    def close(self):
        """Closes the reader."""
        pass

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
             
        # Optimization: Bulk encode if no deletions are required
        # This avoids overhead of N translate calls and N numpy array allocations
        if len(alphabet._delete_bytes) == 0:
            lengths = np.array([len(s) for s in seq_bytes_list], dtype=np.int32)
            n = len(lengths)
            starts = np.zeros(n, dtype=np.int32)
            if n > 1: np.cumsum(lengths[:-1], out=starts[1:])
            
            encoded_data = alphabet.encode(b"".join(seq_bytes_list))
            return alphabet.new_batch(encoded_data, starts, lengths)

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
    def __init__(self, file: Union[str, Path, BinaryIO], mode: Union[str, Xopen.Mode] = Xopen.Mode.WRITE,
                 compression: Union[str, Xopen.Format] = Xopen.Format.INFER,
                 threaded: bool = True, **kwargs):
        """
        Initializes the writer.
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


class FormatSpec:
    """Metadata for a supported file format."""
    __slots__ = ('extensions', 'alphabets', 'reader', 'writer')
    def __init__(self, extensions: list[str] = None, alphabets: dict[str, 'Alphabet'] = None):
        self.extensions = extensions or []
        self.alphabets = alphabets or {}
        self.reader: Type[BaseReader] = None
        self.writer: Type[BaseWriter] = None


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
    VCF = 'vcf'
    EMBL = 'embl'


class SeqFile:
    """
    Main interface for reading sequence files.

    Automatically detects file format (e.g., FASTA, GenBank, GFF) and
    compression (e.g., gzip, bzip2), providing a single, unified way to
    iterate over sequence data.
    """
    __slots__ = ('_opener', '_handle', '_format', '_reader', '_reader_kwargs', '_iterator')
    _REGISTRY = {}
    Format = SeqFileFormat
    def __init__(self, file: Union[str, Path, BinaryIO], fmt: Union[str, Format] = None,
                 alphabet: Alphabet = None, **reader_kwargs):
        """
        Initializes the SeqFile reader.
        """
        self._opener = Xopen(file, mode='rb')
        self._handle = None
        self._format = self.Format(fmt) if fmt else None
        self._reader = None
        self._reader_kwargs = reader_kwargs
        self._iterator = None

        # Auto-detect Alphabet from extension if not provided
        if alphabet is None and isinstance(file, (str, Path)) and self._format:
            if spec := self._REGISTRY.get(self._format):
                path_str = str(file)
                for ext, alpha in spec.alphabets.items():
                    if path_str.endswith(ext):
                        alphabet = alpha
                        break

        if alphabet: self._reader_kwargs['alphabet'] = alphabet

    @property
    def format(self) -> Format: return self._format
    @property
    def name(self) -> str: return self._opener.name

    @classmethod
    def open(cls, file: Union[str, Path, BinaryIO], mode: Union[str, OpenMode] = OpenMode.READ,
             fmt: Union[str, Format] = None, alphabet: Alphabet = None, **kwargs) -> Union['SeqFile', BaseWriter]:
        if fmt is None:
            fmt = kwargs.pop('format', None)
        
        mode_str = str(mode)
        if 'w' in mode_str or 'a' in mode_str or 'x' in mode_str:
            # Writing
            if fmt is None:
                 # Try to infer from filename
                 path = Path(str(file)) # naive
                 if isinstance(file, (str, Path)):
                      path = Path(file)
                      name = path.name.lower()
                      # Remove compression ext
                      if name.endswith(('.gz', '.bz2', '.xz', '.zst')):
                           name = path.stem.lower()
                      
                      for f, spec in cls._REGISTRY.items():
                           if any(name.endswith(ext) for ext in spec.extensions):
                               fmt = f
                               break
            
            if fmt is None:
                raise ValueError("Format must be specified for writing if it cannot be inferred from extension.")
            
            spec = cls._REGISTRY.get(cls.Format(fmt))
            if not spec or not spec.writer:
                raise ValueError(f"No writer available for format {fmt}")
            
            return spec.writer(file, mode=mode, **kwargs)
            
        return cls(file, fmt=fmt, alphabet=alphabet, **kwargs)

    def __iter__(self) -> Generator[Union[Record, Feature, Batch], None, None]:
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

            self._reader = self._REGISTRY[self._format].reader(handle_to_use, **self._reader_kwargs)
            yield from self._reader
        finally:
            if close_on_exit:
                self.close()

    def __next__(self):
        if self._iterator is None:
            self._iterator = self.__iter__()
        return next(self._iterator)

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

            self._reader = self._REGISTRY[self._format].reader(handle_to_use, **self._reader_kwargs)
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
        for fmt, spec in self._REGISTRY.items():
            if spec.reader and spec.reader.sniff(peek_window): return fmt
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

    @classmethod
    def register(cls, fmt: Format, extensions: list[str] = None, alphabets: dict[str, 'Alphabet'] = None):
        """
        Decorator to register a Reader or Writer for a specific format.
        """

        def decorator(func: Callable) -> Callable:
            if (spec := cls._REGISTRY.get(fmt)) is None:
                spec = FormatSpec()
                cls._REGISTRY[fmt] = spec

            if extensions:
                [spec.extensions.append(ext) for ext in extensions if ext not in spec.extensions]

            if alphabets: spec.alphabets.update(alphabets)

            if issubclass(func, BaseReader):
                spec.reader = func
            elif issubclass(func, BaseWriter):
                spec.writer = func
            return func

        return decorator

# Import submodules to populate the registry
from baclib.io import seq, tabular, genbank, motif
