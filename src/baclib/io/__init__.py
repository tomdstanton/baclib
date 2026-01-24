"""
Module for parsing and managing bacterial sequence files and data.
"""
from abc import ABC, abstractmethod
from typing import Union, Generator, BinaryIO
from pathlib import Path

from baclib.core.seq import Alphabet
from baclib.containers.graph import Edge
from baclib.containers.record import Record, Feature
from baclib.align.alignment import Alignment
from baclib.io.open import Xopen


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class ParserError(Exception): pass


# Helpers --------------------------------------------------------------------------------------------------------------
class QualifierParser:
    """
    Parses qualifier strings from various file formats.
    """
    _TYPE_MAP = {b'f': float, b'i': int, b'Z': bytes, b'A': bytes, b'H': bytes, b'B': bytes}
    _ABBREVIATIONS = {b'dp': b'depth', b'ln': b'length', b'kc': b'kmer count'}

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
            val = (val.replace(b'%3B', b';').replace(b'%3D', b'=')
                   .replace(b'%26', b'&').replace(b'%2C', b','))
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


# Base Readers ---------------------------------------------------------------------------------------------------------
class BaseReader(ABC):
    """Abstract base class for sequence file readers."""
    _QUALIFIER_PARSER = QualifierParser()
    _DEFAULT_ALPHABET = Alphabet.dna()
    _CHUNK_SIZE = 65536

    def __init__(self, handle: BinaryIO, **kwargs):
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
    _delim = b'\t'
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
        
        # Optimization: Read large binary chunks
        read = self._handle.read
        
        buf = b""
        while True:
            chunk = read(self._CHUNK_SIZE)
            if not chunk:
                if buf:
                    line = buf.rstrip()
                    if line and not line.startswith(b'#'):
                        parts = line.split(delim)
                        if len(parts) >= min_cols:
                            yield parse(parts)
                break
            
            buf += chunk
            pos = 0
            
            while True:
                nl_pos = buf.find(b'\n', pos)
                if nl_pos == -1:
                    buf = buf[pos:]
                    break
                
                line = buf[pos:nl_pos].rstrip()
                pos = nl_pos + 1
                
                if not line or line.startswith(b'#'): continue
                
                parts = line.split(delim)
                if len(parts) < min_cols: continue
                yield parse(parts)

    @abstractmethod
    def parse_row(self, parts: list[bytes]) -> Union[Feature, Alignment]:
        """
        Parses a single row split by delimiter.

        Args:
            parts: List of column bytes.

        Returns:
            A Feature or Alignment object.
        """
        pass


# Writers --------------------------------------------------------------------------------------------------------------
class BaseWriter(ABC):
    """Abstract base class for sequence file writers."""

    def __init__(self, file: Union[str, Path, BinaryIO], mode: str = 'wb', compression: str = 'guess', **kwargs):
        """
        Initializes the writer.

        Args:
            file: File path or object.
            mode: File mode ('wb' or 'ab').
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
