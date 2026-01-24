from typing import Union, Generator, get_args, Literal, BinaryIO
from pathlib import Path

from baclib.containers.graph import Edge
from baclib.containers.record import Record, Feature
from baclib.core.seq import Alphabet
from baclib.align.alignment import Alignment
from baclib.io.tabular import BedReader, GffReader
from baclib.io.seq import FastaReader, FastqReader, GfaReader
from baclib.io.genbank import GenbankReader
from baclib.io.open import Xopen, PeekableHandle
from baclib.io.align import PafReader


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class SeqFileError(Exception):
    """Exception raised for errors in sequence file processing."""
    pass


# Sniffers -------------------------------------------------------------------------------------------------------------
def _sniff_bed(s: bytes) -> bool:
    try:
        for line in s.splitlines():
            if not line.strip() or line.startswith(b'track') or line.startswith(b'browser') or line.startswith(b'#'):
                continue
            parts = line.split(b'\t')
            return len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit()
        return False
    except Exception: return False

def _sniff_paf(s: bytes) -> bool:
    try:
        line = s.split(b'\n', 1)[0]
        parts = line.split(b'\t')
        return (len(parts) >= 12 and parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit() and
                parts[6].isdigit() and parts[7].isdigit() and parts[8].isdigit())
    except Exception: return False


# Classes --------------------------------------------------------------------------------------------------------------
class SeqFile:
    """
    Main interface for reading sequence files.

    Automatically detects file format (e.g., FASTA, GenBank, GFF) and
    compression (e.g., gzip, bzip2), providing a single, unified way to
    iterate over sequence data.
    """
    _SUPPORTED_FORMATS = Literal['fasta', 'gfa', 'genbank', 'fastq', 'gff', 'bed', 'paf']
    _READERS = dict(zip(get_args(_SUPPORTED_FORMATS),
                        (FastaReader, GfaReader, GenbankReader, FastqReader, GffReader, BedReader, PafReader)))
    _FORMAT_SNIFFERS = [
        ('fasta', lambda s: s.startswith(b'>')),
        ('fastq', lambda s: s.startswith(b'@')),
        ('genbank', lambda s: b'LOCUS' in s[:100]),
        ('gfa', lambda s: s.startswith(b'H\t') or s.startswith(b'S\t')),
        ('gff', lambda s: s.startswith(b'##gff')),
        ('paf', _sniff_paf),
        ('bed', _sniff_bed)
    ]

    def __init__(self, file: Union[str, Path, BinaryIO], format_: _SUPPORTED_FORMATS = None, alphabet: Alphabet = None):
        """
        Initializes the SeqFile reader.

        Args:
            file: File path or object.
            format_: Specific format to use (optional).
            alphabet: Alphabet to use for sequences (optional).
        """
        self._xopen = Xopen(file, mode='rb')
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
        # Auto-open if not used as a context manager
        close_on_exit = False
        if self._handle is None:
            self._handle = self._xopen.__enter__()
            close_on_exit = True

        handle_to_use = self._handle

        try:
            if self.format is None:
                # Only wrap in PeekableHandle if we actually need to sniff
                peekable_handle = PeekableHandle(self._handle)
                self.format = self._sniff_format(peekable_handle)
                handle_to_use = peekable_handle

            if self.format not in self._READERS:
                raise SeqFileError(f"Cannot parse file: format '{self.format}' is unknown or unsupported.")

            self._reader = self._READERS[self.format](handle_to_use, alphabet=self.alphabet)
            yield from self._reader
        finally:
            if close_on_exit:
                self.close()

    def _sniff_format(self, handle: PeekableHandle) -> _SUPPORTED_FORMATS:
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
        self._handle = self._xopen.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
