from pathlib import Path
from typing import Union, Generator, BinaryIO
from re import compile as regex

from baclib.core.seq import Alphabet
from baclib.containers.record import Record
from baclib.containers.graph import Edge
from baclib.io import BaseReader, BaseWriter, ParserError


# Classes --------------------------------------------------------------------------------------------------------------
class FastaWriter(BaseWriter):
    """
    Writer for FASTA format files.

    Examples:
        >>> with FastaWriter("output.fasta") as w:
        ...     w.write_one(record)
    """

    def __init__(self, file: Union[str, Path, BinaryIO], width: int = 60, **kwargs):
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
        
        header = b">" + record.id
        if record.description: header += b" " + record.description
        self._handle.write(header + b"\n")
        
        encoded = record.seq.encoded
        width = self.width
        
        if width > 0:
            alphabet = record.seq.alphabet
            for i in range(0, len(encoded), width):
                self._handle.write(alphabet.decode(encoded[i:i + width]))
                self._handle.write(b"\n")
        else:
            self._handle.write(bytes(record.seq) + b"\n")


class FastaReader(BaseReader):
    """
    Reader for FASTA format files.

    Examples:
        >>> with open("genome.fasta", "rb") as f:
        ...     reader = FastaReader(f)
        ...     for record in reader:
        ...         print(record.id)
    """
    _TOPOLOGY_REGEX = regex(rb'(?i)(\bcircular\b|\bcircular\s*=\s*true\b)')
    _COPY_NUMBER_REGEX = regex(rb'depth=(\d+\.\d+)')

    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None, **kwargs):
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
        # Optimization: Read large binary chunks (Seqtk style)
        # This avoids the overhead of iterating line-by-line in Python
        
        read = self._handle.read
        encode = self.alphabet.encode
        seq_cls = self.alphabet.seq
        
        buf = b""
        header = None
        seq_parts = []
        
        while True:
            chunk = read(self._CHUNK_SIZE)
            if not chunk:
                if header is not None:
                    if buf: seq_parts.append(buf)
                    yield self._make_record(header, seq_parts, encode, seq_cls)
                break
            
            buf += chunk
            pos = 0
            
            while True:
                gt_pos = buf.find(b'>', pos)
                
                if gt_pos == -1:
                    if header is not None:
                        seq_parts.append(buf[pos:])
                    buf = b""
                    break
                
                if header is not None:
                    seq_parts.append(buf[pos:gt_pos])
                    yield self._make_record(header, seq_parts, encode, seq_cls)
                    seq_parts = []
                    header = None
                
                nl_pos = buf.find(b'\n', gt_pos)
                if nl_pos == -1:
                    buf = buf[gt_pos:]
                    break
                
                header = buf[gt_pos+1:nl_pos].rstrip()
                pos = nl_pos + 1

    def _make_record(self, header, seq_parts, encode, seq_cls):
        name, _, desc = header.partition(b' ')
        full_seq = b"".join(seq_parts)
        # Encode the whole sequence at once (releases GIL in Numba)
        encoded = encode(full_seq)
        
        return Record(
            seq_cls(encoded),
            name, desc,
            qualifiers=[
                (b'topology', b'circular' if self._TOPOLOGY_REGEX.search(desc) else b'linear'),
                (b'depth', float(m[1]) if (m := self._COPY_NUMBER_REGEX.search(desc)) else 1.0)
            ]
        )


class GfaWriter(BaseWriter):
    """
    Writer for GFA format files.

    Examples:
        >>> with GfaWriter("graph.gfa") as w:
        ...     w.write_one(record)
        ...     w.write_one(edge)
    """

    def write_one(self, item: Union[Record, Edge]):
        """
        Writes a Record (Segment) or Edge (Link).

        Args:
            item: Record or Edge object.
        """
        if isinstance(item, Record):
            self._handle.write(format(item, 'gfa').encode('ascii'))
        elif isinstance(item, Edge):
            cigar = item.attributes.get(b'cigar', b'0M')
            self._handle.write(b"L\t" + item.u + b"\t" + item.attributes.get(b'u_strand', b'+') + b"\t" +
                               item.v + b"\t" + item.attributes.get(b'v_strand', b'+') + b"\t" + cigar + b"\n")


class GfaReader(BaseReader):
    """
    Reader for GFA (Graphical Fragment Assembly) files.

    Examples:
        >>> with open("graph.gfa", "rb") as f:
        ...     reader = GfaReader(f)
        ...     for item in reader:
        ...         if isinstance(item, Record): print("Segment:", item.id)
        ...         elif isinstance(item, Edge): print("Link:", item.u, "->", item.v)
    """

    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None, min_seq_length: int = 1, **kwargs):
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
        # Optimization: Read large binary chunks
        read = self._handle.read
        seq_factory = self.alphabet.seq
        parse_tags = self._QUALIFIER_PARSER.parse_tags
        min_len = self.min_seq_length

        buf = b""
        while True:
            chunk = read(self._CHUNK_SIZE)
            if not chunk:
                if buf and buf.strip():
                    yield from self._parse_line(buf, seq_factory, parse_tags, min_len)
                break

            buf += chunk
            pos = 0

            while True:
                nl_pos = buf.find(b'\n', pos)
                if nl_pos == -1:
                    buf = buf[pos:]
                    break

                line = buf[pos:nl_pos]
                yield from self._parse_line(line, seq_factory, parse_tags, min_len)
                pos = nl_pos + 1

    def _parse_line(self, line: bytes, seq_factory, parse_tags, min_len):
        line = line.rstrip()
        if not line: return
        first = line[0]

        if first == 83:  # 'S'
            parts = line.split(b'\t')
            if len(parts) < 3: return

            name, seq_bytes = parts[1], parts[2]

            if len(seq_bytes) >= min_len:
                seq = seq_factory(seq_bytes if seq_bytes != b'*' else b'')
                quals = parse_tags(parts[3:]) if len(parts) > 3 else []
                yield Record(seq, name, qualifiers=quals)

        elif first == 76:  # 'L'
            parts = line.split(b'\t')
            if len(parts) < 6: return
            yield Edge(parts[1], parts[3], {b'u_strand': parts[2], b'v_strand': parts[4], b'cigar': parts[5]})


class FastqReader(BaseReader):
    """
    Reader for FASTQ format files.

    Examples:
        >>> with open("reads.fastq", "rb") as f:
        ...     reader = FastqReader(f)
        ...     for record in reader:
        ...         print(record.id)
    """

    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None, **kwargs):
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
        # Optimization: Read large binary chunks
        read = self._handle.read
        encode = self.alphabet.encode
        seq_cls = self.alphabet.seq

        buf = b""
        while True:
            chunk = read(self._CHUNK_SIZE)
            if not chunk:
                if buf.strip():
                    # Ensure last line has a newline to simplify parsing logic
                    if not buf.endswith(b'\n'): buf += b'\n'
                else:
                    break
            else:
                buf += chunk

            pos = 0
            n_len = len(buf)

            while pos < n_len:
                # Skip whitespace between records
                while pos < n_len and buf[pos] in (10, 13, 32, 9):
                    pos += 1

                if pos >= n_len: break

                # Start of record check
                if buf[pos] != 64:  # @
                    raise ParserError(f"Invalid FASTQ header at byte {pos}: expected '@'")

                # Find 4 newlines
                nl1 = buf.find(b'\n', pos)
                if nl1 == -1: break

                nl2 = buf.find(b'\n', nl1 + 1)
                if nl2 == -1: break

                nl3 = buf.find(b'\n', nl2 + 1)
                if nl3 == -1: break

                nl4 = buf.find(b'\n', nl3 + 1)
                if nl4 == -1: break

                # Extract fields
                header = buf[pos + 1:nl1].rstrip()
                seq_bytes = buf[nl1 + 1:nl2].rstrip()
                qual_bytes = buf[nl3 + 1:nl4].rstrip()

                name, _, desc = header.partition(b' ')

                yield Record(seq_cls(encode(seq_bytes)), name, desc, qualifiers=[(b'quality', qual_bytes)])

                pos = nl4 + 1

            if pos > 0: buf = buf[pos:]
            if not chunk: break
