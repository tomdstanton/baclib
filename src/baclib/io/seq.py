from pathlib import Path
from typing import Union, Generator, BinaryIO, Iterable

import numpy as np

from baclib.core.seq import Alphabet
from baclib.containers.record import Record, RecordBatch
from baclib.containers.graph import Edge, EdgeBatch
from baclib.io import BaseReader, BaseWriter, ParserError, SeqFile


# Classes --------------------------------------------------------------------------------------------------------------
class SeqReader(BaseReader):
    __slots__ = ('_alphabet', '_min_seq_length')
    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None, min_seq_length: int = 1, **kwargs):
        """ABC for readers that parse files with sequences (Fasta, Fastq and GFA)"""
        super().__init__(handle, **kwargs)
        self._alphabet = alphabet or self._DEFAULT_ALPHABET
        self._min_seq_length = min_seq_length


@SeqFile.register('fasta')
class FastaReader(SeqReader):
    """
    Reader for FASTA format files.

    Examples:
        >>> with open("genome.fasta", "rb") as f:
        ...     reader = FastaReader(f)
        ...     for record in reader:
        ...         print(record.id)
    """
    __slots__ = ()
    def __iter__(self) -> Generator[Record, None, None]:
        """
        Iterates over FASTA records.

        Yields:
            Record objects.
        """
        for header, seq_parts in self._read_entries():
            yield self._make_record(header, seq_parts)

    def batches(self, size: int = 1024) -> Generator[RecordBatch, None, None]:
        """
        Optimized batch reader that performs bulk encoding.
        """
        entries = []
        for entry in self._read_entries():
            entries.append(entry)
            if len(entries) >= size:
                yield self._make_batch(entries)
                entries = []
        if entries:
            yield self._make_batch(entries)

    def _read_entries(self):
        """Internal generator that yields (header, seq_parts_list)."""
        read = self._handle.read
        min_len = self._min_seq_length
        
        buf = b""
        header = None
        seq_parts = []
        
        while True:
            chunk = read(self._CHUNK_SIZE)
            if not chunk:
                if header is not None:
                    if buf: seq_parts.append(buf)
                    if sum(len(p) for p in seq_parts) >= min_len:
                        yield header, seq_parts
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
                    if sum(len(p) for p in seq_parts) >= min_len:
                        yield header, seq_parts
                    seq_parts = []
                    header = None
                
                nl_pos = buf.find(b'\n', gt_pos)
                if nl_pos == -1:
                    buf = buf[gt_pos:]
                    break
                
                header = buf[gt_pos+1:nl_pos].rstrip()
                pos = nl_pos + 1

    def _make_record(self, header, seq_parts: Iterable[bytes]) -> Record:
        name, _, desc = header.partition(b' ')
        # Encode the whole sequence at once (releases GIL in Numba)
        return Record(self._alphabet.seq(b"".join(seq_parts)), name, desc)

    def _make_batch(self, entries) -> RecordBatch:
        # 1. Bulk Encode
        # Join all parts of all sequences into one massive bytes string
        full_seqs = [b"".join(parts) for _, parts in entries]
        # Vectorized encoding of the entire batch at once
        encoded_data = self._alphabet.encode(b"".join(full_seqs))

        # 2. Build SeqBatch Metadata
        n = len(entries)
        lengths = np.array([len(s) for s in full_seqs], dtype=np.int32)
        starts = np.zeros(n, dtype=np.int32)
        if n > 1:
            np.cumsum(lengths[:-1], out=starts[1:])

        batch = self._alphabet.batch(encoded_data, starts, lengths)

        # 3. Create Records with Views
        records = []
        for i, (header, _) in enumerate(entries):
            name, _, desc = header.partition(b' ')
            
            # Create a Seq that views the batch memory (Zero Copy)
            s_start = starts[i]
            s_view = self._alphabet.seq(encoded_data[s_start : s_start + lengths[i]])
            
            # Parse metadata
            records.append(Record(s_view, name, desc))
            
        return RecordBatch.from_aligned_batch(batch, records)
    
    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b">")


class FastaWriter(BaseWriter):
    """
    Writer for FASTA format files.

    Examples:
        >>> with FastaWriter("output.fasta") as w:
        ...     w.write_one(record)
    """
    __slots__ = ('width',)
    def __init__(self, file: Union[str, Path, BinaryIO], width: int = 0, **kwargs):
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


@SeqFile.register('gfa')
class GfaReader(SeqReader):
    """
    Reader for GFA (Graphical Fragment Assembly) files.

    Examples:
        >>> with open("graph.gfa", "rb") as f:
        ...     reader = GfaReader(f)
        ...     for item in reader:
        ...         if isinstance(item, Record): print("Segment:", item.id)
        ...         elif isinstance(item, Edge): print("Link:", item.u, "->", item.v)
    """
    __slots__ = ()
    def __iter__(self) -> Generator[Union[Record, Edge], None, None]:
        """
        Iterates over GFA lines (Segments as Records, Links as Edges).

        Yields:
            Record or Edge objects.
        """
        # Optimization: Read large binary chunks
        buf = b""
        while True:
            chunk = self._handle.read(self._CHUNK_SIZE)
            if not chunk:
                if buf and buf.strip():
                    yield from self._parse_line(buf)
                break

            buf += chunk
            pos = 0

            while True:
                nl_pos = buf.find(b'\n', pos)
                if nl_pos == -1:
                    buf = buf[pos:]
                    break

                line = buf[pos:nl_pos]
                yield from self._parse_line(line)
                pos = nl_pos + 1

    def batches(self, size: int = 1024):
        rec_entries = []
        edge_u = []
        edge_v = []
        edge_attrs = {b'u_strand': [], b'v_strand': [], b'cigar': []}
        
        buf = b""
        while True:
            chunk = self._handle.read(self._CHUNK_SIZE)
            if not chunk:
                if buf and buf.strip():
                    self._process_line_batch(buf, rec_entries, edge_u, edge_v, edge_attrs)
                break

            buf += chunk
            pos = 0

            while True:
                nl_pos = buf.find(b'\n', pos)
                if nl_pos == -1:
                    buf = buf[pos:]
                    break

                line = buf[pos:nl_pos]
                self._process_line_batch(line, rec_entries, edge_u, edge_v, edge_attrs)
                pos = nl_pos + 1
                
                if len(rec_entries) >= size:
                    yield self._make_record_batch(rec_entries)
                    rec_entries = []
                
                if len(edge_u) >= size:
                    yield self._make_edge_batch(edge_u, edge_v, edge_attrs)
                    edge_u = []
                    edge_v = []
                    edge_attrs = {b'u_strand': [], b'v_strand': [], b'cigar': []}
        
        if rec_entries:
            yield self._make_record_batch(rec_entries)
        if edge_u:
            yield self._make_edge_batch(edge_u, edge_v, edge_attrs)

    def _process_line_batch(self, line, rec_entries, edge_u, edge_v, edge_attrs):
        line = line.rstrip()
        if not line: return
        first = line[0]

        if first == 83:  # 'S'
            parts = line.split(b'\t')
            if len(parts) < 3: return

            name, seq_bytes = parts[1], parts[2]
            if seq_bytes == b'*': seq_bytes = b''

            if len(seq_bytes) >= self._min_seq_length:
                tags = parts[3:] if len(parts) > 3 else []
                rec_entries.append((name, seq_bytes, tags))

        elif first == 76:  # 'L'
            parts = line.split(b'\t')
            if len(parts) < 6: return
            # L name1 ori1 name2 ori2 cigar
            edge_u.append(parts[1])
            edge_attrs[b'u_strand'].append(parts[2])
            edge_v.append(parts[3])
            edge_attrs[b'v_strand'].append(parts[4])
            edge_attrs[b'cigar'].append(parts[5])

    def _make_record_batch(self, entries):
        bulk_bytes = b"".join([e[1] for e in entries])
        encoded_data = self._alphabet.encode(bulk_bytes)

        n = len(entries)
        lengths = np.array([len(e[1]) for e in entries], dtype=np.int32)
        starts = np.zeros(n, dtype=np.int32)
        if n > 1: np.cumsum(lengths[:-1], out=starts[1:])

        batch = self._alphabet.batch(encoded_data, starts, lengths)

        records = []
        for i, (name, _, tags) in enumerate(entries):
            s_start = starts[i]
            s_view = self._alphabet.seq(encoded_data[s_start : s_start + lengths[i]])
            quals = self._QUALIFIER_PARSER.parse_tags(tags) if tags else []
            records.append(Record(s_view, name, qualifiers=quals))
            
        return RecordBatch.from_aligned_batch(batch, records)

    def _make_edge_batch(self, u, v, attrs):
        u_arr = np.array(u, dtype=object)
        v_arr = np.array(v, dtype=object)
        attr_arrays = {k: np.array(val, dtype=object) for k, val in attrs.items()}
        return EdgeBatch(u_arr, v_arr, attr_arrays)

    def _parse_line(self, line: bytes):
        line = line.rstrip()
        if not line: return
        first = line[0]

        if first == 83:  # 'S'
            parts = line.split(b'\t')
            if len(parts) < 3: return

            name, seq_bytes = parts[1], parts[2]

            if len(seq_bytes) >= self._min_seq_length:
                seq = self._alphabet.seq(seq_bytes if seq_bytes != b'*' else b'')
                quals = self._QUALIFIER_PARSER.parse_tags(parts[3:]) if len(parts) > 3 else []
                yield Record(seq, name, qualifiers=quals)

        elif first == 76:  # 'L'
            parts = line.split(b'\t')
            if len(parts) < 6: return
            yield Edge(parts[1], parts[3], {b'u_strand': parts[2], b'v_strand': parts[4], b'cigar': parts[5]})
    
    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b'H\t') or s.startswith(b'S\t')


class GfaWriter(BaseWriter):
    """
    Writer for GFA format files.

    Examples:
        >>> with GfaWriter("graph.gfa") as w:
        ...     w.write_one(record)
        ...     w.write_one(edge)
    """
    __slots__ = ()
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


@SeqFile.register('fastq')
class FastqReader(SeqReader):
    """
    Reader for FASTQ format files.
    Optimized for standard 4-line FASTQ records. Wrapped FASTQ sequences are not supported.

    Examples:
        >>> with open("reads.fastq", "rb") as f:
        ...     reader = FastqReader(f)
        ...     for record in reader:
        ...         print(record.id)
    """
    __slots__ = ()
    def __iter__(self) -> Generator[Record, None, None]:
        """
        Iterates over FASTQ records.

        Yields:
            Record objects.
        """
        for header, seq_bytes, qual_bytes in self._read_entries():
            name, _, desc = header.partition(b' ')
            yield Record(self._alphabet.seq(seq_bytes), name, desc, qualifiers=[(b'quality', qual_bytes)])

    def batches(self, size: int = 1024):
        entries = []
        for entry in self._read_entries():
            entries.append(entry)
            if len(entries) >= size:
                yield self._make_batch(entries)
                entries = []
        if entries:
            yield self._make_batch(entries)

    def _read_entries(self):
        min_len = self._min_seq_length
        buf = b""
        while True:
            chunk = self._handle.read(self._CHUNK_SIZE)
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

                if len(seq_bytes) >= min_len:
                    yield header, seq_bytes, qual_bytes

                pos = nl4 + 1

            if pos > 0: buf = buf[pos:]
            if not chunk: break

    def _make_batch(self, entries):
        # entries: (header, seq_bytes, qual_bytes)
        bulk_bytes = b"".join([e[1] for e in entries])
        encoded_data = self._alphabet.encode(bulk_bytes)

        n = len(entries)
        lengths = np.array([len(e[1]) for e in entries], dtype=np.int32)
        starts = np.zeros(n, dtype=np.int32)
        if n > 1: np.cumsum(lengths[:-1], out=starts[1:])

        batch = self._alphabet.batch(encoded_data, starts, lengths)

        records = []
        for i, (header, _, qual) in enumerate(entries):
            name, _, desc = header.partition(b' ')
            s_start = starts[i]
            s_view = self._alphabet.seq(encoded_data[s_start : s_start + lengths[i]])
            records.append(Record(s_view, name, desc, qualifiers=[(b'quality', qual)]))

        return RecordBatch.from_aligned_batch(batch, records)

    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b"@")
