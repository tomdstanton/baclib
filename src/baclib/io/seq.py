from pathlib import Path
from typing import Union, Generator, BinaryIO, Iterable

import numpy as np

from baclib.core.alphabet import Alphabet
from baclib.core.interval import Strand
from baclib.containers.record import Record, RecordBatch
from baclib.containers.graph import Edge, EdgeBatch
from baclib.io import BaseReader, BaseWriter, ParserError, SeqFile, SeqFileFormat, Qualifier
from baclib.utils import Batch


# Classes --------------------------------------------------------------------------------------------------------------
class SeqReader(BaseReader):
    __slots__ = ('_alphabet', '_min_seq_length')
    _DEFAULT_ALPHABET = Alphabet.DNA
    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None, min_seq_length: int = 1, **kwargs):
        """ABC for readers that parse files with sequences (Fasta, Fastq and GFA)"""
        super().__init__(handle, **kwargs)
        self._alphabet = alphabet or self._DEFAULT_ALPHABET
        self._min_seq_length = min_seq_length


@SeqFile.register(SeqFileFormat.FASTA, extensions=['.fasta', '.fa', '.fna', '.faa'],
                  alphabets={'.faa': Alphabet.AMINO, '.fna': Alphabet.DNA})
class FastaReader(SeqReader):
    """
    Reader for FASTA format files.
    """
    __slots__ = ('_n_seq_lines',)
    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None, min_seq_length: int = 1, n_seq_lines: int = 0, **kwargs):
        super().__init__(handle, alphabet, min_seq_length, **kwargs)
        self._n_seq_lines = n_seq_lines

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
        if self._n_seq_lines > 0:
            yield from self._read_entries_fixed()
            return

        min_len = self._min_seq_length
        
        buf = b""
        header = None
        seq_parts = []
        current_len = 0
        
        for chunk in self.read_chunks(self._CHUNK_SIZE):
            if not chunk:
                if header is not None:
                    if buf: 
                        seq_parts.append(buf)
                        current_len += len(buf)
                    if current_len >= min_len:
                        yield header, seq_parts
                break
            
            buf += chunk
            pos = 0
            
            while True:
                gt_pos = buf.find(b'>', pos)
                
                if gt_pos == -1:
                    if header is not None:
                        part = buf[pos:]
                        seq_parts.append(part)
                        current_len += len(part)
                    buf = b""
                    break
                
                if header is not None:
                    part = buf[pos:gt_pos]
                    seq_parts.append(part)
                    current_len += len(part)
                    if current_len >= min_len:
                        yield header, seq_parts
                    seq_parts = []
                    current_len = 0
                    header = None
                
                nl_pos = buf.find(b'\n', gt_pos)
                if nl_pos == -1:
                    buf = buf[gt_pos:]
                    break
                
                header = buf[gt_pos+1:nl_pos].rstrip()
                pos = nl_pos + 1

    def _read_entries_fixed(self):
        """Optimized reader for FASTA with fixed number of sequence lines (e.g. unwrapped)."""
        n_lines = self._n_seq_lines
        min_len = self._min_seq_length
        buf = b""
        
        state = 0 # 0=Header, 1=Sequence
        header = None
        seq_parts = []
        seq_lines_read = 0
        
        for chunk in self.read_chunks(self._CHUNK_SIZE):
            if not chunk: break
            buf += chunk
            pos = 0
            
            while True:
                nl_pos = buf.find(b'\n', pos)
                if nl_pos == -1:
                    buf = buf[pos:]
                    break
                
                # Check for > at start of line
                is_header_start = (buf[pos] == 62)
                
                if state == 0:
                    if is_header_start:
                        header = buf[pos+1:nl_pos].rstrip()
                        state = 1
                        seq_parts = []
                        seq_lines_read = 0
                
                elif state == 1:
                    if is_header_start:
                        raise ParserError(f"Found '>' inside sequence. Check n_seq_lines={n_lines}.")
                    
                    seq_parts.append(buf[pos:nl_pos])
                    seq_lines_read += 1
                    
                    if seq_lines_read == n_lines:
                        if sum(len(p) for p in seq_parts) >= min_len:
                            yield header, seq_parts
                        state = 0
                
                pos = nl_pos + 1
        
        if state == 1 and seq_parts and sum(len(p) for p in seq_parts) >= min_len:
            yield header, seq_parts

    def _make_record(self, header, seq_parts: Iterable[bytes]) -> Record:
        name, _, desc = header.partition(b' ')
        # Encode the whole sequence at once (releases GIL in Numba)
        return Record(self._alphabet.seq_from(b"".join(seq_parts)), name, desc)

    def _make_batch(self, entries) -> RecordBatch:
        # Join parts for each record and encode individually to ensure correct lengths
        raw_seqs = [b"".join(parts) for _, parts in entries]
        batch = self._build_seq_batch(raw_seqs, self._alphabet)
        encoded_data, starts, lengths = batch.arrays

        # 3. Create Records with Views
        records = []
        for i, (header, _) in enumerate(entries):
            name, _, desc = header.partition(b' ')
            
            # Create a Seq that views the batch memory (Zero Copy)
            s_start = starts[i]
            s_view = self._alphabet.seq_from(encoded_data[s_start: s_start + lengths[i]])
            
            # Parse metadata
            records.append(Record(s_view, name, desc))
            
        return RecordBatch.from_aligned_batch(batch, records)
    
    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b">")


@SeqFile.register(SeqFileFormat.FASTA)
class FastaWriter(BaseWriter):
    """
    Writer for FASTA format files.
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

        # Optimization: Decode once, then slice bytes (faster than decoding slices)
        seq_bytes = bytes(record.seq)
        width = self.width

        if width > 0:
            self._handle.write(b"\n".join(seq_bytes[i:i+width] for i in range(0, len(seq_bytes), width)) + b"\n")
        else:
            self._handle.write(seq_bytes + b"\n")

    def write_batch(self, batch: Batch):
        if isinstance(batch, RecordBatch):
            # Optimization: Decode entire batch data at once to avoid N translate calls
            seqs = batch.seqs
            ids = batch.ids
            decoded = seqs.alphabet.decode(seqs.encoded)
            starts = seqs.starts
            lengths = seqs.lengths
            width = self.width
            
            buffer = []
            
            for i in range(len(batch)):
                buffer.append(b">")
                buffer.append(ids[i])
                buffer.append(b"\n")
                
                s = starts[i]
                l = lengths[i]
                seq_content = decoded[s : s+l]
                
                if width > 0:
                    buffer.append(b"\n".join(seq_content[j:j+width] for j in range(0, l, width)))
                else:
                    buffer.append(seq_content)
                buffer.append(b"\n")
                
                if len(buffer) >= 1000:
                    self._handle.write(b"".join(buffer))
                    buffer = []
            
            if buffer:
                self._handle.write(b"".join(buffer))
        else:
            super().write_batch(batch)

@SeqFile.register(SeqFileFormat.GFA, extensions=['.gfa'])
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
        for chunk in self.read_chunks(self._CHUNK_SIZE):
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
        edge_u_strands = []
        edge_v_strands = []
        edge_attrs = {b'cigar': []}
        
        buf = b""
        for chunk in self.read_chunks(self._CHUNK_SIZE):
            if not chunk:
                if buf and buf.strip():
                    self._process_line_batch(buf, rec_entries, edge_u, edge_v, edge_u_strands, edge_v_strands, edge_attrs)
                break

            buf += chunk
            pos = 0

            while True:
                nl_pos = buf.find(b'\n', pos)
                if nl_pos == -1:
                    buf = buf[pos:]
                    break

                line = buf[pos:nl_pos]
                self._process_line_batch(line, rec_entries, edge_u, edge_v, edge_u_strands, edge_v_strands, edge_attrs)
                pos = nl_pos + 1
                
                if len(rec_entries) >= size:
                    yield self._make_record_batch(rec_entries)
                    rec_entries = []
                
                if len(edge_u) >= size:
                    yield self._make_edge_batch(edge_u, edge_v, edge_u_strands, edge_v_strands, edge_attrs)
                    edge_u = []
                    edge_v = []
                    edge_u_strands = []
                    edge_v_strands = []
                    edge_attrs = {b'cigar': []}
        
        if rec_entries:
            yield self._make_record_batch(rec_entries)
        if edge_u:
            yield self._make_edge_batch(edge_u, edge_v, edge_u_strands, edge_v_strands, edge_attrs)

    def _process_line_batch(self, line, rec_entries, edge_u, edge_v, edge_u_strands, edge_v_strands, edge_attrs):
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
            edge_u_strands.append(Strand.from_symbol(parts[2]))
            edge_v.append(parts[3])
            edge_v_strands.append(Strand.from_symbol(parts[4]))
            edge_attrs[b'cigar'].append(parts[5])

    def _make_record_batch(self, entries):
        batch = self._build_seq_batch([e[1] for e in entries], self._alphabet)
        encoded_data, starts, lengths = batch.arrays
        parse_tags = Qualifier.parse_tags
        records = []
        for i, (name, _, tags) in enumerate(entries):
            s_start = starts[i]
            s_view = self._alphabet.seq_from(encoded_data[s_start: s_start + lengths[i]])
            quals = parse_tags(tags) if tags else []
            records.append(Record(s_view, name, qualifiers=quals))
            
        return RecordBatch.from_aligned_batch(batch, records)

    @staticmethod
    def _make_edge_batch(u, v, u_strands, v_strands, attrs):
        u_arr = np.array(u, dtype=object)
        v_arr = np.array(v, dtype=object)
        us_arr = np.array(u_strands, dtype=np.int8)
        vs_arr = np.array(v_strands, dtype=np.int8)
        attr_arrays = {k: np.array(val, dtype=object) for k, val in attrs.items()}
        return EdgeBatch(u_arr, v_arr, us_arr, vs_arr, attr_arrays)

    def _parse_line(self, line: bytes):
        line = line.rstrip()
        if not line: return
        first = line[0]

        if first == 83:  # 'S'
            parts = line.split(b'\t')
            if len(parts) < 3: return

            name, seq_bytes = parts[1], parts[2]

            if len(seq_bytes) >= self._min_seq_length:
                seq = self._alphabet.seq_from(seq_bytes if seq_bytes != b'*' else b'')
                quals = Qualifier.parse_tags(parts[3:]) if len(parts) > 3 else []
                yield Record(seq, name, qualifiers=quals)

        elif first == 76:  # 'L'
            parts = line.split(b'\t')
            if len(parts) < 6: return
            yield Edge(parts[1], parts[3], 
                       Strand.from_symbol(parts[2]), 
                       Strand.from_symbol(parts[4]), 
                       {b'cigar': parts[5]})
    
    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b'H\t') or s.startswith(b'S\t')


@SeqFile.register(SeqFileFormat.GFA)
class GfaWriter(BaseWriter):
    """
    Writer for GFA format files.
    """
    __slots__ = ()
    def write_one(self, item: Union[Record, Edge]):
        """
        Writes a Record (Segment) or Edge (Link).

        Args:
            item: Record or Edge object.
        """
        if isinstance(item, Record):
            # S name seq
            self._handle.write(b"S\t" + item.id + b"\t" + bytes(item.seq) + b"\n")
        elif isinstance(item, Edge):
            cigar = item.attributes.get(b'cigar', b'0M')
            us = item.u_strand.bytes
            vs = item.v_strand.bytes
            self._handle.write(b"L\t" + item.u + b"\t" + us + b"\t" +
                               item.v + b"\t" + vs + b"\t" + cigar + b"\n")

    def write_batch(self, batch: Batch):
        if isinstance(batch, EdgeBatch):
            # Vectorized write for Edges
            u = batch.u
            v = batch.v
            u_strands = batch.u_strands
            v_strands = batch.v_strands
            
            # Extract columns or use defaults
            cigars = batch.attributes.get(b'cigar')
            
            lines = []
            for i in range(len(batch)):
                us = Strand(u_strands[i]).bytes
                vs = Strand(v_strands[i]).bytes
                cg = cigars[i] if cigars is not None else b'0M'
                
                lines.append(b"L\t" + u[i] + b"\t" + us + b"\t" + v[i] + b"\t" + vs + b"\t" + cg + b"\n")
                
                if len(lines) >= 1000:
                    self._handle.write(b"".join(lines))
                    lines = []
            if lines:
                self._handle.write(b"".join(lines))

        elif isinstance(batch, RecordBatch):
            # Vectorized write for Records (Segments)
            ids = batch.ids
            seqs = batch.seqs
            decoded = seqs.alphabet.decode(seqs.encoded)
            starts = seqs.starts
            lengths = seqs.lengths
            
            lines = []
            for i in range(len(batch)):
                s = starts[i]
                l = lengths[i]
                lines.append(b"S\t" + ids[i] + b"\t" + decoded[s : s+l] + b"\n")
                
                if len(lines) >= 1000:
                    self._handle.write(b"".join(lines))
                    lines = []
            if lines:
                self._handle.write(b"".join(lines))
        else:
            super().write_batch(batch)


@SeqFile.register(SeqFileFormat.FASTQ, extensions=['.fastq', '.fq'])
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
            yield Record(self._alphabet.seq_from(seq_bytes), name, desc, qualifiers=[(b'quality', qual_bytes)])

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
        for chunk in self.read_chunks(self._CHUNK_SIZE):
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

    def _make_batch(self, entries):
        batch = self._build_seq_batch([e[1] for e in entries], self._alphabet)
        encoded_data, starts, lengths = batch.arrays

        records = []
        for i, (header, _, qual) in enumerate(entries):
            name, _, desc = header.partition(b' ')
            s_start = starts[i]
            s_view = self._alphabet.seq_from(encoded_data[s_start: s_start + lengths[i]])
            records.append(Record(s_view, name, desc, qualifiers=[(b'quality', qual)]))

        return RecordBatch.from_aligned_batch(batch, records)

    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b"@")
