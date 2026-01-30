from typing import List, BinaryIO, Generator
from re import compile as regex

import numpy as np

from baclib.core.seq import Alphabet
from baclib.containers.record import Record, Feature, RecordBatch
from baclib.core.seq import Seq
from baclib.core.interval import Interval
from baclib.io import BaseReader, SeqFile


# Classes --------------------------------------------------------------------------------------------------------------
@SeqFile.register('genbank')
class GenbankReader(BaseReader):
    """
    High-performance Genbank Reader (Binary Mode).
    Robust to fuzzy coordinates (<1..>100) and loose indentation.

    Examples:
        >>> with open("genome.gbk", "rb") as f:
        ...     reader = GenbankReader(f)
        ...     for record in reader:
        ...         print(record.id)
    """
    # Relaxed regex to handle <, > and partial entries (Bytes)
    _INTERVAL_REGEX = regex(rb'(?P<partial_start><)?(?P<start>[0-9]+)\.\.(?P<partial_end>>)?(?P<end>[0-9]+)')
    _SUPPORTED_KINDS = frozenset({b'CDS', b'source', b'misc_feature'})

    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None):
        super().__init__(handle)
        self.alphabet = alphabet or self._DEFAULT_ALPHABET
        self._base = 1

    def __iter__(self) -> Generator[Record, None, None]:
        for chunk in self._read_chunks():
            try:
                yield self._parse_record_chunk(chunk)
            except Exception:
                # Fallback for truncated records at EOF or malformed chunks
                pass

    def batches(self, size: int = 1024) -> Generator[RecordBatch, None, None]:
        chunks = []
        for chunk in self._read_chunks():
            chunks.append(chunk)
            if len(chunks) >= size:
                yield self._make_batch(chunks)
                chunks = []
        if chunks:
            yield self._make_batch(chunks)

    def _read_chunks(self):
        # Optimization: Read large binary chunks (Seqtk style)
        read = self._handle.read
        
        buf = b""
        search_pos = 0
        while True:
            chunk = read(self._CHUNK_SIZE)
            if not chunk:
                if buf and (b'LOCUS' in buf[:100] or b'//' in buf):
                    # Try to parse remaining buffer if it looks like a record
                    if len(buf.strip()) > 2:
                        yield buf
                break
            
            buf += chunk
            pos = 0
            # Only search from where we left off (minus delimiter length to catch splits)
            # But we must respect 'pos' which tracks the start of the current record in buf
            scan_start = max(pos, search_pos)
            
            while True:
                # Find end of record marker //
                end_pos = -1
                
                # Check start of buffer
                if scan_start == 0 and buf.startswith(b'//'):
                    end_pos = 0
                else:
                    # Find \n//
                    found = buf.find(b'\n//', scan_start)
                    if found != -1:
                        end_pos = found + 1
                
                if end_pos == -1:
                    search_pos = max(pos, len(buf) - 4) # Update search position for next chunk
                    if pos > 0: buf = buf[pos:]
                    if pos > 0: search_pos -= pos # Adjust search_pos relative to new buf start
                    break
                
                # Found delimiter at end_pos
                # Find end of that line to advance pos
                next_line = buf.find(b'\n', end_pos)
                if next_line == -1:
                    # Incomplete delimiter line
                    if pos > 0: buf = buf[pos:]
                    if pos > 0: search_pos = max(0, search_pos - pos)
                    break
                
                record_bytes = buf[pos:end_pos]
                if record_bytes.strip():
                    yield record_bytes
                
                pos = next_line + 1
                scan_start = pos # Next search starts after this record

    def _parse_record_chunk(self, data: bytes) -> Record:
        # 1. Split Sequence vs Metadata
        origin_idx = data.find(b'\nORIGIN')
        
        seq_data = None
        meta_data = data
        
        if origin_idx != -1:
            # Find end of ORIGIN line
            line_end = data.find(b'\n', origin_idx + 1)
            if line_end != -1:
                meta_data = data[:origin_idx]
                seq_data = data[line_end+1:]
        
        # 2. Encode Sequence
        if seq_data:
            seq = self.alphabet.seq(seq_data)
        else:
            seq = self.alphabet.seq(np.array([], dtype=np.uint8))
            
        # 3. Parse Metadata
        features_idx = meta_data.find(b'\nFEATURES')
        
        if features_idx != -1:
            header_bytes = meta_data[:features_idx]
            feature_bytes = meta_data[features_idx+1:]
            header_lines = header_bytes.splitlines()
            feature_lines = feature_bytes.splitlines()
        elif meta_data.startswith(b'FEATURES'):
            header_lines = []
            feature_lines = meta_data.splitlines()
        else:
            header_lines = meta_data.splitlines()
            feature_lines = []
            
        return self._finalize_record(header_lines, feature_lines, seq)

    def _make_batch(self, chunks: List[bytes]) -> RecordBatch:
        seq_encoded_list = []
        meta_list = []
        
        for data in chunks:
            origin_idx = data.find(b'\nORIGIN')
            seq_data = None
            meta_data = data
            
            if origin_idx != -1:
                line_end = data.find(b'\n', origin_idx + 1)
                if line_end != -1:
                    meta_data = data[:origin_idx]
                    seq_data = data[line_end+1:]
            
            meta_list.append(meta_data)
            if seq_data:
                seq_encoded_list.append(self.alphabet.encode(seq_data))
            else:
                seq_encoded_list.append(np.empty(0, dtype=np.uint8))

        # Build SeqBatch
        lengths = np.array([len(x) for x in seq_encoded_list], dtype=np.int32)
        if len(seq_encoded_list) > 0:
            bulk_data = np.concatenate(seq_encoded_list)
        else:
            bulk_data = np.empty(0, dtype=np.uint8)
            
        n = len(chunks)
        starts = np.zeros(n, dtype=np.int32)
        if n > 1: np.cumsum(lengths[:-1], out=starts[1:])
        
        batch = self.alphabet.batch(bulk_data, starts, lengths)
        
        # Build Records
        records = []
        for i, meta_data in enumerate(meta_list):
            # Create View
            s_start = starts[i]
            s_len = lengths[i]
            s_view = self.alphabet.seq(bulk_data[s_start : s_start + s_len])
            
            # Parse Meta
            features_idx = meta_data.find(b'\nFEATURES')
            if features_idx != -1:
                header_bytes = meta_data[:features_idx]
                feature_bytes = meta_data[features_idx+1:]
                header_lines = header_bytes.splitlines()
                feature_lines = feature_bytes.splitlines()
            elif meta_data.startswith(b'FEATURES'):
                header_lines = []
                feature_lines = meta_data.splitlines()
            else:
                header_lines = meta_data.splitlines()
                feature_lines = []
                
            records.append(self._finalize_record(header_lines, feature_lines, s_view))
            
        return RecordBatch.from_aligned_batch(batch, records)

    def _finalize_record(self, header_lines: List[bytes], feature_lines: List[bytes],
                         seq: Seq) -> Record:
        # 1. Parse Header
        name = b'unknown'
        description = b''

        for line in header_lines:
            if line.startswith(b'LOCUS'):
                parts = line.split()
                if len(parts) > 1: name = parts[1]
            elif line.startswith(b'DEFINITION'):
                description = line[12:].strip()

                # 2. Create Record
                record = Record(
                    seq,
                    name,
                    description
                )

                # 3. Parse Features
                current_lines = []
                for line in feature_lines:
                    if not line.strip(): continue
                    # Heuristic: Feature keys start at col 5 (index 5) and are not qualifiers (/)
                    # Standard GenBank: 5 spaces, Key, ...
                    # line[5] returns int (ASCII), 32 is space.
                    if len(line) > 5 and line[5] != 32 and not line.strip().startswith(b'/'):
                        if current_lines:
                            self._parse_feature_block(record, current_lines)
                        current_lines = [line]
                    else:
                        current_lines.append(line)

                if current_lines:
                    self._parse_feature_block(record, current_lines)

                return record

    def _parse_feature_block(self, record: Record, lines: List[bytes]):
        header_line = lines[0].strip()
        if not header_line: return

        parts = header_line.split(maxsplit=1)
        kind = parts[0]
        if kind not in self._SUPPORTED_KINDS: return

        loc_str = parts[1] if len(parts) > 1 else b""
        qual_start_index = 1

        loc_parts = [loc_str]
        # Accumulate location string
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith(b'/'): break
            loc_parts.append(stripped)
            qual_start_index += 1
        loc_str = b"".join(loc_parts)

        # Try to parse intervals
        strand = -1 if b'complement' in loc_str else 1
        intervals = []
        for m in self._INTERVAL_REGEX.finditer(loc_str):
            intervals.append(Interval(int(m.group('start')) - self._base, int(m.group('end')), strand))

        # Handle 'source' or failed parses
        if not intervals:
            if kind == b'source':
                intervals = [Interval(0, 0, 1)]
            else:
                return

        # Handle compound intervals (join/order) by taking the span
        if len(intervals) > 1:
            # Create a spanning interval
            span_start = min(i.start for i in intervals)
            span_end = max(i.end for i in intervals)
            feat = Feature(Interval(span_start, span_end, strand), kind)
        else:
            feat = Feature(intervals[0], kind)

        # Parse Qualifiers
        current_qual = []
        for line in lines[qual_start_index:]:
            stripped = line.strip()
            if stripped.startswith(b'/'):
                if current_qual: self._parse_and_add_qual(feat, current_qual)
                current_qual = [stripped]
            else:
                current_qual.append(stripped)

        if current_qual: self._parse_and_add_qual(feat, current_qual)

        if kind == b'source':
            record.qualifiers.extend(feat.qualifiers)
        else:
            record.features.append(feat)

    def _parse_and_add_qual(self, feat: Feature, lines: List[bytes]):
        full_text = b"".join(lines)
        # Remove leading '/'
        key, sep, val = full_text[1:].partition(b'=')

        if not sep:
            feat.add_qualifier(key, True)
        else:
            val = val.strip()
            if val.startswith(b'"') and val.endswith(b'"'): val = val[1:-1]
            feat.add_qualifier(key, val)
            
    @classmethod
    def sniff(cls, s: bytes) -> bool: return b'LOCUS' in s[:100]
