from typing import List, BinaryIO, Generator
from re import compile as regex

import numpy as np

from baclib.core.seq import Alphabet
from baclib.containers.record import Record, Feature
from baclib.core.seq import Seq
from baclib.core.interval import Interval
from baclib.io import BaseReader


# Classes --------------------------------------------------------------------------------------------------------------
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
    _TOPOLOGY_REGEX = regex(rb'(?i)(\bcircular\b|\bcircular\s*=\s*true\b)')
    _SUPPORTED_KINDS = frozenset({b'CDS', b'source', b'misc_feature'})

    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None):
        super().__init__(handle)
        self.alphabet = alphabet or self._DEFAULT_ALPHABET
        self._base = 1

    def __iter__(self) -> Generator[Record, None, None]:
        # Optimization: Read large binary chunks (Seqtk style)
        read = self._handle.read
        
        buf = b""
        while True:
            chunk = read(self._CHUNK_SIZE)
            if not chunk:
                if buf and (b'LOCUS' in buf[:100] or b'//' in buf):
                    # Try to parse remaining buffer if it looks like a record
                    if len(buf.strip()) > 2:
                        try: yield self._parse_record_chunk(buf)
                        except Exception: pass
                break
            
            buf += chunk
            pos = 0
            
            while True:
                # Find end of record marker //
                end_pos = -1
                
                # Check start of buffer
                if pos == 0 and buf.startswith(b'//'):
                    end_pos = 0
                else:
                    # Find \n//
                    found = buf.find(b'\n//', pos)
                    if found != -1:
                        end_pos = found + 1
                
                if end_pos == -1:
                    if pos > 0: buf = buf[pos:]
                    break
                
                # Found delimiter at end_pos
                # Find end of that line to advance pos
                next_line = buf.find(b'\n', end_pos)
                if next_line == -1:
                    # Incomplete delimiter line
                    if pos > 0: buf = buf[pos:]
                    break
                
                record_bytes = buf[pos:end_pos]
                if record_bytes.strip():
                    yield self._parse_record_chunk(record_bytes)
                
                pos = next_line + 1

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

    def _finalize_record(self, header_lines: List[bytes], feature_lines: List[bytes],
                         seq: Seq) -> Record:
        # 1. Parse Header
        name = b'unknown'
        description = b''
        topology = b'linear'

        for line in header_lines:
            if line.startswith(b'LOCUS'):
                parts = line.split()
                if len(parts) > 1: name = parts[1]
                if self._TOPOLOGY_REGEX.search(line): topology = b'circular'
            elif line.startswith(b'DEFINITION'):
                description = line[12:].strip()

                # 2. Create Record
                record = Record(
                    seq,
                    name,
                    description,
                    qualifiers=[(b'topology', topology)]
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

        # Accumulate location string
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith(b'/'): break
            loc_str += stripped
            qual_start_index += 1

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
