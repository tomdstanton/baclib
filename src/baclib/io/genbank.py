from typing import List, BinaryIO, Generator, Tuple
from re import compile as regex

from baclib.core.alphabet import Alphabet
from baclib.containers.record import Record, Feature, RecordBatch, FeatureKey
from baclib.containers.seq import Seq
from baclib.core.interval import Interval
from baclib.io import BaseReader, BaseWriter, SeqFile


# Classes --------------------------------------------------------------------------------------------------------------
from baclib.containers.record import Record, Feature, RecordBatch, FeatureKey
from baclib.containers.seq import Seq
from baclib.core.interval import Interval
from baclib.io import BaseReader, SeqFile


# Classes --------------------------------------------------------------------------------------------------------------

class InsdcReader(BaseReader):
    """
    Base reader for INSDC formats (GenBank, EMBL, DDBJ).
    Implements strict column-based feature table parsing.
    """
    __slots__ = ('_alphabet', '_base')
    _INTERVAL_REGEX = regex(rb'(?P<partial_start><)?(?P<start>[0-9]+)\.\.(?P<partial_end>>)?(?P<end>[0-9]+)')
    
    # Subclasses must define these
    _HEADER_START = b''  # e.g., b'LOCUS' or b'ID'
    _FEATURE_START = b'' # e.g., b'FEATURES' or b'FH'
    _FEATURE_TABLE_PREFIX = b'' # e.g. b'FT ' for EMBL, empty for GenBank

    def __init__(self, handle: BinaryIO, alphabet: Alphabet = None):
        super().__init__(handle)
        self._alphabet = alphabet
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
        
        buf = bytearray()
        search_pos = 0
        for chunk in self.read_chunks(self._CHUNK_SIZE):
            if not chunk:
                # EOF
                if buf and (self._HEADER_START in buf[:100] or b'//' in buf):
                    if len(buf.strip()) > 2: yield bytes(buf)
                break
            
            buf.extend(chunk)
            pos = 0
            
            # Find records separated by //
            while True:
                # Find end of record marker //
                # We need to find \n// which is the standard end
                # but handle if it's at start of buffer separately
                
                scan_start = max(pos, search_pos)
                end_pos = -1
                
                if scan_start == 0 and buf.startswith(b'//'): end_pos = 0
                else:
                    found = buf.find(b'\n//', scan_start)
                    if found != -1: end_pos = found + 1
                
                if end_pos == -1:
                    # No complete record yet
                    search_pos = max(pos, len(buf) - 4) 
                    if pos > 0: 
                        del buf[:pos]
                        search_pos -= pos
                        search_pos = max(0, search_pos)
                    break 
                
                # Found delimiter at end_pos
                next_line = buf.find(b'\n', end_pos)
                if next_line == -1:
                    # Incomplete delimiter line, wait for more data
                    if pos > 0: 
                        del buf[:pos]
                        search_pos = max(0, search_pos - pos)
                    break
                
                record_bytes = bytes(buf[pos:end_pos])
                if record_bytes.strip(): yield record_bytes
                
                pos = next_line + 1
                search_pos = pos

    @staticmethod
    def _extract_seq_data(data: bytes) -> Tuple[bytes, bytes]:
        """Returns (metadata, sequence_data)"""
        # Common logic: Sequence usually starts after ORIGIN (GenBank) or SQ (EMBL)
        # But implementations might differ slightly.
        # We'll use a heuristic: look for ORIGIN line or Sequence Header
        pass

    def _parse_record_chunk(self, data: bytes) -> Record:
        meta_data, seq_data = self._split_meta_seq(data)

        # Encode Sequence
        if self._alphabet is None and seq_data:
            self._alphabet = Alphabet.detect(seq_data)
        
        if self._alphabet:
            seq = self._alphabet.seq_from(seq_data) if seq_data else self._alphabet.empty_seq()
        else:
             if not seq_data:
                 self._alphabet = Alphabet.DNA
                 seq = self._alphabet.empty_seq()
             else:
                 seq = self._alphabet.seq_from(seq_data)

        # Parse Metadata (Features)
        header_lines, feature_lines = self._split_features(meta_data)
            
        return self._finalize_record(header_lines, feature_lines, seq)

    def _split_meta_seq(self, data: bytes) -> Tuple[bytes, bytes]:
        raise NotImplementedError

    def _split_features(self, meta_data: bytes) -> Tuple[List[bytes], List[bytes]]:
        raise NotImplementedError

    def _make_batch(self, chunks: List[bytes]) -> RecordBatch:
         seq_bytes_list = []
         meta_list = []
         
         for data in chunks:
             meta_data, seq_data = self._split_meta_seq(data)
             meta_list.append(meta_data)
             seq_bytes_list.append(seq_data or b"")

         if self._alphabet is None:
             # Detect from first non-empty sequence
             for s in seq_bytes_list:
                 if s:
                     self._alphabet = Alphabet.detect(s)
                     break
             if self._alphabet is None and seq_bytes_list:
                  self._alphabet = Alphabet.DNA
         
         batch = self._build_seq_batch(seq_bytes_list, self._alphabet)
         bulk_data, starts, lengths = batch.arrays

         # Build Records
         records = []
         for i, meta_data in enumerate(meta_list):
             # Create View
             s_start = starts[i]
             s_len = lengths[i]
             s_view = self._alphabet.seq_from(bulk_data[s_start: s_start + s_len])
             
             # Parse Meta
             header_lines, feature_lines = self._split_features(meta_data)
                 
             records.append(self._finalize_record(header_lines, feature_lines, s_view))
             
         return RecordBatch.from_aligned_batch(batch, records)

    def _finalize_record(self, header_lines: List[bytes], feature_lines: List[bytes], seq: Seq) -> Record:
        # 1. Parse Header
        name = b'unknown'
        description = b''
        
        name, description, qualifiers = self._parse_header(header_lines)

        # 2. Create Record
        record = Record(seq, name, description, qualifiers=qualifiers)

        # 3. Parse Features (Strict Column Parsing)
        if feature_lines:
            self._parse_feature_table(record, feature_lines)

        return record

    def _parse_header(self, lines: List[bytes]) -> Tuple[bytes, bytes]:
        raise NotImplementedError

    def _parse_feature_table(self, record: Record, lines: List[bytes]):
        # State machine for strict column parsing
        # Key: 5-20 (0-indexed)
        # Data: 21+
        
        current_key = None
        current_loc_parts = []
        current_qualifiers = []      # List of (key, value identifiers)
        current_qual_key = None      # Current qualifier key being parsed
        current_qual_val_parts = []  # Accumulator for qualifier value
        
        def commit_feature():
            nonlocal current_key, current_loc_parts, current_qualifiers
            if not current_key: return
            
            # Parse Location
            loc_str = b"".join(current_loc_parts)
            strand = -1 if b'complement' in loc_str else 1
            intervals = []
            for m in self._INTERVAL_REGEX.finditer(loc_str):
                intervals.append(Interval(int(m.group('start')) - self._base, int(m.group('end')), strand))
            
            if not intervals:
                if current_key == FeatureKey.SOURCE.bytes:
                    intervals = [Interval(0, 0, 1)]
                else:
                    return # Skip if no location? Or default?

            # Span
            if len(intervals) > 1:
                span_start = min(i.start for i in intervals)
                span_end = max(i.end for i in intervals)
                feat = Feature(Interval(span_start, span_end, strand), current_key)
            else:
                feat = Feature(intervals[0], current_key)
            
            # Add Qualifiers
            for qk, qv in current_qualifiers:
                feat.add_qualifier(qk, qv)
            
            if feat.key == FeatureKey.SOURCE:
                record.qualifiers.extend(feat.qualifiers)
            else:
                record.features.append(feat)

        def commit_qualifier():
            nonlocal current_qual_key, current_qual_val_parts
            if current_qual_key:
                val = b"".join(current_qual_val_parts)
                # Strip quotes if present
                if val.startswith(b'"') and val.endswith(b'"'):
                    val = val[1:-1]
                
                # If bool (no value originally), val might be empty or we flag it
                # Logic: if no '=', val is True. But here we accumulated bytes.
                # We need to know if '=' was present. 
                # Our parsing logic below splits on '='.
                
                current_qualifiers.append((current_qual_key, val))
            
            current_qual_key = None
            current_qual_val_parts = []

        
        ft_prefix_len = len(self._FEATURE_TABLE_PREFIX)

        for line in lines:
            if not line.strip(): continue
            
            # Handle EMBL FT prefix if needed
            if ft_prefix_len > 0 and not line.startswith(self._FEATURE_TABLE_PREFIX):
                continue
            
            if len(line) < 21: continue
            
            key_col = line[5:20].strip()
            data_col = line[21:].strip()
            
            if key_col:
                # NEW FEATURE
                commit_qualifier() # Commit previous feature's last qualifier
                commit_feature()   # Commit previous feature
                
                current_key = key_col
                current_loc_parts = [data_col]
                current_qualifiers = []
                current_qual_key = None
                current_qual_val_parts = []
                
            else:
                # CONTINUATION LINE (Feature)
                if not current_key: continue
                
                if data_col.startswith(b'/'):
                    # NEW QUALIFIER
                    commit_qualifier()
                    
                    # Parse key=val
                    # /strain="ATCC 123"
                    # /pseudo
                    content = data_col[1:] # Strip '/'
                    if b'=' in content:
                        q_key, q_val_start = content.split(b'=', 1)
                        current_qual_key = q_key
                        current_qual_val_parts = [q_val_start]
                    else:
                        # Boolean / flag
                        current_qualifiers.append((content, True))
                        current_qual_key = None # No value accumulation needed
                else:
                    # CONTINUATION OF DATA
                    if current_qual_key:
                        # Continue qualifier value
                        # Add space if needed? GenBank/EMBL spec regarding spaces in line wraps:
                        # Usually we just join. But for text, often spaces are implied.
                        # For keys like 'translation', no spaces.
                        # For 'note', spaces?
                        # Implementing standard: if it's a quote-wrapped text, spaces are likely needed 
                        # if the previous line didn't end with hyphen?
                        # Heuristic: just append for now, maybe add space if previous char was not space?
                        # Safest: join with space if text, join without if sequence?
                        # Let's simple join for now to match strict byte preservation.
                        # Actually, strictly, we trim strings. 
                        # Let's append with a space if parsing text, but maybe not for translation?
                        # Ref: "Qualifiers ... enclosed in double quotation marks ... string"
                        # We will append raw stripped parts.
                        
                        if current_qual_key == b'translation':
                             current_qual_val_parts.append(data_col)
                        else:
                             # Generally add space for text fields
                             if current_qual_val_parts: 
                                 # Check if we are inside a quote?
                                 # This is complex without full parser.
                                 # Simple: Join with space.
                                 current_qual_val_parts.append(b" " + data_col)
                             else:
                                 current_qual_val_parts.append(data_col)
                    else:
                        # Continue location
                        current_loc_parts.append(data_col)
        
        # End of loop
        commit_qualifier()
        commit_feature()


@SeqFile.register(SeqFile.Format.GENBANK, extensions=['.gb', '.gbk', '.genbank'])
class GenbankReader(InsdcReader):
    """
    GenBank Format Reader.
    """
    _HEADER_START = b'LOCUS'
    _FEATURE_START = b'FEATURES'
    _FEATURE_TABLE_PREFIX = b''

    def _split_meta_seq(self, data: bytes) -> Tuple[bytes, bytes]:
        origin_idx = data.find(b'\nORIGIN')
        if origin_idx != -1:
            line_end = data.find(b'\n', origin_idx + 1)
            if line_end != -1:
                return data[:origin_idx], data[line_end+1:]
        
        # If no ORIGIN, maybe it's all meta or purely meta (contig?)
        return data, None

    def _split_features(self, meta_data: bytes) -> Tuple[List[bytes], List[bytes]]:
        features_idx = meta_data.find(b'\nFEATURES')
        
        if features_idx != -1:
            header_bytes = meta_data[:features_idx]
            feature_bytes = meta_data[features_idx+1:]
            return header_bytes.splitlines(), feature_bytes.splitlines()
        elif meta_data.strip().startswith(b'FEATURES'):
             return [], meta_data.splitlines()
        
        return meta_data.splitlines(), []

    def _parse_header(self, lines: List[bytes]) -> Tuple[bytes, bytes, List[Tuple[bytes, bytes]]]:
        name = b'unknown'
        description = b''
        qualifiers = []
        for line in lines:
            if line.startswith(b'LOCUS'):
                parts = line.split()
                if len(parts) > 1: name = parts[1]
            elif line.startswith(b'DEFINITION'):
                description = line[12:].strip()
            elif line.startswith(b'SOURCE'):
                qualifiers.append((b'organism', line[12:].strip()))
        return name, description, qualifiers
    
    @classmethod
    def sniff(cls, s: bytes) -> bool: return b'LOCUS' in s[:100]


@SeqFile.register(SeqFile.Format.EMBL, extensions=['.embl'])
class EmblReader(InsdcReader):
    """
    EMBL Format Reader.
    """
    _HEADER_START = b'ID'
    _FEATURE_START = b'FH'
    _FEATURE_TABLE_PREFIX = b'FT' # EMBL lines start with 'FT   '

    def _split_meta_seq(self, data: bytes) -> Tuple[bytes, bytes]:
        # EMBL Sequence starts after 'SQ' line
        # SQ   Sequence 1859 BP; 609 A; 314 C; 355 G; 581 T; 0 other;
        sq_idx = data.find(b'\nSQ')
        if sq_idx != -1:
             line_end = data.find(b'\n', sq_idx + 1)
             if line_end != -1:
                 return data[:sq_idx], data[line_end+1:]
        return data, None

    def _split_features(self, meta_data: bytes) -> Tuple[List[bytes], List[bytes]]:
        # EMBL Features start with FH key (Feature Header) and contain FT lines
        # Stop at SQ or XX?
        # We can just iterate lines and filter.
        
        # But to fit _split_features signature efficiently:
        # Find start of FT lines?
        # Usually internal to metadata.
        # Let's split all lines.
        lines = meta_data.splitlines()
        header = []
        features = []
        
        in_features = False
        for line in lines:
            if line.startswith(b'FH'):
                in_features = True
                continue
            if line.startswith(b'FT'):
                features.append(line)
            else:
                 header.append(line)
        
        return header, features

    def _parse_header(self, lines: List[bytes]) -> Tuple[bytes, bytes, List[Tuple[bytes, bytes]]]:
        name = b'unknown'
        description = b''
        qualifiers = []
        for line in lines:
             if line.startswith(b'ID'):
                 parts = line.split()
                 if len(parts) > 1: name = parts[1].strip(b';')
             elif line.startswith(b'DE'):
                 if not description:
                     description = line[5:].strip()
                 else:
                     description += b" " + line[5:].strip()
             elif line.startswith(b'OS'):
                 val = line[5:].strip()
                 qualifiers.append((b'organism', val))
        return name, description, qualifiers

    @classmethod
    def sniff(cls, s: bytes) -> bool: 
        return b'ID' in s[:100] and b'; SV' in s[:200]


# Writers --------------------------------------------------------------------------------------------------------------

class InsdcWriter(BaseWriter):
    """
    Base writer for INSDC formats.
    """
    _HEADER_WIDTH = 12
    _FT_PREFIX = b'' 
    
    def _write_feature_table(self, record: Record):
        if not record.features: return
        
        # Write Header
        if self._FT_PREFIX:
             # EMBL: "FH   Key             Location/Qualifiers"
             self.write_line(b"FH   Key             Location/Qualifiers")
        else:
             # GenBank: "FEATURES             Location/Qualifiers"
             self.write_line(b"FEATURES             Location/Qualifiers")
             
        for feature in record.features:
            self._write_feature(feature)

    def _write_feature(self, feature: Feature):
        # Format Location
        loc_str = self._format_location(feature.interval)
        
        # Write Feature Line
        # Key: col 5-20 (16 chars?)
        # Data: col 21+
        # GenBank: "     source          1..100"
        # EMBL:    "FT   source          1..100"
        
        key_bytes = feature.key.bytes
        
        if self._FT_PREFIX:
            # EMBL: "FT   key..."
            # Prefix len 5 ("FT   ")
            # Key starts at index 5. matches GenBank col 6.
            # "FT   " is 5 chars.
            # "FT   source          "
            # We need 21 chars total prefix before location.
            # "FT   " + key + padding = 21 chars?
            # 5 + len(key) + padding = 21 -> padding = 16 - len(key)
            padding = b" " * (16 - len(key_bytes))
            line = self._FT_PREFIX + b"   " + key_bytes + padding + loc_str.encode('ascii')
        else:
            # GenBank: "     source          "
            # 5 spaces + key + padding = 21 chars
            # padding = 21 - 5 - len(key) = 16 - len(key)
            padding = b" " * (16 - len(key_bytes))
            line = b"     " + key_bytes + padding + loc_str.encode('ascii')
            
        self.write_line(line)
        
        # Write Qualifiers
        if feature.qualifiers:
            for key, val in feature.qualifiers.items():
                self._write_qualifier(key, val)

    def _format_location(self, interval: Interval) -> str:
        # 1-based inclusive
        start = interval.start + 1
        end = interval.end
        
        if start == end:
             base = f"{start}"
        else:
             base = f"{start}..{end}"
        
        if interval.strand == -1:
            return f"complement({base})"
        return base

    def _write_qualifier(self, key: bytes, value: object):
        # Format: /key="value"
        # Indent: 21 spaces.
        # Wrap at 80.
        
        prefix = b' ' * 21
        if self._FT_PREFIX: prefix = self._FT_PREFIX + b' ' * 16 # EMBL FT lines start with FT + spaces to col 21?
        # EMBL "FT                   /organism=..."
        # FT is 2 chars. + 19 spaces = 21.
        
        if self._FT_PREFIX:
             prefix = self._FT_PREFIX + b' ' * (21 - len(self._FT_PREFIX))
        
        q_key = b'/' + key
        
        # Determine value string
        if value is True:
             # Boolean flag
             self.write_line(prefix + q_key)
             return
             
        val_bytes = str(value).encode('ascii') if not isinstance(value, bytes) else value
        
        # Check if we should quote
        # Generally yes for text.
        # Integer values (e.g. taxid) might not need quotes but usually safe to quote or required?
        # Specification says "most qualifiers take values... enclosed in double quotes".
        # Some like /codon_start are integers (no quotes? or quotes allowed?)
        # We will quote everything for simplicity unless it's strictly forbidden (which is rare).
        
        # Escape quotes in value?
        # Replace " with ""? simple escape.
        val_bytes = val_bytes.replace(b'"', b'""')
        
        full_val = q_key + b'="' + val_bytes + b'"'
        
        # Wrap logic
        # Available width = 80 - 21 = 59 chars.
        
        if len(full_val) + len(prefix) <= 80:
             self.write_line(prefix + full_val)
             return
        
        # Multi-line wrap
        # First line
        # We can split full_val
        
        # Greedy wrap
        current = full_val
        first = True
        
        while current:
             limit = 80 - len(prefix)
             if len(current) <= limit:
                 self.write_line(prefix + current)
                 break
             
             # Need to split
             # Try to split at space if possible?
             # User said "avoid textwrap", implication is purely binary speed or specific logic.
             # INSDC prefers breaking at space.
             
             chunk = current[:limit]
             # Check for space in last 10 chars to break nicely?
             # Or just hard break?
             # Let's do hard break to ensure strict column width compliance safely.
             # "break at arbitrary position" is allowed?
             # Spec: "lines... max 80 chars".
             
             self.write_line(prefix + chunk)
             current = current[limit:]

    def write_line(self, line: bytes):
        self._handle.write(line + b'\n')

    def write_one(self, record: Record):
        # Implemented by subclasses (Header, Sequence, Terminator)
        raise NotImplementedError


@SeqFile.register(SeqFile.Format.GENBANK, extensions=['.gb', '.gbk', '.genbank'])
class GenbankWriter(InsdcWriter):
    def write_one(self, record: Record):
        # Header
        # LOCUS       Action_17                671 bp    DNA     linear   PLN 18-FEB-2026
        # Name: 12 chars max?
        name = record.id or b'Unknown'
        length = len(record.seq)
        
        # Fixed padding is tricky without f-string for bytes or strict formatting
        # LOCUS (col 0-4)
        # Name (col 12-?)
        # Length
        # MolType
        # Div
        # Date
        
        # Simplified LOCUS line
        locus_line = f"LOCUS       {name.decode('ascii'):<20} {length:>10} bp    DNA     linear   UNK {self._get_date()}".encode('ascii')
        self.write_line(locus_line)
        
        # DEFINITION
        desc = record.description or b'.'
        self._write_wrapped_line(b"DEFINITION  ", desc)
        
        # ACCESSION / VERSION
        self.write_line(b"ACCESSION   " + name)
        self.write_line(b"VERSION     " + name) # Placeholder
        
        # KEYWORDS
        self.write_line(b"KEYWORDS    .")
        
        # SOURCE
        # Check source feature?
        source_feat = next((f for f in record.features if f.key == FeatureKey.SOURCE), None)
        organism = b"Unknown"
        if source_feat and b'organism' in source_feat.qualifiers:
             organism = source_feat.qualifiers[b'organism']
        
        self._write_wrapped_line(b"SOURCE      ", organism)
        self.write_line(b"  ORGANISM  " + organism)
        self.write_line(b"            .")
        
        # Features
        self._write_feature_table(record)
        
        # Sequence
        self.write_line(b"ORIGIN")
        self._write_sequence(record.seq)
        
        self.write_line(b"//")

    def _get_date(self) -> str:
        return "01-JAN-1900" # Placeholder, strict date logic needs imports

    def _write_wrapped_line(self, prefix: bytes, content: bytes):
        # Wrap content with prefix
        # Indent subsequent lines equal to prefix len
        indent_len = len(prefix)
        indent = b' ' * 12 # Standard indent for DEFINITION/SOURCE bodies is 12
        
        # Actually logic:
        # Header (12 chars) + Content
        # definition is 12 chars "DEFINITION  "
        
        # Reuse wrap logic?
        full = prefix + content
        if len(full) <= 80:
             self.write_line(full)
        else:
             # Split
             # TODO: Proper wrapping
             self.write_line(full[:80])
             self.write_line(indent + full[80:]) # Simple split

    def _write_sequence(self, seq: Seq):
        # 60 bp per line, blocks of 10
        #        1 ctgctggcgc catcttgctc tggctgtcgg cgatccggcg gccaatgtgc aggcgctggt
        
        # Convert to bytes
        seq_bytes = bytes(seq)
        length = len(seq_bytes)
        
        for i in range(0, length, 60):
            chunk = seq_bytes[i:i+60]
            
            # Format: 9 chars numbering (right align) + space + seq chunks
            # Numbering is 1-based start of line
            line_start = f"{i+1:>9} ".encode('ascii')
            
            # Split into blocks of 10
            blocks = [chunk[j:j+10] for j in range(0, len(chunk), 10)]
            seq_str = b' '.join(blocks)
            
            self.write_line(line_start + seq_str)


@SeqFile.register(SeqFile.Format.EMBL, extensions=['.embl'])
class EmblWriter(InsdcWriter):
    _FT_PREFIX = b'FT'
    
    def write_one(self, record: Record):
        # ID   Name; SV 1; linear; DNA; STD; UNK; Length BP.
        name = record.id or b'Unknown'
        length = len(record.seq)
        
        id_line = f"ID   {name.decode('ascii')}; SV 1; linear; DNA; STD; UNK; {length} BP.".encode('ascii')
        self.write_line(id_line)
        
        # XX
        self.write_line(b"XX")
        
        # DE
        desc = record.description or b'.'
        self.write_line(b"DE   " + desc)
        self.write_line(b"XX")
        
        # Features
        self._write_feature_table(record)
        
        # Sequence
        # SQ   Sequence 1859 BP; 609 A; 314 C; 355 G; 581 T; 0 other;
        # Stats?
        self.write_line(f"SQ   Sequence {length} BP;".encode('ascii'))
        
        # Sequence data
        #      gatc gatc       10
        seq = record.seq
        seq_bytes = bytes(seq) if seq is not None and len(seq) > 0 else b""
        seq_len = len(seq_bytes)
        
        for i in range(0, seq_len, 60):
             chunk = seq_bytes[i:i+60]
             # Blocks of 10
             blocks = [chunk[j:j+10] for j in range(0, len(chunk), 10)]
             seq_content = b' '.join(blocks)
             
             # Padding to 72 chars? Then count.
             # "     " (5 spaces) + content + padding + count
             
             # EMBL format: 5 spaces, seq data, char count at col 72>?
             #      cctttatcgg aatgaaaaaa ttatttattt attagaggaa agaacataca atggacaatg         60
             
             line_prefix = b"     " + seq_content
             # Pad to somewhere?
             # Count at end.
             count_str = f"{min(i+60, seq_len)}".encode('ascii')
             
             # Pad line_prefix to 72 chars?
             padding = b" " * max(1, 72 - len(line_prefix))
             
             self.write_line(line_prefix + padding + count_str)
        
        self.write_line(b"//")


