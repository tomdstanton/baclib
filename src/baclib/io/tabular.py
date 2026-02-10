from abc import abstractmethod
from typing import Union, Generator

import numpy as np

from baclib.containers.alignment import Alignment, AlignmentBatch
from baclib.containers.record import Record, Feature, FeatureBatch, FeatureKey, QualifierBatch
from baclib.core.interval import Interval, IntervalBatch
from baclib.io import BaseWriter, BaseReader, SeqFile, SeqFileFormat, Qualifier


# Classes --------------------------------------------------------------------------------------------------------------
class TabularReader(BaseReader):
    """Base class for readers of tabular formats (GFF, BED, PAF)."""
    _delim = b'\t'
    _min_cols: int = 1
    __slots__ = ('_handle',)

    def _read_parts(self) -> Generator[list[bytes], None, None]:
        """Internal generator that yields split lines."""
        delim = self._delim
        min_cols = self._min_cols
        
        buf = b""
        for chunk in self.read_chunks(self._CHUNK_SIZE):
            if not chunk:
                if buf:
                    line = buf.rstrip()
                    if line and not line.startswith(b'#'):
                        parts = line.split(delim)
                        if len(parts) >= min_cols:
                            yield parts
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
                yield parts

    def __iter__(self) -> Generator:
        """
        Iterates over lines, parsing valid rows.

        Yields:
            Parsed Feature or Alignment objects.
        """
        parse = self.parse_row
        for parts in self._read_parts(): yield parse(parts)

    def batches(self, size: int = 1024):
        batch_parts = []
        for parts in self._read_parts():
            batch_parts.append(parts)
            if len(batch_parts) >= size:
                yield self._make_batch_from_parts(batch_parts)
                batch_parts = []
        if batch_parts: yield self._make_batch_from_parts(batch_parts)

    def _make_batch(self, items: list):
        """
        Creates a batch from a list of items.
        """
        return items
    
    def _make_batch_from_parts(self, parts_list: list[list[bytes]]):
        """Creates a batch directly from parsed columns."""
        items = [self.parse_row(p) for p in parts_list]
        return self._make_batch(items)

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


@SeqFile.register(SeqFileFormat.BED, extensions=['.bed'])
class BedReader(TabularReader):
    """
    Reader for BED format files.

    Examples:
        >>> with open("features.bed", "rb") as f:
        ...     reader = BedReader(f)
        ...     for feature in reader:
        ...         print(feature.key)
    """
    _min_cols = 3

    def parse_row(self, parts: list[bytes]) -> Feature:
        """
        Parses a BED row.

        Args:
            parts: List of column strings.

        Returns:
            A Feature object.
        """
        start, end = int(parts[1]), int(parts[2])
        n_cols = len(parts)
        kind = parts[3] if n_cols > 3 else b'feature'
        score = float(parts[4]) if n_cols > 4 and parts[4] != b'.' else 0.0
        strand = parts[5] if n_cols > 5 else b'.'
        quals = [(b'source', parts[0])]
        if score: quals.append((b'score', score))
        if n_cols > 9: quals.append((b'blocks', b','.join(parts[9:])))
        return Feature(Interval(start, end, strand), kind, qualifiers=quals)

    @classmethod
    def sniff(cls, s: bytes) -> bool:
        try:
            for line in s.splitlines():
                if not line.strip() or line.startswith((b'track', b'browser', b'#')): continue
                parts = line.split(b'\t')
                return len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit()
            return False
        except Exception: return False
        
    def _make_batch(self, items: list) -> FeatureBatch:
        return FeatureBatch.from_features(items)

    def _make_batch_from_parts(self, parts_list: list[list[bytes]]) -> FeatureBatch:
        n = len(parts_list)
        if n == 0: 
            return FeatureBatch.empty_batch()

        starts = np.empty(n, dtype=np.int32)
        ends = np.empty(n, dtype=np.int32)
        strands = np.empty(n, dtype=np.int32)
        keys = np.empty(n, dtype=np.int16)
        
        from_bytes = FeatureKey.from_bytes
        
        qualifiers_collection = []
        
        for i, parts in enumerate(parts_list):
            starts[i] = int(parts[1])
            ends[i] = int(parts[2])
            
            n_cols = len(parts)
            s_char = parts[5] if n_cols > 5 else b'.'
            strands[i] = 1 if s_char == b'+' else (-1 if s_char == b'-' else 0)
            
            kind_bytes = parts[3] if n_cols > 3 else b'feature'
            keys[i] = from_bytes(kind_bytes).value
            
            quals = [(b'source', parts[0])]
            score = float(parts[4]) if n_cols > 4 and parts[4] != b'.' else 0.0
            if score: quals.append((b'score', score))
            if n_cols > 9: quals.append((b'blocks', b','.join(parts[9:])))
            qualifiers_collection.append(quals)
            
        intervals = IntervalBatch(starts, ends, strands, sort=False)
        qualifiers = QualifierBatch.from_qualifiers(qualifiers_collection)
        
        return FeatureBatch(intervals, keys, qualifiers)


@SeqFile.register(SeqFileFormat.BED)
class BedWriter(BaseWriter):
    """
    Writer for BED format files.

    Examples:
        >>> with BedWriter("output.bed") as w:
        ...     w.write_one(record)
    """

    def write_one(self, record: Record):
        """
        Writes features of a Record in BED format.

        Args:
            record: The Record object.
        """
        if not isinstance(record, Record): raise TypeError("BedWriter expects Record objects")
        for feature in record.features:
            # Resolve Name/ID/Gene/Key for BED column 4 (Name)
            name = feature.get(b'Name', feature.get(b'ID', feature.get(
                b'gene', feature.key.bytes if isinstance(feature.key, FeatureKey) else feature.key)))
            line = b"\t".join([
                record.id,
                str(feature.interval.start).encode('ascii'),
                str(feature.interval.end).encode('ascii'),
                name,
                str(feature.get(b'score', 0)).encode('ascii'), feature.interval.strand.bytes
            ]) + b"\n"
            self._handle.write(line)


@SeqFile.register(SeqFileFormat.GFF, extensions=['.gff', '.gff3'])
class GffReader(TabularReader):
    """
    Reader for GFF3 format files.

    Examples:
        >>> with open("features.gff", "rb") as f:
        ...     reader = GffReader(f)
        ...     for feature in reader:
        ...         print(feature.key)
    """
    _min_cols = 9

    def parse_row(self, parts: list[bytes]) -> Feature:
        """
        Parses a GFF3 row.

        Args:
            parts: List of column strings.

        Returns:
            A Feature object.
        """
        start, end = int(parts[3]) - 1, int(parts[4])
        quals = Qualifier.parse_gff_attributes(parts[8])
        quals.append((b'source', parts[0]))
        if parts[1] != b'.': quals.append((b'tool', parts[1]))
        if parts[5] != b'.': quals.append((b'score', float(parts[5])))
        if parts[7] != b'.': quals.append((b'phase', int(parts[7])))
        return Feature(Interval(start, end, parts[6]), parts[2], qualifiers=quals)
    
    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b'##gff')

    def _make_batch(self, items: list) -> FeatureBatch:
        return FeatureBatch.from_features(items)

    def _make_batch_from_parts(self, parts_list: list[list[bytes]]) -> FeatureBatch:
        n = len(parts_list)
        if n == 0: 
            return FeatureBatch.empty_batch()

        starts = np.empty(n, dtype=np.int32)
        ends = np.empty(n, dtype=np.int32)
        strands = np.empty(n, dtype=np.int32)
        keys = np.empty(n, dtype=np.int16)
        
        from_bytes = FeatureKey.from_bytes
        parse_attrs = Qualifier.parse_gff_attributes
        
        qualifiers_collection = []
        
        for i, parts in enumerate(parts_list):
            # GFF is 1-based inclusive -> 0-based half-open
            starts[i] = int(parts[3]) - 1
            ends[i] = int(parts[4])
            
            s_char = parts[6]
            strands[i] = 1 if s_char == b'+' else (-1 if s_char == b'-' else 0)
            
            keys[i] = from_bytes(parts[2]).value
            
            quals = parse_attrs(parts[8])
            quals.append((b'source', parts[0]))
            if parts[1] != b'.': quals.append((b'tool', parts[1]))
            if parts[5] != b'.': quals.append((b'score', float(parts[5])))
            if parts[7] != b'.': quals.append((b'phase', int(parts[7])))
            qualifiers_collection.append(quals)
            
        intervals = IntervalBatch(starts, ends, strands, sort=False)
        qualifiers = QualifierBatch.from_qualifiers(qualifiers_collection)
        
        return FeatureBatch(intervals, keys, qualifiers)


@SeqFile.register(SeqFileFormat.GFF)
class GffWriter(BaseWriter):
    """
    Writer for GFF3 format files.

    Examples:
        >>> with GffWriter("output.gff") as w:
        ...     w.write_header()
        ...     w.write_one(record)
    """

    def write_header(self):
        """Writes the GFF3 header."""
        self._handle.write(b"##gff-version 3\n")

    def write_one(self, record: Record):
        """
        Writes a Record and its features in GFF3 format.

        Args:
            record: The Record object.
        """
        if not isinstance(record, Record): raise TypeError("GffWriter expects Record objects")
        self._handle.write(b"##sequence-region " + record.id + b" 1 %b" % len(record) + b"\n")
        for feature in record.features: self._write_feature(record.id, feature)

    def _write_feature(self, seq_id: bytes, feature: Feature):
        """
        Writes a single feature.

        Args:
            seq_id: ID of the sequence containing the feature.
            feature: The Feature object.
        """
        source = feature.get(b'source', b'baclib')
        start = feature.interval.start + 1
        end = feature.interval.end
        score = feature.get(b'score', b'.')

        strand = feature.interval.strand.bytes

        phase = feature.get(b'phase', b'.')
        kind_bytes = feature.key.bytes if isinstance(feature.key, FeatureKey) else feature.key

        attr_strings = []
        if val := feature.get(b'ID'): attr_strings.append(b"ID=" + val)
        if val := feature.get(b'Name'): attr_strings.append(b"Name=" + val)
        for key, value in feature.qualifiers:
            if key in {b'source', b'score', b'phase', b'ID', b'Name'}: continue
            safe_val = val.replace(b';', b'%3B').replace(b'=', b'%3D').replace(b'&', b'%26')
            attr_strings.append(key + b"=" + safe_val)
        attr_block = b";".join(attr_strings) if attr_strings else b"."

        self._handle.write(
            b"\t".join([seq_id, source, kind_bytes, str(start).encode('ascii'), str(end).encode('ascii'), str(score).encode('ascii'), strand, str(phase).encode('ascii'), attr_block]) + b"\n")


@SeqFile.register(SeqFileFormat.PAF, extensions=['.paf'])
class PafReader(TabularReader):
    """
    Reader for PAF (Pairwise mApping Format) files.

    Examples:
        >>> with open("alignments.paf", "rb") as f:
        ...     reader = PafReader(f)
        ...     for aln in reader:
        ...         print(aln.score)
    """
    _min_cols = 12

    def parse_row(self, parts: list[bytes]) -> Alignment:
        """
        Parses a PAF row.

        Args:
            parts: List of column strings.

        Returns:
            An Alignment object.
        """
        q_len, t_len, block_len = int(parts[1]), int(parts[6]), int(parts[10])
        n_matches = int(parts[9])
        cigar, score, quals = None, None, []
        for key, value in Qualifier.parse_tags(parts[12:]):
            if key == b'cg': cigar = value
            elif key == b'AS': score = value
            else: quals.append((key, value))

        return Alignment(
            query=parts[0], query_interval=Interval(int(parts[2]), int(parts[3]), 1),
            target=parts[5], interval=Interval(int(parts[7]), int(parts[8]), parts[4].decode('ascii')),
            query_length=q_len, target_length=t_len, length=block_len, score=score,
            cigar=cigar, n_matches=n_matches, quality=int(parts[11]), qualifiers=quals
        )

    def _make_batch(self, items: list) -> AlignmentBatch:
        return AlignmentBatch.from_alignments(items)
    
    def _make_batch_from_parts(self, parts_list: list[list[bytes]]) -> AlignmentBatch:
        n = len(parts_list)
        if n == 0: return AlignmentBatch()
        
        # PAF cols: 
        # 0:q, 1:q_len, 2:q_s, 3:q_e, 4:strand, 5:t, 6:t_len, 7:t_s, 8:t_e, 9:match, 10:aln_len, 11:qual
        
        # 1. IDs
        q_names = [p[0] for p in parts_list]
        t_names = [p[5] for p in parts_list]
        q_ids, q_indices = np.unique(q_names, return_inverse=True)
        t_ids, t_indices = np.unique(t_names, return_inverse=True)
        
        # 2. Numeric Columns
        # Use list comprehensions for speed on small batches
        q_lens = np.array([int(p[1]) for p in parts_list], dtype=np.int32)
        q_starts = np.array([int(p[2]) for p in parts_list], dtype=np.int32)
        q_ends = np.array([int(p[3]) for p in parts_list], dtype=np.int32)
        
        t_lens = np.array([int(p[6]) for p in parts_list], dtype=np.int32)
        t_starts = np.array([int(p[7]) for p in parts_list], dtype=np.int32)
        t_ends = np.array([int(p[8]) for p in parts_list], dtype=np.int32)
        
        matches = np.array([int(p[9]) for p in parts_list], dtype=np.int32)
        aln_lens = np.array([int(p[10]) for p in parts_list], dtype=np.int32)
        qualities = np.array([int(p[11]) for p in parts_list], dtype=np.uint8)
        
        # Strands: PAF uses '+' and '-'
        strands = np.array([1 if p[4] == b'+' else -1 for p in parts_list], dtype=np.int8)
        
        # 3. Tags (Score, Cigar)
        scores = np.zeros(n, dtype=np.float32)
        cigars = np.full(n, None, dtype=object)
        quals = np.full(n, None, dtype=object)

        parse_tags = Qualifier.parse_tags
        for i, parts in enumerate(parts_list):
            if len(parts) > 12:
                row_quals = []
                for key, value in parse_tags(parts[12:]):
                    if key == b'cg': cigars[i] = value
                    elif key == b'AS': scores[i] = value
                    else: row_quals.append((key, value))
                if row_quals: quals[i] = row_quals

        # 4. Construct Batch
        data = np.zeros(n, dtype=AlignmentBatch._DTYPE)
        data['q_idx'] = q_indices; data['t_idx'] = t_indices; data['score'] = scores
        data['q_start'] = q_starts; data['q_end'] = q_ends; data['q_len'] = q_lens; data['q_strand'] = 1
        data['t_start'] = t_starts; data['t_end'] = t_ends; data['t_len'] = t_lens; data['t_strand'] = strands
        data['matches'] = matches; data['quality'] = qualities; data['aln_len'] = aln_lens
        
        return AlignmentBatch(data=data, cigars=cigars, qualifiers=quals, query_ids=q_ids, target_ids=t_ids)
    
    @classmethod
    def sniff(cls, s: bytes) -> bool:
        try:
            line = s.split(b'\n', 1)[0]
            parts = line.split(b'\t')
            return (len(parts) >= 12 and parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit() and
                    parts[6].isdigit() and parts[7].isdigit() and parts[8].isdigit())
        except Exception:
            return False
    

@SeqFile.register(SeqFileFormat.PAF)
class PafWriter(BaseWriter):
    """
    Writer for PAF (Pairwise mApping Format) files.

    Examples:
        >>> with PafWriter("alignments.paf") as w:
        ...     w.write(alignment_batch)
    """
    def write_one(self, item: Union[Alignment, AlignmentBatch]):
        if isinstance(item, AlignmentBatch):
            self.write_batch(item)
        elif isinstance(item, Alignment):
            self._write_alignment(item)
        else:
            raise TypeError(f"PafWriter expects Alignment or AlignmentBatch objects, got {type(item)}")

    def _write_alignment(self, aln: Alignment):
        q_name = aln.query
        if isinstance(q_name, (int, np.integer)): q_name = str(q_name).encode('ascii')
        elif isinstance(q_name, str): q_name = q_name.encode('ascii')

        t_name = aln.target
        if isinstance(t_name, (int, np.integer)): t_name = str(t_name).encode('ascii')
        elif isinstance(t_name, str): t_name = t_name.encode('ascii')

        strand = b'+' if aln.query_interval.strand == aln.interval.strand else b'-'
        
        parts = [
            q_name,
            b"%d" % aln.query_length,
            b"%d" % aln.query_interval.start,
            b"%d" % aln.query_interval.end,
            strand,
            t_name,
            b"%d" % aln.target_length,
            b"%d" % aln.interval.start,
            b"%d" % aln.interval.end,
            b"%d" % aln.n_matches,
            b"%d" % aln.length,
            b"%d" % aln.quality
        ]
        
        self._append_tags(parts, aln.score, aln.cigar, aln.qualifiers)
        self._handle.write(b"\t".join(parts) + b"\n")

    def write_batch(self, batch: AlignmentBatch):
        n = len(batch)
        if n == 0: return
        
        q_ids = batch.query.ids
        t_ids = batch.target.ids
        is_same_strand = (batch.q_strands == batch.t_strands)
        qualities = batch._data['quality']
        
        lines = []
        for i in range(n):
            q_name = q_ids[i]
            if isinstance(q_name, str): q_name = q_name.encode('ascii')
            elif isinstance(q_name, (int, np.integer)): q_name = str(q_name).encode('ascii')

            t_name = t_ids[i]
            if isinstance(t_name, str): t_name = t_name.encode('ascii')
            elif isinstance(t_name, (int, np.integer)): t_name = str(t_name).encode('ascii')
            
            parts = [
                q_name,
                b"%d" % batch.q_lens[i],
                b"%d" % batch.q_starts[i],
                b"%d" % batch.q_ends[i],
                b'+' if is_same_strand[i] else b'-',
                t_name,
                b"%d" % batch.t_lens[i],
                b"%d" % batch.t_starts[i],
                b"%d" % batch.t_ends[i],
                b"%d" % batch.matches[i],
                b"%d" % batch.aln_lens[i],
                b"%d" % qualities[i]
            ]
            
            self._append_tags(parts, batch.scores[i], batch.cigars[i], batch._qualifiers[i])
            lines.append(b"\t".join(parts) + b"\n")
            
            if len(lines) >= 1000:
                self._handle.write(b"".join(lines))
                lines = []
        
        if lines:
            self._handle.write(b"".join(lines))

    def _append_tags(self, parts: list, score, cigar, qualifiers):
        # Score
        if score is not None:
            if isinstance(score, (int, np.integer)) or score.is_integer():
                parts.append(b"AS:i:%d" % int(score))
            else:
                parts.append(b"AS:f:%.4f" % score)
        
        # Cigar
        if cigar:
            parts.append(b"cg:Z:" + cigar)
            
        # Qualifiers
        if qualifiers:
            for k, v in qualifiers:
                if k in (b'AS', b'cg'): continue
                
                val_bytes = b""
                type_char = b"Z"
                
                if isinstance(v, (int, np.integer)):
                    type_char = b"i"
                    val_bytes = b"%d" % v
                elif isinstance(v, (float, np.floating)):
                    type_char = b"f"
                    val_bytes = b"%.4f" % v
                elif isinstance(v, bytes):
                    type_char = b"Z"
                    val_bytes = v
                elif isinstance(v, str):
                    type_char = b"Z"
                    val_bytes = v.encode('ascii')
                else:
                    continue
                
                parts.append(k + b":" + type_char + b":" + val_bytes)
