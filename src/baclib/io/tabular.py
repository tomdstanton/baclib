from abc import abstractmethod
from typing import Union, Generator

from baclib.align.alignment import Alignment, AlignmentBatch
from baclib.containers.record import Record, Feature, FeatureBatch
from baclib.core.interval import Interval
from baclib.io import BaseWriter, BaseReader, SeqFile


# Classes --------------------------------------------------------------------------------------------------------------
class TabularReader(BaseReader):
    """Base class for readers of tabular formats (GFF, BED, PAF)."""
    _delim = b'\t'
    _min_cols: int = 1
    __slots__ = ('_handle',)

    def __iter__(self) -> Generator:
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

    def _make_batch(self, items: list):
        """
        Creates a batch from a list of items.
        """
        return items

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


@SeqFile.register('bed')
class BedReader(TabularReader):
    """
    Reader for BED format files.

    Examples:
        >>> with open("features.bed", "rb") as f:
        ...     reader = BedReader(f)
        ...     for feature in reader:
        ...         print(feature.kind)
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
                if not line.strip() or line.startswith(b'track') or line.startswith(b'browser') or line.startswith(
                        b'#'):
                    continue
                parts = line.split(b'\t')
                return len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit()
            return False
        except Exception:
            return False
        
    def _make_batch(self, items: list) -> FeatureBatch:
        return FeatureBatch.from_features(items)


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
            line = b"\t".join([
                record.id, feature.interval.start, feature.interval.end,
                feature.get(b'Name', feature.get(b'ID', feature.get(b'gene', feature.kind))),
                feature.get(b'score', 0), feature.interval.strand.token
            ]) + b"\n"
            self._handle.write(line)


@SeqFile.register('gff')
class GffReader(TabularReader):
    """
    Reader for GFF3 format files.

    Examples:
        >>> with open("features.gff", "rb") as f:
        ...     reader = GffReader(f)
        ...     for feature in reader:
        ...         print(feature.kind)
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
        quals = self._QUALIFIER_PARSER.parse_gff_attributes(parts[8])
        quals.append((b'source', parts[0]))
        if parts[1] != b'.': quals.append((b'tool', parts[1]))
        if parts[5] != b'.': quals.append((b'score', float(parts[5])))
        if parts[7] != b'.': quals.append((b'phase', int(parts[7])))
        return Feature(Interval(start, end, parts[6]), parts[2], qualifiers=quals)
    
    @classmethod
    def sniff(cls, s: bytes) -> bool: return s.startswith(b'##gff')

    def _make_batch(self, items: list) -> FeatureBatch:
        return FeatureBatch.from_features(items)


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

        strand = feature.interval.strand.token

        phase = feature.get(b'phase', b'.')

        attr_strings = []
        if val := feature.get(b'ID'): attr_strings.append(b"ID=" + val)
        if val := feature.get(b'Name'): attr_strings.append(b"Name=" + val)
        for key, value in feature.qualifiers:
            if key in {b'source', b'score', b'phase', b'ID', b'Name'}: continue
            safe_val = val.replace(b';', b'%3B').replace(b'=', b'%3D').replace(b'&', b'%26')
            attr_strings.append(key + b"=" + safe_val)
        attr_block = b";".join(attr_strings) if attr_strings else b"."

        self._handle.write(
            b"\t".join([seq_id, source, feature.kind, start, end, score, strand, phase, attr_block]) + b"\n")


@SeqFile.register('paf')
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
        for key, value in self._QUALIFIER_PARSER.parse_tags(parts[12:]):
            if key == b'cg': cigar = value
            elif key == b'AS': score = value
            else: quals.append((key, value))

        return Alignment(
            query=parts[0], query_interval=Interval(int(parts[2]), int(parts[3])),
            target=parts[5], interval=Interval(int(parts[7]), int(parts[8]), parts[4].decode('ascii')),
            query_length=q_len, target_length=t_len, length=block_len, score=score,
            cigar=cigar, n_matches=n_matches, quality=int(parts[11]), qualifiers=quals
        )

    def _make_batch(self, items: list) -> AlignmentBatch:
        return AlignmentBatch(items)
    
    @classmethod
    def sniff(cls, s: bytes) -> bool:
        try:
            line = s.split(b'\n', 1)[0]
            parts = line.split(b'\t')
            return (len(parts) >= 12 and parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit() and
                    parts[6].isdigit() and parts[7].isdigit() and parts[8].isdigit())
        except Exception:
            return False
    
