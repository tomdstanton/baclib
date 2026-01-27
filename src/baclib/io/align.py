from baclib.align.alignment import Alignment
from baclib.core.interval import Interval
from baclib.io import TabularReader


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
