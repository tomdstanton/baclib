from typing import Generator, BinaryIO

import numpy as np

from baclib.align.motif import Motif, Background
from baclib.core.seq import Alphabet
from baclib.io import BaseReader


class MotifReader(BaseReader):
    """Base class for motif readers."""
    def __init__(self, handle: BinaryIO, background: Background = None, **kwargs):
        super().__init__(handle, **kwargs)
        self.background = background or Background.uniform(Alphabet.dna())


class MemeReader(MotifReader):
    """Reader for MEME format files."""

    def __iter__(self) -> Generator[Motif, None, None]:
        lines = (line.strip() for line in self._handle)

        current_name = b"Unknown"
        matrix_lines = []
        in_matrix = False
        alphabet_map = [2, 1, 3, 0]  # Default MEME (ACGT) -> BacLib DNA (TCAG)

        for line in lines:
            if not line: continue

            if line.startswith(b'ALPHABET='):
                parts = line.split(b'=')
                if len(parts) > 1:
                    meme_alpha = parts[1].strip()
                    new_map = []
                    for char_code in meme_alpha:
                        enc = self.background.alphabet.encode(bytes([char_code]))
                        if len(enc) > 0 and enc[0] != self.background.alphabet.INVALID:
                            new_map.append(enc[0])
                        else:
                            new_map.append(-1)
                    alphabet_map = new_map

            elif line.startswith(b'MOTIF'):
                if matrix_lines:
                    yield self._make_motif(current_name, matrix_lines, alphabet_map)
                    matrix_lines = []
                    in_matrix = False

                parts = line.split()
                if len(parts) >= 2:
                    current_name = parts[1]
                    if len(parts) > 2:
                        current_name = parts[1] + b" " + parts[2]
                else:
                    current_name = b"Unknown"

            elif line.startswith(b'letter-probability matrix:'):
                in_matrix = True

            elif in_matrix:
                if line.startswith(b'URL') or line.startswith(b'MOTIF') or line.startswith(b'--------'):
                    in_matrix = False
                    if matrix_lines:
                        yield self._make_motif(current_name, matrix_lines, alphabet_map)
                        matrix_lines = []

                    if line.startswith(b'MOTIF'):
                        parts = line.split()
                        if len(parts) >= 2:
                            current_name = parts[1]
                            if len(parts) > 2:
                                current_name = parts[1] + b" " + parts[2]
                        else:
                            current_name = b"Unknown"
                else:
                    try:
                        float(line.split()[0])
                        matrix_lines.append(line)
                    except ValueError:
                        in_matrix = False

        if matrix_lines:
            yield self._make_motif(current_name, matrix_lines, alphabet_map)

    def _make_motif(self, name, lines, mapping):
        data = []
        for l in lines:
            data.append([float(x) for x in l.split()])

        raw_freqs = np.array(data, dtype=np.float32)
        n_rows, n_cols = raw_freqs.shape
        target_len = len(self.background.alphabet)

        mapped_freqs = np.zeros((n_rows, target_len), dtype=np.float32)

        for i in range(min(n_cols, len(mapping))):
            target_idx = mapping[i]
            if target_idx != -1 and target_idx < target_len:
                mapped_freqs[:, target_idx] = raw_freqs[:, i]

        return Motif.from_frequencies(name, mapped_freqs.T, self.background)


class TransfacReader(MotifReader):
    """Reader for TRANSFAC format files."""

    def __iter__(self) -> Generator[Motif, None, None]:
        lines = (line.strip() for line in self._handle)

        current_id = b"Unknown"
        counts = []
        base_order = None

        for line in lines:
            if not line: continue

            if line.startswith(b'ID'):
                parts = line.split(maxsplit=1)
                current_id = parts[1] if len(parts) > 1 else b"Unknown"
                counts = []
                base_order = None

            elif line.startswith(b'P0'):
                parts = line.split()
                base_order = [self.background.alphabet.encode(b)[0] if self.background.alphabet.encode(b).size > 0 else -1 for b in parts[1:]]

            elif line[0] in b'0123456789' and base_order is not None:
                parts = line.split()
                if len(parts) < 2 or not parts[0].isdigit(): continue
                try:
                    row_counts = [float(x) for x in parts[1:1 + len(base_order)]]
                    mapped_row = np.zeros(len(self.background.alphabet), dtype=np.float32)
                    for i, count in enumerate(row_counts):
                        if i < len(base_order) and base_order[i] != -1:
                            mapped_row[base_order[i]] = count
                    counts.append(mapped_row)
                except ValueError: continue

            elif line.startswith(b'//'):
                if counts:
                    yield Motif.from_counts(current_id, np.array(counts, dtype=np.float32).T, self.background)
                current_id = b"Unknown"
                counts = []
                base_order = None