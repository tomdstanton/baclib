from typing import Union, Dict, BinaryIO
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

from baclib.utils.resources import RESOURCES
from baclib.io.dispatcher import SeqFile
from baclib.core.seq import Alphabet
from baclib.containers.record import Record, RecordBatch
from baclib.containers.graph import Graph, Edge


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class GenomeError(Exception): pass


# Classes --------------------------------------------------------------------------------------------------------------
class Genome:
    """
    A class representing a single bacterial genome assembly in memory.
    It holds contigs as `Record` objects and the connections between them as `Edge` objects.

    Attributes:
        id (bytes): The genome identifier.
        contigs (Dict[bytes, Record]): A dictionary mapping contig IDs to Record objects.
        edges (list[Edge]): A list of edges representing the assembly graph.

    Examples:
        >>> g = Genome(id_=b'Ecoli_K12')
        >>> r = Record(Alphabet.dna().seq("ACGT"), id_=b'contig1')
        >>> g.contigs[b'contig1'] = r
        >>> len(g)
        4
    """
    __slots__ = ('id', 'contigs', 'edges', '_cached_graph')
    _ALPHABET = Alphabet.dna()
    def __init__(self, id_: bytes = b'unknown', contigs: Dict[bytes, Record] = None, edges: list[Edge] = None):
        """Represents a single bacterial genome to be loaded into memory from a file"""
        self.id: bytes = id_
        self.contigs: Dict[bytes, Record] = contigs or {}
        self.edges: list[Edge] = edges or []
        self._cached_graph: Optional[Graph] = None

    def __len__(self): return sum(len(i) for i in self.contigs.values())
    def __iter__(self): return iter(self.contigs.values())
    def __str__(self): return self.id
    def __getitem__(self, item: bytes) -> 'Record': return self.contigs[item]

    @classmethod
    def from_file(cls, file: Union[str, Path, BinaryIO, SeqFile], annotations: Union[str, Path, BinaryIO, SeqFile] = None):
        """
        Loads a genome from a file with optional annotations.

        Args:
            file: The sequence file (FASTA, GFA, GenBank).
            annotations: Optional annotation file (GFF, BED).

        Returns:
            A Genome object.

        Raises:
            GenomeError: If file formats are incompatible.
        """
        self = cls()
        if isinstance(file, str): file = Path(file)
        if isinstance(file, Path):
            self.id = file.stem.encode()
            file = SeqFile(file)

        elif isinstance(file, BinaryIO):
            self.id = file.name.encode()
            file = SeqFile(file)

        for record in file:
            if isinstance(record, Record):
                self.contigs[record.id] = record
                # Auto-circularize if topology=circular
                is_circular = False
                for k, v in record.qualifiers:
                    if k == b'topology' and v == b'circular':
                        is_circular = True
                        break
                if is_circular and file.format != 'gfa':
                    self.edges.append(Edge(record.id, record.id, {b'type': b'circular'}))

            elif isinstance(record, Edge): self.edges.append(record)

        # 2. Load Annotations
        if annotations:
            if file.format not in {'fasta', 'gfa'}:
                raise GenomeError(f'Can only provide annotations to FASTA/GFA files, not {file.format}')
            if not isinstance(annotations, SeqFile): annotations = SeqFile(annotations)
            if annotations.format not in {'gff', 'bed'}:
                raise GenomeError(f'Annotations must be in GFF or BED format, not {annotations.format}')
            # Iterate features and link to contigs via 'source' qualifier
            for feature in annotations:
                # Find the contig ID in the qualifiers
                target_id = None
                for k, v in feature.qualifiers:
                    if k == b'source':
                        target_id = v
                        break

                if target_id and target_id in self.contigs: self.contigs[target_id].features.append(feature)

        return self

    @classmethod
    def random(cls, rng: np.random.Generator = None, n_contigs: int = None, min_contigs: int = 1,
               max_contigs: int = 1000, length: int = None, min_len: int = 10, max_len: int = 5_000_000, weights=None):
        """
        Generates a random genome assembly for testing purposes.

        Args:
            rng: Random number generator.
            n_contigs: Number of contigs.
            min_contigs: Min contigs if n_contigs is None.
            max_contigs: Max contigs if n_contigs is None.
            length: Total length of the genome.
            min_len: Min length per contig.
            max_len: Max length per contig.
            weights: Symbol weights.

        Returns:
            A random Genome object.
        """
        if rng is None: rng = RESOURCES.rng
        if n_contigs is None: n_contigs = int(rng.integers(min_contigs, max_contigs))

        genome = cls(b"random_genome_%b" % rng.integers(0, 100_000))

        # Optimization: Generate contigs directly instead of shredding a massive sequence
        if length is not None:
            # Partition total length into n_contigs
            if n_contigs > 1:
                cuts = np.sort(rng.integers(0, length, size=n_contigs - 1))
                bounds = np.concatenate(([0], cuts, [length]))
                lengths = np.diff(bounds)
            else:
                lengths = [length]
        else:
            lengths = rng.integers(min_len, max_len, size=n_contigs)

        lengths = np.maximum(1, lengths)
        for i, seq in enumerate(cls._ALPHABET.random_many(lengths, rng=rng, weights=weights)):
            contig = Record(seq, id_=b"contig_%d" % (i+1))
            genome.contigs[contig.id] = contig

        return genome

    def annotated(self) -> bool:
        """Fast check if any contig has features."""
        return any(bool(c.features) for c in self.contigs.values())

    def as_graph(self, node_attributes: Iterable[bytes] = ()) -> Graph:
        """
        Returns the assembly graph representation.

        Args:
            node_attributes: List of qualifier keys to include as node attributes.

        Returns:
            A Graph object.
        """
        if self._cached_graph is None:
            g = Graph()
            # Add Nodes
            for contig in self.contigs.values():
                # Extract specific qualifiers if requested
                attrs = {k: v for k, v in contig.qualifiers if k in node_attributes}
                g.add_node(contig.id, attrs)
            # Add Edges
            if self.edges: g.add_edges(self.edges)
            self._cached_graph = g
        return self._cached_graph

    def to_batch(self) -> RecordBatch:
        """Converts the Genome into a read-only, optimized BatchRecord."""
        return RecordBatch(self.contigs.values())
