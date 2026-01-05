from typing import Literal, IO, Union, Optional, Iterable, Dict
from pathlib import Path
from itertools import chain

import numpy as np

from . import RESOURCES
from .io import SeqFile
from .seq import Record, Alphabet
from .graph import Graph, Edge


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class GenomeError(Exception): pass


# Classes --------------------------------------------------------------------------------------------------------------
class Genome:
    """
    A class representing a single bacterial genome assembly in memory.

    It holds contigs as `Record` objects and the connections between them as `Edge` objects.

    Attributes:
        id (str): The identifier for the genome.
        contigs (Dict[str, Record]): A dictionary mapping contig IDs to Record objects.
        edges (list[Edge]): A list of edges representing connections between contigs.

    Examples:
        >>> from baclib.genome import Genome
        >>> # Load a genome from a FASTA file
        >>> # genome = Genome.from_file("my_genome.fasta")
        >>> # print(f"Genome ID: {genome.id}")
        >>> # print(f"Total length: {len(genome)}")
        >>> # Access a specific contig
        >>> # if "contig_1" in genome.contigs:
        ...     # print(genome["contig_1"])

    """
    _ALPHABET = Alphabet.dna()
    __slots__ = ('id', 'contigs', 'edges', '_cached_graph')

    def __init__(self, id_: str = 'unknown', contigs: Dict[str, Record] = None, edges: list[Edge] = None):
        """Represents a single bacterial genome to be loaded into memory from a file"""
        self.id: str = id_
        self.contigs: Dict[str, Record] = contigs or {}
        self.edges: list[Edge] = edges or []
        self._cached_graph: Optional[Graph] = None

    def __len__(self): return sum(len(i) for i in self.contigs.values())
    def __iter__(self): return iter(self.contigs.values())
    def __str__(self): return self.id
    def __getitem__(self, item: str) -> 'Record': return self.contigs[item]

    @classmethod
    def from_file(cls, file: Union[str, Path, IO, SeqFile], annotations: Union[str, Path, SeqFile] = None):
        """
        Loads a genome from a file with optional annotations.
        """
        self = cls()
        if isinstance(file, str): file = Path(file)
        if isinstance(file, Path): self.id = file.stem
        if isinstance(file, IO): self.id = file.name
        if not isinstance(file, SeqFile): file = SeqFile(file)
        for record in file:
            if isinstance(record, Record):
                self.contigs[record.id] = record
                # Auto-circularize if topology=circular
                is_circular = False
                for k, v in record.qualifiers:
                    if k == 'topology' and v == 'circular':
                        is_circular = True
                        break
                if is_circular and file.format != 'gfa':
                    self.edges.append(Edge(record.id, record.id, {'type': 'circular'}))

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
                    if k == 'source':
                        target_id = v
                        break

                if target_id and target_id in self.contigs: self.contigs[target_id].features.append(feature)

        return self

    @classmethod
    def random(cls, rng: np.random.Generator = None, n_contigs: int = None, min_contigs: int = 1, max_contigs: int = 1000,
               length: int = None, min_len: int = 10, max_len: int = 5000000, weights=None):
        """Generates a random genome assembly for testing purposes."""
        if rng is None: rng = RESOURCES.rng
        if n_contigs is None: n_contigs = rng.randint(min_contigs, max_contigs)

        # Create one massive sequence then shred it
        full_seq = cls._ALPHABET.random(rng, length, min_len, max_len, weights)
        master_record = Record(cls._ALPHABET.seq(full_seq), id_='random_genome')

        genome = cls(master_record.id)
        # Assuming Record.shred returns an iterable of Records
        for contig in master_record.shred(rng, n_contigs):
            genome.contigs[contig.id] = contig

        return genome

    def annotated(self) -> bool:
        """Fast check if any contig has features."""
        return any(bool(c.features) for c in self.contigs.values())

    def as_graph(self, qualifiers: Iterable[str] = ()) -> Graph:
        if self._cached_graph is None:
            g = Graph()
            # Add Nodes
            for contig in self.contigs.values():
                # Extract specific qualifiers if requested
                attrs = {k: v for k, v in contig.qualifiers if k in qualifiers}
                g.add_node(contig.id, attrs)
            # Add Edges
            for edge in self.edges: g.add_edge(edge)
            self._cached_graph = g
        return self._cached_graph
