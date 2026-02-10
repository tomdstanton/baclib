from typing import Union, Dict, BinaryIO
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

from baclib.utils.resources import RESOURCES
from baclib.io import SeqFile
from baclib.core.alphabet import Alphabet
from baclib.containers.record import Record, RecordBatch, FeatureKey
from baclib.containers.graph import Graph, Edge, EdgeBatch


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
    """
    __slots__ = ('id', 'contigs', 'edges', '_cached_graph')
    _ALPHABET = Alphabet.DNA
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
    def __contains__(self, item: bytes): return item in self.contigs
    def __repr__(self): return f"<Genome: {self.id.decode()}, {len(self.contigs)} contigs, {len(self.edges)} edges>"

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
        elif isinstance(file, SeqFile):
            # Try to infer ID from the underlying file handle/opener
            if hasattr(file, '_opener') and hasattr(file._opener, 'name') and file._opener.name:
                self.id = Path(file._opener.name).stem.encode()

        # Use batch processing for speed
        for batch in file.batches():
            if isinstance(batch, RecordBatch):
                for i in range(len(batch)):
                    record = batch[i]
                    self.contigs[record.id] = record
                    
                    # Auto-circularize if topology=circular
                    # Check qualifiers efficiently
                    is_circular = False
                    for k, v in record.qualifiers:
                        if k == b'topology' and v == b'circular':
                            is_circular = True
                            break
                    if is_circular and file.format != 'gfa':
                        self.edges.append(Edge(record.id, record.id, attributes={b'type': b'circular'}))

            elif isinstance(batch, EdgeBatch): self.edges.extend(batch)

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

    def write(self, file: Union[str, Path], format: str = 'fasta', **kwargs):
        """
        Writes the genome to a file.
        """
        from baclib.io.seq import FastaWriter, GfaWriter

        if format == 'fasta':
            with FastaWriter(file, **kwargs) as w:
                w.write(self.to_batch())
        elif format == 'gfa':
            with GfaWriter(file, **kwargs) as w:
                w.write(self.to_batch())
                if self.edges:
                    # GfaWriter handles lists of Edges
                    w.write(self.edges)
        else:
            raise ValueError(f"Unsupported write format: {format}")

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

        genome = cls(b"random_genome_%b" % rng.integers(0, 100_000))

        # Delegate sequence generation to the optimized batch method
        batch = cls._ALPHABET.random_batch(
            rng=rng,
            n_seqs=n_contigs,
            min_seqs=min_contigs,
            max_seqs=max_contigs,
            length=length,
            min_len=min_len,
            max_len=max_len,
            weights=weights
        )

        for i, seq in enumerate(batch):
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

    def extract_features(self, key: FeatureKey = FeatureKey.CDS) -> 'SeqBatch':
        """
        Extracts sequences for all features of a specific kind across the genome.
        Returns a single concatenated SeqBatch.
        """
        batches = []
        for record in self.contigs.values():
            # We can't easily use RecordBatch.extract_features here because records are separate.
            # But we can collect features and batch them.
            batches.extend(f.extract(record.seq) for f in record.features if f.key == key)
        return self._ALPHABET.batch_from(batches)
