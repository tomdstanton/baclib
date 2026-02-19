"""Container for representing bacterial genome assemblies composed of contigs and edges."""
from typing import Union, BinaryIO, Iterable
from pathlib import Path

import numpy as np

from baclib.io import SeqFile, SeqFileFormat
from baclib.core.alphabet import Alphabet
from baclib.core.interval import Strand
from baclib.containers.record import Record, RecordBatch, FeatureKey
from baclib.containers.feature import FeatureBatch
from baclib.containers.graph import Graph, Edge, EdgeBatch


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class GenomeAssemblyError(Exception):
    """Raised when a genome assembly operation fails (e.g. missing contigs, format mismatch)."""


# Classes --------------------------------------------------------------------------------------------------------------
class GenomeAssembly:
    """
    An immutable representation of a bacterial genome assembly.

    Backed by a single ``RecordBatch`` for efficient, vectorized access to sequences
    and features. Contigs are keyed by ID for O(1) lookup.

    Args:
        contigs: A ``RecordBatch`` containing the contig records.
        edges: Optional ``EdgeBatch`` of connectivity edges (e.g. from GFA).

    Examples:
        >>> assembly = GenomeAssembly.from_file('genome.fasta')
        >>> assembly.n_contigs
        5
        >>> assembly[b'contig_1']  # O(1) lookup by ID
        Record(...)
    """
    __slots__ = ('_contigs', '_edges', '_cached_graph', '_id_index')
    _ALPHABET = Alphabet.DNA
    _SEQ_FORMATS = {SeqFileFormat.FASTA, SeqFileFormat.GFA, SeqFileFormat.GENBANK}
    _ANNOTATION_FORMATS = {SeqFileFormat.BED, SeqFileFormat.GFF}

    def __init__(self, contigs: RecordBatch, edges: EdgeBatch = None):
        self._contigs = contigs
        self._edges: EdgeBatch = edges or EdgeBatch.empty()
        self._cached_graph = None
        self._id_index = {contigs.ids[i]: i for i in range(len(contigs))}

    # --- Dunder methods ---
    def __len__(self): return int(np.sum(self._contigs.seqs.lengths))
    def __iter__(self): return iter(self._contigs)
    def __getitem__(self, item: bytes) -> Record: return self._contigs[self._id_index[item]]
    def __contains__(self, item: bytes): return item in self._id_index
    def __repr__(self): return f"<Genome: {len(self._contigs)} contigs, {len(self._edges)} edges>"

    # --- Properties ---
    @property
    def contigs(self) -> RecordBatch:
        """Returns the underlying ``RecordBatch`` of all contigs.

        Returns:
            The contig ``RecordBatch``.
        """
        return self._contigs

    @property
    def n_contigs(self) -> int:
        """Returns the number of contigs in the assembly.

        Returns:
            Contig count.
        """
        return len(self._contigs)

    @property
    def ids(self) -> np.ndarray:
        """Returns the contig IDs as a numpy object array.

        Returns:
            An object array of ``bytes`` contig IDs.
        """
        return self._contigs.ids

    @property
    def edges(self) -> EdgeBatch:
        """Returns the connectivity edges between contigs.

        Returns:
            An ``EdgeBatch`` of assembly edges.
        """
        return self._edges

    # --- Construction ---
    @classmethod
    def from_file(cls, file: Union[str, Path, BinaryIO, SeqFile],
                  annotations: Union[str, Path, BinaryIO, SeqFile] = None,
                  deduplicate: bool = True):
        """Loads a genome assembly from a sequence file with optional annotations.

        Supports FASTA, GFA, and GenBank sequence formats, with optional
        BED or GFF annotation overlays. Circular contigs are auto-detected
        from ``topology=circular`` qualifiers and expressed as self-loop edges.

        Args:
            file: Path, file handle, or ``SeqFile`` for the sequence data.
            annotations: Optional path or handle to a BED or GFF annotation file.
            deduplicate: Whether to deduplicate identical contig sequences
                (default ``True``).

        Returns:
            A new ``GenomeAssembly``.

        Raises:
            GenomeAssemblyError: If the file format is unsupported or contains
                no sequence records.

        Examples:
            >>> assembly = GenomeAssembly.from_file('genome.gfa')
            >>> assembly = GenomeAssembly.from_file('genome.fasta', annotations='features.gff')
        """
        if not isinstance(file, SeqFile): file = SeqFile(file)
        if file.format not in cls._SEQ_FORMATS:
            raise GenomeAssemblyError(f'Genome assemblies must come from {cls._SEQ_FORMATS}, not {file.format}')

        contigs, edges = [], []
        for batch in file.batches():
            if isinstance(batch, RecordBatch):
                contigs.append(batch)
            elif isinstance(batch, EdgeBatch):
                edges.append(batch)

        if not contigs:
            raise GenomeAssemblyError('File contained no sequence records')

        contigs = RecordBatch.concat(contigs, deduplicate=deduplicate)

        # Auto-circularize contigs with topology=circular (FASTA/GenBank)
        if file.format != SeqFileFormat.GFA:
            for i in range(len(contigs)):
                for k, v in contigs.get_qualifiers(i):
                    if k == b'topology' and v == b'circular':
                        rid = contigs.ids[i]
                        edges.append(EdgeBatch.build(Edge(rid, rid, i) for i in (Strand.FORWARD, Strand.REVERSE)))
                        break

        # Load annotations
        if annotations:
            if not isinstance(annotations, SeqFile): annotations = SeqFile(annotations)
            if annotations.format not in cls._ANNOTATION_FORMATS:
                raise GenomeAssemblyError(
                    f'Annotations must come from {cls._ANNOTATION_FORMATS}, not {annotations.format}')

            feature_batches = [b for b in annotations.batches() if isinstance(b, FeatureBatch)]
            if feature_batches:
                all_features = FeatureBatch.concat(feature_batches) if len(feature_batches) > 1 else feature_batches[0]
                contigs = contigs.add_features(all_features, pivot_key=FeatureKey.SOURCE)

        if edges:
            edges = EdgeBatch.concat(edges) if len(edges) > 1 else edges[0]
        else:
            edges = EdgeBatch.empty()

        return cls(contigs, edges)

    def n_features(self) -> int:
        """Returns the total number of features across all contigs.

        Returns:
            Total feature count.
        """
        return self._contigs.n_features

    def as_graph(self, node_attributes: Iterable[bytes] = ()) -> Graph:
        """Returns the assembly graph representation.

        Nodes correspond to contigs, edges represent physical connectivity
        (e.g. GFA links, circular topology). The graph is cached and rebuilt
        only if the assembly changes.

        Args:
            node_attributes: Qualifier keys to include as node attributes
                (e.g. ``[b'topology', b'length']``).

        Returns:
            A ``Graph`` object.

        Examples:
            >>> g = assembly.as_graph(node_attributes=[b'topology'])
            >>> g.nodes
            [b'contig_1', b'contig_2']
        """
        if self._cached_graph is None:
            g = Graph()
            for i in range(len(self._contigs)):
                rec_id = self._contigs.ids[i]
                quals = self._contigs.get_qualifiers(i)
                attrs = {k: v for k, v in quals if k in node_attributes}
                g.add_node(rec_id, attrs)
            if self._edges: g.add_edges(self._edges)
            self._cached_graph = g
        return self._cached_graph

    def extract_features(self, key: FeatureKey = FeatureKey.CDS) -> 'SeqBatch':
        """Extracts sequences for all features of a given type across the genome.

        Uses the vectorized ``RecordBatch.extract_features`` kernel for
        high-performance extraction.

        Args:
            key: The ``FeatureKey`` to filter by (default ``CDS``).

        Returns:
            A ``SeqBatch`` of the extracted feature sequences.

        Examples:
            >>> cds_seqs = assembly.extract_features(FeatureKey.CDS)
            >>> trna_seqs = assembly.extract_features(FeatureKey.TRNA)
        """
        return self._contigs.extract_features(key.bytes)
