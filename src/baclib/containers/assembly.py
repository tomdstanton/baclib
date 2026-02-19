from typing import Union, Dict, BinaryIO
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

from baclib.lib.resources import RESOURCES
from baclib.io import SeqFile
from baclib.core.alphabet import Alphabet
from baclib.core.interval import Strand
from baclib.containers.record import Record, RecordBatch, FeatureKey
from baclib.containers.graph import Graph, Edge, EdgeBatch


# Exceptions and Warnings ----------------------------------------------------------------------------------------------
class GenomeAssemblyError(Exception): pass


# Classes --------------------------------------------------------------------------------------------------------------
class GenomeAssembly:
    """
    A class representing a single bacterial genome assembly in memory.
    It holds contigs as `Record` objects and the connections between them as `Edge` objects.
    """
    __slots__ = ('_contigs', '_edges', '_cached_graph', '_seq_map', '_redundancy_map')
    _ALPHABET = Alphabet.DNA
    _SEQ_FORMATS = {SeqFile.Format.FASTA, SeqFile.Format.GFA, SeqFile.Format.GENBANK}
    _ANNOTATION_FORMATS = {SeqFile.Format.BED, SeqFile.Format.GFF}
    def __init__(self):
        """Represents a single bacterial genome to be loaded into memory from a file"""
        self._contigs: Dict[bytes, Record] = {}
        self._edges: list[Edge] = []
        self._cached_graph: Optional[Graph] = None
        self._seq_map: Dict[int, bytes] = {}
        self._redundancy_map: Dict[bytes, bytes] = {}

    def __len__(self): return sum(len(i) for i in self._contigs.values())
    def __iter__(self): return iter(self._contigs.values())
    def __str__(self): return self.id
    def __getitem__(self, item: bytes) -> 'Record': return self._contigs[item]
    def __contains__(self, item: bytes): return item in self._contigs
    def __repr__(self): return f"<Genome: {len(self._contigs)} contigs, {len(self._edges)} edges>"

    @property
    def contigs(self) -> Dict[bytes, Record]: return self._contigs
    @property
    def edges(self) -> list[Edge]: return self._edges
    @property
    def redundancy_map(self) -> Dict[bytes, bytes]: return self._redundancy_map

    @classmethod
    def from_file(cls, file: Union[str, Path, BinaryIO, SeqFile], annotations: Union[str, Path, BinaryIO, SeqFile] = None):
        """
        Loads a genome from a file with optional annotations.
        """
        if not isinstance(file, SeqFile): file = SeqFile(file)
        if file.format not in cls._SEQ_FORMATS:
            raise GenomeAssemblyError(f'Genome assemblies must come from {cls._SEQ_FORMATS}, not {file.format}')

        self = cls()  # cls(file.name.encode())

        # Use batch processing for speed
        for batch in file.batches():
            if isinstance(batch, RecordBatch):
                self.add_batch(batch)
                
                # Auto-circularize if topology=circular (FASTA/GenBank)
                if file.format != SeqFile.Format.GFA:
                    for i in range(len(batch)):
                        # Check qualifiers efficiently without full record reconstruction
                        quals = batch.get_qualifiers(i)
                        for k, v in quals:
                            if k == b'topology' and v == b'circular':
                                rec_id = batch.ids[i]
                                self.add_edge(Edge(rec_id, rec_id, Strand.FORWARD, Strand.FORWARD))
                                self.add_edge(Edge(rec_id, rec_id, Strand.REVERSE, Strand.REVERSE))
                                break

            elif isinstance(batch, EdgeBatch): 
                for e in batch: self.add_edge(e)

        # 2. Load Annotations
        if annotations:
            if not isinstance(annotations, SeqFile): annotations = SeqFile(annotations)
            if annotations.format not in cls._ANNOTATION_FORMATS:
                raise GenomeAssemblyError(f'Annotations must come from {cls._ANNOTATION_FORMATS}, not {annotations.format}')
            # Iterate features and link to contigs via 'source' qualifier
            for feature in annotations:
                # Find the contig ID in the qualifiers
                target_id = None
                for k, v in feature.qualifiers:
                    if k == b'source':
                        target_id = v
                        break

                if target_id and target_id in self._contigs: self._contigs[target_id].features.append(feature)

        return self

    @classmethod
    def from_batch(cls, batch: RecordBatch) -> 'GenomeAssembly':
        """
        Creates a GenomeAssembly from a RecordBatch.
        """
        assembly = cls()
        assembly.add_batch(batch)
        return assembly

    def add_record(self, record: Record):
        """
        Adds a contig to the genome, handling redundancy.
        If an identical sequence exists, the new record shares the sequence data.
        """
        # 1. Check for Sequence Redundancy
        seq_hash = hash(record.seq)
        
        if canonical_id := self._seq_map.get(seq_hash):
            canonical = self._contigs[canonical_id]
            # Verify hash collision
            if canonical.seq == record.seq:
                # Redundant: Point to canonical sequence to save memory/ensure consistency
                # We create a new Record wrapper to keep the ID/Features of the new one,
                # but sharing the sequence view of the old one.
                if record.seq is not canonical.seq:
                    new_rec = Record(canonical.seq, record.id, record.description, qualifiers=record.qualifiers)
                    new_rec.features = record.features
                    record = new_rec
                self._redundancy_map[record.id] = canonical_id
        else:
            self._seq_map[seq_hash] = record.id
        self._contigs[record.id] = record
        self._cached_graph = None # Invalidate graph

    def add_batch(self, batch: RecordBatch):
        """
        Adds a batch of records to the assembly.
        Optimized to handle deduplicated batches by caching sequence hashes.
        """
        # Optimization: Cache sequence hashes by (start, length) to avoid re-hashing
        # identical sequences within the batch (e.g. if batch is deduplicated).
        seqs = batch.seqs
        starts = seqs.starts
        lengths = seqs.lengths
        hash_cache = {}

        for i in range(len(batch)):
            record = batch[i]
            # Use (start, length) as key for content identity within this batch
            key = (starts[i], lengths[i])
            
            if key in hash_cache:
                # Inject cached hash into Seq object to avoid re-computation
                # Accessing private _hash slot is safe within the library
                record.seq._hash = hash_cache[key]
            else:
                # Compute and cache
                hash_cache[key] = hash(record.seq)
            
            self.add_record(record)

    def add_edge(self, edge: Edge):
        """Adds an edge to the genome graph."""
        self._edges.append(edge)
        self._cached_graph = None

    def annotated(self) -> bool:
        """Fast check if any contig has features."""
        return any(bool(c.features) for c in self._contigs.values())

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
            for contig in self._contigs.values():
                # Extract specific qualifiers if requested
                attrs = {k: v for k, v in contig.qualifiers if k in node_attributes}
                g.add_node(contig.id, attrs)
            # Add Edges
            if self._edges: g.add_edges(self._edges)
            self._cached_graph = g
        return self._cached_graph

    def to_batch(self) -> RecordBatch:
        """Converts the Genome into a read-only, optimized BatchRecord."""
        return RecordBatch(self._contigs.values(), deduplicate=True)

    def extract_features(self, key: FeatureKey = FeatureKey.CDS) -> 'SeqBatch':
        """
        Extracts sequences for all features of a specific kind across the genome.
        Returns a single concatenated SeqBatch.
        """
        batches = []
        for record in self._contigs.values():
            # We can't easily use RecordBatch.extract_features here because records are separate.
            # But we can collect features and batch them.
            batches.extend(f.extract(record.seq) for f in record.features if f.key == key)
        return self._ALPHABET.batch_from(batches)
