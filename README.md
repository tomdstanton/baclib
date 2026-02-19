# baclib

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/gpl-3.0/)
[![PyPI](https://img.shields.io/pypi/v/baclib.svg?style=flat-square&maxAge=3600&logo=PyPI)](https://pypi.org/project/baclib)
[![Wheel](https://img.shields.io/pypi/wheel/baclib.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/baclib/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/tomdstanton/baclib/)
[![Issues](https://img.shields.io/github/issues/tomdstanton/baclib.svg?style=flat-square&maxAge=600)](https://github.com/tomdstanton/baclib/issues)
[![pages-build-deployment](https://github.com/tomdstanton/baclib/actions/workflows/docs.yml/badge.svg?style=flat-square)](https://github.com/tomdstanton/baclib/actions/workflows/docs.yml)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/tomdstanton/baclib/blob/main/CHANGELOG.md)

> [!WARNING]
> 🚧 This package is currently under construction, proceed with caution 🚧

**High-Performance Python Library for Bacterial Genomics**

`baclib` is a modern, Numba-accelerated library designed for high-throughput bacterial genomics. Unlike traditional bioinformatics libraries that prioritize flexibility over speed, `baclib` focuses on raw performance and memory efficiency. It utilizes **Structure-of-Arrays (SoA)** layouts, **binary string processing**, and **JIT compilation** to handle large-scale genomic data structures, alignments, and assembly graphs efficiently.

## Key Features

*   **Binary-First I/O**: All file parsing (FASTA, FASTQ, GenBank, GFA, PAF, GFF3) is performed in binary mode. This avoids costly string encoding/decoding overhead and minimizes memory usage.
*   **Numba-Accelerated Kernels**: Core algorithms for alignment, seeding (Minimizers, Syncmers), and interval arithmetic are JIT-compiled for C-like performance.
*   **Vectorized Containers**:
    *   `SeqBatch`: Stores thousands of sequences in contiguous memory for SIMD-friendly processing.
    *   `AlignmentBatch`: Manages millions of alignment records with NumPy-backed storage.
    *   `IntervalIndex`: Fast overlap queries and set operations for genomic features.
*   **Alignment Engine**:
    *   Built-in pairwise aligner (Smith-Waterman, Needleman-Wunsch) with affine gap penalties.
    *   Fast K-mer indexing (MinHash, Minimizers) for rapid sequence comparison.
    *   Seamless integration with **Minimap2** via direct process streaming.
*   **Assembly Graph Toolkit**: Native support for GFA graphs, including pathfinding, simplification, and topological analysis uses efficient `RecordBatch` backends.
*   **API Clients**: Includes a client for the PRODORIC database (`ProdoricClient`) for retrieving transcriptional regulation data.

## ⚠️ Important: Binary Strings

To achieve maximum performance, `baclib` operates almost exclusively with **bytes** (`b'string'`) rather than Python unicode strings (`'string'`).

*   **Identifiers**: `record.id`, `feature.kind`, and dictionary keys are `bytes`.
*   **Sequences**: DNA/Protein data is stored as `uint8` numpy arrays but converts to `bytes`.
*   **I/O**: Readers expect and return binary data.

**Example:**
```python
# Correct
record_id = b"contig_1"
feature_type = b"CDS"

# Incorrect (will likely fail lookups or comparisons)
record_id = "contig_1"
```

This design choice ensures zero-copy compatibility with low-level parsers and external tools.

## Installation

Requires Python 3.11+.

```bash
python -m pip install baclib
```

For documentation generation support:
```bash
python -m pip install baclib --group docs
```

For developers:
```bash
python -m pip install baclib -e --group dev
```

## Quick Start

### Reading Sequences

```python
from baclib.io import SeqFile

# Read a FASTA file (gzip supported automatically)
# Use SeqFile.open() as the unified entry point
with SeqFile.open("genome.fasta.gz") as reader:
    for record in reader:
        # record.id is bytes!
        print(f"ID: {record.id.decode()}, Length: {len(record)}")
        print(f"Sequence: {record.seq[:50]}...")

# Read a GenBank file with features
with SeqFile.open("annotation.gbk") as reader:
    for record in reader:
        for feature in record.features:
            if feature.key == b'CDS':
                # Access qualifiers using bytes keys
                gene = feature.get(b'gene')
                print(f"Gene: {gene}")
```

### Sequence Manipulation

```python
from baclib.containers.seq import Alphabet, GeneticCode

dna = Alphabet.DNA

# Create a sequence from string
seq = dna.seq_from("ATGCGTAGCTAG")

# Or generate a random sequence
seq = dna.random_seq(length=100)

# Reverse complement
rc_seq = dna.reverse_complement(seq)
print(rc_seq)

# Translate to protein using Bacterial code (Table 11)
gc = GeneticCode.from_code(11)
protein = gc.translate(seq)
print(protein)
```

### Pairwise Alignment

Perform local, global, or glocal alignment using the built-in high-performance aligner.

```python
from baclib.engines.pairwise import Aligner
from baclib.containers.seq import Alphabet, SeqBatch

dna = Alphabet.DNA

# Create random sequences
targets = dna.random_batch(n_seqs=5, length=1000)
queries = dna.random_batch(n_seqs=2, length=100)

# Initialize aligner (Glocal mode: Global in Query, Local in Target)
aligner = Aligner(mode='glocal', compute_traceback=True)

# Build index on targets
aligner.build(targets)

# Map queries to targets
hits = aligner.map(queries, min_score=50)

for hit in hits:
    print(f"Query {hit.query} maps to Target {hit.target}")
    print(f"Score: {hit.score}, CIGAR: {hit.cigar}")
```

### Using Minimap2

Ensure `minimap2` is installed and in your PATH.

```python
from baclib.lib.external import Minimap2
from baclib.containers.record import RecordBatch
from baclib.containers.seq import Alphabet

dna = Alphabet.DNA
ref = RecordBatch.random(dna, n_seqs=100, length=1000)
queries = RecordBatch.random(dna, n_seqs=10, length=100)

# Align using Minimap2 wrapper (handles indexing automatically)
with Minimap2(ref) as mapper:
    for alignment in mapper.align(queries):
        print(f"Query: {alignment.queries} -> Target: {alignment.target}")
        print(f"Matches: {alignment.n_matches}")
```

### Genome Assembly Graph

Load and analyze assembly graphs (GFA format).

```python
from baclib.containers.genome import GenomeAssembly

# Load GFA graph
assembly = GenomeAssembly.from_file("assembly.gfa")

print(f"Edges: {assembly.edges}")
```

### PRODORIC Client

Fetch regulatory data directly from the PRODORIC database.

```python
from baclib.apis.prodoric import ProdoricClient

client = ProdoricClient()

# Search for motifs
motifs = client.search_motifs("LexA")
for m in motifs:
    print(f"Found motif: {m.name} ({m.accession})")

# Get a matrix details
motif = client.get_motif(motifs[0].accession)
print(motif.name)
```

## Dependencies

*   `numpy`
*   `scipy`
*   `numba` (Optional, but highly recommended for performance)

## License

This project is licensed under the terms of the license found in the `LICENSE` file.
