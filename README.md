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

**Lightweight Python Library for Bacterial Genomics**

`baclib` is a high-performance, modern Python library designed for bioinformatics and bacterial genomics. 
It provides efficient data structures and algorithms for handling sequences, alignments, assembly graphs, and genomic 
features, with a focus on speed and ease of use.

## Features

*   **Unified I/O**: Read and write common formats (FASTA, FASTQ, GFF3, BED, GenBank, GFA, PAF) with automatic format 
                     detection and compression support (gzip, bzip2, xz, zstd).
*   **Efficient Sequence Handling**:
    *   DNA and Amino Acid alphabets with validation.
    *   Fast reverse complement and translation.
    *   Interval arithmetic and feature management.
*   **Alignment Engine**:
    *   Pairwise alignment (Smith-Waterman, Needleman-Wunsch) with affine gap penalties.
    *   Fast K-mer indexing and Jaccard similarity search.
    *   Integration with **Minimap2** for high-throughput alignment.
*   **Graph Algorithms**:
    *   Assembly graph processing (GFA support).
    *   Dijkstra's algorithm, connected components, and greedy set cover.
*   **Genomic Features**:
    *   Rich representation of Genomes, Contigs, and Features (genes, CDS, etc.).
    *   Mutation detection (SNP/Indel) from alignments.
    *   Motif finding (e.g., promoters).
*   **External Tool Wrappers**:
    *   Seamless interfaces for `Minimap2` and `FragGeneScanRs`.

## Installation

Requires Python 3.11+.

```bash
pip install baclib
```

For documentation generation support:
```bash
pip install baclib[docs]
```

## Quick Start

### Reading Sequences

`baclib` automatically detects file formats and compression.

```python
from baclib.io import SeqFile

# Read a FASTA file (gzip supported automatically)
with SeqFile("genome.fasta.gz") as reader:
    for record in reader:
        print(f"ID: {record.id}, Length: {len(record)}")
        print(f"Sequence: {record.seq[:50]}...")

# Read a GenBank file with features
with SeqFile("annotation.gbk") as reader:
    for record in reader:
        for feature in record.features:
            if feature.kind == 'CDS':
                # Access qualifiers easily
                gene = feature['gene']
                print(f"Gene: {gene}")
```

### Sequence Manipulation

```python
from baclib.seq import Alphabet

dna = Alphabet.dna()
seq = dna.seq("ATGCGTAGCTAG")

# Or generate a random seq
seq = dna.seq(dna.generate_seq())

# Reverse complement
# Note: Returns a new Seq object
rc_seq = seq.reverse_complement()
print(rc_seq)

# Translate to protein
protein = dna.translate(seq)
print(protein)
```

### Pairwise Alignment

Perform local (Smith-Waterman), global (Needleman-Wunsch) or glocal alignment.

```python
from baclib.alignment import PairwiseAligner
from baclib.seq import Alphabet, Record


dna = Alphabet.dna()
aligner = PairwiseAligner(dna, k=5, flavour='local', compute_traceback=True)

target = Record(dna.seq(dna.generate_seq(length=1000)), "target")
query = Record(dna.seq(dna.generate_seq(length=100)), "query")

# Add targets to the index
if alignment := aligner.align(query, target):
    print(f"Score: {alignment.score}")
    print(f"CIGAR: {alignment.cigar}")
```

### Using Minimap2

Ensure `minimap2` is installed and in your PATH.

```python
from baclib.external import Minimap2
from baclib.seq import Record, Alphabet

# Generate random records
dna = Alphabet.dna()
ref = Record(dna.seq(dna.generate_seq(length=1000)), "ref")
query = Record(dna.seq(dna.generate_seq(length=100)), "query")

# Align using Minimap2 wrapper (handles indexing automatically)
with Minimap2(ref) as mapper:
    for alignment in mapper.align(query):
        print(f"Query: {alignment.query} -> Target: {alignment.target}")
        print(f"Matches: {alignment.n_matches}")
```

## Dependencies

*   `numpy`
*   `scipy`
*   `numba` (Optional, but highly recommended for performance)

## License

This project is licensed under the terms of the license found in the `LICENSE` file.
