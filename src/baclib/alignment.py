"""
Module for aligning sequences and managing alignments.
"""
from typing import Generator, Union, Optional, List, Dict, Literal
from concurrent.futures import Executor
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from . import jit, RESOURCES
from .seq import Alphabet, AlphabetError, Record, Feature, Qualifier, IntervalIndex, Interval
from .graph import Graph, Edge, WeightingPolicy


# Classes --------------------------------------------------------------------------------------------------------------
class Hit:
    """
    Intermediate representation of a similarity event.
    Lighter than an Alignment (no sequence/CIGAR), heavier than a tuple.
    """
    __slots__ = ('query', 'target', 'score', 'q_start', 'q_end', 't_start', 't_end')

    def __init__(self, query: str, target: str, score: float,
                 q_start: int = -1, q_end: int = -1, t_start: int = -1, t_end: int = -1):
        """
        Initializes a Hit.

        Args:
            query: Query sequence ID.
            target: Target sequence ID.
            score: Alignment score.
            q_start: Query start position.
            q_end: Query end position.
            t_start: Target start position.
            t_end: Target end position.
        """
        self.query = query
        self.target = target
        self.score = score
        self.q_start = q_start
        self.q_end = q_end
        self.t_start = t_start
        self.t_end = t_end

    def __repr__(self):
        return (f"Hit({self.query} -> {self.target}, score={self.score:.1f}, "
                f"range=[{self.t_start}:{self.t_end}])")


class HitTable:
    """
    A unified, high-performance container for alignment hits (Jaccard or SW).
    """
    __slots__ = ('_data', '_count', 'query_ids', 'target_ids', '_q_map', '_t_map')
    _HIT_DTYPE = np.dtype([
        ('query_idx', np.uint32), ('target_idx', np.uint32),
        ('score', np.float32),
        ('q_start', np.int32), ('q_end', np.int32),
        ('t_start', np.int32), ('t_end', np.int32),
    ])

    def __init__(self, size: int = 0):
        """
        Initializes a HitTable.

        Args:
            size: Initial size of the data array.
        """
        self._data = np.zeros(size, dtype=self._HIT_DTYPE)
        self._count = 0
        self.query_ids: List[str] = []
        self.target_ids: List[str] = []
        # Maps for fast integer lookup when adding Hit objects
        self._q_map = {}
        self._t_map = {}

    @property
    def data(self):
        """Returns the valid portion of the data array."""
        return self._data[:self._count]

    def _get_or_add_id(self, name: str, is_query: bool) -> int:
        """Helper to manage ID mapping on the fly."""
        id_list = self.query_ids if is_query else self.target_ids
        id_map = self._q_map if is_query else self._t_map

        if name not in id_map:
            id_map[name] = len(id_list)
            id_list.append(name)
        return id_map[name]

    def add(self, q_idx: int, t_idx: int, score: float, q_interval: tuple = None, t_interval: tuple = None):
        """
        Low-level appender using pre-calculated indices.

        Args:
            q_idx: Query index.
            t_idx: Target index.
            score: Alignment score.
            q_interval: Tuple of (start, end) for query.
            t_interval: Tuple of (start, end) for target.
        """
        if self._count >= len(self._data):
            new_size = max(1024, len(self._data) * 2)
            try: self._data.resize(new_size, refcheck=True)
            except ValueError:
                new_arr = np.zeros(new_size, dtype=self._HIT_DTYPE)
                new_arr[:self._count] = self._data[:self._count]
                self._data = new_arr

        row = self._data[self._count]
        row['query_idx'] = q_idx
        row['target_idx'] = t_idx
        row['score'] = score
        if q_interval: row['q_start'], row['q_end'] = q_interval
        else: row['q_start'], row['q_end'] = -1, -1
        if t_interval: row['t_start'], row['t_end'] = t_interval
        else: row['t_start'], row['t_end'] = -1, -1
        self._count += 1

    def add_hit(self, hit: Hit):
        """
        High-level appender for Hit objects.

        Args:
            hit: The Hit object to add.
        """
        q_idx = self._get_or_add_id(hit.query, is_query=True)
        t_idx = self._get_or_add_id(hit.target, is_query=False)

        self.add(
            q_idx, t_idx, hit.score,
            q_interval=(hit.q_start, hit.q_end),
            t_interval=(hit.t_start, hit.t_end),
        )

    def to_alignments(self) -> Generator['Alignment', None, None]:
        """
        Converts hits in the table to Alignment objects.

        Yields:
            Alignment objects.
        """
        valid_rows = self.data
        for r in valid_rows:
            q_id = self.query_ids[r['query_idx']]
            t_id = self.target_ids[r['target_idx']]
            yield Alignment(
                query=q_id,
                target=t_id,
                query_interval=Interval(max(0, r['q_start']), max(0, r['q_end'])),
                interval=Interval(max(0, r['t_start']), max(0, r['t_end'])),
                score=r['score']
            )


class JaccardResult:
    """
    Stores the results of a Jaccard similarity search.
    """
    __slots__ = ('query_ids', 'target_ids', 'matrix', 'encoded_queries', '_q_map')

    def __init__(self, query_ids: np.ndarray, target_ids: np.ndarray, matrix: csr_matrix,
                 encoded_queries: dict[str, np.ndarray] = None):
        """
        Initializes a JaccardResult.

        Args:
            query_ids: Array of query IDs.
            target_ids: Array of target IDs.
            matrix: Sparse matrix of Jaccard scores.
            encoded_queries: Dictionary of encoded query sequences.
        """
        self.query_ids = query_ids
        self.target_ids = target_ids
        self.matrix = matrix
        self.encoded_queries = encoded_queries or {}
        self._q_map = {qid: i for i, qid in enumerate(query_ids)}

    def get_hits(self, query_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves hits for a specific query.

        Args:
            query_id: The query ID.

        Returns:
            A tuple of (target_ids, scores).
        """
        row_idx = self._q_map.get(query_id)
        if row_idx is None: return np.array([]), np.array([])
        start_ptr = self.matrix.indptr[row_idx]
        end_ptr = self.matrix.indptr[row_idx + 1]
        if start_ptr == end_ptr: return np.array([]), np.array([])
        return self.target_ids[self.matrix.indices[start_ptr:end_ptr]], self.matrix.data[start_ptr:end_ptr]

    def get_encoded_query(self, query_id: str) -> Optional[np.ndarray]:
        """Returns the encoded sequence for a query ID."""
        return self.encoded_queries.get(query_id)

    def to_hit_table(self, min_score: float = 0.0) -> HitTable:
        """
        Converts Jaccard sparse matrix to a dense HitTable.
        Useful for feeding Jaccard results into an AlignmentCollection.

        Args:
            min_score: Minimum score threshold.

        Returns:
            A HitTable object.
        """
        # Estimate size (number of non-zeros)
        nnz = self.matrix.getnnz()
        table = HitTable(size=nnz)

        # Pre-populate IDs in the table to match ours
        table.query_ids = list(self.query_ids)
        table.target_ids = list(self.target_ids)
        # Rebuild lookups
        table._q_map = {q: i for i, q in enumerate(table.query_ids)}
        table._t_map = {t: i for i, t in enumerate(table.target_ids)}

        # Iterate and fill
        # Note: We iterate via arrays for speed
        rows, cols = self.matrix.nonzero()
        data = self.matrix.data

        for r, c, score in zip(rows, cols, data):
            if score < min_score: continue
            # Add directly to internal struct to bypass dictionary lookups
            table.add(r, c, score)  # Jaccard score ~ identity approximation

        return table

    def __getitem__(self, query_id: str) -> dict[str, float]:
        """Convenience access: result['geneA'] -> {'contig1': 0.5}"""
        targets, scores = self.get_hits(query_id)
        if len(targets) == 0: return {}
        return dict(zip(targets, scores))

    def __iter__(self) -> Generator[tuple[str, dict[str, float]], None, None]:
        """
        Yields (query_id, hits_dict) one by one.
        Memory efficient: does not materialize the whole table.
        """
        # Iterate only over rows (queries) that have hits
        for row_idx in np.flatnonzero(self.matrix.getnnz(axis=1)):
            start_ptr = self.matrix.indptr[row_idx]
            end_ptr = self.matrix.indptr[row_idx + 1]
            scores = self.matrix.data[start_ptr:end_ptr]
            indices = self.matrix.indices[start_ptr:end_ptr]
            matched_targets = self.target_ids[indices]
            # Yield tuple immediately
            yield self.query_ids[row_idx], dict(zip(matched_targets, scores))

    def get_hit_indices(self, query_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (target_indices, scores) directly, skipping string conversion.

        Args:
            query_id: The query ID.

        Returns:
            A tuple of (target_indices, scores).
        """
        row_idx = self._q_map.get(query_id)
        if row_idx is None: return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

        start_ptr = self.matrix.indptr[row_idx]
        end_ptr = self.matrix.indptr[row_idx + 1]

        if start_ptr == end_ptr:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

        # Return the indices directly (Faster!)
        return self.matrix.indices[start_ptr:end_ptr], self.matrix.data[start_ptr:end_ptr]


class Alignment(Feature):
    """
    Represents a pairwise sequence alignment.
    """
    __slots__ = (
        'query', 'query_interval', 'query_length', 'target', 'target_length', 'length', 'cigar', 'score',
        'n_matches', 'quality'
    )

    def __init__(self, query: str, query_interval: Interval, target: str, interval: Interval,
                 query_length: int = 0, target_length: int = 0, length: int = 0, cigar: str = None,
                 n_matches: int = 0, quality: int = 0, qualifiers: list[Qualifier] = None, score: float = 0
        ):
        """
        Initializes an Alignment.

        Args:
            query: Query sequence ID.
            query_interval: Interval on the query sequence.
            target: Target sequence ID.
            interval: Interval on the target sequence.
            query_length: Total length of the query sequence.
            target_length: Total length of the target sequence.
            length: Alignment block length.
            cigar: CIGAR string.
            n_matches: Number of matching bases.
            quality: Mapping quality.
            qualifiers: List of qualifiers.
            score: Alignment score.
        """
        super().__init__(interval, kind='Alignment', qualifiers=qualifiers)
        self.query = str(query)
        self.target = str(target)
        self.query_interval = query_interval
        self.query_length = query_length
        self.target_length = target_length
        self.length = length
        self.cigar = cigar
        self.score = score
        self.n_matches = n_matches
        self.quality = quality

    def query_coverage(self) -> float:
        """Returns the fraction of the query sequence covered by the alignment."""
        if self.query_length > 0: return len(self.query_interval) / self.query_length
        return 0.0

    def target_coverage(self) -> float:
        """Returns the fraction of the target sequence covered by the alignment."""
        if self.target_length > 0: return len(self.interval) / self.target_length
        return 0.0

    def identity(self) -> float:
        """Returns the sequence identity (matches / alignment length)."""
        if self.length > 0: return self.n_matches / self.length
        return 0.0

    @classmethod
    def from_hit(cls, hit: Hit):
        """
        Factory to upgrade a Hit to an Alignment.

        Args:
            hit: The Hit object.

        Returns:
            An Alignment object.
        """
        # Note: Hits from SW Score or Jaccard usually lack start coords or cigar
        # We create best-effort intervals.
        q_start = max(0, hit.q_start)
        q_end = max(0, hit.q_end)
        t_start = max(0, hit.t_start)
        t_end = max(0, hit.t_end)

        return cls(hit.query, Interval(q_start, q_end), hit.target, Interval(t_start, t_end), score=hit.score)

    def flip(self) -> 'Alignment':
        """Flips the query and the target."""
        return Alignment(
            query=self.target,
            query_interval=self.interval,
            query_length=self.target_length,
            target=self.query,
            interval=self.query_interval,
            target_length=self.query_length,
            length=self.length,
            cigar=self.cigar,
            score=self.score,
            n_matches=self.n_matches,
            quality=self.quality,
            # SAFETY FIX: Copy qualifiers to prevent shared mutable state
            qualifiers=list(self.qualifiers) if self.qualifiers else None
        )

    def __repr__(self): return f"Alignment({self.query}:{self.query_interval} -> {self.target}:{self.interval})"


class AlignmentCollection:
    """
    A high-performance container for large sets of Alignments.
    Hybrid Architecture:
    1. Topology View (Graph): Fast 'Best Hit', 'Reciprocal Best', and 'Connectivity' operations.
    2. Geometry View (IntervalIndex): Fast 'Overlap', 'Pileup', and 'Depth' operations per target.
    """

    def __init__(self, alignments: Union[List[Alignment], HitTable]):
        """
        Initializes an AlignmentCollection.

        Args:
            alignments: List of Alignment objects or a HitTable.
        """
        if isinstance(alignments, HitTable): self._master_list = list(alignments.to_alignments())
        else: self._master_list = alignments
        self.graph = self._build_graph()
        self._spatial_indices: Dict[str, IntervalIndex] = {}
        # Helper to quickly grab subsets without rebuilding indices
        self._target_map: Dict[str, List[int]] = defaultdict(list)
        for idx, aln in enumerate(self._master_list): self._target_map[aln.target].append(idx)

    def _build_graph(self) -> Graph:
        """Builds a graph representation of the alignments."""
        edge_data = {}
        for idx, aln in enumerate(self._master_list):
            key = (aln.query, aln.target)
            if key not in edge_data: edge_data[key] = {'weight': aln.score, 'best_idx': float(idx), 'count': 1.0}
            else:
                attrs = edge_data[key]
                attrs['count'] += 1.0
                if aln.score > attrs['weight']:
                    attrs['weight'] = aln.score
                    attrs['best_idx'] = float(idx)
        edges = [Edge(u, v, attrs) for (u, v), attrs in edge_data.items()]
        return Graph(*edges, directed=True)

    def filter_best_hits(self, policy: WeightingPolicy = None) -> 'AlignmentCollection':
        """
        Filters to keep only the best hit for each query.

        Args:
            policy: WeightingPolicy to determine 'best'.

        Returns:
            A new AlignmentCollection.
        """
        if policy is None: policy = WeightingPolicy(attr='weight', aggregator='max')
        matrix = self.graph.get_matrix(policy)
        best_col_indices = matrix.argmax(axis=1).A.flatten()
        idx_matrix = self.graph.get_matrix(WeightingPolicy(attr='best_idx', default=-1.0))
        best_aln_indices = idx_matrix[np.arange(matrix.shape[0]), best_col_indices].A.flatten()
        keep_indices = best_aln_indices[best_aln_indices >= 0].astype(int)
        return AlignmentCollection([self._master_list[i] for i in keep_indices])

    def filter_best_query_per_target(self, policy: WeightingPolicy = None) -> 'AlignmentCollection':
        """
        Retains the Best Hit for each TARGET (e.g., identifying the best gene for each contig).

        Args:
            policy: WeightingPolicy to determine 'best'.

        Returns:
            A new AlignmentCollection.
        """
        if policy is None: policy = WeightingPolicy(attr='weight', aggregator='max')
        # Get the matrix (Rows=Query, Cols=Target)
        matrix = self.graph.get_matrix(policy)
        # argmax(axis=0) looks down columns (Targets) to find the best Row (Query)
        best_row_indices = matrix.argmax(axis=0).A.flatten()
        col_indices = np.arange(matrix.shape[1])
        # Recover Alignment Indices
        idx_matrix = self.graph.get_matrix(WeightingPolicy(attr='best_idx', default=-1.0))
        # Note the indexing: [best_row, current_col]
        best_aln_indices = idx_matrix[best_row_indices, col_indices].A.flatten()
        keep_indices = best_aln_indices[best_aln_indices >= 0].astype(int)
        return AlignmentCollection([self._master_list[i] for i in keep_indices])

    def _get_spatial_index(self, target_id: str) -> IntervalIndex:
        """Retrieves or builds the spatial index for a target."""
        if target_id not in self._spatial_indices:
            indices = self._target_map.get(target_id, [])
            intervals = [self._master_list[i].interval for i in indices]
            self._spatial_indices[target_id] = IntervalIndex(*intervals)
        return self._spatial_indices[target_id]

    def pileup(self, target_id: str, length: int = None) -> np.ndarray:
        """
        Calculates coverage depth across the target sequence.

        Args:
            target_id: The target sequence ID.
            length: Length of the target sequence (optional).

        Returns:
            A numpy array of depths.
        """
        idx = self._get_spatial_index(target_id)
        if len(idx) == 0: return np.zeros(length or 0, dtype=np.int32)
        size = length or (idx.ends.max() + 1)
        diffs = np.zeros(size + 1, dtype=np.int32)
        valid_starts = idx.starts[idx.starts < size]
        valid_ends = idx.ends[idx.ends < size]
        np.add.at(diffs, valid_starts, 1)
        np.add.at(diffs, valid_ends, -1)
        return np.add.accumulate(diffs)[:-1]

    def merge_overlaps(self, target_id: str, tolerance: int = 0) -> IntervalIndex:
        """
        Merges overlapping intervals on a target.

        Args:
            target_id: The target sequence ID.
            tolerance: Gap tolerance for merging.

        Returns:
            An IntervalIndex of merged intervals.
        """
        return self._get_spatial_index(target_id).merge(tolerance=tolerance)

    def cull_overlaps(self, max_overlap_fraction: float = 0.1, key: str = 'score') -> 'AlignmentCollection':
        """
        Greedily removes alignments that overlap with higher-scoring alignments on the same target.

        Args:
            max_overlap_fraction: Maximum allowed overlap fraction.
            key: Attribute to use for sorting (default: 'score').

        Returns:
            A new AlignmentCollection.
        """
        kept_indices = []
        # Optimization: Process each target independently to avoid N^2 checks across the whole list
        for target_id, indices in self._target_map.items():
            if not indices: continue
            accepted_for_target = []
            # Extract tuples of (Alignment, OriginalIndex)
            # Sort by score descending (or provided key)
            for aln, original_idx in sorted(
                    ((self._master_list[i], i) for i in indices), key=lambda x: getattr(x[0], key), reverse=True):
                reject = False
                # Check overlap against higher-scoring alignments we've already accepted
                for keeper in accepted_for_target:
                    # Note: We rely on Alignment inheriting .overlap() from Feature
                    if aln.overlap(keeper) / aln.length >= max_overlap_fraction:
                        reject = True
                        break

                if not reject:
                    accepted_for_target.append(aln)
                    kept_indices.append(original_idx)

        # Return a new filtered collection
        # We sort indices to maintain stable order relative to input
        return AlignmentCollection([self._master_list[i] for i in sorted(kept_indices)])


class KmerEncoder:
    """Encodes sequences into k-mer hashes."""
    _DEFAULT_ALPHABET = Alphabet.amino()
    def __init__(self, k: int = 7, alphabet: Alphabet = None):
        """
        Initializes the KmerEncoder.

        Args:
            k: K-mer size.
            alphabet: Alphabet to use.
        """
        self.k: int = k
        self.alphabet: Alphabet = alphabet or self._DEFAULT_ALPHABET
        self._bits_per_symbol: int = (len(self.alphabet) - 1).bit_length()
        total_bits = self.k * self._bits_per_symbol
        if total_bits <= 32: self.dtype = np.uint32
        elif total_bits <= 64: self.dtype = np.uint64
        else: raise ValueError(f"k={self.k} is too large to fit in 64-bit integer.")
        self.mask: int = (1 << (self._bits_per_symbol * (self.k - 1))) - 1
        self._lookup_table = np.zeros(256, dtype=self.dtype)
        for code, char in enumerate(self.alphabet): self._lookup_table[ord(char)] = code

    @property
    def bits_per_symbol(self): return self._bits_per_symbol

    def encode(self, record: Record) -> np.ndarray:
        """
        Encodes a record's sequence into integer representation.

        Args:
            record: The Record object.

        Returns:
            Numpy array of encoded symbols.
        """
        if record.seq.alphabet != self.alphabet:
            raise AlphabetError(f"Seq alphabet is {record.seq.alphabet}, not {self.alphabet}")
        try:
            raw_bytes = np.frombuffer(str(record.seq).encode('ascii'), dtype=np.uint8)
            return self._lookup_table[raw_bytes]
        except UnicodeEncodeError as e: raise ValueError(f"Record {record.id} contains non-ASCII characters.") from e

    def kmers(self, record: Union[Record, np.ndarray]) -> np.ndarray:
        """
        Generates k-mer hashes for a record.

        Args:
            record: Record object or encoded sequence array.

        Returns:
            Numpy array of k-mer hashes.
        """
        if isinstance(record, Record): record = self.encode(record)
        return _rolling_hash_kernel(record, self.k, self._bits_per_symbol, self.mask, self.dtype)


class KmerIndex:
    """
    Index for fast k-mer based sequence search.
    """
    _TARGET_RESIDUES = 500_000
    _DEFAULT_ENCODER = KmerEncoder()
    def __init__(self, encoder: KmerEncoder = None, pool: Executor = None):
        """
        Initializes the KmerIndex.

        Args:
            encoder: KmerEncoder instance.
            pool: Executor for parallel processing.
        """
        self.encoder: KmerEncoder = encoder or self._DEFAULT_ENCODER
        self._records: list[str] = []
        self._kmers: Optional[list[set[int]]] = []
        self._vocab: dict[int, int] = {}
        self._matrix: Optional[csr_matrix] = None
        self._db_counts: Optional[np.ndarray] = None
        self._pool = pool or RESOURCES.pool

    def __len__(self): return len(self._records)
    def __iter__(self): return iter(self._records)
    def __getitem__(self, item: int): return self._records[item]

    @property
    def pool(self) -> Executor: return self._pool
    @property
    def is_sparse(self) -> bool: return self._kmers is None
    @property
    def records(self) -> np.ndarray: return np.array(self._records)

    def make_sparse(self):
        """Finalizes the index, converting to sparse matrix format."""
        self._build()
        self._kmers = None

    def add(self, *records: Record) -> list[Record]:
        """
        Adds records to the index.

        Args:
            *records: Record objects to add.

        Returns:
            List of added records.
        """
        if self.is_sparse: raise RuntimeError("Cannot add to a Sparse (frozen) index.")
        self._matrix = None
        added = []
        existing = set(self._records)
        for r in records:
            if r.id in existing: continue
            self._records.append(r.id)
            self._kmers.append(set(self.encoder.kmers(r)))
            existing.add(r.id)
            added.append(r)
        return added

    def _build(self):
        """Builds the internal sparse matrix."""
        if self._matrix is not None: return
        if self.is_sparse: return
        all_kmers = set.union(*self._kmers) if self._kmers else set()
        self._vocab = {kmer: i for i, kmer in enumerate(all_kmers)}
        if not self._vocab:
            self._matrix = csr_matrix((len(self._records), 0), dtype=np.uint32)
        else:
            rows, cols = [], []
            for i, kmer_set in enumerate(self._kmers):
                rows.extend([i] * len(kmer_set))
                cols.extend(self._vocab[k] for k in kmer_set if k in self._vocab)
            data = np.ones(len(rows), dtype=np.uint32)
            self._matrix = csr_matrix((data, (rows, cols)), shape=(len(self._records), len(self._vocab)))
        self._db_counts = np.array(self._matrix.sum(axis=1)).flatten()

    @staticmethod
    def _map_batch(args: tuple[int, list[Record]]):
        """Helper for parallel encoding."""
        start_index, batch_records, encoder, vocab = args
        b_rows, b_cols, b_ids, b_encoded = [], [], [], []
        current_idx, vocab_get, kmers_func = start_index, vocab.get, encoder.kmers
        encode_func = encoder.encode
        for record in batch_records:
            b_ids.append(record.id)
            enc = encode_func(record)
            b_encoded.append(enc)
            if len(k_hashes := kmers_func(enc)) > 0:
                unique_kmers = np.unique(k_hashes)
                matched_cols = [c for k in unique_kmers if (c := vocab_get(k)) is not None]
                if matched_cols:
                    b_cols.extend(matched_cols)
                    b_rows.extend([current_idx] * len(matched_cols))
            current_idx += 1
        return b_rows, b_cols, b_ids, b_encoded

    def encode_records(self, *records: Record) -> tuple[np.ndarray, Optional[csr_matrix], dict[str, np.ndarray]]:
        """
        Encodes a batch of records and maps them to the index vocabulary.

        Args:
            *records: Records to encode.

        Returns:
            Tuple of (ids, sparse_matrix, encoded_map).
        """
        if not (records := list(records)): return np.array([]), None, {}
        self._build()
        if not self._vocab: return np.array([r.id for r in records]), None, {r.id: self.encoder.encode(r) for r in records}

        batch_size = max(1, min(int(self._TARGET_RESIDUES / (sum(len(r) for r in records[:100]) / 100 or 1)), 5000))
        tasks = [(i, records[i: i + batch_size]) for i in range(0, len(records), batch_size)]
        rows, cols, ids, idx_offset = [], [], [None] * len(records), 0
        encoded_map = {}

        for b_rows, b_cols, b_ids, b_enc_list in self._pool.map(self._map_batch, tasks):
            rows.extend(b_rows)
            cols.extend(b_cols)
            ids[idx_offset: idx_offset + len(b_ids)] = b_ids
            for rid, renc in zip(b_ids, b_enc_list):
                encoded_map[rid] = renc
            idx_offset += len(b_ids)

        id_array = np.array(ids)
        if not rows: return id_array, None, encoded_map
        data = np.ones(len(rows), dtype=np.uint32)
        matrix = csr_matrix((data, (rows, cols)), shape=(len(records), len(self._vocab)))
        return id_array, matrix, encoded_map

    def jaccard(self, *queries: Record, against_self: bool = False) -> JaccardResult:
        """
        Calculate the Jaccard similarity between queries and the index. If against_self is true, then
        the queries become the index itself.

        Args:
            *queries: Query records.
            against_self: If True, compares index against itself.

        Returns:
            JaccardResult object.
        """
        self._build()
        if against_self:  # 1. Self-Comparison Optimization
            q_matrix, q_ids, q_enc_map, q_counts = self._matrix, self.records, {},self._db_counts
            intersection = q_matrix.dot(self._matrix.T).tocoo()  # Dot product of Matrix vs Itself
        else:  # 2. Standard Query vs DB
            q_ids, q_matrix, q_enc_map = self.encode_records(*queries)
            if self._matrix is None or self._matrix.shape[1] == 0 or q_matrix is None:
                empty_matrix = csr_matrix((len(q_ids), len(self.records)))
                return JaccardResult(q_ids, self.records, empty_matrix, q_enc_map)
            q_counts = np.array(q_matrix.sum(axis=1)).flatten()
            intersection = q_matrix.dot(self._matrix.T).tocoo()
        db_counts = self._db_counts
        unions = q_counts[intersection.row] + db_counts[intersection.col] - intersection.data
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = intersection.data / unions
        similarities[np.isnan(similarities)] = 0.0
        result_matrix = csr_matrix((similarities, (intersection.row, intersection.col)), shape=intersection.shape)
        return JaccardResult(q_ids, self.records, result_matrix, q_enc_map)


class PairwiseAligner:
    """
    Performs pairwise sequence alignment using Smith-Waterman or Needleman-Wunsch algorithms.
    """
    _MATRICES = {
        'blosum62': np.reshape([
            4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2,
            0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2,
            -2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3,
            -1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2,
            -2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3,
            0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3,
            -2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2,
            -1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1,
            -1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -3, 2, 0, -3, -3, 1, 0, -3, -1, 2,
            -1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1,
            -1, -1, -3, -2, 0, -3, -2, 1, -3, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1,
            -2, -3, 1, 0, -3, 0, 1, -3, 2, -3, -2, 6, -2, -4, -4, -1, 0, -3, -1, -3,
            -1, -3, -1, -1, -4, -2, -2, -3, 0, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3,
            -1, -3, 0, 2, -3, -2, 0, -3, -3, -2, 0, -4, -1, 5, 1, 0, -1, -2, -2, -1,
            -1, -3, -2, 0, -3, -2, 0, -3, -3, -2, -1, -4, -2, 1, 5, -1, -1, -3, -3, -2,
            1, -1, 0, 0, -2, 0, -1, -2, 1, -2, -1, -1, -1, 0, -1, 4, 1, -2, -3, -2,
            0, -1, -1, -1, -2, -2, -2, -1, 0, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2,
            0, -1, -3, -2, -1, -3, -3, 3, -3, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1,
           -3, -2, -4, -3, 1, -2, -2, -3, -1, -2, -1, -1, -4, -2, -3, -3, -2, -3, 11, 2,
           -2, -2, -3, -2, 3, -3, 2, -1, 2, -1, -1, -3, -3, -1, -2, -2, -2, -1, 2, 7
        ], (20, 20))
    }
    _DEFAULTS = {
        Alphabet.amino(): {'match': 1, 'mismatch': -1, 'gap_open': 11, 'gap_extend': 1, 'band_padding': 20},
        Alphabet.dna(): {'match': 2, 'mismatch': -3, 'gap_open': 5, 'gap_extend': 2, 'band_padding': 10}
    }
    def __init__(
            self, alphabet: Alphabet, k: int, flavour: Literal['local', 'global', 'glocal'] = 'local',
            min_similarity: float = 0.0, compute_traceback: bool = True, matrix: Literal['blosum62'] = None,
            match: int = None, mismatch: int = None, gap_open: int = None, gap_extend: int = None,
            band_padding: int = None
    ):
        """
        Initializes the PairwiseAligner.

        Args:
            alphabet: Sequence alphabet.
            k: K-mer size for seeding.
            flavour: Alignment type ('local', 'global', 'glocal').
            min_similarity: Minimum Jaccard similarity to trigger alignment.
            compute_traceback: Whether to compute CIGAR strings.
            matrix: Substitution matrix name (e.g., 'blosum62').
            match: Match score.
            mismatch: Mismatch penalty.
            gap_open: Gap open penalty.
            gap_extend: Gap extension penalty.
            band_padding: Padding for the alignment band.
        """
        self._index = KmerIndex(KmerEncoder(k, alphabet))
        self._records: list[np.ndarray] = []

        # Set alignment parameters
        defaults = self._DEFAULTS[alphabet]
        n = len(alphabet)
        if matrix is None:
            self.matrix = np.full((n, n), mismatch or defaults['mismatch'], dtype=np.int8)
            np.fill_diagonal(self.matrix, match or defaults['match'])
        else:
            if (matrix := self._MATRICES.get(matrix)) is None: raise ValueError('Invalid matrix')
            if np.shape(matrix)[1] != n: raise ValueError('Invalid matrix')
            self.matrix = matrix
        self.min_similarity = min_similarity
        self.gap_open = gap_open or defaults['gap_open']
        self.gap_extend = gap_extend or defaults['gap_extend']
        self.band_padding = band_padding or defaults['gap_open']
        # Set alignment and traceback kernels
        if (flavour := _KERNEL_REGISTRY.get(flavour)) is None: raise ValueError('Invalid flavour')
        align_kernels, traceback_kernel = flavour
        self._align_kernel = align_kernels[compute_traceback]
        self._traceback_kernel = traceback_kernel

    def __len__(self) -> int: return len(self._records)
    def jaccard(self, *args, **kwargs) -> JaccardResult: return self._index.jaccard(*args, **kwargs)

    def add(self, *records: Record):
        """
        Adds records to the aligner's index.

        Args:
            *records: Record objects to add.
        """
        to_add = []
        existing_ids = set(self._index.records)
        for r in records:
            if r.id not in existing_ids:
                to_add.append(r)
                existing_ids.add(r.id)
        if not to_add: return
        self._index.add(*to_add)
        encode = self._index.encoder.encode
        for r in to_add: self._records.append(encode(r))

    def search(self, *queries: Record, jaccard_result: JaccardResult = None, against_self: bool = False) -> Generator[
        Union[Alignment, Hit], None, None]:
        """
        Searches for alignments between queries and indexed records.

        Args:
            *queries: Query records.
            jaccard_result: Pre-computed Jaccard results (optional).
            against_self: If True, aligns index against itself.

        Yields:
            Alignment or Hit objects.
        """

        # 1. Handle Jaccard Logic
        if jaccard_result is None:
            jaccard_result = self._index.jaccard(*queries, against_self=against_self)

        # 2. Pre-process Queries
        query_payloads = {}
        target_work_map = defaultdict(list)
        min_sim = self.min_similarity

        # Unified Iterator
        if against_self:
            input_iterator = zip(range(len(self._records)), self._index.records, self._records)
        else:
            if len(queries) == 0: return
            input_iterator = enumerate(queries)

        for item in input_iterator:
            if against_self:
                q_idx, q_id, q_enc = item
                q_obj = type('RecordProxy', (object,), {'id': q_id})()
            else:
                q_idx, q_obj = item
                q_id = q_obj.id
                q_enc = None

            # CHANGED: Get indices directly
            # No more string lookups!
            t_indices, scores = jaccard_result.get_hit_indices(q_id)
            if len(t_indices) == 0: continue

            # Filter
            mask = scores >= min_sim
            if not np.any(mask): continue

            # These are already INTEGER INDICES
            valid_t_indices = t_indices[mask]

            if q_id not in query_payloads:
                if q_enc is None:
                    q_enc = jaccard_result.get_encoded_query(q_id)
                    if q_enc is None: q_enc = self._index.encoder.encode(q_obj)

                kmers = self._index.encoder.kmers(q_enc)
                sorter = np.argsort(kmers)
                query_payloads[q_id] = (q_obj, q_enc, kmers[sorter], sorter.astype(np.uint32))

            # Map Target Index -> Query ID
            for t_idx in valid_t_indices:
                # Optimized Self-Check is now just an integer compare
                if against_self and t_idx < q_idx: continue

                target_work_map[t_idx].append(q_id)

        if not target_work_map: return

        # 3. Create Batch Tasks
        target_indices = list(target_work_map.keys())
        batch_size = 10

        tasks = [(
            [(self._index.records[t_idx], self._records[t_idx], target_work_map[t_idx])
             for t_idx in target_indices[i: i + batch_size]],
            query_payloads, self.matrix, self.gap_open, self.gap_extend, self.band_padding, self._index.encoder,
            self._align_kernel, self._traceback_kernel
        ) for i in range(0, len(target_indices), batch_size)]

        # 4. Execute
        for batch_results in self._index.pool.map(self._align_batch, tasks):
            yield from batch_results

    @staticmethod  # We need to keep this static so it can be pickleable for threading
    def _align_batch(args: tuple) -> list:
        """
        targets_batch, query_payloads, matrix, gap_open, gap_extend, band_padding, encoder, align_kernel, traceback_kernel
        Worker function.
        Calculates Target Hash ONCE, then aligns multiple Queries against it.
        """
        targets_batch, query_payloads, matrix, gap_open, gap_extend, band_padding, encoder, align_kernel, traceback_kernel = args
        k, bits, mask, dtype = encoder.k, encoder.bits_per_symbol, encoder.mask, encoder.dtype
        results = []

        for t_id, t_seq, q_ids in targets_batch:
            # OPTIMIZATION: Hash Target Once
            t_kmers = _rolling_hash_kernel(t_seq, k, bits, mask, dtype)

            for q_id in q_ids:
                q_obj, q_seq, q_k_sorted, q_pos_sorted = query_payloads[q_id]
                
                res = PairwiseAligner._align_core(
                    q_id, q_seq, q_k_sorted, q_pos_sorted,
                    t_id, t_seq, t_kmers,
                    matrix, gap_open, gap_extend, band_padding,
                    align_kernel, traceback_kernel
                )
                if res:
                    results.append(res)

        return results

    @staticmethod
    def _align_core(q_id, q_seq, q_k_sorted, q_pos_sorted,
                    t_id, t_seq, t_kmers,
                    matrix, gap_open, gap_extend, band_padding,
                    align_kernel, traceback_kernel) -> Optional[Union[Alignment, Hit]]:
        """
        Core alignment logic.
        """
        
        # Seeding (Band calculation) using pre-calculated kmer positions
        min_diag, max_diag = _fast_band_kernel(q_k_sorted, q_pos_sorted, t_kmers, band_padding)

        # If no shared kmers or band invalid
        if min_diag > max_diag: return None

        score, pos, H, E, F = align_kernel(q_seq, t_seq, matrix, gap_open, gap_extend, min_diag, max_diag)

        if score == 0: return None

        if traceback_kernel is None:
            r, c = pos
            return Hit(query=q_id, target=t_id, score=float(score), q_end=r, t_end=c, q_start=-1, t_start=-1)

        # Traceback
        align1, align2, r_start, r_end, c_start, c_end, matches, length = traceback_kernel(
            H, E, F, q_seq, t_seq, pos, matrix, gap_open, gap_extend
        )

        if length == 0: return None

        return Alignment(
            query=q_id, query_interval=Interval(r_start, r_end), query_length=len(q_seq),
            target=t_id, interval=Interval(c_start, c_end), target_length=len(t_seq),
            length=length, cigar="".join(_make_cigar(align1, align2)), score=score, n_matches=matches
        )

    def align(self, query: Record, target: Union[Record, str]) -> Optional[Alignment]:
        """
        Aligns a query to a target.
        If the target is a string, it will be used to retrieve an existing target in the index.

        Args:
            query: Query record.
            target: Target record or ID.

        Returns:
            Alignment object or None.
        """
        encoder = self._index.encoder

        # 1. Prepare Query
        q_enc = encoder.encode(query)
        q_kmers = encoder.kmers(q_enc)
        q_sorter = np.argsort(q_kmers)
        q_k_sorted = q_kmers[q_sorter]
        q_pos_sorted = q_sorter.astype(np.uint32)

        # 2. Prepare Target
        if isinstance(target, str):
            try:
                t_idx = self._index._records.index(target)
                t_seq = self._records[t_idx]
                t_id = target
            except ValueError:
                return None
        else:
            t_id = target.id
            t_seq = encoder.encode(target)

        # 3. Hash Target
        t_kmers = _rolling_hash_kernel(t_seq, encoder.k, encoder.bits_per_symbol, encoder.mask, encoder.dtype)

        # 4. Run Core
        return self._align_core(
            query.id, q_enc, q_k_sorted, q_pos_sorted,
            t_id, t_seq, t_kmers,
            self.matrix, self.gap_open, self.gap_extend, self.band_padding,
            self._align_kernel, self._traceback_kernel
        )


# Kernels --------------------------------------------------------------------------------------------------------------
def _make_cigar(seq1_ali: np.ndarray, seq2_ali: np.ndarray) -> Generator[str, None, None]:
    """
    Generates CIGAR string from aligned sequences.

    Args:
        seq1_ali: Aligned query sequence (integers).
        seq2_ali: Aligned target sequence (integers).

    Yields:
        CIGAR operations (e.g., "10M", "2D").
    """
    if len(seq1_ali) == 0: return
    ops = ('D' if s1 == -1 else 'I' if s2 == -1 else 'M' for s1, s2 in zip(seq1_ali, seq2_ali))
    try: current_op = next(ops)
    except StopIteration: return
    count = 1
    for op in ops:
        if op == current_op: count += 1
        else:
            yield f"{count}{current_op}"
            current_op = op
            count = 1
    yield f"{count}{current_op}"


@jit(nopython=True, cache=True, nogil=True)
def _rolling_hash_kernel(int_seq: np.ndarray, k: int, bits_per_symbol: int, mask: int, out_dtype):
    """
    Computes rolling hash k-mers for an integer sequence.
    """
    if len(int_seq) < k: return np.empty(0, dtype=out_dtype)
    n_kmers = len(int_seq) - k + 1
    result = np.empty(n_kmers, dtype=out_dtype)
    current_kmer = 0
    for i in range(k): current_kmer = (current_kmer << bits_per_symbol) | int_seq[i]
    result[0] = current_kmer
    for i in range(1, n_kmers):
        new_symbol = int_seq[i + k - 1]
        current_kmer = ((current_kmer & mask) << bits_per_symbol) | new_symbol
        result[i] = current_kmer
    return result


@jit(nopython=True, cache=True, nogil=True)
def _fast_band_kernel(q_kmers_sorted: np.ndarray, q_pos_sorted: np.ndarray, t_kmers: np.ndarray, padding: int):
    """
    Calculates the alignment band based on shared k-mers.
    """
    min_diag = 2_000_000_000
    max_diag = -2_000_000_000
    found = False
    n_q = len(q_kmers_sorted)

    for t_i in range(len(t_kmers)):
        t_val = t_kmers[t_i]
        idx = np.searchsorted(q_kmers_sorted, t_val)
        while idx < n_q and q_kmers_sorted[idx] == t_val:
            q_i = q_pos_sorted[idx]
            diag = t_i - q_i
            if diag < min_diag: min_diag = diag
            if diag > max_diag: max_diag = diag
            found = True
            idx += 1

    if not found: return 1, 0
    return min_diag - padding, max_diag + padding


@jit(nopython=True, cache=True, nogil=True)
def _local_kernel(seq1: np.ndarray, seq2: np.ndarray, matrix: np.ndarray,
               gap_open: int, gap_extend: int, min_diag: int, max_diag: int):
    """Local pairwise alignment with the Smith-Waterman algorithm"""
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    H = np.zeros((rows, cols), dtype=np.int32)
    E = np.zeros((rows, cols), dtype=np.int32)
    F = np.zeros((rows, cols), dtype=np.int32)
    max_score = 0
    max_pos = (0, 0)
    for r in range(1, rows):
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        for c in range(start_c, end_c):
            match = H[r - 1, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]
            e_score = max(E[r, c - 1] - gap_extend, H[r, c - 1] - gap_open)
            E[r, c] = e_score
            f_score = max(F[r - 1, c] - gap_extend, H[r - 1, c] - gap_open)
            F[r - 1, c] = f_score
            score = max(0, match, e_score, f_score)
            H[r, c] = score
            if score > max_score:
                max_score = score
                max_pos = (r, c)
    return max_score, max_pos, H, E, F


@jit(nopython=True, cache=True, nogil=True, boundscheck=False)
def _local_score_kernel(seq1: np.ndarray, seq2: np.ndarray, matrix: np.ndarray,
                        gap_open: int, gap_extend: int, min_diag: int, max_diag: int):
    """Local pairwise scoring with the Smith-Waterman algorithm"""
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    H = np.zeros((2, cols), dtype=np.int32)
    F = np.zeros((2, cols), dtype=np.int32)
    max_score = 0
    max_pos = (0, 0)
    prev, curr = 0, 1

    for r in range(1, rows):
        H[curr, :] = 0
        F[curr, :] = 0
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        E_curr = 0

        for c in range(start_c, end_c):
            match = H[prev, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]
            h_curr_prev_col = H[curr, c - 1]
            E_curr = max(E_curr - gap_extend, h_curr_prev_col - gap_open)
            f_score = max(F[prev, c] - gap_extend, H[prev, c] - gap_open)
            F[curr, c] = f_score
            score = max(0, match, E_curr, f_score)
            H[curr, c] = score
            if score > max_score:
                max_score = score
                max_pos = (r, c)
        prev, curr = curr, prev
    return max_score, max_pos, H, F, F


@jit(nopython=True, cache=True, nogil=True)
def _local_traceback_kernel(H, E, F, seq1, seq2, max_pos, matrix, gap_open, gap_extend):
    """Traceback for local alignment."""
    r, c = max_pos
    if r == 0 and c == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0, 0, 0, 0, 0, 0

    end_r, end_c = r, c
    matches = 0

    # PERF FIX: Pre-allocate max possible size
    max_len = r + c
    out1 = np.empty(max_len, dtype=np.int32)
    out2 = np.empty(max_len, dtype=np.int32)
    k = 0

    while r > 0 and c > 0 and H[r, c] > 0:
        score = H[r, c]
        if score == H[r - 1, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]:
            s1, s2 = seq1[r - 1], seq2[c - 1]
            out1[k] = s1
            out2[k] = s2
            if s1 == s2: matches += 1
            r -= 1
            c -= 1
        elif score == F[r, c]:
            out1[k] = seq1[r - 1]
            out2[k] = -1
            r -= 1
        else:
            out1[k] = -1
            out2[k] = seq2[c - 1]
            c -= 1
        k += 1

    return (out1[:k][::-1].copy(), out2[:k][::-1].copy(),
            r, end_r, c, end_c, matches, k)


@jit(nopython=True, cache=True, nogil=True)
def _global_kernel(seq1: np.ndarray, seq2: np.ndarray, matrix: np.ndarray,
               gap_open: int, gap_extend: int, min_diag: int, max_diag: int):
    """
    Needleman-Wunsch with Affine Gap Penalties and K-mer Banding.
    """
    rows = len(seq1) + 1
    cols = len(seq2) + 1

    # 1. Initialize with -Infinity (unlike SW which uses 0)
    # We use a sufficiently small integer, e.g., -1 billion
    NEG_INF = -1_000_000_000
    H = np.full((rows, cols), NEG_INF, dtype=np.int32)
    E = np.full((rows, cols), NEG_INF, dtype=np.int32)
    F = np.full((rows, cols), NEG_INF, dtype=np.int32)

    # 2. Base Case: Top-Left is 0
    H[0, 0] = 0

    # 3. Initialize First Row and Column (Gap Penalties)
    # Only if they fall within the band/logic, though usually we force them for NW
    for c in range(1, cols):
        # Cost to gap from (0,0) to (0,c)
        H[0, c] = -(gap_open + (c - 1) * gap_extend)
        E[0, c] = H[0, c]  # Technically part of the gap

    for r in range(1, rows):
        # Cost to gap from (0,0) to (r,0)
        H[r, 0] = -(gap_open + (r - 1) * gap_extend)
        F[r, 0] = H[r, 0]

    # 4. Fill Matrix
    for r in range(1, rows):
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)

        for c in range(start_c, end_c):
            # Match/Mismatch
            match = H[r - 1, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]

            # Insertion (Gap in seq1)
            # Existing gap extension OR opening a new gap from H
            e_score = max(E[r, c - 1] - gap_extend, H[r, c - 1] - gap_open)
            E[r, c] = e_score

            # Deletion (Gap in seq2)
            f_score = max(F[r - 1, c] - gap_extend, H[r - 1, c] - gap_open)
            F[r - 1, c] = f_score

            # Global: No max(0, ...), scores can go negative
            H[r, c] = max(match, e_score, f_score)

    # Return the score at the bottom-right corner
    # Note: We return (rows-1, cols-1) as the 'pos', though it's implicit in NW
    return H[rows - 1, cols - 1], (rows - 1, cols - 1), H, E, F


@jit(nopython=True, cache=True, nogil=True)
def _global_score_kernel(seq1: np.ndarray, seq2: np.ndarray, matrix: np.ndarray,
                     gap_open: int, gap_extend: int, min_diag: int, max_diag: int):
    """
    Memory-efficient Needleman-Wunsch (Score Only).
    Uses 2 rows of memory instead of the full matrix.
    """
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000  # Sufficiently small integer

    # We only need 2 rows for H and F
    # H[0] is 'prev', H[1] is 'curr'
    H = np.full((2, cols), NEG_INF, dtype=np.int32)
    F = np.full((2, cols), NEG_INF, dtype=np.int32)

    # 1. Initialize Top Row (Row 0)
    # H[0,0] = 0, rest are horizontal gap penalties
    H[0, 0] = 0
    for c in range(1, cols):
        # Cost to gap from (0,0) to (0,c)
        H[0, c] = -(gap_open + (c - 1) * gap_extend)
        # F[0, c] remains NEG_INF (cannot vertically gap from nowhere)

    prev, curr = 0, 1

    for r in range(1, rows):
        # 2. Initialize Left Boundary (Column 0) for current row
        # Cost to gap from (0,0) to (r,0)
        val_r = -(gap_open + (r - 1) * gap_extend)
        H[curr, 0] = val_r
        F[curr, 0] = val_r  # Effectively a vertical gap from the cell above

        # 3. Reset the rest of the current row to NEG_INF
        # This is CRITICAL for banding. Cells outside the band must be -inf.
        H[curr, 1:] = NEG_INF
        F[curr, 1:] = NEG_INF

        # Determine band limits
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)

        # 'running_E' is the Horizontal Gap score for the current cell
        # We initialize it to NEG_INF.
        # Inside the loop, it will check H[curr, c-1] to see if it can open/extend.
        running_E = NEG_INF

        for c in range(start_c, end_c):
            # --- Vertical Gap (F) ---
            # Extend existing F from row above OR Open new from H above
            f_score = max(F[prev, c] - gap_extend, H[prev, c] - gap_open)
            F[curr, c] = f_score

            # --- Horizontal Gap (E) ---
            # Extend existing E from col left OR Open new from H left
            # We use 'running_E' as the scalar E variable
            h_left = H[curr, c - 1]
            running_E = max(running_E - gap_extend, h_left - gap_open)

            # --- Match/Mismatch ---
            match = H[prev, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]

            # --- Final Score ---
            # Note: No max(0, ...) because this is Global
            H[curr, c] = max(match, running_E, f_score)

        # Flip rows
        prev, curr = curr, prev

    # After the loop, 'prev' holds the last processed row (the bottom of the matrix)
    # The Global Alignment score is strictly at the bottom-right corner.
    final_score = H[prev, cols - 1]

    # We return (rows-1, cols-1) as the position to match the SW signature
    return final_score, (rows - 1, cols - 1), H, F, F


@jit(nopython=True, cache=True, nogil=True)
def _global_traceback_kernel(H, E, F, seq1, seq2, matrix, gap_open, gap_extend):
    """
    Traceback for Global Alignment.
    Starts at bottom-right (N, M) and walks to (0, 0).
    """
    r = len(seq1)
    c = len(seq2)

    # Pre-allocate worst-case length (sum of both sequences)
    max_len = r + c
    out1 = np.empty(max_len, dtype=np.int32)
    out2 = np.empty(max_len, dtype=np.int32)

    k = 0
    matches = 0

    # Stop when we hit the top-left
    while r > 0 or c > 0:
        # If we reached the edge, we must strictly gap-walk to (0,0)
        if r == 0:
            out1[k] = -1
            out2[k] = seq2[c - 1]
            c -= 1
        elif c == 0:
            out1[k] = seq1[r - 1]
            out2[k] = -1
            r -= 1
        else:
            # Standard traceback logic
            score = H[r, c]
            # Check Match
            if score == H[r - 1, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]:
                s1, s2 = seq1[r - 1], seq2[c - 1]
                out1[k] = s1
                out2[k] = s2
                if s1 == s2: matches += 1
                r -= 1
                c -= 1
            # Check Deletion (Up)
            elif score == F[r, c]:  # Matches F logic
                out1[k] = seq1[r - 1]
                out2[k] = -1
                r -= 1
            # Check Insertion (Left)
            else:  # Matches E logic
                out1[k] = -1
                out2[k] = seq2[c - 1]
                c -= 1
        k += 1

    # Return formatted similarly to SW traceback
    return (out1[:k][::-1].copy(), out2[:k][::-1].copy(),
            0, len(seq1), 0, len(seq2), matches, k)


@jit(nopython=True, cache=True, nogil=True)
def _glocal_kernel(seq1: np.ndarray, seq2: np.ndarray, matrix: np.ndarray,
                   gap_open: int, gap_extend: int, min_diag: int, max_diag: int):
    """
    Glocal Alignment (Global Query, Local Target).
    Query must align fully. Target can be entered/exited freely.
    """
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000

    H = np.full((rows, cols), NEG_INF, dtype=np.int32)
    E = np.full((rows, cols), NEG_INF, dtype=np.int32)
    F = np.full((rows, cols), NEG_INF, dtype=np.int32)

    # 1. Initialization
    # Top Row: 0. Free entry into Target (Local behavior)
    H[0, :] = 0
    E[0, :] = NEG_INF  # Cannot extend a gap from nowhere

    # Left Column: Gap Penalties. Force start of Query (Global behavior)
    for r in range(1, rows):
        gap_cost = -(gap_open + (r - 1) * gap_extend)
        H[r, 0] = gap_cost
        F[r, 0] = gap_cost

    # 2. Fill Matrix
    for r in range(1, rows):
        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)

        for c in range(start_c, end_c):
            # Standard Gotoh Recurrence (Same as NW/SW)
            match = H[r - 1, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]
            e_score = max(E[r, c - 1] - gap_extend, H[r, c - 1] - gap_open)
            E[r, c] = e_score
            f_score = max(F[r - 1, c] - gap_extend, H[r - 1, c] - gap_open)
            F[r - 1, c] = f_score
            H[r, c] = max(match, e_score, f_score)

    # 3. Find Max Score (End of Query)
    # We strictly look at the LAST ROW (rows - 1).
    # The max score indicates the best exit point in the Target.
    max_score = NEG_INF
    best_c = 0

    # Optimization: Only search within the valid band of the last row
    last_r = rows - 1
    start_c = max(1, last_r + min_diag)
    end_c = min(cols, last_r + max_diag + 1)

    for c in range(start_c, end_c):
        if H[last_r, c] > max_score:
            max_score = H[last_r, c]
            best_c = c

    return max_score, (rows - 1, best_c), H, E, F


@jit(nopython=True, cache=True, nogil=True)
def _glocal_score_kernel(seq1: np.ndarray, seq2: np.ndarray, matrix: np.ndarray,
                         gap_open: int, gap_extend: int, min_diag: int, max_diag: int):
    """
    Memory-efficient Glocal Scoring.
    """
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    NEG_INF = -1_000_000_000

    H = np.full((2, cols), NEG_INF, dtype=np.int32)
    F = np.full((2, cols), NEG_INF, dtype=np.int32)

    # 1. Init Top Row (Free Entry)
    H[0, :] = 0

    prev, curr = 0, 1

    for r in range(1, rows):
        # 2. Init Left Col (Global Query Penalty)
        gap_cost = -(gap_open + (r - 1) * gap_extend)
        H[curr, 0] = gap_cost
        F[curr, 0] = gap_cost

        # Reset current row buffer
        H[curr, 1:] = NEG_INF
        F[curr, 1:] = NEG_INF

        start_c = max(1, r + min_diag)
        end_c = min(cols, r + max_diag + 1)
        running_E = NEG_INF

        for c in range(start_c, end_c):
            f_score = max(F[prev, c] - gap_extend, H[prev, c] - gap_open)
            F[curr, c] = f_score

            h_left = H[curr, c - 1]
            running_E = max(running_E - gap_extend, h_left - gap_open)

            match = H[prev, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]
            H[curr, c] = max(match, running_E, f_score)

        prev, curr = curr, prev

    # 3. Find Max in 'prev' (which holds the last row)
    max_score = NEG_INF
    best_c = 0

    # Check valid band in last row
    start_c = max(1, (rows - 1) + min_diag)
    end_c = min(cols, (rows - 1) + max_diag + 1)

    for c in range(start_c, end_c):
        if H[prev, c] > max_score:
            max_score = H[prev, c]
            best_c = c

    return max_score, (rows - 1, best_c), H, F, F


@jit(nopython=True, cache=True, nogil=True)
def _glocal_traceback_kernel(H, E, F, seq1, seq2, max_pos, matrix, gap_open, gap_extend):
    """
    Traceback for Glocal.
    Starts at (rows-1, best_c).
    Ends when r == 0.
    """
    r, c = max_pos
    end_r, end_c = r, c

    max_len = r + c
    out1 = np.empty(max_len, dtype=np.int32)
    out2 = np.empty(max_len, dtype=np.int32)

    k = 0
    matches = 0

    # We MUST consume the entire Query (r > 0).
    # Once r hits 0, we are done (we implicitly soft-clip the rest of Target).
    while r > 0:
        if c == 0:
            # Hit left edge, must delete up to (0,0)
            out1[k] = seq1[r - 1]
            out2[k] = -1
            r -= 1
        else:
            score = H[r, c]
            # Match
            if score == H[r - 1, c - 1] + matrix[seq1[r - 1], seq2[c - 1]]:
                s1, s2 = seq1[r - 1], seq2[c - 1]
                out1[k] = s1
                out2[k] = s2
                if s1 == s2: matches += 1
                r -= 1
                c -= 1
            # Deletion (Vertical)
            elif score == F[r, c]:
                out1[k] = seq1[r - 1]
                out2[k] = -1
                r -= 1
            # Insertion (Horizontal)
            else:
                out1[k] = -1
                out2[k] = seq2[c - 1]
                c -= 1
        k += 1

    # r_start is always 0 for Glocal
    # c_start is wherever we stopped in the target
    return (out1[:k][::-1].copy(), out2[:k][::-1].copy(),
            0, end_r, c, end_c, matches, k)


_KERNEL_REGISTRY = {
    'local': ({True: _local_kernel, False: _local_score_kernel}, _local_traceback_kernel),
    'global': ({True: _global_kernel, False: _global_score_kernel}, _global_traceback_kernel),
    'glocal': ({True: _glocal_kernel, False: _glocal_score_kernel}, _glocal_traceback_kernel)
}


# Functions ------------------------------------------------------------------------------------------------------------
def otsu(similarity_matrix) -> float:
    """
    Calculates Otsu's threshold for a similarity matrix.

    Args:
        similarity_matrix: Array of similarity scores.

    Returns:
        The calculated threshold.
    """
    if len(similarity_matrix) == 0: return 0.5
    hist, bin_edges = np.histogram(similarity_matrix, bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight = hist.cumsum()
    mean = (hist * bin_centers).cumsum()
    total_mean = mean[-1]
    total_weight = weight[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_bg = mean / weight
        mean_fg = (total_mean - mean) / (total_weight - weight)
    mean_bg[np.isnan(mean_bg)] = 0.0
    mean_fg[np.isnan(mean_fg)] = 0.0
    w0 = weight
    w1 = total_weight - weight
    between_class_variance = w0 * w1 * (mean_bg - mean_fg) ** 2
    idx = np.argmax(between_class_variance)
    return bin_centers[idx]
