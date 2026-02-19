"""Containers for representing and batch-processing genomic mutations (SNPs, indels)."""
from typing import Iterable, Any, Optional
from enum import IntEnum, auto

import numpy as np

from baclib.containers.feature import Feature, FeatureKey
from baclib.core.interval import IntervalBatch, Interval
from baclib.core.alphabet import Alphabet
from baclib.containers.seq import Seq, SeqBatch
from baclib.containers import Batch
from baclib.lib.protocols import HasIntervals
from baclib.containers.qualifier import QualifierList, QualifierBatch, QualifierType


# Classes --------------------------------------------------------------------------------------------------------------
class MutationEffect(IntEnum):
    """
    Predicted functional impact of a mutation on the gene product.

    Examples:
        >>> MutationEffect.MISSENSE
        <MutationEffect.MISSENSE: 3>
    """
    UNKNOWN = auto()
    SYNONYMOUS = auto()
    MISSENSE = auto()
    NONSENSE = auto()
    FRAMESHIFT = auto()
    INTERGENIC = auto()
    NON_CODING = auto()


class Mutation(Feature):
    """
    A discrete sequence change: SNP, insertion, or deletion.

    Extends ``Feature`` with reference and alternative allele sequences,
    a predicted functional effect, and an optional amino acid change string.

    Args:
        interval: Location on the reference sequence.
        ref_seq: The reference allele.
        alt_seq: The alternative allele.
        effect: Predicted functional impact (default ``UNKNOWN``).
        aa_change: Optional amino acid change annotation (e.g. ``b'S123A'``).
        qualifiers: Optional ``(key, value)`` qualifier tuples.

    Examples:
        >>> mut = Mutation(Interval(100, 101, 1), ref, alt)
        >>> mut.is_snp
        True
    """
    __slots__ = ('ref_seq', 'alt_seq', 'effect', 'aa_change')

    def __init__(self, interval: Interval, ref_seq: Seq, alt_seq: Seq,
                 effect: MutationEffect = MutationEffect.UNKNOWN,
                 aa_change: bytes = None,
                 qualifiers: Iterable[tuple[bytes, Any]] = None):
        super().__init__(interval, key=FeatureKey.MISC_DIFFERENCE, qualifiers=qualifiers)
        self.ref_seq = ref_seq
        self.alt_seq = alt_seq
        self.effect = effect
        self.aa_change = aa_change

    @property
    def is_snp(self) -> bool:
        """Returns ``True`` if the mutation is a single-nucleotide polymorphism.

        Returns:
            ``True`` when both ref and alt are exactly 1 base long.
        """
        return len(self.ref_seq) == 1 and len(self.alt_seq) == 1

    @property
    def is_indel(self) -> bool:
        """Returns ``True`` if the mutation is an insertion or deletion.

        Returns:
            ``True`` when ref and alt differ in length.
        """
        return len(self.ref_seq) != len(self.alt_seq)

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``MutationBatch`` class.
        """
        return MutationBatch

    @property
    def diff(self) -> int:
        """Returns the net change in sequence length (alt − ref).

        Returns:
            Positive for insertions, negative for deletions, zero for SNPs.
        """
        return len(self.alt_seq) - len(self.ref_seq)


class MutationBatch(Batch, HasIntervals):
    """
    Columnar batch of mutations for efficient bulk operations.

    Stores intervals, ref/alt sequences, effects, and qualifiers in
    parallel numpy arrays for vectorized filtering and application.

    Args:
        intervals: An ``IntervalBatch`` of mutation positions.
        ref_seqs: A ``SeqBatch`` of reference alleles.
        alt_seqs: A ``SeqBatch`` of alternative alleles.
        effects: Optional ``uint8`` array of ``MutationEffect`` values.
        aa_changes: Optional object array of amino acid change annotations.
        qualifiers: Optional ``QualifierBatch`` of per-mutation qualifiers.

    Examples:
        >>> batch = MutationBatch.build([mut1, mut2])
        >>> len(batch)
        2
    """
    __slots__ = ('_intervals', '_ref_seqs', '_alt_seqs', '_effects', '_aa_changes', '_qualifiers')

    def __init__(self, intervals: IntervalBatch, ref_seqs: SeqBatch, alt_seqs: SeqBatch,
                 effects: np.ndarray = None, aa_changes: np.ndarray = None, qualifiers: QualifierBatch = None):
        self._intervals = intervals
        self._ref_seqs = ref_seqs
        self._alt_seqs = alt_seqs
        n = len(intervals)
        self._effects = effects if effects is not None else np.zeros(n, dtype=np.uint8)
        self._aa_changes = aa_changes if aa_changes is not None else np.full(n, None, dtype=object)
        self._qualifiers = qualifiers if qualifiers is not None else QualifierBatch.zeros(n)

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``Mutation`` class.
        """
        return Mutation

    @classmethod
    def empty(cls) -> 'MutationBatch':
        """Creates an empty MutationBatch with zero mutations.

        Returns:
            An empty ``MutationBatch``.
        """
        return cls(
            IntervalBatch.empty(),
            Alphabet.DNA.empty_batch(),
            Alphabet.DNA.empty_batch()
        )

    @classmethod
    def build(cls, components: Iterable[Mutation]) -> 'MutationBatch':
        """Constructs a MutationBatch from an iterable of Mutation objects.

        Args:
            components: An iterable of ``Mutation`` objects.

        Returns:
            A new ``MutationBatch``.

        Examples:
            >>> batch = MutationBatch.build([mut1, mut2])
            >>> len(batch)
            2
        """
        mutations = list(components)
        if not mutations: return cls.empty()
        
        # Manual extraction to ensure sort=False (preserve order)
        n = len(mutations)
        starts = np.empty(n, dtype=np.int32)
        ends = np.empty(n, dtype=np.int32)
        strands = np.empty(n, dtype=np.int32)
        
        for i, m in enumerate(mutations):
            iv = m.interval
            starts[i] = iv.start
            ends[i] = iv.end
            strands[i] = iv.strand
            
        intervals = IntervalBatch(starts, ends, strands, sort=False)
        
        ref_seqs = SeqBatch.build([m.ref_seq for m in mutations])
        alt_seqs = SeqBatch.build([m.alt_seq for m in mutations])
        
        effects = np.array([m.effect for m in mutations], dtype=np.uint8)
        aa_changes = np.array([m.aa_change for m in mutations], dtype=object)
        
        qualifiers = QualifierBatch.build(m.qualifiers for m in mutations)

        return cls(intervals, ref_seqs, alt_seqs, effects, aa_changes, qualifiers)

    @classmethod
    def concat(cls, batches: Iterable['MutationBatch']) -> 'MutationBatch':
        """Concatenates multiple MutationBatch objects into one.

        Args:
            batches: An iterable of ``MutationBatch`` objects.

        Returns:
            A single concatenated ``MutationBatch``.

        Examples:
            >>> combined = MutationBatch.concat([batch_a, batch_b])
        """
        batches = list(batches)
        if not batches: return cls.empty()
        
        # Manual interval concat to maintain sync with other arrays (IntervalBatch.concat sorts)
        i_batches = [b.intervals for b in batches]
        starts = np.concatenate([b.starts for b in i_batches])
        ends = np.concatenate([b.ends for b in i_batches])
        strands = np.concatenate([b.strands for b in i_batches])
        intervals = IntervalBatch(starts, ends, strands, sort=False)
        
        ref_seqs = SeqBatch.concat([b._ref_seqs for b in batches])
        alt_seqs = SeqBatch.concat([b._alt_seqs for b in batches])
        effects = np.concatenate([b._effects for b in batches])
        aa_changes = np.concatenate([b._aa_changes for b in batches])
        qualifiers = QualifierBatch.concat([b._qualifiers for b in batches])
        
        return cls(intervals, ref_seqs, alt_seqs, effects, aa_changes, qualifiers)

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by all internal arrays.
        """
        return (self._intervals.nbytes + self._ref_seqs.nbytes + 
                self._alt_seqs.nbytes + self._effects.nbytes + 
                self._aa_changes.nbytes + self._qualifiers.nbytes)

    def copy(self) -> 'MutationBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``MutationBatch`` with copied arrays.
        """
        return MutationBatch(
            self._intervals.copy(),
            self._ref_seqs.copy(),
            self._alt_seqs.copy(),
            self._alt_seqs.copy(),
            self._effects.copy(),
            self._aa_changes.copy(),
            self._qualifiers.copy()
        )

    def __repr__(self):
        return f"<MutationBatch: {len(self)} mutations>"

    def __len__(self):
        return len(self._intervals)

    @property
    def intervals(self) -> IntervalBatch:
        """Returns the interval array for all mutations.

        Returns:
            An ``IntervalBatch`` of mutation positions.
        """
        return self._intervals

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            interval = self._intervals[item]
            ref = self._ref_seqs[item]
            alt = self._alt_seqs[item]
            eff = MutationEffect(self._effects[item])
            aa = self._aa_changes[item]
            quals = self._qualifiers[item]
            return Mutation(interval, ref, alt, eff, aa, quals)

        if isinstance(item, slice):
            return MutationBatch(
                self._intervals[item],
                self._ref_seqs[item],
                self._alt_seqs[item],
                self._effects[item],
                self._aa_changes[item],
                self._qualifiers[item]
            )
        raise TypeError(f"Invalid index type: {type(item)}")

    @classmethod
    def zeros(cls, n: int) -> 'MutationBatch':
        """Creates a MutationBatch with *n* zero-length placeholder mutations.

        Args:
            n: Number of placeholder slots.

        Returns:
            A ``MutationBatch`` with empty sequences and default effects.

        Examples:
            >>> batch = MutationBatch.zeros(5)
            >>> len(batch)
            5
        """
        return cls(
            IntervalBatch.zeros(n),
            Alphabet.DNA.zeros_batch(n),
            Alphabet.DNA.zeros_batch(n),
            effects=None,
            aa_changes=None,
            qualifiers=QualifierBatch.zeros(n)
        )

    def apply_to(self, reference: Seq) -> Any:
        """Applies the mutations to a reference sequence.

        Args:
            reference: The reference ``Seq`` to mutate.

        Returns:
            A ``SparseSeq`` representing the mutated sequence.

        Raises:
            NotImplementedError: SparseSeq is not yet implemented.
        """
        raise NotImplementedError("SparseSeq is not yet implemented.")
