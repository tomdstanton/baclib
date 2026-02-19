"""
Module for managing mutations.
"""
from typing import Iterable, Any, Optional
from enum import IntEnum, auto

import numpy as np

from baclib.containers.feature import Feature, FeatureKey
from baclib.core.interval import IntervalBatch, Interval
from baclib.containers.seq import Seq, SeqBatch
from baclib.containers import Batch
from baclib.lib.protocols import HasIntervals
from baclib.containers.qualifier import QualifierList, QualifierBatch, QualifierType


# Classes --------------------------------------------------------------------------------------------------------------
class MutationEffect(IntEnum):
    """
    Enum for efficient integer storage of mutation effects.
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
    Represents a discrete change: SNP, Insertion, or Deletion.

    Attributes:
        interval (Interval): The location on the REFERENCE sequence.
        ref_seq (Seq): The reference sequence content.
        alt_seq (Seq): The alternative sequence content.
        effect (MutationEffect): The predicted functional impact.
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

    # def __repr__(self):
    #     # VCF-style notation: Pos Ref>Alt (1-based for display)
    #     return f"{self.interval.start + 1}:{self.ref_seq}>{self.alt_seq}"

    @property
    def is_snp(self): return len(self.ref_seq) == 1 and len(self.alt_seq) == 1
    @property
    def is_indel(self): return len(self.ref_seq) != len(self.alt_seq)
    @property
    def batch(self) -> type['Batch']: return MutationBatch
    @property
    def diff(self) -> int:
        """Returns the net change in sequence length (Alt - Ref)."""
        return len(self.alt_seq) - len(self.ref_seq)


class MutationBatch(Batch, HasIntervals):
    """
    Efficient storage for a collection of mutations.
    Can be used to reconstruct sequences via SparseSeq.
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
    def component(self): return Mutation

    @classmethod
    def empty(cls) -> 'MutationBatch':
        return cls(
            IntervalBatch.empty(),
            SeqBatch.empty(),
            SeqBatch.empty()
        )

    @classmethod
    def build(cls, components: Iterable[object]) -> 'MutationBatch':
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
        
        qualifiers = QualifierBatch.from_qualifiers((m.qualifiers for m in mutations))

        return cls(intervals, ref_seqs, alt_seqs, effects, aa_changes, qualifiers)

    @classmethod
    def concat(cls, batches: Iterable['MutationBatch']) -> 'MutationBatch':
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
        return (self._intervals.nbytes + self._ref_seqs.nbytes + 
                self._alt_seqs.nbytes + self._effects.nbytes + 
                self._aa_changes.nbytes + self._qualifiers.nbytes)

    def copy(self) -> 'MutationBatch':
        return MutationBatch(
            self._intervals.copy(),
            self._ref_seqs.copy(),
            self._alt_seqs.copy(),
            self._alt_seqs.copy(),
            self._effects.copy(),
            self._aa_changes.copy(),
            self._qualifiers.copy()
        )

    @classmethod
    def empty(cls) -> 'MutationBatch':
        return MutationBatch(
            IntervalBatch.empty(),
            SeqBatch.empty(),
            SeqBatch.empty(),
            qualifiers=QualifierBatch.empty()
        )

    def __repr__(self):
        return f"<MutationBatch: {len(self)} mutations>"

    def __len__(self):
        return len(self._intervals)

    @property
    def intervals(self) -> IntervalBatch:
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
        return cls(
            IntervalBatch.zeros(n),
            SeqBatch.zeros(n),
            SeqBatch.zeros(n),
            effects=None,
            aa_changes=None,
            qualifiers=QualifierBatch.zeros(n)
        )

    def apply_to(self, reference: Seq) -> Any:
        """
        Applies the mutations to a reference sequence, returning a SparseSeq.
        """
        # We can iterate self because __iter__ yields Mutation objects
        # which SparseSeq accepts.
        # return SparseSeq(reference, self)
        raise NotImplementedError("SparseSeq is not yet implemented.")
