from typing import Protocol, runtime_checkable


@runtime_checkable
class HasAlphabet(Protocol):
    """Protocol for objects that possess an Alphabet."""
    @property
    def alphabet(self) -> 'Alphabet': ...


@runtime_checkable
class HasInterval(Protocol):
    """Protocol for objects that possess a genomic Interval (e.g. Feature, Alignment)."""

    @property
    def interval(self) -> 'Interval': ...


@runtime_checkable
class HasIntervals(Protocol):
    """Protocol for objects that possess a batch of Intervals (e.g. FeatureBatch)."""

    @property
    def intervals(self) -> 'IntervalBatch': ...

