"""
This module contains abstract containers for various components in bacterial genomics. Each component may have a
batched counterpart for efficient batch processing.
"""
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


# Classes --------------------------------------------------------------------------------------------------------------
class Batchable(ABC):
    """Components of a batch - calling the batch method returns the batch type"""
    __slots__ = ()
    @property
    @abstractmethod
    def batch(self) -> type['Batch']: ...


class Batch(ABC):
    """
    Abstract base class for all batch containers.
    Enforces the Sequence protocol (len, getitem, iter).
    """
    __slots__ = ()
    @abstractmethod
    def __len__(self) -> int: ...
    @classmethod
    @abstractmethod
    def empty(cls) -> 'Batch': ...
    @property
    @abstractmethod
    def component(self): ...
    @classmethod
    @abstractmethod
    def build(cls, components: Iterable[object]) -> 'Batch': ...
    @classmethod
    @abstractmethod
    def concat(cls, batches: Iterable['Batch']) -> 'Batch': ...
    @classmethod
    def zeros(cls, n: int) -> 'Batch':
        """
        Creates a batch of size n filled with "empty" or default components.
        For RaggedBatches, this means n empty lists.
        For FixedBatches, this means n default/null entries.
        """
        if n == 0: return cls.empty()
        raise NotImplementedError(f"{cls.__name__} does not support zeros(n>0)")
    @abstractmethod
    def __getitem__(self, item): ...
    def __iter__(self):
        for i in range(len(self)): yield self[i]
    def __bool__(self):
        return len(self) > 0
    @property
    def nbytes(self) -> int:
        """Returns the approximate memory usage of the batch in bytes."""
        return 0
    def copy(self) -> 'Batch':
        """Returns a copy of the batch."""
        raise NotImplementedError


class RaggedBatch(Batch):
    """
    Base class for batches that store variable-length items in a flattened format (CSR-like).
    Manages the offsets array and length calculation.
    """
    __slots__ = ('_offsets', '_length')
    def __init__(self, offsets: np.ndarray, validate: bool = False):
        self._offsets = offsets
        self._length = len(offsets) - 1
        if validate: self._validate()

    def _validate(self):
        if self._length < 0:
            raise ValueError("Offsets array must contain at least one element (usually [0]).")
        if self._offsets[0] != 0:
            raise ValueError("Offsets must start with 0.")
        if not np.all(self._offsets[:-1] <= self._offsets[1:]):
            raise ValueError("Offsets must be non-decreasing.")

    @property
    def total_size(self) -> int:
        """Returns the total size of the flattened data."""
        return self._offsets[-1] if self._length >= 0 else 0

    @classmethod
    def zeros(cls, n: int) -> 'RaggedBatch':
        """
        Creates a batch of n empty lists.
        """
        # [0, 0, ..., 0] -> all lengths are 0
        offsets = np.zeros(n + 1, dtype=np.int32)
        return cls(offsets)

    @classmethod
    def empty(cls) -> 'RaggedBatch':
        return cls.zeros(0)

    @property
    def lengths(self) -> np.ndarray:
        """Returns the lengths of the components."""
        return np.diff(self._offsets)

    @property
    def nbytes(self) -> int: return self._offsets.nbytes

    def __len__(self) -> int: return self._length

    def _get_slice_info(self, item: slice) -> tuple[np.ndarray, int, int]:
        start, stop, step = item.indices(len(self))
        if step != 1: raise NotImplementedError("Batch slicing with step != 1 not supported")
        val_start = self._offsets[start]
        val_end = self._offsets[stop]
        new_offsets = self._offsets[start:stop+1] - val_start
        return new_offsets, val_start, val_end

    @staticmethod
    def _stack_offsets(batches: Iterable['RaggedBatch']) -> np.ndarray:
        """Helper to concatenate offset arrays from multiple batches."""
        offsets_list = [b._offsets for b in batches]
        if not offsets_list: return np.zeros(1, dtype=np.int32)
        
        # We must shift subsequent offsets by the total count of previous batches
        # offsets[0] is always 0, so we slice [1:] for subsequent appends
        shifts = np.cumsum([0] + [b._offsets[-1] for b in batches[:-1]])
        return np.concatenate([o[:-1] + s for o, s in zip(offsets_list, shifts)] + [offsets_list[-1][-1:] + shifts[-1]])
