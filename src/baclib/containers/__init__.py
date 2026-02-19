"""
This module contains abstract containers for various components in bacterial genomics. Each component may have a
batched counterpart for efficient batch processing.
"""
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


# Classes --------------------------------------------------------------------------------------------------------------
class Batchable(ABC):
    """
    Interface for components that can be batched.
    
    Classes implementing this interface must define a ``batch`` property
    that returns their corresponding ``Batch`` class.
    """
    __slots__ = ()
    @property
    @abstractmethod
    def batch(self) -> type['Batch']:
        """Returns the Batch class associated with this component type."""
        ...


class Batch(ABC):
    """
    Abstract base class for all batch containers.

    Batches are columnar containers that store multiple instances of a component
    efficiently (usually using SoA layout with NumPy arrays). They enforce the
    Sequence protocol (len, getitem, iter).
    """
    __slots__ = ()
    @abstractmethod
    def __len__(self) -> int: ...
    @classmethod
    @abstractmethod
    def empty(cls) -> 'Batch':
        """Creates an empty batch."""
        ...
    @property
    @abstractmethod
    def component(self):
        """Returns the component class stored in this batch."""
        ...
    @classmethod
    @abstractmethod
    def build(cls, components: Iterable[object]) -> 'Batch':
        """Constructs a batch from an iterable of components."""
        ...
    @classmethod
    @abstractmethod
    def concat(cls, batches: Iterable['Batch']) -> 'Batch':
        """Concatenates multiple batches into one."""
        ...
    @classmethod
    def zeros(cls, n: int) -> 'Batch':
        """
        Creates a batch of *n* filled with "empty" or default components.

        For RaggedBatches, this means *n* empty lists.
        For FixedBatches, this means *n* default/null entries.

        Args:
            n: Size of the batch.

        Returns:
            A new batch of size *n*.

        Raises:
            NotImplementedError: If the subclass does not support zero-initialization.
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
        """Returns a deep copy of the batch."""
        raise NotImplementedError


class RaggedBatch(Batch):
    """
    Base class for batches that store variable-length items in a flattened format (CSR-like).
    
    Manages the offsets array and length calculation. Subclasses typically add
    flat data arrays.

    Args:
        offsets: Array of offsets (size *N* + 1).
        validate: If ``True``, checks offset validity.
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
        """Returns the total number of elements flattened across all components."""
        return self._offsets[-1] if self._length >= 0 else 0

    @classmethod
    def zeros(cls, n: int) -> 'RaggedBatch':
        """Creates a batch of *n* empty components (length 0)."""
        # [0, 0, ..., 0] -> all lengths are 0
        offsets = np.zeros(n + 1, dtype=np.int32)
        return cls(offsets)

    @classmethod
    def empty(cls) -> 'RaggedBatch':
        """Creates an empty batch."""
        return cls.zeros(0)

    @property
    def lengths(self) -> np.ndarray:
        """Returns the lengths of the components as a numpy array."""
        return np.diff(self._offsets)

    @property
    def nbytes(self) -> int:
        """Returns the memory usage of the offset array."""
        return self._offsets.nbytes

    def __len__(self) -> int: return self._length

    def _get_slice_info(self, item: slice) -> tuple[np.ndarray, int, int]:
        """Helper to calculate new offsets and data slice indices for slicing."""
        start, stop, step = item.indices(len(self))
        if step != 1: raise NotImplementedError("Batch slicing with step != 1 not supported")
        val_start = self._offsets[start]
        val_end = self._offsets[stop]
        new_offsets = self._offsets[start:stop+1] - val_start
        return new_offsets, val_start, val_end

    @staticmethod
    def _stack_offsets(batches: Iterable['RaggedBatch']) -> np.ndarray:
        """Helper to concatenate offset arrays from multiple batches."""
        batches = list(batches)
        offsets_list = [b._offsets for b in batches]
        if not offsets_list: return np.zeros(1, dtype=np.int32)
        
        # We must shift subsequent offsets by the total count of previous batches
        # offsets[0] is always 0, so we slice [1:] for subsequent appends
        shifts = np.cumsum([0] + [b._offsets[-1] for b in batches[:-1]])
        if len(batches) == 1:
            return offsets_list[0]
            
        parts = [offsets_list[0]]
        for i in range(1, len(batches)):
             parts.append(offsets_list[i][1:] + shifts[i])
             
        return np.concatenate(parts)
