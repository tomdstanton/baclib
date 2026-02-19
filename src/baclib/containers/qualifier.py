from typing import Union, MutableSequence, Iterable
import itertools

import numpy as np

from baclib.containers import RaggedBatch


# Classes --------------------------------------------------------------------------------------------------------------
QualifierType = Union[int, float, bytes, bool]


class QualifierList(MutableSequence):
    """
    A list-like container for (key, value) tuples that also supports dictionary-style access.
    Maintains insertion order and allows duplicate keys.
    """
    __slots__ = ('_data',)

    def __init__(self, items: Union[Iterable[tuple[bytes, QualifierType]], dict] = None):
        if isinstance(items, dict):
            self._data = list(items.items())
        else:
            self._data = list(items) if items else []

    def __getitem__(self, item):
        # List-style access
        if isinstance(item, (int, slice)): return self._data[item]
        # Dict-style access (First match)
        for k, v in self._data:
            if k == item: return v
        return None

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            self._data[key] = value
            return

        # Dict-style set: Replace first occurrence or append
        for i, (k, v) in enumerate(self._data):
            if k == key:
                self._data[i] = (key, value)
                return
        self._data.append((key, value))

    def __delitem__(self, index): del self._data[index]
    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)
    def __repr__(self):
        if len(self) > 6:
            return f"[{', '.join(repr(x) for x in self[:3])}, ..., {', '.join(repr(x) for x in self[-3:])}]"
        return repr(self._data)
    def insert(self, index, value): self._data.insert(index, value)

    def __eq__(self, other):
        if isinstance(other, QualifierList): return self._data == other._data
        if isinstance(other, list): return self._data == other
        return False

    def get(self, key, default=None):
        """Returns the first value for a key, or default."""
        for k, v in self._data:
            if k == key: return v
        return default

    def get_all(self, key: bytes) -> list[QualifierType]:
        """Returns all values for a key."""
        return [v for k, v in self._data if k == key]

    def add(self, key: bytes, value: QualifierType):
        """Adds a new key-value pair."""
        self._data.append((key, value))

    def to_dict(self) -> dict[bytes, QualifierType]:
        """Converts the list to a standard dictionary (lossy for duplicates)."""
        return {k: v for k, v in self._data}

    def items(self):
        """Returns occurrances of (key, value) tuples."""
        return iter(self._data)


class QualifierBatch(RaggedBatch):
    """
    A batch of qualifier lists, stored in a columnar format (Structure-of-Arrays).
    """
    __slots__ = ('_key_vocab', '_key_ids', '_values')

    def __init__(self, key_vocab, key_ids, values, offsets):
        super().__init__(offsets)
        self._key_vocab = key_vocab
        self._key_ids = key_ids
        self._values = values

    @classmethod
    def build(cls, qualifiers_list: Iterable[Iterable[tuple[bytes, QualifierType]]]) -> 'QualifierBatch':
        key_to_id = {}
        key_vocab = []
        flat_key_ids = []
        flat_values = []

        offsets = [0]
        curr_idx = 0

        for quals in qualifiers_list:
            for k, v in quals:
                if k not in key_to_id:
                    key_to_id[k] = len(key_vocab)
                    key_vocab.append(k)
                flat_key_ids.append(key_to_id[k])
                flat_values.append(v)
                curr_idx += 1
            offsets.append(curr_idx)

        return cls(
            np.array(key_vocab, dtype=object),
            np.array(flat_key_ids, dtype=np.int32),
            np.array(flat_values, dtype=object),
            np.array(offsets, dtype=np.int32)
        )

    @classmethod
    def concat(cls, batches: Iterable['QualifierBatch']) -> 'QualifierBatch':
        # Rebuild strategy to handle vocabulary merging
        # This iterates all items, which is O(N), but safe.
        def iterator():
            for b in batches:
                for item in b:
                    yield item
        return cls.build(iterator())

    @classmethod
    def nbytes(self) -> int:
        return super().nbytes + self._key_vocab.nbytes + self._key_ids.nbytes + self._values.nbytes

    def copy(self) -> 'QualifierBatch':
        return self.__class__(self._key_vocab.copy(), self._key_ids.copy(), self._values.copy(), self._offsets.copy())

    @classmethod
    def empty(cls) -> 'QualifierBatch':
        return cls(
            np.array([], dtype=object),
            np.array([], dtype=np.int32),
            np.array([], dtype=object),
            np.zeros(1, dtype=np.int32)
        )

    @classmethod
    def zeros(cls, n: int) -> 'QualifierBatch':
        return cls(
            np.array([], dtype=object),
            np.array([], dtype=np.int32),
            np.array([], dtype=object),
            np.zeros(n + 1, dtype=np.int32)
        )

    @property
    def component(self): return tuple # Qualifiers are usually lists of tuples

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            start = self._offsets[item]
            end = self._offsets[item + 1]
            if start == end: return []
            keys = self._key_vocab[self._key_ids[start:end]]
            values = self._values[start:end]
            return list(zip(keys, values))

        if isinstance(item, slice):
            new_offsets, val_start, val_end = self._get_slice_info(item)
            new_key_ids = self._key_ids[val_start:val_end]
            new_values = self._values[val_start:val_end]

            # Create new batch (Zero Copy views where possible)
            obj = object.__new__(QualifierBatch)
            obj._key_vocab = self._key_vocab
            obj._key_ids = new_key_ids
            obj._values = new_values
            obj._offsets = new_offsets
            return obj

        raise TypeError(f"Invalid index type: {type(item)}")
