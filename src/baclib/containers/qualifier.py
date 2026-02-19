"""List and batch containers for key-value qualifier annotations on genomic features."""
from typing import Union, MutableSequence, Iterable
import itertools

import numpy as np

from baclib.containers import RaggedBatch


# Classes --------------------------------------------------------------------------------------------------------------
QualifierType = Union[int, float, bytes, bool]


class QualifierList(MutableSequence):
    """
    A list-like container for (key, value) tuples that also supports dictionary-style access.

    Maintains insertion order and allows duplicate keys. Used to store
    feature and record qualifiers from formats like GenBank and GFF.

    Args:
        items: Initial qualifier data as an iterable of ``(key, value)`` tuples
            or a dictionary.

    Examples:
        >>> quals = QualifierList([(b'gene', b'dnaA'), (b'product', b'replication initiator')])
        >>> quals[b'gene']
        b'dnaA'
        >>> quals.add(b'note', b'essential')
        >>> len(quals)
        3
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

    def insert(self, index, value):
        """Inserts a ``(key, value)`` tuple at the given index.

        Args:
            index: Position to insert at.
            value: A ``(key, value)`` tuple to insert.

        Examples:
            >>> quals = QualifierList([(b'gene', b'dnaA')])
            >>> quals.insert(0, (b'locus_tag', b'b0001'))
            >>> quals[0]
            (b'locus_tag', b'b0001')
        """
        self._data.insert(index, value)

    def __eq__(self, other):
        if isinstance(other, QualifierList): return self._data == other._data
        if isinstance(other, list): return self._data == other
        return False

    def get(self, key: bytes, default=None) -> QualifierType:
        """Returns the first value for a key, or *default* if not found.

        Args:
            key: The qualifier key to look up.
            default: Value to return if key is absent.

        Returns:
            The first matching value, or *default*.

        Examples:
            >>> quals = QualifierList([(b'gene', b'dnaA')])
            >>> quals.get(b'gene')
            b'dnaA'
            >>> quals.get(b'missing', b'N/A')
            b'N/A'
        """
        for k, v in self._data:
            if k == key: return v
        return default

    def get_all(self, key: bytes) -> list[QualifierType]:
        """Returns all values for a key.

        Args:
            key: The qualifier key to look up.

        Returns:
            A list of all matching values (empty if key is absent).

        Examples:
            >>> quals = QualifierList([(b'db_xref', b'GI:123'), (b'db_xref', b'UniProt:P0A')])
            >>> quals.get_all(b'db_xref')
            [b'GI:123', b'UniProt:P0A']
        """
        return [v for k, v in self._data if k == key]

    def add(self, key: bytes, value: QualifierType):
        """Appends a new key-value pair.

        Args:
            key: The qualifier key.
            value: The qualifier value.

        Examples:
            >>> quals = QualifierList()
            >>> quals.add(b'gene', b'dnaA')
            >>> len(quals)
            1
        """
        self._data.append((key, value))

    def to_dict(self) -> dict[bytes, QualifierType]:
        """Converts to a standard dictionary (lossy for duplicate keys).

        Returns:
            A dict mapping each key to its *last* value.

        Examples:
            >>> quals = QualifierList([(b'gene', b'dnaA'), (b'product', b'initiator')])
            >>> quals.to_dict()
            {b'gene': b'dnaA', b'product': b'initiator'}
        """
        return {k: v for k, v in self._data}

    def items(self):
        """Returns an iterator over ``(key, value)`` tuples.

        Returns:
            An iterator of ``(key, value)`` tuples.

        Examples:
            >>> quals = QualifierList([(b'gene', b'dnaA')])
            >>> list(quals.items())
            [(b'gene', b'dnaA')]
        """
        return iter(self._data)


class QualifierBatch(RaggedBatch):
    """
    A batch of qualifier lists stored in columnar format (Structure-of-Arrays).

    Internally uses a shared vocabulary for keys (``_key_vocab``), integer key IDs
    (``_key_ids``), and an object array of values (``_values``), with ragged offsets
    to delineate each qualifier list.

    Args:
        key_vocab: Array of unique qualifier key bytes.
        key_ids: Integer array mapping each qualifier entry to its vocab index.
        values: Object array of qualifier values.
        offsets: Integer array of cumulative offsets (length = n_items + 1).

    Examples:
        >>> batch = QualifierBatch.build([
        ...     [(b'gene', b'dnaA')],
        ...     [(b'gene', b'dnaN'), (b'product', b'polymerase')],
        ... ])
        >>> len(batch)
        2
        >>> batch[0]
        [(b'gene', b'dnaA')]
    """
    __slots__ = ('_key_vocab', '_key_ids', '_values')

    def __init__(self, key_vocab, key_ids, values, offsets):
        super().__init__(offsets)
        self._key_vocab = key_vocab
        self._key_ids = key_ids
        self._values = values

    @classmethod
    def build(cls, qualifiers_list: Iterable[Iterable[tuple[bytes, QualifierType]]]) -> 'QualifierBatch':
        """Constructs a QualifierBatch from an iterable of qualifier iterables.

        Args:
            qualifiers_list: An iterable where each element is an iterable of
                ``(key, value)`` tuples representing one qualifier list.

        Returns:
            A new QualifierBatch.

        Examples:
            >>> batch = QualifierBatch.build([
            ...     [(b'gene', b'dnaA')],
            ...     [(b'gene', b'dnaN')],
            ... ])
            >>> len(batch)
            2
        """
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
            np.array(key_vocab),
            np.array(flat_key_ids, dtype=np.int32),
            np.array(flat_values, dtype=object),
            np.array(offsets, dtype=np.int32)
        )

    @classmethod
    def concat(cls, batches: Iterable['QualifierBatch']) -> 'QualifierBatch':
        """Concatenates multiple QualifierBatch objects into one.

        Rebuilds the shared vocabulary to handle merging across batches.

        Args:
            batches: An iterable of QualifierBatch objects.

        Returns:
            A single concatenated QualifierBatch.

        Examples:
            >>> a = QualifierBatch.build([[(b'gene', b'dnaA')]])
            >>> b = QualifierBatch.build([[(b'gene', b'dnaN')]])
            >>> combined = QualifierBatch.concat([a, b])
            >>> len(combined)
            2
        """
        def iterator():
            for b in batches:
                for item in b:
                    yield item
        return cls.build(iterator())

    @classmethod
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by vocabulary, key IDs, values, and offsets.
        """
        return super().nbytes + self._key_vocab.nbytes + self._key_ids.nbytes + self._values.nbytes

    def copy(self) -> 'QualifierBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new QualifierBatch with copied arrays.
        """
        return self.__class__(self._key_vocab.copy(), self._key_ids.copy(), self._values.copy(), self._offsets.copy())

    @classmethod
    def empty(cls) -> 'QualifierBatch':
        """Creates an empty QualifierBatch with zero items.

        Returns:
            An empty QualifierBatch.

        Examples:
            >>> batch = QualifierBatch.empty()
            >>> len(batch)
            0
        """
        return cls(
            np.array([], dtype='S1'),
            np.array([], dtype=np.int32),
            np.array([], dtype=object),
            np.zeros(1, dtype=np.int32)
        )

    @classmethod
    def zeros(cls, n: int) -> 'QualifierBatch':
        """Creates a QualifierBatch with *n* empty qualifier lists.

        Args:
            n: Number of empty qualifier slots.

        Returns:
            A QualifierBatch where each of the *n* items has zero qualifiers.

        Examples:
            >>> batch = QualifierBatch.zeros(3)
            >>> len(batch)
            3
            >>> batch[0]
            []
        """
        return cls(
            np.array([], dtype='S1'),
            np.array([], dtype=np.int32),
            np.array([], dtype=object),
            np.zeros(n + 1, dtype=np.int32)
        )

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``tuple`` type.
        """
        return tuple

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
