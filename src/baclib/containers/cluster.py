"""Container for sequence clusters and their batched representation."""
from typing import Iterable

import numpy as np

from baclib.containers import RaggedBatch


# Classes --------------------------------------------------------------------------------------------------------------
class Cluster:
    """
    A group of related nodes (e.g. connected component, clique, or community).

    Args:
        nodes: An iterable of member node IDs.
        id_: Optional cluster identifier.
        score: Optional cluster quality score (default ``0.0``).
        representative: Optional ID of the representative member.

    Examples:
        >>> c = Cluster([b'seq1', b'seq2', b'seq3'], id_=b'cluster_0')
        >>> len(c)
        3
    """
    __slots__ = ('_members', 'id', 'score', 'representative')

    def __init__(self, nodes: Iterable, id_: bytes = None, score: float = 0.0, representative: bytes = None):
        self._members = list(nodes)
        self.id = id_
        self.score = score
        self.representative = representative

    @property
    def batch(self) -> type['Batch']:
        """Returns the batch type for this class.

        Returns:
            The ``ClusterBatch`` class.
        """
        return ClusterBatch

    def __len__(self): return len(self._members)

    def __iter__(self): return iter(self._members)

    def __getitem__(self, item): return self._members[item]

    def __repr__(self):
        rep = f", rep={self.representative.decode()}" if self.representative else ""
        return f"Cluster(size={len(self._members)}{rep})"


class ClusterBatch(RaggedBatch):
    """
    Columnar batch of clusters using ragged array storage.

    Stores all member node IDs in a flat array with per-cluster offsets,
    scores, representative IDs, and cluster IDs.

    Args:
        flat_nodes: Flat object array of all node IDs concatenated.
        offsets: ``int32`` offset array (length = ``n_clusters + 1``).
        scores: Optional ``float32`` array of per-cluster scores.
        representatives: Optional object array of representative node IDs.
        ids: Optional object array of cluster IDs.

    Examples:
        >>> batch = ClusterBatch.build([cluster1, cluster2])
        >>> len(batch)
        2
    """
    __slots__ = ('_flat_nodes', '_scores', '_representatives', '_ids')

    def __init__(self, flat_nodes: np.ndarray, offsets: np.ndarray, scores: np.ndarray = None,
                 representatives: np.ndarray = None, ids: np.ndarray = None):
        super().__init__(offsets)
        self._flat_nodes = flat_nodes
        n = len(offsets) - 1
        self._scores = scores if scores is not None else np.zeros(n, dtype=np.float32)
        self._representatives = representatives if representatives is not None else np.full(n, b'', dtype='S1')
        self._ids = ids if ids is not None else np.full(n, b'', dtype='S1')

    @classmethod
    def empty(cls) -> 'ClusterBatch':
        """Creates an empty ClusterBatch with zero clusters.

        Returns:
            An empty ``ClusterBatch``.
        """
        return cls.zeros(0)

    @classmethod
    def zeros(cls, n: int) -> 'ClusterBatch':
        """Creates a ClusterBatch with *n* empty placeholder clusters.

        Args:
            n: Number of placeholder cluster slots.

        Returns:
            A ``ClusterBatch`` with zero-score, empty clusters.
        """
        return cls(
            np.empty(0, dtype='S1'),
            np.zeros(n + 1, dtype=np.int32)
        )

    @property
    def component(self):
        """Returns the scalar type represented by this batch.

        Returns:
            The ``Cluster`` class.
        """
        return Cluster

    @classmethod
    def concat(cls, batches: Iterable['ClusterBatch']) -> 'ClusterBatch':
        """Concatenates multiple ClusterBatch objects into one.

        Args:
            batches: An iterable of ``ClusterBatch`` objects.

        Returns:
            A single concatenated ``ClusterBatch``.
        """
        batches = list(batches)
        if not batches: return cls.empty()
        flat_nodes = np.concatenate([b._flat_nodes for b in batches])
        scores = np.concatenate([b._scores for b in batches])
        reps = np.concatenate([b._representatives for b in batches])
        ids = np.concatenate([b._ids for b in batches])
        offsets = cls._stack_offsets(batches)
        return cls(flat_nodes, offsets, scores, reps, ids)

    @property
    def nbytes(self) -> int:
        """Returns the total memory usage in bytes.

        Returns:
            Total bytes consumed by all internal arrays.
        """
        return super().nbytes + self._flat_nodes.nbytes + self._scores.nbytes + self._representatives.nbytes + self._ids.nbytes

    def copy(self) -> 'ClusterBatch':
        """Returns a deep copy of this batch.

        Returns:
            A new ``ClusterBatch`` with copied arrays.
        """
        return self.__class__(self._flat_nodes.copy(), self._offsets.copy(), self._scores.copy(), self._representatives.copy(), self._ids.copy())

    def __repr__(self):
        return f"<ClusterBatch: {len(self)} clusters>"

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            start = self._offsets[item]
            end = self._offsets[item + 1]
            nodes = self._flat_nodes[start:end]
            return Cluster(nodes, id_=self._ids[item], score=self._scores[item],
                           representative=self._representatives[item])

        if isinstance(item, slice):
            new_offsets, val_start, val_end = self._get_slice_info(item)

            return ClusterBatch(
                self._flat_nodes[val_start:val_end],
                new_offsets,
                self._scores[item],
                self._representatives[item],
                self._ids[item]
            )
        raise TypeError(f"Invalid index type: {type(item)}")

    @classmethod
    def build(cls, components: Iterable['Cluster']) -> 'ClusterBatch':
        """Constructs a ClusterBatch from an iterable of Cluster objects.

        Args:
            components: An iterable of ``Cluster`` objects.

        Returns:
            A new ``ClusterBatch``.

        Examples:
            >>> batch = ClusterBatch.build([cluster1, cluster2])
        """
        clusters = list(components)
        n = len(clusters)
        if n == 0:
            return cls(np.empty(0, dtype='S1'), np.array([0], dtype=np.int32))

        flat_nodes = []
        offsets = [0]
        curr = 0
        for c in clusters:
            flat_nodes.extend(c._members)
            curr += len(c._members)
            offsets.append(curr)

        return cls(
            np.array(flat_nodes),
            np.array(offsets, dtype=np.int32),
            np.array([c.score for c in clusters], dtype=np.float32),
            np.array([c.representative or b'' for c in clusters]),
            np.array([c.id or b'' for c in clusters])
        )
