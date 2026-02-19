from typing import Iterable

import numpy as np

from baclib.containers import RaggedBatch


# Classes --------------------------------------------------------------------------------------------------------------
class Cluster:
    """
    Represents a group of nodes (e.g. connected component, clique, or community).
    """
    __slots__ = ('_members', 'id', 'score', 'representative')

    def __init__(self, nodes: Iterable, id_: bytes = None, score: float = 0.0, representative: bytes = None):
        self._members = list(nodes)
        self.id = id_
        self.score = score
        self.representative = representative

    @property
    def batch(self) -> type['Batch']: return ClusterBatch

    def __len__(self): return len(self._members)

    def __iter__(self): return iter(self._members)

    def __getitem__(self, item): return self._members[item]

    def __repr__(self):
        rep = f", rep={self.representative.decode()}" if self.representative else ""
        return f"Cluster(size={len(self._members)}{rep})"


class ClusterBatch(RaggedBatch):
    """
    Efficient storage for many clusters (Structure of Arrays).
    """
    __slots__ = ('_flat_nodes', '_scores', '_representatives', '_ids')

    def __init__(self, flat_nodes: np.ndarray, offsets: np.ndarray, scores: np.ndarray = None,
                 representatives: np.ndarray = None, ids: np.ndarray = None):
        super().__init__(offsets)
        self._flat_nodes = flat_nodes
        n = len(offsets) - 1
        self._scores = scores if scores is not None else np.zeros(n, dtype=np.float32)
        self._representatives = representatives if representatives is not None else np.full(n, None, dtype=object)
        self._ids = ids if ids is not None else np.full(n, None, dtype=object)

    @classmethod
    def empty(cls) -> 'ClusterBatch':
        return cls.zeros(0)

    @classmethod
    def zeros(cls, n: int) -> 'ClusterBatch':
        return cls(
            np.empty(0, dtype=object),
            np.zeros(n + 1, dtype=np.int32)
        )

    @property
    def component(self): return Cluster

    @classmethod
    def concat(cls, batches: Iterable['ClusterBatch']) -> 'ClusterBatch':
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
        return super().nbytes + self._flat_nodes.nbytes + self._scores.nbytes + self._representatives.nbytes + self._ids.nbytes

    def copy(self) -> 'ClusterBatch':
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
        clusters = list(components)
        n = len(clusters)
        if n == 0:
            return cls(np.empty(0, dtype=object), np.array([0], dtype=np.int32))

        flat_nodes = []
        offsets = [0]
        curr = 0
        for c in clusters:
            flat_nodes.extend(c._members)
            curr += len(c._members)
            offsets.append(curr)

        return cls(
            np.array(flat_nodes, dtype=object),
            np.array(offsets, dtype=np.int32),
            np.array([c.score for c in clusters], dtype=np.float32),
            np.array([c.representative for c in clusters], dtype=object),
            np.array([c.id for c in clusters], dtype=object)
        )
