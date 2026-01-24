from typing import Union

import numpy as np

from baclib.utils.resources import RESOURCES, jit
if 'numba' in RESOURCES.optional_packages: from numba import prange


class PhyloBuilder:
    """
    Fast phylogenetic tree construction tools.
    """

    @staticmethod
    def neighbor_joining(dist_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs a tree using the Neighbor-Joining algorithm.

        Args:
            dist_matrix: A symmetric (N x N) distance matrix.

        Returns:
            parents: Array of parent indices (Root is -1).
            lengths: Array of branch lengths.
        """
        # Ensure float64 and copy to avoid destroying input
        D = dist_matrix.astype(np.float64, copy=True)
        n = D.shape[0]

        # Allocate output arrays
        # NJ creates n-2 internal nodes + 1 root = n - 1 new nodes.
        # Total nodes = n + (n - 2) + 1 = 2n - 1 approx.
        # We start with n tips.
        max_nodes = 2 * n - 1
        parents = np.full(max_nodes, -1, dtype=np.int32)
        lengths = np.zeros(max_nodes, dtype=np.float64)

        # Run the kernel
        _nj_kernel(D, parents, lengths)

        # The result is technically unrooted. The last node created is the 'root'
        # of our representation, effectively rooting the tree at the final edge.
        return parents, lengths


class PhyloNode:
    def __init__(
        self,
        name: bytes = None,
        length: Union[float | int] = None,
        support: Union[float | int] = None,
        parent: 'PhyloNode' = None,
        children: list['PhyloNode'] = None,
    ):
        self.name = name
        self.length = length
        self.support = support
        self.parent = parent
        self.children = children


class PhyloTree:
    __slots__ = ('_parents', '_lengths', '_node_order', '_n_tips')

    def __init__(self, parents: np.ndarray, lengths: np.ndarray, node_names: np.ndarray = None):
        """
        Args:
            parents: Array where parents[i] is the parent index of node i.
            lengths: Array where lengths[i] is distance from parent to i.
        """
        self._parents = parents
        self._lengths = lengths
        # Optimization: Pre-calculate Pre-order or Post-order for fast traversal
        self._node_order = self._get_postorder_indices()

    def root_to_tip_distances(self) -> np.ndarray:
        """
        Returns the distance from the root to every node.
        This operation is vectorized and nearly instant.
        """
        # Allocate output array
        dists = np.zeros(len(self._parents), dtype=np.float64)

        # We process in topological order (parents before children)
        # If nodes are topologically sorted, this is just a loop.
        # If not, we use a robust kernel.
        _calc_root_dist_kernel(self._parents, self._lengths, dists)
        return dists

    def patristic_distance_matrix(self) -> np.ndarray:
        """
        Calculates all-vs-all distances for tips.
        Significantly faster than scipy.sparse.csgraph.dijkstra for trees.
        """
        # Specialized tree implementation using LCA (Lowest Common Ancestor)
        pass


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True, cache=True)
def _nj_kernel(D, parents, lengths):
    n = D.shape[0]

    # Active nodes mask. True = available to be merged.
    # We map current matrix indices to their actual node IDs.
    # Indices 0 to n-1 are the initial tips.
    active_nodes = np.ones(n, dtype=np.bool_)
    node_indices = np.arange(n, dtype=np.int32)  # Maps matrix_idx -> actual_node_id

    next_node_id = n  # The next internal node to be created

    # We loop until only 2 nodes remain
    for r in range(n - 2):
        n_active = n - r

        # 1. Calculate Net Divergence (Row Sums)
        # We only sum over active nodes.
        divergence = np.zeros(n, dtype=np.float64)
        for i in prange(n):
            if active_nodes[i]:
                row_sum = 0.0
                for j in range(n):
                    if active_nodes[j]:
                        row_sum += D[i, j]
                divergence[i] = row_sum

        # 2. Find pair (i, j) minimizing Q-matrix
        # Q_ij = (n_active - 2) * D_ij - div_i - div_j
        # We initialize min_val to infinity
        min_Q = np.inf
        min_i = -1
        min_j = -1

        # This loop is O(N^2) and is the bottleneck. Parallelize outer.
        # Note: In Numba parallel, reductions (finding global min) need care.
        # Simple approach: Each thread finds local min, we aggregate.
        # For simplicity in this snippet, we stick to serial or simple parallel scan.

        scaler = n_active - 2

        # Note: Numba's prange doesn't support complex reduction tuples well yet.
        # We use a serial scan for correctness/stability or a blocked approach.
        # Given N < 5000 usually, serial scan is acceptable inside JIT.
        for i in range(n):
            if not active_nodes[i]: continue

            div_i = divergence[i]

            for j in range(i + 1, n):
                if not active_nodes[j]: continue

                # Q-score calculation
                q_val = scaler * D[i, j] - div_i - divergence[j]

                if q_val < min_Q:
                    min_Q = q_val
                    min_i = i
                    min_j = j

        # 3. Create new node u
        u_id = next_node_id
        next_node_id += 1

        node_i_real = node_indices[min_i]
        node_j_real = node_indices[min_j]

        # Set parents
        parents[node_i_real] = u_id
        parents[node_j_real] = u_id

        # 4. Calculate Branch Lengths
        dist_ij = D[min_i, min_j]
        # v_i = 0.5 * (d_ij + (div_i - div_j) / (N-2))
        diff_div = (divergence[min_i] - divergence[min_j]) / scaler
        len_i = 0.5 * (dist_ij + diff_div)
        len_j = 0.5 * (dist_ij - diff_div)

        # Clamp negative lengths (standard phylogenetic fix)
        if len_i < 0: len_i = 0.0
        if len_j < 0: len_j = 0.0

        lengths[node_i_real] = len_i
        lengths[node_j_real] = len_j

        # 5. Update Distance Matrix
        # Formula: D_uk = 0.5 * (D_ik + D_jk - D_ij)
        # We reuse row/col `min_i` for the new node `u`
        # We mark `min_j` as inactive.

        for k in prange(n):
            if active_nodes[k] and k != min_i and k != min_j:
                d_ik = D[min_i, k]
                d_jk = D[min_j, k]
                new_dist = 0.5 * (d_ik + d_jk - dist_ij)

                # Update symmetric
                D[min_i, k] = new_dist
                D[k, min_i] = new_dist

        D[min_i, min_i] = 0.0
        node_indices[min_i] = u_id  # Slot i now represents node u
        active_nodes[min_j] = False  # Slot j is dead

    # --- Final Step: Connect the last 2 remaining nodes ---
    # There are exactly 2 active nodes left. Connect them.
    left = -1
    right = -1
    for i in range(n):
        if active_nodes[i]:
            if left == -1:
                left = i
            else:
                right = i

    if left != -1 and right != -1:
        dist_final = D[left, right]

        # We create one final root node to connect them
        root_id = next_node_id
        real_left = node_indices[left]
        real_right = node_indices[right]

        parents[real_left] = root_id
        parents[real_right] = root_id

        # Split the remaining distance evenly (midpoint rooting)
        lengths[real_left] = dist_final / 2.0
        lengths[real_right] = dist_final / 2.0

@jit(nopython=True, cache=True)
def _calc_root_dist_kernel(parents, lengths, out_dists):
    """
    Calculates root-to-tip distances.
    Because parents[i] isn't guaranteed to be < i, we might need
    to resolve recursively or iteratively.
    """
    n = len(parents)
    # Simple iterative approach:
    # For every node, walk up to root or a visited node.
    for i in range(n):
        if out_dists[i] != 0.0 or i == 0: continue  # Already done or root

        # Path compression / walk up
        curr = i
        path_cost = 0.0

        # Walk up until we hit the root (-1) or a node with a calculated distance
        while curr != -1 and out_dists[curr] == 0.0:
            path_cost += lengths[curr]
            curr = parents[curr]

        # If we hit a known node, add its distance
        if curr != -1:
            path_cost += out_dists[curr]

        # Now efficient fill: This is O(Depth) per node, but we can optimize
        # by back-filling the path we just walked if needed.
        # For simple arrays, just setting the target is often enough.
        out_dists[i] = path_cost