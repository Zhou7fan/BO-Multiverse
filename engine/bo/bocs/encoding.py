# Encoding between the unified choice-index assignment and BOCS one-hot blocks.
#
# Unified representation (used by all four optimizers and the experiment driver):
#   assignment : 1-D int array of length = n_fork, each entry in [0, n_vertices[f])
#                i.e. the choice index picked at each fork.
#
# BOCS internal representation (contract section 8):
#   binary vector : flattened one-hot.  Fork f occupies a contiguous block of
#                   length n_vertices[f]; exactly one bit in the block is 1.
#
# Invalid assignments (0 bits or >1 bit in a block) are never produced here;
# if an external evaluator receives one it applies a large penalty.

import numpy as np


class BlockEncoder(object):
    """Maps choice-index assignments to/from flattened one-hot blocks."""

    def __init__(self, n_vertices):
        self.n_vertices = np.asarray(n_vertices, dtype=int)
        self.n_forks = len(self.n_vertices)
        # block start offsets in the flattened binary vector
        self.offsets = np.concatenate([[0], np.cumsum(self.n_vertices)])
        self.n_bits = int(self.offsets[-1])

    # ---- assignment (n_fork,) int  ->  binary row (n_bits,) ---------------
    def encode_onehot(self, assignment):
        assignment = np.asarray(assignment, dtype=int).reshape(-1)
        assert assignment.shape[0] == self.n_forks
        x = np.zeros(self.n_bits)
        for f in range(self.n_forks):
            c = assignment[f]
            assert 0 <= c < self.n_vertices[f], f"fork {f} choice {c} out of range"
            x[self.offsets[f] + c] = 1.0
        return x

    def encode_matrix(self, assignments):
        assignments = np.asarray(assignments, dtype=int)
        return np.vstack([self.encode_onehot(a) for a in assignments])

    # ---- binary row (n_bits,) -> assignment (n_fork,) int (argmax block) -
    def decode_onehot(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        assignment = np.zeros(self.n_forks, dtype=int)
        for f in range(self.n_forks):
            block = x[self.offsets[f]:self.offsets[f + 1]]
            # pick the active bit; if none/multiple, fall back to argmax
            assignment[f] = int(np.argmax(block))
        return assignment

    def random_valid_assignment(self, rng):
        return np.array([rng.integers(0, self.n_vertices[f]) for f in range(self.n_forks)], dtype=int)

    # ---- block-internal flip: move the hot bit in fork `f` to another choice
    def block_flip(self, x, f, rng):
        x = x.copy()
        k = self.n_vertices[f]
        start = self.offsets[f]
        cur = int(np.argmax(x[start:start + k]))
        choices = [c for c in range(k) if c != cur]
        if not choices:
            return x
        new = choices[rng.integers(0, len(choices))]
        x[start + cur] = 0.0
        x[start + new] = 1.0
        return x
