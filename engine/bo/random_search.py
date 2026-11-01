# Uniform-random search baseline over the valid (one-hot-per-fork) space.
#
# Unified interface identical to the BO optimizers:
#   __init__(n_vertices, warmups=None, seed=0, ...)
#   suggest(observed_X, observed_y) -> assignment  (1-D int array, length n_fork)
#
# At every suggest we draw a uniformly valid assignment (each fork picks a
# choice independently uniform in [0, n_vertices[f])). We record best-vs-query
# downstream for comparison. No surrogate / no posterior sampling.

import numpy as np


class RandomSearch(object):
    acquisition_name = 'RandomSearch'

    def __init__(self, n_vertices, warmups=None, seed=0, **kwargs):
        self.n_vertices = list(n_vertices)
        self.n_forks = len(self.n_vertices)
        self.warmups = warmups
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.suggest_count = 0
        self.log_lines = []

    def suggest(self, observed_X, observed_y):
        self.suggest_count += 1
        assignment = np.array([self.rng.integers(0, self.n_vertices[f])
                               for f in range(self.n_forks)], dtype=int)
        self.log_lines.append(f"[RandomSearch suggest#{self.suggest_count}] sampled assignment={assignment.tolist()}")
        return assignment
