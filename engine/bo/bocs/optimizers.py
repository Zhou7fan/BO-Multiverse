# Unified BOCS optimizers (SA and SMC acquisition) over the block-one-hot space.
#
# Both classes share the contract:
#   __init__(n_vertices, warmups=None, seed=0, ...algo hyper-params)
#   suggest(observed_X, observed_y) -> assignment  (1-D int array, length n_fork)
#
# observed_X : iterable of assignments; each assignment is a length-n_fork int
#              vector of choice indices (one entry per fork).
# observed_y : 1-D float array of the external objective at those assignments
#              (LOWER is better; invalid assignments carry a large penalty).
#
# The statistical surrogate is always the BOCS order-2 Bayesian linear model
# with the horseshoe prior (LinReg + bhs) -- never a plain GP.

import numpy as np

from .LinReg import LinReg
from .encoding import BlockEncoder
from .smc import binary_smc


class _BaseBOCS(object):
    acquisition_name = 'base'

    def __init__(self, n_vertices, warmups=None, seed=0, order=2, lam=1e-4,
                 sa_iter=200, sa_reruns=5, smc_particles=20, smc_budget=200):
        self.n_vertices = list(n_vertices)
        self.n_forks = len(self.n_vertices)
        self.warmups = warmups
        self.seed = seed
        self.order = order
        self.lam = lam
        self.sa_iter = sa_iter
        self.sa_reruns = sa_reruns
        self.smc_particles = smc_particles
        self.smc_budget = smc_budget
        self.encoder = BlockEncoder(self.n_vertices)
        self.rng = np.random.default_rng(seed)
        self.suggest_count = 0
        self.log_lines = []

    def _log(self, msg):
        line = f"[{self.acquisition_name} suggest#{self.suggest_count}] {msg}"
        self.log_lines.append(line)

    # -- fit the BOCS surrogate on all observed points ----------------------
    def _fit_surrogate(self, observed_X, observed_y):
        Xbin = self.encoder.encode_matrix(observed_X)      # (n_obs, n_bits)
        y = np.asarray(observed_y, dtype=float).reshape(-1)
        inputs = {'n_vars': self.encoder.n_bits,
                  'x_vals': Xbin, 'y_vals': y}
        LR = LinReg(self.encoder.n_bits, self.order)
        LR.train(inputs)
        self._log(f"retrained order-{self.order} Bayesian(horseshoe) linreg on "
                  f"{Xbin.shape[0]} obs, n_bits={self.encoder.n_bits}; drew posterior alpha")
        return LR

    # -- block-internal hill-climb polish on the surrogate -----------------
    def _block_polish(self, x0, surr, rng, max_passes=10):
        enc = self.encoder
        x = x0.copy()
        cur = surr(x)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            order = list(range(enc.n_forks))
            rng.shuffle(order)
            for f in order:
                k = enc.n_vertices[f]
                start = enc.offsets[f]
                cur_choice = int(np.argmax(x[start:start + k]))
                for c in range(k):
                    if c == cur_choice:
                        continue
                    xn = x.copy()
                    xn[start + cur_choice] = 0.0
                    xn[start + c] = 1.0
                    on = surr(xn)
                    if on < cur:
                        x, cur = xn, on
                        improved = True
            passes += 1
        return x, cur

    # -- simulated annealing acquisition on the surrogate -------------------
    def _sa_on_surrogate(self, LR, alpha, rng):
        enc = self.encoder

        def surr(xrow):
            return float(LR.surrogate_model(xrow.reshape(1, -1), alpha))

        best_x, best_obj = None, np.inf
        for j in range(self.sa_reruns):
            x = enc.encode_onehot(enc.random_valid_assignment(rng))
            cur_obj = surr(x)
            T = 1.0
            for t in range(self.sa_iter):
                T = 0.8 * T
                f = int(rng.integers(0, enc.n_forks))
                xn = enc.block_flip(x, f, rng)
                on = surr(xn)
                if (on < cur_obj) or (rng.random() < np.exp((cur_obj - on) / T)):
                    x, cur_obj = xn, on
                if cur_obj < best_obj:
                    best_x, best_obj = x.copy(), cur_obj
        return best_x

    def suggest(self, observed_X, observed_y):  # pragma: no cover - overridden
        raise NotImplementedError


class BOCS_SA(_BaseBOCS):
    acquisition_name = 'BOCS-SA'

    def suggest(self, observed_X, observed_y):
        self.suggest_count += 1
        LR = self._fit_surrogate(observed_X, observed_y)
        alpha = LR.alpha

        def surr(xrow):
            return float(LR.surrogate_model(xrow.reshape(1, -1), alpha))

        x_bin = self._sa_on_surrogate(LR, alpha, self.rng)
        # guaranteed valid (block-internal moves), but polish anyway
        x_bin, _ = self._block_polish(x_bin, surr, self.rng)
        assignment = self.encoder.decode_onehot(x_bin)
        self._log(f"SA acquisition over {self.sa_reruns} reruns x {self.sa_iter} iters "
                  f"(block-internal flips); proposed assignment={assignment.tolist()}")
        return assignment


class BOCS_SMC(_BaseBOCS):
    acquisition_name = 'BOCS-SMC'

    def suggest(self, observed_X, observed_y):
        self.suggest_count += 1
        LR = self._fit_surrogate(observed_X, observed_y)
        alpha = LR.alpha

        def surr(xrow):
            return float(LR.surrogate_model(xrow.reshape(1, -1), alpha))

        # faithful SMC acquisition on the flattened binary space
        x_bin_raw, _ = binary_smc(surr, self.encoder.n_bits,
                                  n_models=self.smc_particles,
                                  evalBudget=self.smc_budget, rng=self.rng)
        # project onto the one-hot manifold and polish with block-internal moves
        x_bin = self.encoder.encode_onehot(self.encoder.decode_onehot(x_bin_raw))
        x_bin, _ = self._block_polish(x_bin, surr, self.rng)
        assignment = self.encoder.decode_onehot(x_bin)
        self._log(f"SMC acquisition ({self.smc_particles} particles, geometric-bridge "
                  f"tempering + logistic move kernel) + block-internal polish; "
                  f"proposed assignment={assignment.tolist()}")
        return assignment
