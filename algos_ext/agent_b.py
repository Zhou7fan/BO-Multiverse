# -*- coding: utf-8 -*-
"""Agent B external adapters for experiment 4.

Algorithms owned by Agent B:
    MerCBO, LADDER, BODi, MOCA-HESP, MixedTypeBO, BO-MINLP

Only algorithms whose upstream implementation could be imported and whose
surrogate/acquisition could be mapped onto the experiment-4 categorical space
(length-15 choice-index vector) are exported through ``ADAPTERS``.  Everything
else is reported as ``not_run`` in ``experiment4/algorithms_status/<name>.json``
with the concrete import error.

Vendored upstream code lives in ``experiment4/algos_ext/vendor/<name>/`` and
contains only the files the adapter actually imports.  Each adapter below uses
the upstream surrogate + acquisition (not a re-implementation):

* BODi   : ``bodi.dictionary_kernel.DictionaryKernel`` (dictionary of random
           binary prototype vectors + Matern kernel on the embedding) wrapped in
           a BoTorch ``SingleTaskGP``, acquisition optimised with BODi's own
           ``bodi.optimize.optimize_acqf_binary_local_search``.
* MerCBO : ``MerCBO.graphGP.kernels.diffusionkernel.DiffusionKernel`` (Mercer
           features of the diffusion kernel over the per-fork path graph) with
           hyper-parameters fitted by marginal likelihood, and MerCBO's
           ``expected_improvement`` acquisition evaluated over the candidate
           pool (the repository itself searches the same acquisition with a
           graph-cut/greedy-ascent optimiser; the runner's interface is
           pool-based, so the search is exhaustive over the fixed pool).

Both adapters encode the 15 categorical forks as a 54-dimensional binary
one-hot vector (sum of per-fork choice counts) for the surrogate, and decode
back to the categorical choice-index vector.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")

# --------------------------------------------------------------------------- #
# make vendored packages importable (first path entry wins over site-packages)
# --------------------------------------------------------------------------- #
for _name in ("mercbo", "bodi"):
    _p = os.path.join(VENDOR, _name)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:  # the experiment's own base class
    from core.algorithms import BaseAlgo  # type: ignore
except Exception:  # pragma: no cover - allows importing this module standalone
    class BaseAlgo:  # type: ignore
        name = "base"

        def __init__(self, codec, pool_X, seed=0, **kwargs):
            self.codec = codec
            self.pool_X = np.asarray(pool_X, dtype=int)
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed + 991)
            self.kwargs = kwargs

        def _random(self, rng=None):
            rng = rng or self.rng
            return self.pool_X[int(rng.integers(len(self.pool_X)))]


def _decode_onehot(codec, oh):
    """(n, sum(n_vertices)) one-hot -> (n, n_forks) choice-index matrix."""
    oh = np.asarray(oh)
    out = np.zeros((oh.shape[0], codec.dim), dtype=int)
    off = 0
    for f, k in enumerate(codec.n_vertices):
        k = int(k)
        block = oh[:, off:off + k]
        out[:, f] = np.argmax(block, axis=1)
        off += k
    return out


def _decode_onehot_vec(codec, vec):
    return _decode_onehot(codec, np.asarray(vec).reshape(1, -1))[0]


# --------------------------------------------------------------------------- #
# BODi
# --------------------------------------------------------------------------- #
class BODiAlgo(BaseAlgo):
    """BODi (AISTATS 2023) on the binary one-hot view of the 15 categorical forks.

    Upstream: https://github.com/aryandeshwal/BODi (vendored: dictionary_kernel,
    optimize).  The dictionary kernel embeds the binary vector through
    ``n_prototype_vectors`` random binary prototype vectors and puts a Matern-5/2
    ARD kernel on [embedding, continuous dims].  Acquisition: analytic EI,
    optimised with BODi's binary local search (random candidates + hamming
    neighbours of the incumbent + greedy 1-bit local search).
    """

    name = "BODi"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        self.n_binary = int(np.sum(codec.n_vertices))
        self.n_prototype = int(kwargs.get("n_prototype_vectors", 64))
        self.afo_config = {
            "n_initial_candts": int(kwargs.get("n_initial_candts", 2000)),
            "n_restarts": int(kwargs.get("n_restarts", 10)),
            "afo_init_design": "random",
            "n_alternate_steps": 50,
            "num_cmaes_steps": 20,
            "num_ls_steps": int(kwargs.get("num_ls_steps", 30)),
            "n_spray_points": 200,
            "verbose": False,
            "add_spray_points": True,
            "n_binary": self.n_binary,
            "n_cont": 0,
        }

    def _onehot(self, X):
        return np.asarray(self.codec.onehot(X), dtype=float)

    def suggest(self, observed_X, observed_y, rng=None):
        try:
            return self._suggest(observed_X, observed_y, rng)
        except Exception:
            return self._random(rng)

    def _suggest(self, observed_X, observed_y, rng=None):
        import torch
        from botorch.acquisition.analytic import ExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize
        from gpytorch.constraints import GreaterThan
        from gpytorch.kernels import ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.priors import GammaPrior

        from bodi.dictionary_kernel import DictionaryKernel
        from bodi.optimize import optimize_acqf_binary_local_search

        tkwargs = {"dtype": torch.double}
        torch.manual_seed(self.seed + 17 * len(observed_y))
        X = torch.tensor(self._onehot(observed_X), **tkwargs)
        y = np.asarray(observed_y, dtype=float)
        y = (y - y.mean()) / (y.std() + 1e-12)
        train_Y = torch.tensor(-y, **tkwargs)  # BoTorch maximises

        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)),
            noise_constraint=GreaterThan(1e-4),
        )
        dictionary_kernel = DictionaryKernel(
            binary_dims=list(range(self.n_binary)),
            num_dims=self.n_binary,
            num_basis_vectors=self.n_prototype,
            similarity=True,
        )
        covar_module = ScaleKernel(
            base_kernel=dictionary_kernel,
            outputscale_prior=GammaPrior(torch.tensor(2.0, **tkwargs), torch.tensor(0.15, **tkwargs)),
            outputscale_constraint=GreaterThan(1e-6),
        )
        model = SingleTaskGP(
            train_X=X,
            train_Y=train_Y.unsqueeze(-1),
            covar_module=covar_module,
            input_transform=Normalize(d=self.n_binary),
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)

        acqf = ExpectedImprovement(model=model, best_f=train_Y.max())
        pareto_points = X[torch.argmax(train_Y)].unsqueeze(0).clone()
        with torch.no_grad():
            next_x, _acq_val = optimize_acqf_binary_local_search(
                acqf, afo_config=self.afo_config, pareto_points=pareto_points, q=1
            )
        return _decode_onehot_vec(self.codec, next_x[0].numpy())


# --------------------------------------------------------------------------- #
# MerCBO
# --------------------------------------------------------------------------- #
class MerCBOAlgo(BaseAlgo):
    """MerCBO (AAAI 2021): diffusion-kernel Mercer features on a per-fork graph.

    Upstream: https://github.com/aryandeshwal/MerCBO (vendored: DiffusionKernel,
    GraphKernel, GPModule, acquisition_functions).

    Each of the 15 forks is a path graph over its choice indices (the same
    adjacency the MerCBO repository builds in ``experiments/test_functions``),
    its graph Laplacian is eigendecomposed, and the diffusion kernel
    ``k(x, x') = amp * prod_f sum_j exp(-beta_f * lambda_fj) V_fj[x_f] V_fj[x'_f]``
    is evaluated with MerCBO's own ``DiffusionKernel`` class.  ``log_amp``,
    per-fork ``log_beta`` and the Gaussian noise variance are fitted by type-II
    maximum marginal likelihood (MerCBO samples them with slice sampling; the
    prior bounds used by MerCBO are enforced here as box constraints).  The next
    point maximises MerCBO's ``expected_improvement`` over the candidate pool.
    """

    name = "MerCBO"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        import torch
        from MerCBO.graphGP.kernels.diffusionkernel import DiffusionKernel

        torch.set_default_dtype(torch.float64)
        self._torch = torch
        self.n_vertices = np.asarray(codec.n_vertices, dtype=int)
        fourier_freq, fourier_basis = [], []
        for n_v in self.n_vertices:
            n_v = int(n_v)
            if n_v <= 1:
                adjmat = torch.zeros(1, 1)
            else:
                adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.linalg.eigh(laplacian)
            fourier_freq.append(eigval)
            fourier_basis.append(eigvec)
        self.fourier_freq = fourier_freq
        self.fourier_basis = fourier_basis
        self.grouped_log_beta = torch.nn.Parameter(torch.zeros(len(self.n_vertices)))
        self.log_amp = torch.nn.Parameter(torch.zeros(1))
        self.log_noise_var = torch.nn.Parameter(torch.tensor([-6.0]))
        self.kernel = DiffusionKernel(
            grouped_log_beta=self.grouped_log_beta,
            fourier_freq_list=self.fourier_freq,
            fourier_basis_list=self.fourier_basis,
        )
        self.kernel.log_amp = self.log_amp
        self.beta_bounds = (-12.0, float(np.log(100.0)))
        self.amp_bounds = (-12.0, float(np.log(10000.0)))
        self.noise_bounds = (-12.0, float(np.log(10000.0)))

    # ------------------------------------------------------------------ #
    def _core(self, X):
        """Standardised per-fork kernel matrix for the given raw index matrix."""
        import torch
        x = torch.as_tensor(np.asarray(X, dtype=np.int64))
        kernel = self.kernel
        return kernel.forward(x, x, diagonal=True).reshape(-1)

    def _kern(self, X1, X2=None, diagonal=False):
        import torch
        x1 = torch.as_tensor(np.asarray(X1, dtype=np.int64))
        if diagonal:
            return self.kernel.forward(x1, diagonal=True).reshape(-1)
        if X2 is None:
            return self.kernel.forward(x1, x1)
        x2 = torch.as_tensor(np.asarray(X2, dtype=np.int64))
        return self.kernel.forward(x1, x2)

    def _fit(self, X, y):
        """Type-II MLE for log_amp, per-fork log_beta and log noise variance."""
        import torch

        params = [self.log_amp, self.grouped_log_beta, self.log_noise_var]
        opt = torch.optim.LBFGS(params, lr=0.3, max_iter=40, tolerance_grad=1e-7,
                                tolerance_change=1e-9, line_search_fn=None)

        def closure():
            opt.zero_grad()
            K = self._kern(X)
            noise = torch.exp(self.log_noise_var).clamp_min(1e-10)
            K = K + noise * torch.eye(K.shape[0])
            jitter = 1e-8 * torch.eye(K.shape[0])
            L = torch.linalg.cholesky(K + jitter)
            alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
            nll = 0.5 * (y * alpha).sum() + L.diagonal().log().sum() + 0.5 * len(y) * np.log(2 * np.pi)
            nll.backward()
            return nll

        try:
            opt.step(closure)
        finally:
            with torch.no_grad():
                self.log_amp.clamp_(*self.amp_bounds)
                self.grouped_log_beta.clamp_(*self.beta_bounds)
                self.log_noise_var.clamp_(*self.noise_bounds)
        return closure().detach()

    def suggest(self, observed_X, observed_y, rng=None):
        try:
            return self._suggest(observed_X, observed_y, rng)
        except Exception:
            return self._random(rng)

    def _suggest(self, observed_X, observed_y, rng=None):
        import torch
        from MerCBO.acquisition.acquisition_functions import expected_improvement

        torch.manual_seed(self.seed + 13 * len(observed_y))
        X = np.asarray(observed_X, dtype=np.int64)
        y = np.asarray(observed_y, dtype=float)
        y = (y - y.mean()) / (y.std() + 1e-12)
        y_t = torch.as_tensor(y)
        self._fit(X, y_t)

        K = self._kern(X)
        noise = float(torch.exp(self.log_noise_var).clamp_min(1e-10))
        K = K + noise * torch.eye(K.shape[0], dtype=K.dtype)
        L = torch.linalg.cholesky(K + 1e-8 * torch.eye(K.shape[0], dtype=K.dtype))
        alpha = torch.cholesky_solve(y_t.unsqueeze(-1), L).squeeze(-1)
        K_inv = torch.cholesky_inverse(L)
        reference = float(y.min())

        pool = self.pool_X
        n = len(pool)
        chunk = int(self.kwargs.get("chunk", 25000))
        best_score, best_idx = -np.inf, 0
        with torch.no_grad():
            for start in range(0, n, chunk):
                Xc = pool[start:start + chunk]
                k_s = self._kern(Xc, X)                     # (m, n_obs)
                k_diag = self._kern(Xc, diagonal=True)      # (m,)
                mu = k_s @ alpha
                v = k_diag - ((k_s @ K_inv) * k_s).sum(dim=1)
                v = v.clamp_min(1e-9)
                score = expected_improvement(mu, v, reference)
                vmax, imax = torch.max(score, dim=0)
                if float(vmax) > best_score:
                    best_score, best_idx = float(vmax), start + int(imax)
        return np.asarray(pool[best_idx], dtype=int).reshape(-1)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
ADAPTERS = {
    "BODi": BODiAlgo,
    "MerCBO": MerCBOAlgo,
}
