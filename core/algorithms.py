# -*- coding: utf-8 -*-
"""Algorithms for experiment 4.

All optimizers share one interface:
    suggest(observed_X, observed_y, rng) -> active-fork choice-index vector
`observed_X` has shape (n_obs, n_forks), values are choice indices.  Pool-based
optimizers score a fixed candidate pool and return the argmax.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    _SKLEARN = True
except Exception:
    _SKLEARN = False


def _norm_y(y):
    y = np.asarray(y, dtype=float)
    mu = float(np.mean(y))
    sd = float(np.std(y))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (y - mu) / sd, mu, sd


class BaseAlgo:
    name = "base"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        self.codec = codec
        self.pool_X = np.asarray(pool_X, dtype=int)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 991)
        self.kwargs = kwargs

    def suggest(self, observed_X, observed_y, rng=None):
        raise NotImplementedError

    def _random(self, rng=None):
        rng = rng or self.rng
        return self.pool_X[int(rng.integers(len(self.pool_X)))]


class RandomSearch(BaseAlgo):
    name = "Random"

    def suggest(self, observed_X, observed_y, rng=None):
        return self._random(rng)


class GPBase(BaseAlgo):
    name = "gp"
    acquisition = "ei"

    def _gp(self, observed_X, observed_y):
        if not _SKLEARN:
            raise RuntimeError("sklearn_missing")
        X = self.codec.ordinal(observed_X)
        y, _, _ = _norm_y(observed_y)
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            Matern(length_scale=self.kwargs.get("length_scale", 0.5), nu=2.5,
                   length_scale_bounds="fixed") + \
            WhiteKernel(noise_level=self.kwargs.get("noise", 0.05), noise_level_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False,
                                      optimizer=None, random_state=self.seed)
        gp.fit(X, y)
        return gp

    def _predict(self, gp, X, chunk=50000):
        mus, sigmas = [], []
        for i in range(0, len(X), chunk):
            mu, sd = gp.predict(X[i:i + chunk], return_std=True)
            mus.append(mu)
            sigmas.append(sd)
        return np.concatenate(mus), np.concatenate(sigmas)

    def suggest(self, observed_X, observed_y, rng=None):
        rng = rng or self.rng
        try:
            gp = self._gp(observed_X, observed_y)
        except Exception:
            return self._random(rng)
        mu, sd = self._predict(gp, self.codec.ordinal(self.pool_X))
        sd = np.maximum(sd, 1e-9)
        if self.acquisition == "ei":
            best = float(np.min(_norm_y(observed_y)[0]))
            z = (best - mu) / sd
            from scipy.stats import norm
            score = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        elif self.acquisition == "ucb":
            beta = float(self.kwargs.get("beta", 2.0))
            score = -(mu - beta * sd)
        elif self.acquisition == "ts":
            score = -(mu + sd * rng.standard_normal(len(mu)))
        elif self.acquisition == "pi":
            best = float(np.min(_norm_y(observed_y)[0]))
            from scipy.stats import norm
            score = norm.cdf((best - mu) / sd)
        else:
            raise ValueError(self.acquisition)
        return self.pool_X[int(np.argmax(score))]


class BoTorchGP(BaseAlgo):
    name = "BoTorch"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        self._torch = None

    def suggest(self, observed_X, observed_y, rng=None):
        rng = rng or self.rng
        try:
            import torch
            from botorch.models import SingleTaskGP
            from botorch.fit import fit_gpytorch_mll
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
            torch.manual_seed(self.seed + len(observed_y))
            X = torch.as_tensor(self.codec.ordinal(observed_X), dtype=torch.double)
            y = torch.as_tensor(np.asarray(observed_y, dtype=float), dtype=torch.double).view(-1, 1)
            y = (y - y.mean()) / (y.std() + 1e-9)
            model = SingleTaskGP(X, y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            if self.kwargs.get("acq", "ei") == "ucb":
                acq = UpperConfidenceBound(model, beta=float(self.kwargs.get("beta", 2.0)))
            else:
                acq = ExpectedImprovement(model, best_f=float(y.min()))
            pool = torch.as_tensor(self.codec.ordinal(self.pool_X), dtype=torch.double)
            best_score, best_idx = -float("inf"), 0
            with torch.no_grad():
                for i in range(0, len(pool), 20000):
                    s = acq(pool[i:i + 20000].unsqueeze(1)).reshape(-1)
                    v, j = torch.max(s, dim=0)
                    if float(v) > best_score:
                        best_score, best_idx = float(v), i + int(j)
            return self.pool_X[best_idx]
        except Exception:
            return self._random(rng)


class GPEI(GPBase):
    name = "GP-EI"
    acquisition = "ei"


class GPUCB(GPBase):
    name = "GP-UCB"
    acquisition = "ucb"


class GPTS(GPBase):
    name = "GP-TS"
    acquisition = "ts"


class GPPI(GPBase):
    name = "GP-PI"
    acquisition = "pi"


class COMBOAlgo(BaseAlgo):
    name = "COMBO"

    def suggest(self, observed_X, observed_y, rng=None):
        try:
            from engine.bo.combo.combo_optimizer import COMBOOptimizer
            opt = COMBOOptimizer(list(map(int, self.codec.n_vertices)),
                                 warmups=None, seed=self.seed,
                                 n_post=int(self.kwargs.get("n_post", 3)),
                                 n_burn=int(self.kwargs.get("n_burn", 10)))
            vec = opt.suggest(np.asarray(observed_X), np.asarray(observed_y))
            return np.asarray(vec, dtype=int).reshape(-1)
        except Exception:
            return self._random(rng)


class ExternalAlgo(BaseAlgo):
    """Placeholder used when an external package adapter is available."""

    name = "external"

    def __init__(self, codec, pool_X, seed=0, adapter=None, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        self.adapter = adapter

    def suggest(self, observed_X, observed_y, rng=None):
        return self.adapter.suggest(np.asarray(observed_X), np.asarray(observed_y))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
ALGORITHMS = {
    "Random": RandomSearch,
    "RandomUniform": RandomSearch,
    "GP-EI": GPEI,
    "GP-UCB": GPUCB,
    "GP-TS": GPTS,
    "GP-PI": GPPI,
    "COMBO": COMBOAlgo,
    "BoTorch": BoTorchGP,
}

# The 18 GP algorithms in the plan.  `available` is filled by adapters; entries
# not in ALGORITHMS are reported as not-run with a reason instead of faked.
PLANNED = [
    "COMBO", "CASMOPOLITAN", "CoCaBO", "MerCBO", "LADDER", "BODi", "Bounce",
    "MOCA-HESP", "MixedTypeBO", "BO-MINLP", "HEBO", "bo_pr", "Gryffin",
    "Summit", "BayBE", "bayes_optim", "BoTorch", "Ax",
]


_EXTERNAL = None


def external_registry():
    global _EXTERNAL
    if _EXTERNAL is not None:
        return _EXTERNAL
    import importlib
    out = {}
    for mod in ("algos_ext.registry_a", "algos_ext.registry_b", "algos_ext.registry_c"):
        try:
            m = importlib.import_module(mod)
            out.update(getattr(m, "ADAPTERS", {}) or {})
        except Exception:
            pass
    _EXTERNAL = out
    return out


def make_optimizer(name: str, codec, pool_X, seed: int, **kwargs):
    ext = external_registry()
    if name in ext:
        return ext[name](codec, pool_X, seed=seed, **kwargs)
    cls = ALGORITHMS.get(name)
    if cls is None:
        raise NotImplementedError(f"algorithm_not_available:{name}")
    return cls(codec, pool_X, seed=seed, **kwargs)
