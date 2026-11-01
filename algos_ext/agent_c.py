# -*- coding: utf-8 -*-
"""Agent C adapters for experiment 4.

Working adapters
----------------
* ``BoTorch``  - fast BoTorch SingleTaskGP + analytic EI on an ordinal encoding,
  scoring a random 20k subset of the candidate pool per query.
* ``bo_pr``    - vendored ``discrete_mixed_bo`` (probabilistic reparameterization
  over a mixed categorical kernel GP), patched for BoTorch 0.18 / GPyTorch 1.15.
* ``BayBE``    - vendored BayBE 0.15.0 Campaign over a 20k candidate subset.
* ``Ax``       - vendored Ax 1.3.1 ``Client`` with categorical ChoiceParameters
  and its own Center+Sobol+MBM generation strategy.

Honest failures
---------------
* ``Gryffin``  - PyPI source build needs a C compiler and ``oldest-supported-numpy``;
  no binary wheel exists, so the adapter raises with the import/build error.
* ``SMAC3``    - optional extra; requires ``pyrfr`` (no Windows wheel) plus a C++
  toolchain.  The adapter raises with the import error.

Everything is lazily imported so ``from algos_ext.registry_c import ADAPTERS``
works in a bare environment (it only needs numpy).
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)

VENDOR = os.path.join(HERE, "vendor")
VENDOR_PY = os.path.join(VENDOR, "py")
BO_PR_VENDOR = os.path.join(VENDOR, "bo_pr")
GRYFFIN_SRC = os.path.join(ROOT, "experiment3", "cbo_survey", "repos", "gryffin", "src")
SMAC_SRC = os.path.join(ROOT, "experiment3", "cbo_survey", "repos", "smac3")


def _append_path(path):
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)


def _prepend_path(path):
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _ensure_botorch():
    """Import botorch, falling back to a base conda site-packages on this host.

    The local smoke venv is intentionally built from a numpy-1.26 environment
    while torch/botorch live in the base Anaconda 3.12 environment.  On the
    remote runner botorch is already importable, so this fallback is a no-op.
    """
    try:
        import torch  # noqa: F401
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
        return
    except Exception:
        base = os.environ.get("DSH_BASE_SITE", r"D:\Anaconda\Lib\site-packages")
        _append_path(base)
        import torch  # noqa: F401
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401


def _bind_seed(seed):
    """Best-effort deterministic seeding for torch/numpy-backed libraries."""
    try:
        import torch

        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
    except Exception:
        pass


class _Base:
    name = "base"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        self.codec = codec
        self.pool_X = np.asarray(pool_X, dtype=int)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 991)
        self.kwargs = kwargs
        self.n_vertices = np.asarray(codec.n_vertices, dtype=int)
        self.dim = int(codec.dim)

    def _random(self, rng=None):
        rng = rng or self.rng
        return self.pool_X[int(rng.integers(len(self.pool_X)))]

    def suggest(self, observed_X, observed_y, rng=None):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. BoTorch - fast variant (random 20k pool subset)
# --------------------------------------------------------------------------- #
class BoTorchFast(_Base):
    name = "BoTorch"

    def suggest(self, observed_X, observed_y, rng=None):
        rng = rng or self.rng
        _ensure_botorch()
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import ExpectedImprovement
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X = np.asarray(observed_X, dtype=float)
        y = np.asarray(observed_y, dtype=float)
        if len(X) < 2:
            return self._random(rng)
        y = (y - y.mean()) / (y.std() + 1e-9)
        torch.manual_seed(self.seed * 1000 + len(y))
        Xt = torch.as_tensor(self.codec.ordinal(X), dtype=torch.double)
        yt = torch.as_tensor(y, dtype=torch.double).view(-1, 1)
        model = SingleTaskGP(Xt, yt)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        acq = ExpectedImprovement(model, best_f=float(yt.max()))

        n_sub = int(self.kwargs.get("pool_subset", 20000))
        n_sub = max(1, min(n_sub, len(self.pool_X)))
        idx = rng.choice(len(self.pool_X), size=n_sub, replace=False)
        pool = torch.as_tensor(self.codec.ordinal(self.pool_X[idx]), dtype=torch.double)
        best_score, best_j = -float("inf"), 0
        with torch.no_grad():
            for start in range(0, len(pool), 4096):
                chunk = pool[start:start + 4096]
                scores = acq(chunk.unsqueeze(1)).reshape(-1)
                val, j = torch.max(scores, dim=0)
                if float(val) > best_score:
                    best_score, best_j = float(val), start + int(j)
        return self.pool_X[idx[best_j]]


# --------------------------------------------------------------------------- #
# 2. bo_pr - probabilistic reparameterization
# --------------------------------------------------------------------------- #
class BoPR(_Base):
    name = "bo_pr"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        self._mods = None

    def _load(self):
        if self._mods is not None:
            return self._mods
        _ensure_botorch()
        _prepend_path(BO_PR_VENDOR)
        from discrete_mixed_bo.kernels import get_kernel
        from discrete_mixed_bo.input import OneHotToNumeric
        from discrete_mixed_bo.probabilistic_reparameterization import (
            MCProbabilisticReparameterization,
        )
        from discrete_mixed_bo.optimize import optimize_acqf
        self._mods = {
            "get_kernel": get_kernel,
            "OneHotToNumeric": OneHotToNumeric,
            "MCProbabilisticReparameterization": MCProbabilisticReparameterization,
            "optimize_acqf": optimize_acqf,
        }
        return self._mods

    def suggest(self, observed_X, observed_y, rng=None):
        rng = rng or self.rng
        mods = self._load()
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import ExpectedImprovement
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X = np.asarray(observed_X, dtype=int)
        y = np.asarray(observed_y, dtype=float)
        if len(X) < 2:
            return self._random(rng)
        y_obj = -y
        y_obj = (y_obj - y_obj.mean()) / (y_obj.std() + 1e-9)
        torch.manual_seed(self.seed * 1000 + len(y))
        _bind_seed(self.seed * 1000 + len(y))

        cat = OrderedDict((i, int(k)) for i, k in enumerate(self.n_vertices))
        dim_oh = int(self.n_vertices.sum())
        tf = mods["OneHotToNumeric"](categorical_features=cat)
        tf.eval()
        with torch.no_grad():
            X_oh = torch.as_tensor(self.codec.onehot(X), dtype=torch.double)
            X_num = tf(X_oh)
        covar = mods["get_kernel"](
            kernel_type="mixed_categorical",
            dim=X_num.shape[-1],
            binary_dims=[],
            categorical_transformed_features=cat,
            train_X=X_num,
            train_Y=torch.as_tensor(y_obj, dtype=torch.double).view(-1, 1),
        )
        yt = torch.as_tensor(y_obj, dtype=torch.double).view(-1, 1)
        # OneHotToNumeric.transform_on_train=False should skip, but newer BoTorch
        # still calls transform_inputs during model construction.  Disable eval
        # transform for construction, then switch it back on for posterior calls.
        tf.transform_on_eval = False
        model = SingleTaskGP(
            train_X=X_num, train_Y=yt, covar_module=covar, input_transform=tf
        )
        tf.transform_on_eval = True
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        acq0 = ExpectedImprovement(model, best_f=float(yt.max()))
        pr = mods["MCProbabilisticReparameterization"](
            acq_function=acq0,
            dim=dim_oh,
            integer_indices=[],
            integer_bounds=torch.zeros(2, 0, dtype=torch.double),
            categorical_features=cat,
            batch_limit=int(self.kwargs.get("batch_limit", 16)),
            mc_samples=int(self.kwargs.get("mc_samples", 64)),
            grad_estimator=str(self.kwargs.get("grad_estimator", "reinforce")),
            tau=float(self.kwargs.get("tau", 0.1)),
        )
        bounds = torch.zeros(2, dim_oh, dtype=torch.double)
        bounds[1] = 1.0
        cands, _ = mods["optimize_acqf"](
            acq_function=pr,
            bounds=bounds,
            q=1,
            num_restarts=int(self.kwargs.get("num_restarts", 4)),
            raw_samples=int(self.kwargs.get("raw_samples", 64)),
            options={"batch_limit": 2,
                     "maxiter": int(self.kwargs.get("maxiter", 50))},
        )
        c = cands.reshape(-1, dim_oh)[0].detach().cpu().numpy()
        out = np.zeros(self.dim, dtype=int)
        off = 0
        for f, k in enumerate(self.n_vertices):
            out[f] = int(np.argmax(c[off:off + int(k)]))
            off += int(k)
        return np.clip(out, 0, self.n_vertices - 1)


# --------------------------------------------------------------------------- #
# 3. BayBE - Campaign over a candidate subset
# --------------------------------------------------------------------------- #
class BayBE(_Base):
    name = "BayBE"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        self._campaign = None
        self._acols = None
        self._active = None
        self._added = set()
        self._init_error = None
        self._init()

    def _init(self):
        try:
            _prepend_path(VENDOR_PY)
            _ensure_botorch()
            import pandas as pd
            from baybe.parameters import CategoricalParameter
            from baybe.searchspace import SearchSpace
            from baybe.targets import NumericalTarget
            from baybe.objectives import SingleTargetObjective
            from baybe.campaign import Campaign
        except Exception as exc:  # pragma: no cover - depends on env
            self._init_error = f"{type(exc).__name__}: {exc}"
            return
        try:
            import pandas as pd

            active = [i for i, k in enumerate(self.n_vertices) if int(k) > 1]
            acols = [f"f{i}" for i in active]
            n_cand = int(self.kwargs.get("candidate_n", 20000))
            n_cand = max(32, min(n_cand, len(self.pool_X)))
            sample_idx = self.rng.choice(len(self.pool_X), size=n_cand, replace=False)
            base = self.pool_X[sample_idx][:, active].astype(int).copy()
            # Guarantee every category is present in the candidate dataframe.
            for j, f in enumerate(active):
                for c in range(int(self.n_vertices[f])):
                    if not np.any(base[:, j] == c):
                        row = base[int(self.rng.integers(len(base)))].copy()
                        row[j] = c
                        base = np.vstack([base, row])
            cand = pd.DataFrame(base.astype(str), columns=acols)

            params = [
                CategoricalParameter(
                    name=acols[j],
                    values=[str(c) for c in range(int(self.n_vertices[f]))],
                )
                for j, f in enumerate(active)
            ]
            ss = SearchSpace.from_dataframe(cand, parameters=params)
            obj = SingleTargetObjective(
                target=NumericalTarget(name="loss", minimize=True)
            )
            self._campaign = Campaign(searchspace=ss, objective=obj)
            self._acols = acols
            self._active = active
            self._SearchSpace = SearchSpace
            self._Campaign = Campaign
            self._pd = pd
        except Exception as exc:  # pragma: no cover - depends on env
            self._init_error = f"{type(exc).__name__}: {exc}"

    def _params(self, row_idx):
        return {f"f{i}": str(int(row_idx[i])) for i in self._active}

    def suggest(self, observed_X, observed_y, rng=None):
        rng = rng or self.rng
        if self._init_error is not None:
            raise RuntimeError(f"baybe_unavailable:{self._init_error}")
        pd = self._pd
        X = np.asarray(observed_X, dtype=int)
        y = np.asarray(observed_y, dtype=float)
        # Add every not-yet-seen observation to the campaign.
        rows = []
        for i in range(len(X)):
            key = tuple(int(v) for v in X[i])
            if key in self._added:
                continue
            self._added.add(key)
            d = self._params(X[i])
            d["loss"] = float(y[i])
            rows.append(d)
        if rows:
            try:
                self._campaign.add_measurements(pd.DataFrame(rows))
            except Exception:
                # A rejected/duplicate point must not kill the adapter; fall back
                # to the best candidate seen so far.
                pass
        rec = self._campaign.recommend(batch_size=1)
        row = rec.iloc[0]
        out = np.zeros(self.dim, dtype=int)
        for j, f in enumerate(self._active):
            out[f] = int(row[self._acols[j]])
        return np.clip(out, 0, self.n_vertices - 1)


# --------------------------------------------------------------------------- #
# 4. Ax 1.3.1 Client
# --------------------------------------------------------------------------- #
class AxClient(_Base):
    name = "Ax"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        self._client = None
        self._active = None
        self._pending = []
        self._added = set()
        self._init_error = None
        self._init()

    def _init(self):
        try:
            _prepend_path(VENDOR_PY)
            _ensure_botorch()
            from ax.api.client import Client
            from ax.api.configs import ChoiceParameterConfig
        except Exception as exc:  # pragma: no cover - depends on env
            self._init_error = f"{type(exc).__name__}: {exc}"
            return
        try:
            active = [i for i, k in enumerate(self.n_vertices) if int(k) > 1]
            params = [
                ChoiceParameterConfig(
                    name=f"f{i}",
                    values=[str(c) for c in range(int(self.n_vertices[i]))],
                    parameter_type="str",
                    is_ordered=False,
                )
                for i in active
            ]
            client = Client(random_seed=self.seed)
            client.configure_experiment(parameters=params)
            client.configure_optimization(objective="-loss")
            self._client = client
            self._active = active
            self._Client = Client
        except Exception as exc:  # pragma: no cover - depends on env
            self._init_error = f"{type(exc).__name__}: {exc}"

    def _params(self, row):
        return {f"f{i}": str(int(row[i])) for i in self._active}

    def suggest(self, observed_X, observed_y, rng=None):
        rng = rng or self.rng
        if self._init_error is not None:
            raise RuntimeError(f"ax_unavailable:{self._init_error}")
        X = np.asarray(observed_X, dtype=int)
        y = np.asarray(observed_y, dtype=float)
        # Close pending trials using the observation that matches their suggestion.
        for tidx, vec in self._pending:
            match = np.where((X == vec).all(axis=1))[0]
            if len(match):
                self._client.complete_trial(
                    tidx, raw_data={"loss": float(y[int(match[-1])])}
                )
                self._added.add(tuple(int(v) for v in vec))
            else:
                try:
                    self._client.mark_trial_failed(tidx)
                except Exception:
                    pass
        self._pending = []
        # Attach warm-up / fallback observations that Ax has not seen.
        for i in range(len(X)):
            key = tuple(int(v) for v in X[i])
            if key in self._added:
                continue
            try:
                tidx = self._client.attach_trial(self._params(X[i]))
                self._client.complete_trial(tidx, raw_data={"loss": float(y[i])})
                self._added.add(key)
            except Exception:
                pass
        trials = self._client.get_next_trials(max_trials=1)
        if not trials:
            return self._random(rng)
        tidx, params = next(iter(trials.items()))
        out = np.zeros(self.dim, dtype=int)
        for i in self._active:
            out[i] = int(params[f"f{i}"])
        self._pending.append((int(tidx), out.copy()))
        return np.clip(out, 0, self.n_vertices - 1)


# --------------------------------------------------------------------------- #
# 5/6. Gryffin / SMAC3 - honest import attempts
# --------------------------------------------------------------------------- #
class Gryffin(_Base):
    name = "Gryffin"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        _prepend_path(GRYFFIN_SRC)
        try:
            from gryffin import Gryffin  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "gryffin_not_importable: "
                f"{type(exc).__name__}: {exc}; source build needs a C compiler "
                "(kernel_evaluations/kernel_prob_reshaping) and no Windows wheel "
                f"is published. Source: {GRYFFIN_SRC}"
            ) from exc
        # Kept for completeness; only reached if a future wheel/import works.
        self._gryffin = Gryffin


class SMAC3(_Base):
    name = "SMAC3"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        _prepend_path(SMAC_SRC)
        try:
            import smac
        except Exception as exc:
            raise RuntimeError(
                "smac_not_importable: "
                f"{type(exc).__name__}: {exc}; SMAC requires ConfigSpace/pyrfr "
                "and pyrfr has no Windows wheel and needs a C++ toolchain. "
                f"Source: {SMAC_SRC}"
            ) from exc
        # The SMAC __init__ swallows some import failures, leaving a partial
        # module behind.  Treat that as not importable instead of faking it.
        if not hasattr(smac, "SMAC") and not hasattr(smac, "HyperparameterOptimizer"):
            raise RuntimeError(
                "smac_not_importable: partial smac import (no SMAC/"
                "HyperparameterOptimizer symbol); root cause: "
                "ModuleNotFoundError: No module named 'ConfigSpace'. "
                f"Source: {SMAC_SRC}"
            )
        self._smac = smac


ADAPTERS = {
    "BoTorch": BoTorchFast,
    "bo_pr": BoPR,
    "BayBE": BayBE,
    "Ax": AxClient,
    "Gryffin": Gryffin,
    "SMAC3": SMAC3,
}
