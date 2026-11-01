# -*- coding: utf-8 -*-
"""Agent A adapters for experiment 4.

Six GP-family CBO algorithms were assigned to Agent A:
    CASMOPOLITAN, CoCaBO, Bounce, HEBO, Summit, bayes_optim

Only algorithms that could be imported and mapped onto the 15-fork
categorical space are exported here (ADAPTERS): CASMOPOLITAN, Bounce, HEBO,
Summit, bayes_optim.  See ``experiment4/algorithms_status/*.json`` for the
honest not-run record of CoCaBO (its BaseBO builds ``np.vstack`` over the
continuous-variable bounds and therefore requires at least one continuous
dimension; the 15-fork space is purely categorical).

Every adapter subclasses/duck-types ``core.algorithms.BaseAlgo``:

    suggest(self, observed_X, observed_y, rng=None) -> np.ndarray of shape (15,)

The adapters are *stateful*: the runner hands them the full observation
history on every call, so each adapter tracks how many rows it has already
consumed (``n_seen``) and keeps the algorithm's own state (trust region,
incumbent history, etc.) across calls.

Vendored upstream code lives in ``algos_ext/vendor/<name>/`` and is added to
``sys.path`` lazily.  Minimal compatibility patches applied to the vendored
copies are documented in the ``_vendor_notes`` entries below.
"""
from __future__ import annotations

import os
import sys

import numpy as np

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")

_vendor_notes = {
    "casmopolitan": [
        "copied bo/ (kernels.py, localbo_cat.py, localbo_utils.py, optimizer.py)",
        "localbo_utils.py: `from collections import Callable` -> `from collections.abc import Callable` (py3.12)",
        "localbo_utils.py: removed `from test_funcs import *` (test-only import, package not vendored)",
        "added bo/__init__.py (the repo relies on namespace-package discovery)",
    ],
    "bounce": [
        "copied bounce/ (bounce.py, candidates.py, gaussian_process.py, neighbors.py, projection.py, settings.py, trust_region.py, kernel/, util/)",
        "no source changes; requires the pure-python `gin-config` package (pip installed)",
    ],
    "hebo": [
        "copied HEBO/hebo/ (upstream repo version, not the incompatible pip HEBO 0.3.6)",
        "no source changes; requires pymoo (pip installed, used by HEBO's EvolutionOpt)",
    ],
    "bayes_optim": [
        "copied bayes_optim/ package",
        "vendored dependency `pyDOE` 0.3.6 (pyDOE.lhs is the only function used by bayes_optim.search_space)",
    ],
    "summit": [
        "copied the minimal module set: summit/domain.py, summit/utils/dataset.py,"
        " summit/strategies/{base,random,sobo}.py",
        "replaced summit/__init__.py and summit/strategies/__init__.py with empty files:"
        " the upstream __init__ eagerly imports TSEMO/benchmarks (fastprogress, skorch,"
        " sklearn private APIs that no longer exist); SOBO itself does not need them",
    ],
}


def _add_vendor(name):
    path = os.path.join(VENDOR, name)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


class _BaseAlgo:
    name = "base"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        self.codec = codec
        self.pool_X = np.asarray(pool_X, dtype=int)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 991)
        self.kwargs = kwargs
        self.n_vertices = np.asarray(codec.n_vertices, dtype=int)
        self.dim = int(codec.dim)
        self.n_seen = 0

    # ------------------------------------------------------------------ #
    def _seed_everything(self):
        import random

        random.seed(self.seed)
        np.random.seed(self.seed % (2 ** 32 - 1))
        try:
            import torch

            torch.manual_seed(self.seed)
        except Exception:
            pass

    def suggest(self, observed_X, observed_y, rng=None):
        raise NotImplementedError

    def _random(self, rng=None):
        rng = rng or self.rng
        return self.pool_X[int(rng.integers(len(self.pool_X)))]


# --------------------------------------------------------------------------- #
# 1. CASMOPOLITAN
# --------------------------------------------------------------------------- #
class CASMOPOLITANAlgo(_BaseAlgo):
    """CASMOPOLITAN (Wan et al., 2021) -- categorical local BO.

    Uses the repo's own ``bo.localbo_cat.CASMOPOLITANCat`` and
    ``bo.localbo_utils.train_gp`` (``TransformedCategorical`` kernel trained by
    the GPyTorch MLL) together with CASMOPOLITAN's discrete trust region and
    Thompson-sampling candidate generation.

    Trust-region bookkeeping calls the class' own ``_adjust_length``
    (failtol=40, succtol=3, multiplier=1.5), exactly as the repo does.
    ``length_init_discrete`` is scaled to the problem dimension
    (min(20, dim // 3) = 5 for the 15-fork space), because the paper's default
    of 20 was tuned for 20-40 dimensional problems and would degenerate to a
    fully random proposal on a 15 dimensional space.
    """

    name = "CASMOPOLITAN"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        _add_vendor("casmopolitan")
        self._seed_everything()
        from bo.localbo_cat import CASMOPOLITANCat

        self.config = self.n_vertices.astype(int)
        self.n_training_steps = int(kwargs.get("n_training_steps", 100))
        self.acq = str(kwargs.get("acq", "thompson"))
        self.length_init_discrete = int(
            kwargs.get("length_init_discrete", max(2, min(20, self.dim // 3)))
        )
        self._cat = CASMOPOLITANCat(
            dim=self.dim,
            n_init=2 * self.dim + 1,
            max_evals=10 ** 7,
            config=self.config,
            batch_size=1,
            verbose=False,
            acq=self.acq,
            kernel_type="transformed_overlap",
            n_training_steps=self.n_training_steps,
            length_init_discrete=self.length_init_discrete,
            length_max_discrete=self.dim,
            length_min_discrete=1,
            failtol=int(kwargs.get("failtol", 40)),
            succtol=int(kwargs.get("succtol", 3)),
            n_cand=int(kwargs.get("n_cand", min(100 * self.dim, 5000))),
        )
        self._prev_best = None

    def suggest(self, observed_X, observed_y, rng=None):
        from bo.optimizer import copula_standardize

        X = np.asarray(observed_X, dtype=float)
        y = np.asarray(observed_y, dtype=float)
        # ---- CASMOPOLITAN trust-region update (repo's _adjust_length) ----
        if self._prev_best is not None and len(y) > self.n_seen:
            self._cat._fX = np.asarray([[self._prev_best]], dtype=float)
            self._cat._adjust_length(np.asarray([float(y[-1])], dtype=float))
        self._prev_best = float(np.min(y))
        self.n_seen = len(y)
        fX = copula_standardize(y.copy())
        cand = self._cat._create_and_select_candidates(
            X, fX,
            length=self._cat.length_discrete,
            n_training_steps=self.n_training_steps,
            hypers={},
        )
        sug = np.asarray(cand[0], dtype=int).reshape(-1)
        return np.clip(sug, 0, self.n_vertices - 1)


# --------------------------------------------------------------------------- #
# 2. Bounce
# --------------------------------------------------------------------------- #
class BounceAlgo(_BaseAlgo):
    """Bounce (Papenmeier et al., NeurIPS 2023) -- categorical/one-hot BO.

    Drives the repo's own components exactly like ``bounce.bounce.Bounce.run``
    does for a discrete benchmark:

        AxUS random embedding (identity here: one bin per fork)
        -> Bounce MixtureKernel/GP (``bounce.gaussian_process.get_gp``)
        -> GP MLL fit (``fit_mll``)
        -> analytical ExpectedImprovement
        -> ``bounce.candidates.create_candidates_discrete`` (5000 candidates
           within the Hamming trust region + neighbour local search)
        -> ``bounce.trust_region.update_tr_state``

    The ``AxUS`` projection with ``n_bins == dim`` is a bijection for our 15
    categorical forks, so the low-dimensional target space is the 54-dim
    one-hot encoding of the 15-fork assignment.
    """

    name = "Bounce"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        _add_vendor("bounce")
        self._seed_everything()
        from bounce.projection import AxUS
        from bounce.trust_region import TrustRegion
        from bounce.util.benchmark import Parameter, ParameterType

        # Forks with a single choice carry no decision (Bounce's Parameter
        # requires lower_bound < upper_bound); they are held fixed at 0.
        self._active = [i for i, k in enumerate(self.n_vertices) if int(k) > 1]
        params = [
            Parameter(
                name=f"f{i}",
                type=ParameterType.CATEGORICAL,
                lower_bound=0.0,
                upper_bound=float(int(self.n_vertices[i]) - 1),
            )
            for i in self._active
        ]
        # n_bins == number of active forks -> one bin per fork (identity AxUS)
        self.axus = AxUS(parameters=params, n_bins=len(self._active))
        # (fork index, one-hot index tensor) per bin
        self._bins = []
        for bin_, inds in zip(self.axus.bins, self.axus.bin_indices):
            fname = bin_.parameters[0].name
            self._bins.append((int(fname[1:]), inds))
        if len(self._bins) != len(self._active):
            raise RuntimeError(f"bounce_axus_bins:{len(self._bins)}")
        # Bounce's published config: TrustRegion.length_init_discrete = 10
        self.tr = TrustRegion(
            dimensionality=int(self.axus.target_dim),
            length_init_discrete=int(kwargs.get("length_init_discrete", 10)),
        )
        self._target_dim = int(self.axus.target_dim)
        self._prev_best = None
        self._seen_count = 0
        self.budget = int(kwargs.get("budget", 100))

    # ---- encode / decode -------------------------------------------------- #
    def _encode(self, X):
        X = np.asarray(X, dtype=int)
        x = -np.ones((len(X), self._target_dim), dtype=float)
        for fi, inds in self._bins:
            x[np.arange(len(X)), inds[0] + X[:, fi]] = 1.0
        return x

    def _decode(self, x):
        x = np.asarray(x.internal if hasattr(x, "internal") else x, dtype=float).reshape(-1)
        out = np.zeros(self.dim, dtype=int)
        for fi, inds in self._bins:
            out[fi] = int(np.argmax(x[inds]))
        return np.clip(out, 0, self.n_vertices - 1)

    def suggest(self, observed_X, observed_y, rng=None):
        import torch
        from botorch.acquisition import ExpectedImprovement
        from bounce.candidates import create_candidates_discrete
        from bounce.gaussian_process import fit_mll, get_gp
        from bounce.trust_region import update_tr_state

        X = np.asarray(observed_X, dtype=int)
        y = np.asarray(observed_y, dtype=float)
        x_down = self._encode(X)                      # one-hot, -1 / +1
        x_scaled = torch.tensor((x_down + 1.0) / 2.0, dtype=torch.float64)  # 0 / 1

        # ---- Bounce trust-region update (as in Bounce.run) ----
        if self._seen_count > 0 and len(y) > self._seen_count:
            tr_ = self.tr
            remaining = max(1, self.budget - len(y))
            factor = np.clip(
                (tr_.length_min_discrete / float(tr_.length_discrete_continuous))
                ** (1.0 / remaining), 1e-10, None)
            update_tr_state(
                trust_region=tr_,
                fx_next=torch.tensor(float(y[-1])),
                fx_incumbent=torch.tensor(float(self._prev_best)),
                adjustment_factor=torch.tensor(float(factor)),
            )

        mean = float(np.mean(y))
        std = float(np.std(y))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        fx_scaled = (y - mean) / std
        fx_t = torch.tensor(fx_scaled, dtype=torch.float64)

        model, train_x, train_fx = get_gp(
            axus=self.axus, x=x_scaled, fx=-fx_t,
        )
        # Adam fallback is used because the botorch>=0.9 fit_gpytorch_mll API
        # no longer accepts the train_x/train_fx kwargs Bounce passes.
        fit_mll(model=model, train_x=train_x, train_fx=-train_fx,
                max_cholesky_size=int(self.kwargs.get("max_cholesky_size", 1000)),
                use_scipy_lbfgs=False)
        acq = ExpectedImprovement(model=model, best_f=float((-fx_t).max()))
        x_best, fx_best, tr_state = create_candidates_discrete(
            x_scaled=x_scaled,
            fx_scaled=fx_t,
            model=model,
            axus=self.axus,
            trust_region=self.tr,
            device="cpu",
            batch_size=1,
            acquisition_function=acq,
        )
        self._prev_best = float(np.min(y))
        self._seen_count = len(y)
        self.n_seen = len(y)
        return self._decode(x_best)


# --------------------------------------------------------------------------- #
# 3. HEBO
# --------------------------------------------------------------------------- #
class HEBOAlgo(_BaseAlgo):
    """HEBO (Cowen-Rivers et al., 2022) -- repo implementation ``HEBO/hebo``.

    Uses HEBO's own ``DesignSpace`` (15 categorical parameters), its ``gp``
    surrogate (torch GP with categorical embeddings trained for 100 epochs),
    ``MACE`` acquisition and ``EvolutionOpt`` (NSGA-II) acquisition optimiser.
    All observations are replayed into the optimizer through HEBO's public
    ``observe_new_data`` API.

    ``rand_sample`` is set to 5 (HEBO's default is 1 + n_params = 16) so that
    the surrogate is exercised as soon as a handful of warm-up points exist.
    """

    name = "HEBO"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        _add_vendor("hebo")
        self._seed_everything()
        from hebo.design_space.design_space import DesignSpace
        from hebo.optimizers.hebo import HEBO

        space = DesignSpace().parse([
            {"name": f"f{i}", "type": "cat",
             "categories": [int(j) for j in range(int(k))]}
            for i, k in enumerate(self.n_vertices)
        ])
        self.space = space
        self.opt = HEBO(
            space,
            model_name=str(kwargs.get("model_name", "gp")),
            rand_sample=max(2, int(kwargs.get("rand_sample", 5))),
            scramble_seed=self.seed,
        )
        self._init_rand_sample = max(2, int(kwargs.get("rand_sample", 5)))

    def suggest(self, observed_X, observed_y, rng=None):
        import pandas as pd

        X = np.asarray(observed_X, dtype=int)
        y = np.asarray(observed_y, dtype=float)
        if len(y) > self.n_seen:
            newX = X[self.n_seen:]
            newy = y[self.n_seen:]
            df = pd.DataFrame({f"f{i}": [int(v) for v in newX[:, i]]
                               for i in range(self.dim)})
            self.opt.observe_new_data(df, newy.reshape(-1, 1))
            self.n_seen = len(y)
        rec = self.opt.suggest(1).iloc[0]
        out = np.array([int(float(rec[f"f{i}"])) for i in range(self.dim)], dtype=int)
        return np.clip(out, 0, self.n_vertices - 1)


# --------------------------------------------------------------------------- #
# 4. bayes_optim
# --------------------------------------------------------------------------- #
class BayesOptimAlgo(_BaseAlgo):
    """bayes_optim (Hao Wang) -- Kriging + EI, ``bayes_optim.BO``.

    The 15 forks are declared as ``Integer`` variables on [0, n_choices-1]
    (bayes_optim has no nominal variable type; its ``Ordinal``/``Discrete``
    variables are restricted to string levels and cannot carry the integer
    choice indices).  Surrogate: bayes_optim's own ``GaussianProcess``
    (Matern correlation, concentrated likelihood, BFGS hyper-parameter
    search).  Acquisition: EI, maximised by bayes_optim's ``MIES``
    mixed-integer evolution strategy for non-Real spaces.
    """

    name = "bayes_optim"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        _add_vendor("bayes_optim")
        self._seed_everything()
        from bayes_optim.search_space import IntegerSpace, Integer
        from bayes_optim.surrogate import GaussianProcess, trend

        self._active = [i for i, k in enumerate(self.n_vertices) if int(k) > 1]
        self.space = IntegerSpace(
            bounds=[(0, int(self.n_vertices[i]) - 1) for i in self._active],
            var_name=[f"f{i}" for i in self._active],
        )
        n_act = len(self._active)
        self._model_kwargs = dict(
            mean=trend.constant_trend(n_act),
            corr="matern",
            thetaL=np.full(n_act, 0.01),
            thetaU=np.full(n_act, 10.0),
            nugget=1e-6,
            noise_estim=False,
            optimizer="BFGS",
            random_start=int(kwargs.get("random_start", 3)),
            wait_iter=3,
            likelihood="concentrated",
            eval_budget=int(kwargs.get("eval_budget", 500)),
            random_state=self.seed,
        )
        self._bo = None

    def _make_bo(self, X, y):
        from bayes_optim.bayes_opt import BO
        from bayes_optim.surrogate import GaussianProcess

        model = GaussianProcess(**self._model_kwargs)
        return BO(
            search_space=self.space,
            obj_fun=None,
            model=model,
            warm_data=([[int(row[i]) for i in self._active] for row in X],
                       [float(v) for v in y]),
            DoE_size=None,
            eval_type="list",
            max_FEs=10 ** 6,
            n_point=1,
            acquisition_fun="EI",
            verbose=False,
            random_seed=self.seed,
        )

    def suggest(self, observed_X, observed_y, rng=None):
        try:
            return self._suggest(observed_X, observed_y, rng)
        except Exception:
            return self._random(rng)

    def _suggest(self, observed_X, observed_y, rng=None):
        X = np.asarray(observed_X, dtype=int)
        y = np.asarray(observed_y, dtype=float)
        if self._bo is None:
            self._bo = self._make_bo(X, y)
            self.n_seen = len(y)
        elif len(y) > self.n_seen:
            self._bo.tell(
                [[int(row[i]) for i in self._active] for row in X[self.n_seen:]],
                [float(v) for v in y[self.n_seen:]],
            )
            self.n_seen = len(y)
        cand = self._bo.ask(n_point=1)[0]
        out = np.zeros(self.dim, dtype=int)
        for j, fi in enumerate(self._active):
            out[fi] = int(cand[j])
        return np.clip(out, 0, self.n_vertices - 1)


# --------------------------------------------------------------------------- #
# 5. Summit (SOBO over GPyOpt)
# --------------------------------------------------------------------------- #
class SummitAlgo(_BaseAlgo):
    """Summit SOBO (Hase et al.) on the 15-fork categorical space.

    Builds a Summit ``Domain`` with one ``CategoricalVariable`` per
    non-degenerate fork (levels = the fork's choice ids) plus a continuous
    ``loss`` objective, and calls Summit's single-objective BO strategy
    ``SOBO`` -- a GPyOpt wrapper using a GPy Matern-5/2 GP over the
    integer-coded categorical levels and Expected Improvement (lbfgs
    acquisition optimiser) -- with the full observation history as
    ``prev_res``.

    Only Summit's minimal module set is vendored (its ``__init__`` pulls in
    benchmarks/TSEMO which need fastprogress/skorch/old sklearn APIs).
    """

    name = "Summit"

    def __init__(self, codec, pool_X, seed=0, **kwargs):
        super().__init__(codec, pool_X, seed, **kwargs)
        _add_vendor("summit")
        self._seed_everything()
        from summit.domain import CategoricalVariable, ContinuousVariable, Domain
        from summit.strategies.sobo import SOBO

        self._active = [i for i, k in enumerate(self.n_vertices) if int(k) > 1]
        self._fork_ids = [fk.fork_id for fk in codec.forks]
        domain = Domain()
        for i in self._active:
            levels = [c.choice_id for c in codec.choices[i]]
            domain += CategoricalVariable(
                name=self._fork_ids[i],
                description=f"fork {i}",
                levels=levels,
            )
        domain += ContinuousVariable(
            name="loss", description="evaluation loss",
            bounds=[0.0, 1e9], is_objective=True, maximize=False,
        )
        self.domain = domain
        self.strat = SOBO(
            domain,
            gp_model_type=str(kwargs.get("gp_model_type", "GP")),
            acquisition_type=str(kwargs.get("acquisition_type", "EI")),
            optimizer_type=str(kwargs.get("optimizer_type", "lbfgs")),
            evaluator_type=str(kwargs.get("evaluator_type", "random")),
            exact_feval=bool(kwargs.get("exact_feval", False)),
            ARD=bool(kwargs.get("ARD", True)),
            standardize_outputs=bool(kwargs.get("standardize_outputs", True)),
        )
        self._fixed = {
            self._fork_ids[i]: codec.choices[i][0].choice_id
            for i in range(self.dim) if i not in self._active
        }

    def suggest(self, observed_X, observed_y, rng=None):
        import pandas as pd
        from summit.utils.dataset import DataSet

        X = np.asarray(observed_X, dtype=int)
        y = np.asarray(observed_y, dtype=float)
        data = {}
        for i in self._active:
            data[self._fork_ids[i]] = [
                self.codec.choices[i][int(v)].choice_id for v in X[:, i]
            ]
        data["loss"] = y
        ds = DataSet.from_df(pd.DataFrame(data))
        # reset the cached GPyOpt parameters: we always pass the full history
        self.strat.reset()
        nxt = self.strat.suggest_experiments(1, prev_res=ds)
        asg = dict(self._fixed)
        for i in self._active:
            asg[self._fork_ids[i]] = str(nxt[self._fork_ids[i]].iloc[0])
        vec = np.asarray(self.codec.to_index(asg), dtype=int).reshape(-1)
        return np.clip(vec, 0, self.n_vertices - 1)


ADAPTERS = {
    "CASMOPOLITAN": CASMOPOLITANAlgo,
    "Bounce": BounceAlgo,
    "HEBO": HEBOAlgo,
    "Summit": SummitAlgo,
    "bayes_optim": BayesOptimAlgo,
}
