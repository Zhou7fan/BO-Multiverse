# -*- coding: utf-8 -*-
"""optimizer_adapter: evaluator, BOCS/COMBO encoding/decoding, diverse warm-up, unified Optimizer protocol."""
from __future__ import annotations

import abc
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import common, gpu_stats
from .choice_runtime import ChoiceExecutor
from .space_loader import SpaceLoader, SKIP

log = logging.getLogger("multiverse.optimizer_adapter")


class Evaluator:
    """evaluate(assignment)->objective: run one universe and return the scalar loss (lower is better)."""

    def __init__(self, loader: SpaceLoader, executor: ChoiceExecutor,
                 targets_json: Optional[str] = None):
        self.loader = loader
        self.executor = executor
        self.cfg = common.SupportConfig.from_space(
            loader.raw,
            targets_json=json.load(open(targets_json, encoding="utf-8")) if targets_json else None)
        self._cache: Dict[str, float] = {}
        self._records: Dict[str, Dict[str, Any]] = {}
        # NOTE: reference_penalty was removed in engine v0.2.  Strict exp2
        # spaces pin every paper-explicit fork, so deviating from the reference
        # is impossible by construction; result distance alone is the objective.
        # SpaceLoader rejects meta.reference_penalty != 0 when strict_roles.
        penalty = float((loader.raw.get("meta", {}) or {}).get("reference_penalty", 0.0) or 0.0)
        if penalty:
            log.warning("ignoring meta.reference_penalty=%s (removed in v0.2)", penalty)

    def __call__(self, assignment: Dict[str, str]) -> float:
        return self.evaluate(assignment)

    def evaluate(self, assignment: Dict[str, str]) -> float:
        uid = self.loader.universe_id(assignment)
        if uid in self._cache:
            return self._cache[uid]
        t0 = time.time()
        res = self.executor.execute(assignment)
        runtime = time.time() - t0
        if not res["ok"]:
            obj = common.objective(None, "", True, {}, 0, self.cfg)
            rec = common.invalid_result(uid, assignment, res.get("invalid_reason", ""), runtime=runtime)
        else:
            out = gpu_stats.estimate_one(res["df"], res["spec"], self.executor.exp_dir)
            estimator = res["spec"].get("estimator", "")
            invalid = bool(out.get("invalid"))
            sd = common.judge_support(out.get("estimate"), out.get("ci_low"),
                                      out.get("ci_high"), estimator, self.cfg) if not invalid else {
                "support": "inconclusive", "wrong_sign": False, "significant": False,
                "close": False, "target": self.cfg.target_for(estimator)}
            obj = common.objective(out.get("estimate"), estimator, invalid, sd,
                                   int(out.get("model_size", 0)), self.cfg)
            rec = {"universe_id": uid, "estimate": out.get("estimate"),
                   "std_error": out.get("std_error"), "support": sd["support"],
                   "objective": obj, "invalid": int(invalid),
                   "invalid_reason": out.get("invalid_reason", ""),
                   "runtime_seconds": runtime, "assignment": dict(assignment)}
        self._cache[uid] = obj
        self._records[uid] = rec
        return obj

    def record(self, assignment: Dict[str, str]) -> Dict[str, Any]:
        uid = self.loader.universe_id(assignment)
        return self._records.get(uid, {})


# --------------------------------------------------------------------------- #
# Encoding / decoding
# --------------------------------------------------------------------------- #
class Encoder:
    def __init__(self, loader: SpaceLoader):
        self.loader = loader
        self.block_sizes = loader.block_sizes()

    def encode(self, assignment: Dict[str, str]) -> np.ndarray:
        return self.loader.encode_onehot(assignment)

    def decode(self, vec: np.ndarray) -> Dict[str, str]:
        return self.loader.decode_onehot(vec)

    @property
    def dim(self) -> int:
        return int(sum(self.block_sizes))


# --------------------------------------------------------------------------- #
# Diverse warm-up (covers every fork value, fixed seed, never contains the reference path)
# --------------------------------------------------------------------------- #
def diverse_warmups(loader: SpaceLoader, n: int = 12, seed: int = 20261020
                    ) -> List[Dict[str, str]]:
    ref = loader.reference_assignment() or {}
    rng = np.random.default_rng(seed)
    out: List[Dict[str, str]] = []
    seen = set()

    def add(asg: Dict[str, str]):
        key = tuple(asg.get(f.fork_id, SKIP) for f in loader.forks)
        if key in seen:
            return
        if ref and all(asg.get(k) == v for k, v in ref.items() if v is not None):
            return  # never contains the reference path
        seen.add(key); out.append(asg)

    # 1) every choice of every fork is covered at least once (other forks take the first legal value / skip)
    for fk in loader.forks:
        base = {}
        for other in loader.forks:
            base[other.fork_id] = SKIP if other.optional else other.choices[0].choice_id
        for c in fk.choices:
            if c.is_skip:
                continue
            a = dict(base); a[fk.fork_id] = c.choice_id
            add(a)
    # 2) fill up to n points with random diversity
    while len(out) < n:
        a = {}
        for fk in loader.forks:
            choices = [c.choice_id for c in fk.choices]
            a[fk.fork_id] = choices[rng.integers(len(choices))]
        add(a)
    return out[:n]


# --------------------------------------------------------------------------- #
# Unified Optimizer protocol (algorithms in engine/bo implement this interface)
# --------------------------------------------------------------------------- #
class Optimizer(abc.ABC):
    def __init__(self, fork_dims: List[int], warmups: List[Dict[str, str]],
                 seed: int = 20261020, **kwargs):
        self.fork_dims = fork_dims
        self.warmups = warmups
        self.seed = seed

    @abc.abstractmethod
    def suggest(self, observed_X: List[np.ndarray], observed_y: List[float]
                ) -> Dict[str, str]:
        """Propose the next assignment based on the observed (X=onehot, y=objective)."""
        raise NotImplementedError


class RandomSearchOptimizer(Optimizer):
    """A naive reference optimizer (random search), used to validate the run_optimizer driver loop."""

    def __init__(self, fork_dims, warmups, seed=20261020, **kwargs):
        super().__init__(fork_dims, warmups, seed)
        self.rng = np.random.default_rng(seed + 1)
        self._idx = 0

    def suggest(self, observed_X, observed_y):
        # first exhaust the warmups, then go random
        if self._idx < len(self.warmups):
            a = self.warmups[self._idx]; self._idx += 1
            return a
        dims = self.fork_dims
        # fork-order info is needed; the index vector returned here is decoded externally by the adapter
        vec = np.zeros(sum(dims))
        off = 0
        for d in dims:
            vec[off + self.rng.integers(d)] = 1
            off += d
        return {"__vec__": vec}
