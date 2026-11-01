# -*- coding: utf-8 -*-
"""Evaluator: run one multiverse assignment, return Cohen's d and loss."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.choice_runtime import ChoiceExecutor  # noqa: E402
from engine.space_loader import SpaceLoader  # noqa: E402

INVALID_PENALTY = 1000.0


class Evaluator:
    def __init__(self, space_yaml: str, targets_json: Optional[str] = None):
        self.space_yaml = os.path.abspath(space_yaml)
        self.loader = SpaceLoader(space_yaml)
        self.executor = ChoiceExecutor(self.loader)
        tpath = targets_json or os.path.join(os.path.dirname(self.space_yaml), "targets.json")
        self.targets = json.load(open(tpath, encoding="utf-8"))
        self.target_d = float(self.targets["target_d"])
        self.tol = float(self.targets["tol"])
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._feas: Dict[str, Any] = {}
        self._ops_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def _resolve(self, spec):
        fq = spec.get("custom_func", "")
        if ":" not in fq:
            return None
        mod_name, fn = fq.split(":", 1)
        exp_dir = self.executor.exp_dir
        if exp_dir not in sys.path:
            sys.path.insert(0, exp_dir)
        if mod_name not in self._ops_cache:
            stale = [m for m in list(sys.modules) if m == "ops" or m.startswith("ops.")]
            for m in stale:
                del sys.modules[m]
            self._ops_cache[mod_name] = importlib.import_module(mod_name)
        return getattr(self._ops_cache[mod_name], fn)

    def _run(self, assignment):
        res = self.executor.execute(assignment)
        if not res.get("ok"):
            return res, None
        spec = res.get("spec") or {}
        fn = self._resolve(spec)
        if fn is None:
            return res, {"invalid": 1, "invalid_reason": "no_custom_estimator"}
        try:
            out = fn(res["df"], spec, {})
        except Exception as exc:
            out = {"invalid": 1, "invalid_reason": f"estimator_exception:{type(exc).__name__}:{exc}"}
        return res, out

    def evaluate(self, assignment: Dict[str, str]) -> Dict[str, Any]:
        uid = self.loader.universe_id(assignment)
        if uid in self._cache:
            return self._cache[uid]
        t0 = time.time()
        res, out = self._run(assignment)
        runtime = time.time() - t0
        if not res.get("ok"):
            rec = {"universe_id": uid, "d": None, "loss": INVALID_PENALTY,
                   "invalid": 1, "invalid_reason": res.get("invalid_reason", "execute_failed"),
                   "t": None, "df": None, "n": None, "note": "", "runtime_s": runtime,
                   "assignment": dict(assignment)}
        elif out is None or out.get("invalid"):
            rec = {"universe_id": uid, "d": None, "loss": INVALID_PENALTY,
                   "invalid": 1, "invalid_reason": (out or {}).get("invalid_reason", "estimator_invalid"),
                   "t": None, "df": None, "n": (out or {}).get("n_obs"), "note": (out or {}).get("note", ""),
                   "runtime_s": runtime, "assignment": dict(assignment)}
        else:
            d = float(out["estimate"])
            rec = {"universe_id": uid, "d": d, "loss": abs(d - self.target_d),
                   "invalid": 0, "invalid_reason": "", "t": out.get("t_stat"),
                   "df": out.get("df"), "n": out.get("n_obs"), "note": out.get("note", ""),
                   "runtime_s": runtime, "assignment": dict(assignment)}
        self._cache[uid] = rec
        return rec

    def feasible(self, assignment: Dict[str, str]):
        """Deterministic feasibility oracle: requires + cheap precheck (no estimator)."""
        uid = self.loader.universe_id(assignment)
        if uid in self._feas:
            return self._feas[uid]
        res, _ = self._run(assignment)
        if not res.get("ok"):
            ans = (False, res.get("invalid_reason", "execute_failed"))
            self._feas[uid] = ans
            return ans
        spec = res.get("spec") or {}
        df = res.get("df")
        try:
            ok, reason = precheck(df, spec)
        except Exception as exc:
            ok, reason = False, f"precheck_exception:{type(exc).__name__}:{exc}"
        self._feas[uid] = (ok, reason)
        return ok, reason

    def reset_cache(self):
        self._cache.clear()
        self._feas.clear()


def precheck(df, spec):
    y = spec.get("y")
    t = spec.get("treatment")
    if not y or y not in df.columns:
        return False, "missing_y"
    if not t or t not in df.columns:
        return False, "missing_treatment"
    cluster = spec.get("cluster")
    fe = list(spec.get("fe_dummies") or [])
    controls = [c for c in (spec.get("controls") or []) if c in df.columns]
    cols = [y, t] + controls + fe
    if cluster and cluster in df.columns:
        cols.append(cluster)
    sub = df[cols].dropna()
    n = len(sub)
    min_n = int(spec.get("min_n", 30) or 30)
    if n < min_n:
        return False, f"small_n:{n}<{min_n}"
    if np.nanstd(sub[y].to_numpy(float)) <= 1e-12:
        return False, "constant_y"
    tv = sub[t]
    if tv.nunique() < 2:
        return False, "treatment_no_variation"
    if tv.nunique() == 2:
        counts = tv.value_counts()
        if counts.min() < 5:
            return False, f"small_treatment_group:{int(counts.min())}"
    if cluster:
        if cluster not in sub.columns:
            return False, "missing_cluster"
        G = sub[cluster].nunique()
        if G < 2:
            return False, f"cluster_G:{G}"
    if fe:
        for c in fe:
            if c not in sub.columns:
                return False, f"missing_fe:{c}"
            if sub[c].nunique() < 2:
                return False, f"fe_no_variation:{c}"
    for term in (spec.get("linear_combo") or {}):
        if term not in [t] + controls and term not in sub.columns:
            return False, f"missing_combo_term:{term}"
    return True, ""
