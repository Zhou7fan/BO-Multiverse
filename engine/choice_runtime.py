# -*- coding: utf-8 -*-
"""choice_runtime: executes forks in assignment order, builds ctx, prefix memoization.

Design notes:
- Node cache key = the (index, choice_id) tuple of executed forks; the same prefix df is computed only once.
- Any runtime error / validity False / unmet requires / too-small sample / missing column ->
  record invalid and short-circuit that universe, never aborting the enumeration.
- Model-style forks register settings via ctx.spec and are not estimated here (estimation is batched by gpu_stats).
"""
from __future__ import annotations

import copy
import importlib
import logging
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import ENGINE_SEED
from .space_loader import SpaceLoader, Fork, Choice, SKIP

log = logging.getLogger("multiverse.choice_runtime")


def build_ctx(space: Dict[str, Any], exp_dir: str, rng_seed: int = ENGINE_SEED) -> SimpleNamespace:
    """Build ctx (contract §2). spec is the model settings registry."""
    return SimpleNamespace(
        params={},
        space={},
        tables={},
        spec={},
        rng=np.random.default_rng(rng_seed),
        exp_dir=exp_dir,
    )


class ChoiceExecutor:
    def __init__(self, loader: SpaceLoader, base_df: Optional[pd.DataFrame] = None,
                 extra_tables: Optional[Dict[str, pd.DataFrame]] = None):
        self.loader = loader
        self.exp_dir = loader.exp_dir
        self.base_df = base_df if base_df is not None else self._load_base_data()
        self.extra_tables = extra_tables or {}
        self._ops_cache: Dict[str, Any] = {}
        # Bounded LRU: prefix nodes are shared across universes; when full, evict the least recently used node (including its df).
        # The root node () is never evicted. Target: per-executor peak RSS <= 4GB.
        self.cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
        self.cache_maxsize = 150
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # root node: empty prefix = base_df + empty ctx
        root_ctx = build_ctx(loader.raw, self.exp_dir)
        root_ctx.tables = dict(self.extra_tables)
        self.cache[()] = {"df": self.base_df, "spec": {}, "space": {}, "invalid": None}

    # ---------------- LRU access ---------------- #
    def _cache_get(self, key: tuple) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def _cache_set(self, key: tuple, val: Dict[str, Any]) -> None:
        self.cache[key] = val
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_maxsize:
            old_key, old_val = self.cache.popitem(last=False)
            if old_key == ():
                # root node must not be evicted: put it back at the tail
                self.cache[()] = old_val
                self.cache.move_to_end(())
                continue
            # explicitly release the DataFrame held by the evicted node
            old_val["df"] = None
            self.evictions += 1

    # ---------------- base data loading ---------------- #
    def _load_base_data(self) -> pd.DataFrame:
        ds = self.loader.raw.get("dataset", {}) or {}
        code = ds.get("read_code")
        ns: Dict[str, Any] = {"pd": pd, "np": np, "exp_dir": self.exp_dir,
                               "data_dir": os.path.join(self.exp_dir, "data")}
        if code:
            old_cwd = os.getcwd()
            os.chdir(self.exp_dir)  # relative paths inside read_code (../data/...) resolve against exp_dir
            try:
                exec(compile(code, "<read_code>", "exec"), ns)
            except Exception as e:
                raise RuntimeError("dataset.read_code execution failed: %s" % e) from e
            finally:
                os.chdir(old_cwd)
        df = ns.get("df")
        if df is None:
            raise RuntimeError("dataset.read_code produced no df")
        return df

    # ---------------- ops module import ---------------- #
    def _load_ops_func(self, ops_module: str, func: str):
        key = (self.exp_dir, ops_module)
        if key in self._ops_cache:
            return getattr(self._ops_cache[key], func)
        if self.exp_dir not in sys.path:
            sys.path.insert(0, self.exp_dir)
        # all three spaces have a top-level ops package: drop ops.* leftovers of other spaces from sys.modules,
        # otherwise import_module would pick up the previous space's ops module.
        stale = [m for m in list(sys.modules) if m == "ops" or m.startswith("ops.")]
        for m in stale:
            del sys.modules[m]
        mod = importlib.import_module(ops_module)
        self._ops_cache[key] = mod
        return getattr(mod, func)

    # ---------------- single-choice execution ---------------- #
    def _exec_choice(self, c: Choice, df: pd.DataFrame, ctx: SimpleNamespace
                     ) -> pd.DataFrame:
        if c.ops_module and c.func:
            f = self._load_ops_func(c.ops_module, c.func)
            if c.params:
                try:
                    return f(df, ctx, **c.params)
                except TypeError as e:
                    # some ops function signatures only accept (df, ctx=None) and read params from ctx.params;
                    # on unexpected keyword, fall back to calling without kwargs (ctx.params is already filled above).
                    if "unexpected keyword argument" in str(e) or "got an unexpected" in str(e):
                        return f(df, ctx)
                    raise
            return f(df, ctx)
        ns = {"pd": pd, "np": np, "ctx": ctx, "__builtins__": __builtins__}
        exec(compile(c.code, "<choice_code>", "exec"), ns)
        apply_fn = ns["apply"]
        return apply_fn(df, ctx)

    # ---------------- replay of one assignment ---------------- #
    def execute(self, assignment: Dict[str, str]) -> Dict[str, Any]:
        prefix = ()
        node = self._cache_get(prefix)
        if node["invalid"]:
            return {"ok": False, **node["invalid"], "universe": assignment}
        df = node["df"]
        spec: Dict[str, Any] = dict(node["spec"])
        space_state: Dict[str, str] = dict(node["space"])

        for fi, fk in enumerate(self.loader.forks):
            cid = assignment.get(fk.fork_id, SKIP)
            choice = next((c for c in fk.choices if c.choice_id == cid), None)
            if choice is None:
                return self._fail(assignment, prefix, fk, cid, "unknown_choice")
            child_key = prefix + ((fi, cid),)
            space_state[fk.fork_id] = cid

            if choice.is_skip:
                if child_key not in self.cache:
                    self._cache_set(child_key, {"df": df, "spec": dict(spec),
                                                "space": dict(space_state), "invalid": None})
                prefix = child_key
                node = self._cache_get(prefix)
                df, spec, space_state = node["df"], node["spec"], node["space"]
                continue

            # cache hit?
            node = self._cache_get(child_key)
            if node is not None:
                self.hits += 1
                if node["invalid"]:
                    return {"ok": False, **node["invalid"], "universe": assignment}
                df, spec, space_state = node["df"], node["spec"], node["space"]
                prefix = child_key
                continue
            self.misses += 1

            # build the ctx for this choice
            ctx = build_ctx(self.loader.raw, self.exp_dir)
            ctx.tables = dict(self.extra_tables)
            ctx.space = dict(space_state)
            ctx.spec = dict(spec)
            ctx.params = dict(choice.params)

            # requires: column:
            for col in choice.column_requires():
                if col not in df.columns:
                    return self._cache_fail(child_key, fk, choice,
                                            "missing_column:%s" % col, assignment)
            # requires: fork: at runtime only "already assigned" upstream forks are checked;
            # forward references (requiring a future fork to take some value) are handled by the loader's static pin pruning, not judged here.
            for (tf, tv) in choice.fork_requires():
                if tf in ctx.space and tv is not None and ctx.space.get(tf) != tv:
                    return self._cache_fail(child_key, fk, choice,
                                            "fork_requires:%s!=%s" % (tf, tv), assignment)
            # validity (evaluated before execution)
            vns = {"df": df, "np": np, "pd": pd, "ctx": ctx}
            try:
                ok = bool(eval(choice.validity, {"__builtins__": {}}, vns))
            except NameError as e:
                # bare-name semantics: e.g. "hrwage in df.columns" means column-name existence,
                # retry after binding the undefined bare name to its own name string ('hrwage' in df.columns).
                import re
                m = re.search(r"name '([^']+)' is not defined", str(e))
                if m:
                    vns2 = dict(vns)
                    vns2[m.group(1)] = m.group(1)
                    try:
                        ok = bool(eval(choice.validity, {"__builtins__": {}}, vns2))
                    except Exception as e2:
                        return self._cache_fail(child_key, fk, choice,
                                                "validity_error:%s" % type(e2).__name__,
                                                assignment)
                else:
                    return self._cache_fail(child_key, fk, choice,
                                            "validity_error:NameError", assignment)
            except Exception as e:
                return self._cache_fail(child_key, fk, choice,
                                        "validity_error:%s" % type(e).__name__, assignment)
            if not ok:
                return self._cache_fail(child_key, fk, choice,
                                        "validity_false:%s" % choice.validity, assignment)

            # execute apply (defensive copy to avoid polluting the cached parent node; nearly free under pandas3 CoW)
            try:
                out = self._exec_choice(choice, df.copy(), ctx)
            except Exception as e:
                return self._cache_fail(child_key, fk, choice,
                                        "apply_exception:%s:%s" % (type(e).__name__, e),
                                        assignment)
            if not isinstance(out, pd.DataFrame):
                return self._cache_fail(child_key, fk, choice,
                                        "apply_not_dataframe", assignment)

            self._cache_set(child_key, {"df": out, "spec": dict(ctx.spec),
                                        "space": dict(ctx.space), "invalid": None})
            prefix = child_key
            df, spec, space_state = out, dict(ctx.spec), dict(ctx.space)

        return {"ok": True, "df": df, "spec": spec, "space": space_state,
                "universe": assignment}

    # ---------------- failure recording ---------------- #
    def _fail(self, assignment, prefix, fk: Fork, cid, reason):
        return {"ok": False, "invalid": 1, "invalid_reason": reason,
                "fail_fork": fk.fork_id if fk else None, "universe": assignment}

    def _cache_fail(self, child_key, fk: Fork, c: Choice, reason, assignment):
        rec = {"invalid": 1, "invalid_reason": reason,
               "fail_fork": fk.fork_id, "fail_choice": c.choice_id}
        self._cache_set(child_key, {"df": None, "spec": {}, "space": {}, "invalid": rec})
        return {"ok": False, **rec, "universe": assignment}
