# -*- coding: utf-8 -*-
"""full_space.py — enumerate and evaluate *every* universe of a space.

Exp2 requires the full space (not a 50k sample) so that:
  * the exact success rate and best path are known;
  * RandomSearch/BOCS can be compared against the ground truth;
  * the human baseline is compared on the same, completely enumerated space.

The driver streams DFS assignments, reuses ChoiceExecutor prefix memoisation,
and writes parquet shards incrementally.  It is intentionally CPU-only and
paper-agnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine import common, gpu_stats  # noqa: E402
from engine.choice_runtime import ChoiceExecutor  # noqa: E402
from engine.space_loader import SpaceLoader, SKIP  # noqa: E402


def _record(loader: SpaceLoader, executor: ChoiceExecutor, cfg: common.SupportConfig,
            assignment: Dict[str, str]) -> Dict[str, Any]:
    t0 = time.time()
    uid = loader.universe_id(assignment)
    res = executor.execute(assignment)
    if not res["ok"]:
        rec = common.invalid_result(uid, assignment, res.get("invalid_reason", "unknown"),
                                    runtime=time.time() - t0)
        rec["assignment"] = dict(assignment)
        return rec
    out = gpu_stats.estimate_one(res["df"], res["spec"], executor.exp_dir)
    estimator = res["spec"].get("estimator", "")
    invalid = bool(out.get("invalid"))
    if invalid:
        sd = {"support": "inconclusive", "wrong_sign": False, "significant": False,
              "close": False, "target": cfg.target_for(estimator)}
    else:
        sd = common.judge_support(out.get("estimate"), out.get("ci_low"),
                                  out.get("ci_high"), estimator, cfg)
    obj = common.objective(out.get("estimate"), estimator, invalid, sd,
                           int(out.get("model_size", 0)), cfg)
    rec = {k: out.get(k) for k in
           ["estimate", "std_error", "p_value", "ci_low", "ci_high", "effect_measure",
            "n_obs", "invalid_reason", "model_size"]}
    rec.update({
        "universe_id": uid, "assignment": dict(assignment),
        "support": sd["support"], "invalid": 1 if invalid else 0,
        "runtime_seconds": time.time() - t0, "estimator": estimator,
        "target": sd.get("target"), "objective": obj,
    })
    return rec


def run_full(space: str, targets: Optional[str], out_dir: str, exp: str,
             shard_size: int = 20000, limit: Optional[int] = None,
             verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    loader = SpaceLoader(space)
    executor = ChoiceExecutor(loader)
    cfg = common.SupportConfig.from_space(
        loader.raw,
        targets_json=json.load(open(targets, encoding="utf-8")) if targets else None)
    os.makedirs(out_dir, exist_ok=True)

    total = int(loader.universe_count)
    n_target = min(total, limit) if limit else total
    if verbose:
        print("[full] space=%s forks=%d universes=%d roles=%s"
              % (space, len(loader.forks), total, loader.role_counts()), flush=True)

    rows: List[Dict[str, Any]] = []
    shard_paths: List[str] = []
    n_done = n_valid = n_success = 0
    best_obj = float("inf")
    best_assignment = None
    best_estimate = None
    support_counts: Dict[str, int] = {}
    invalid_reasons: Dict[str, int] = {}

    def flush():
        nonlocal rows, shard_paths
        if not rows:
            return
        df = pd.DataFrame(rows)
        for fk in loader.forks:
            df[fk.fork_id] = df["assignment"].apply(lambda a: a.get(fk.fork_id, SKIP))
        df["assignment"] = df["assignment"].apply(
            lambda a: json.dumps(a, ensure_ascii=False))
        p = os.path.join(out_dir, "%s_full_shard_%04d.parquet" % (exp, len(shard_paths)))
        try:
            df.to_parquet(p, index=False)
        except Exception:
            p = p.replace(".parquet", ".csv")
            df.to_csv(p, index=False, encoding="utf-8-sig")
        shard_paths.append(p)
        rows = []

    for asg in loader.iter_assignments():
        if limit and n_done >= limit:
            break
        rec = _record(loader, executor, cfg, asg)
        n_done += 1
        support_counts[rec.get("support", "inconclusive")] = \
            support_counts.get(rec.get("support", "inconclusive"), 0) + 1
        if rec.get("invalid"):
            reason = str(rec.get("invalid_reason", ""))[:120]
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        else:
            n_valid += 1
            if rec.get("objective") is not None and rec["objective"] <= 1.0:
                n_success += 1
            obj = rec.get("objective")
            if obj is not None and obj < best_obj:
                best_obj = float(obj)
                best_assignment = dict(asg)
                best_estimate = rec.get("estimate")
        rows.append(rec)
        if len(rows) >= shard_size:
            flush()
            if verbose:
                print("[full] %d/%d valid=%d success=%d best=%.4f (%.0fs)"
                      % (n_done, n_target, n_valid, n_success, best_obj,
                         time.time() - t0), flush=True)
    flush()

    ref_info = None
    ref = loader.reference_assignment()
    if ref:
        try:
            rr = _record(loader, executor, cfg, ref)
            tgt = cfg.target_for(rr.get("estimator", ""))
            diff = (abs(rr["estimate"] - tgt)
                    if rr.get("estimate") is not None and tgt is not None else None)
            ref_info = {"assignment": ref, "estimate": rr.get("estimate"),
                        "target": tgt, "abs_diff": diff,
                        "objective": rr.get("objective"),
                        "support": rr.get("support"), "invalid": rr.get("invalid")}
        except Exception as e:
            ref_info = {"assignment": ref, "error": "%s: %s" % (type(e).__name__, e)}

    summary = {
        "exp": exp, "space": os.path.abspath(space),
        "universe_count": total, "n_enumerated": n_done, "n_valid": n_valid,
        "n_invalid": n_done - n_valid, "n_success": n_success,
        "success_rate_valid": (n_success / n_valid) if n_valid else None,
        "success_rate_total": (n_success / n_done) if n_done else None,
        "best_objective": best_obj if np.isfinite(best_obj) else None,
        "best_assignment": best_assignment, "best_estimate": best_estimate,
        "support_counts": support_counts,
        "invalid_reason_top": dict(sorted(invalid_reasons.items(),
                                          key=lambda kv: -kv[1])[:15]),
        "targets": cfg.targets_by_estimator, "close_tol": cfg.close_tol,
        "objective_mode": cfg.objective_mode, "success_metric": cfg.success_metric,
        "roles": loader.role_counts(),
        "reference": ref_info,
        "shards": shard_paths,
        "wall_seconds": round(time.time() - t0, 3),
        "complete": bool(n_done == total),
    }
    with open(os.path.join(out_dir, "%s_full_summary.json" % exp), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    if verbose:
        print("[full] DONE %d/%d success=%d (%.2f%%) best=%.4f wall=%.0fs"
              % (n_done, total, n_success,
                 100.0 * n_success / max(n_valid, 1),
                 best_obj if np.isfinite(best_obj) else -1,
                 time.time() - t0), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", default="exp2")
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_full(args.space, args.targets, args.out, args.exp, args.shard_size, args.limit)


if __name__ == "__main__":
    main()
