# -*- coding: utf-8 -*-
"""reverse_benchmark.py — BO convergence benchmark from a reverse-engineering perspective (one space per paper).

Core conventions (different from the old multiverse enumeration):
  * The goal is not to "run the whole space", but to find the analysis path consistent with the paper's
    reported result in as few evaluations as possible (inferring which steps / options / choices the paper took).
  * Target objective value: near 0 = full reproduction of the paper; larger = less alike.
    success threshold defaults to 0.1 (overridable via --target-obj).
  * Main metrics: best_so_far(query), queries_to_target, first_support_query,
    success rate, wall-clock. warm-up counts toward the initial sample and is flagged separately.

Usage (one method per seed, easy to parallelize / run in background):
  python engine/bo/reverse_benchmark.py --space ... --targets ... --exp exp2 \
      --out <dir> --method BOCS-SA --seed 0 --queries 60 --warmup 20
Aggregation / plotting:
  python engine/bo/reverse_benchmark.py --space ... --targets ... --exp exp2 \
      --out <dir> --methods BOCS-SA,COMBO,RandomSearch --seeds 0,1,2 --aggregate-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(ENGINE, ".."))
for _p in (ENGINE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.choice_runtime import ChoiceExecutor  # noqa: E402
from engine.optimizer_adapter import Evaluator  # noqa: E402
from engine.space_loader import SpaceLoader  # noqa: E402
from engine.bo.bocs import BOCS_SA, BOCS_SMC  # noqa: E402
from engine.bo.combo.combo_optimizer import COMBOOptimizer  # noqa: E402
from engine.bo.random_search import RandomSearch  # noqa: E402

METHOD_LABEL = {
    "BOCS-SA": "BOCS (SA)",
    "BOCS-SMC": "BOCS (SMC)",
    "COMBO": "COMBO",
    "RandomSearch": "Random search",
}
COLORS = {
    "BOCS-SA": "#c0392b",
    "BOCS-SMC": "#e67e22",
    "COMBO": "#2471a3",
    "RandomSearch": "#7f8c8d",
}


class SpaceCodec:
    """assignment(dict fork->choice_id) <-> choice-index vector."""

    def __init__(self, loader: SpaceLoader):
        self.loader = loader
        self.forks = loader.forks
        self.choices = []
        self.index_of = []
        for fk in self.forks:
            real = [c for c in fk.choices if not c.is_skip]
            self.choices.append(real)
            self.index_of.append({c.choice_id: i for i, c in enumerate(real)})
        # Only forks with >1 choice are searched by BOCS/COMBO.  Forks with a
        # single choice (inactive menu options) are fixed to that choice; this
        # keeps "every menu option has a choice" while avoiding 70+ inert bits.
        self.search_pos = [f for f in range(len(self.forks)) if len(self.choices[f]) > 1]
        self.search_n_vertices = [len(self.choices[f]) for f in self.search_pos]

    @property
    def n_vertices(self) -> List[int]:
        return [len(c) for c in self.choices]

    def to_search(self, full_index: np.ndarray) -> np.ndarray:
        return np.asarray(full_index, dtype=int).reshape(-1)[self.search_pos]

    def from_search(self, search_index: np.ndarray) -> np.ndarray:
        full = np.zeros(len(self.forks), dtype=int)
        for k, f in enumerate(self.search_pos):
            full[f] = int(search_index[k])
        return full

    def to_index(self, assignment: Dict[str, str]) -> np.ndarray:
        out = []
        for f, fk in enumerate(self.forks):
            cid = assignment.get(fk.fork_id)
            if cid not in self.index_of[f]:
                raise KeyError("fork %s: unknown/absent choice %r" % (fk.fork_id, cid))
            out.append(self.index_of[f][cid])
        return np.asarray(out, dtype=int)

    def to_string(self, index: np.ndarray) -> Dict[str, str]:
        index = np.asarray(index, dtype=int).reshape(-1)
        return {fk.fork_id: self.choices[f][int(index[f])].choice_id
                for f, fk in enumerate(self.forks)}


def make_warmups(loader: SpaceLoader, codec: "SpaceCodec", n: int,
                 seed: int, coverage: bool = True) -> List[Dict[str, str]]:
    """Fair, diverse warm-up points.

    The old diverse_warmups started from a base of "every fork takes its first choice" and moved only one
    fork at a time; on exp3 (where the first choice is near the reference) it generated a warm-up approximating
    the reference, causing warmup_hit=True, queries_to_target=0 and making BO convergence incomparable.
    Here we changed it to: coverage first (each choice of each fork once, other forks random), then random fill-up;
    every point is random, and the exact reference path is explicitly excluded.
    """
    rng = np.random.default_rng(seed)
    ref = loader.reference_assignment() or {}
    seen = set()
    out: List[Dict[str, str]] = []

    def add(a: Dict[str, str]) -> bool:
        key = tuple(a.get(f.fork_id) for f in loader.forks)
        if key in seen:
            return False
        if ref and all(a.get(k) == v for k, v in ref.items()):
            return False
        seen.add(key)
        out.append(a)
        return True

    # Coverage pass (optional): every choice of every fork appears at least once, other forks random.
    # For the new spaces ("full menu 81 forks, mostly only skip"), the coverage pass would stuff the reference
    # combination straight into the warm-up, causing q2t=0; the new experiments use coverage=False (pure random warm-up).
    if coverage:
        for f, fk in enumerate(loader.forks):
            for c in codec.choices[f]:
                a = {}
                for g, fk2 in enumerate(loader.forks):
                    if g == f:
                        a[fk2.fork_id] = c.choice_id
                    else:
                        cs = codec.choices[g]
                        a[fk2.fork_id] = cs[int(rng.integers(len(cs)))].choice_id
                add(a)
                if len(out) >= n:
                    return out[:n]

    # random fill-up
    attempts = 0
    while len(out) < n and attempts < 10000:
        attempts += 1
        a = {fk.fork_id: codec.choices[g][int(rng.integers(len(codec.choices[g])))].choice_id
             for g, fk in enumerate(loader.forks)}
        add(a)
    return out[:n]


def build_optimizer(method: str, n_vertices: List[int], warmups: List[np.ndarray],
                    seed: int):
    if method == "BOCS-SA":
        # bhs is already vectorized (one train ~0.2s), so restore BOCS's original SA budget;
        # a more thorough acquisition search finds the target faster for the same query count.
        return BOCS_SA(n_vertices, warmups=warmups, seed=seed,
                       sa_iter=200, sa_reruns=5)
    if method == "BOCS-SMC":
        return BOCS_SMC(n_vertices, warmups=warmups, seed=seed,
                        smc_particles=12, smc_budget=100)
    if method == "COMBO":
        return COMBOOptimizer(n_vertices, warmups=warmups, seed=seed,
                              n_post=3, n_burn=10)
    if method == "RandomSearch":
        return RandomSearch(n_vertices, warmups=warmups, seed=seed)
    raise ValueError("unknown method: %s" % method)


def _atomic_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def run_one(space: str, targets: Optional[str], exp: str, out_dir: str,
            method: str, seed: int, n_warmup: int, n_queries: int,
            target_obj: float = 0.05, warmup_seed: int = 20261020,
            max_seconds: float = 7200.0, verbose: bool = True,
            warmup_coverage: bool = True, exclude_warmup_hits: bool = False
            ) -> Dict[str, Any]:
    """Run one method × seed. Writes CSV per query, returns a summary dict."""
    t_start = time.time()
    loader = SpaceLoader(space)
    executor = ChoiceExecutor(loader)
    evaluator = Evaluator(loader, executor, targets_json=targets)
    codec = SpaceCodec(loader)
    n_vertices = codec.search_n_vertices

    ref_asg = loader.reference_assignment()
    ref_obj = None
    if ref_asg:
        try:
            ref_obj = float(evaluator.evaluate(ref_asg))
        except Exception:
            ref_obj = None

    # Shared warm-up: algorithm-independent, fixed seed, random diversity, never contains the reference path.
    # With exclude_warmup_hits=True, if the warm-up already hit the target, resample with a new seed;
    # otherwise BOCS's q2t would always be 0 (in the full-menu space there are many inert forks, and exact-reference
    # exclusion is not enough — the warm-up could still equal the reference on all active forks).
    warmups = []
    obs_X: List[np.ndarray] = []
    obs_y: List[float] = []
    for attempt in range(25):
        cand = make_warmups(loader, codec, n=n_warmup,
                            seed=warmup_seed + attempt,
                            coverage=warmup_coverage)
        cand_y = [float(evaluator.evaluate(a)) for a in cand]
        warmups = cand
        obs_X = [codec.to_search(codec.to_index(a)) for a in cand]
        obs_y = cand_y
        if (not exclude_warmup_hits) or (min(cand_y) > target_obj):
            break

    method_dir = os.path.join(out_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    csv_path = os.path.join(method_dir, "seed%d.csv" % seed)
    sum_path = os.path.join(method_dir, "seed%d_summary.json" % seed)

    fh = open(csv_path, "w", newline="", encoding="utf-8-sig")
    w = csv.writer(fh)
    w.writerow(["phase", "query", "objective", "best_so_far", "support",
                "estimate", "std_error", "suggest_seconds", "eval_seconds",
                "assignment"])

    best = min(obs_y) if obs_y else float("inf")
    warmup_best = best
    for i, (a, y) in enumerate(zip(warmups, obs_y)):
        rec = evaluator.record(a)
        w.writerow(["warmup", i - len(warmups), y, best, rec.get("support"),
                    rec.get("estimate"), rec.get("std_error"), 0.0, 0.0,
                    json.dumps(a, ensure_ascii=False)])
    fh.flush()

    opt = build_optimizer(method, n_vertices, obs_X, seed)
    first_support_query = None
    support_hits = 0
    best_query = -1
    rows: List[Dict[str, Any]] = []
    status = "ok"
    err = None
    query_times: List[float] = []
    suggest_times: List[float] = []
    eval_times: List[float] = []

    try:
        for q in range(n_queries):
            if time.time() - t_start > max_seconds:
                status = "timeout"
                err = "max_seconds=%.0f" % max_seconds
                break
            t_s = time.time()
            sug = opt.suggest(np.asarray(obs_X), np.asarray(obs_y))
            sug = np.asarray(sug, dtype=int).reshape(-1)
            sug = np.clip(sug, 0, np.asarray(n_vertices) - 1)
            t_suggest = time.time() - t_s

            asg = codec.to_string(codec.from_search(sug))
            t_e = time.time()
            y = float(evaluator.evaluate(asg))
            t_eval = time.time() - t_e
            rec = evaluator.record(asg)

            obs_X.append(sug.copy())
            obs_y.append(y)
            if y < best:
                best = y
                best_query = q
            support = rec.get("support", "inconclusive")
            if support == "supports":
                support_hits += 1
                if first_support_query is None:
                    first_support_query = q

            row = {"query": q, "objective": y, "best_so_far": best,
                   "support": support, "estimate": rec.get("estimate"),
                   "std_error": rec.get("std_error"),
                   "suggest_seconds": round(t_suggest, 4),
                   "eval_seconds": round(t_eval, 4)}
            rows.append(row)
            query_times.append(t_suggest + t_eval)
            suggest_times.append(t_suggest)
            eval_times.append(t_eval)
            w.writerow(["bo", q, y, best, support, rec.get("estimate"),
                        rec.get("std_error"), round(t_suggest, 4),
                        round(t_eval, 4), json.dumps(asg, ensure_ascii=False)])
            fh.flush()
            if verbose and (q + 1) % 5 == 0:
                print("[%s seed%d] q=%d obj=%.4f best=%.4f support=%s "
                      "(suggest %.2fs eval %.2fs)"
                      % (method, seed, q + 1, y, best, support, t_suggest, t_eval),
                      flush=True)
    except Exception as e:
        status = "error"
        err = "%s: %s" % (type(e).__name__, e)
        traceback.print_exc()
    finally:
        fh.close()

    # first BO query reaching the target (0 if the warm-up already hit, and flag warmup_hit)
    warmup_hit = bool(warmup_best <= target_obj)
    q_to_target = 0 if warmup_hit else None
    for r in rows:
        if r["best_so_far"] <= target_obj:
            q_to_target = r["query"]
            break

    # Exp2 attempt-matched metrics: success after k *total* evaluations
    # (warm-up counts as attempts) and after k adaptive queries.
    budget_success_total: Dict[str, bool] = {}
    budget_success_adaptive: Dict[str, bool] = {}
    warmup_curve = [float(y) for y in obs_y[:len(warmups)]]
    for k in (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50):
        if k <= len(warmup_curve):
            best_k = min(warmup_curve[:k])
        elif k - len(warmup_curve) <= len(rows):
            best_k = min(warmup_curve + [r["best_so_far"] for r in rows[:k - len(warmup_curve)]])
        else:
            continue
        budget_success_total[str(k)] = bool(best_k <= target_obj)
        if k <= len(rows):
            budget_success_adaptive[str(k)] = bool(rows[k - 1]["best_so_far"] <= target_obj)

    summary = {
        "exp": exp,
        "space": os.path.abspath(space),
        "method": method,
        "seed": seed,
        "n_forks": len(loader.forks),
        "n_vertices": n_vertices,
        "n_bits": int(sum(n_vertices)),
        "universe_count": int(loader.universe_count),
        "n_warmup": len(warmups),
        "n_queries_requested": n_queries,
        "n_queries_done": len(rows),
        "ref_obj": ref_obj,
        "target_obj": target_obj,
        "warmup_best": float(warmup_best),
        "warmup_hit": warmup_hit,
        "best_obj": float(best),
        "best_query": best_query,
        "first_support_query": first_support_query,
        "support_hits": support_hits,
        "queries_to_target": q_to_target,
        "budget_success_total": budget_success_total,
        "budget_success_adaptive": budget_success_adaptive,
        "wall_seconds": round(time.time() - t_start, 3),
        "mean_suggest_seconds": float(np.mean(suggest_times)) if suggest_times else None,
        "mean_eval_seconds": float(np.mean(eval_times)) if eval_times else None,
        "status": status,
        "error": err,
    }
    _atomic_json(sum_path, summary)
    if verbose:
        print("[%s seed%d] DONE status=%s best=%.4f q_to_target=%s wall=%.0fs"
              % (method, seed, status, best, q_to_target, summary["wall_seconds"]),
              flush=True)
    return summary


# --------------------------------------------------------------------------- #
# aggregation + plotting
# --------------------------------------------------------------------------- #
def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def aggregate(out_dir: str, exp: str, methods: List[str], seeds: List[int],
              target_obj: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    for m in methods:
        for s in seeds:
            p = os.path.join(out_dir, m, "seed%d_summary.json" % s)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    summaries.append(json.load(fh))

    # curves: aligned by query, best_so_far includes warm-up
    curves: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    for m in methods:
        for s in seeds:
            p = os.path.join(out_dir, m, "seed%d.csv" % s)
            if not os.path.exists(p):
                continue
            rows = [r for r in _read_csv_rows(p) if r["phase"] == "bo"]
            if not rows:
                continue
            curves[m].append(np.asarray([float(r["best_so_far"]) for r in rows]))

    curve_rows = []
    for m in methods:
        arrs = curves[m]
        if not arrs:
            continue
        L = min(len(a) for a in arrs)
        C = np.vstack([a[:L] for a in arrs])
        for q in range(L):
            curve_rows.append([m, q, float(C[:, q].mean()),
                               float(C[:, q].min()), float(C[:, q].max()), C.shape[0]])
    with open(os.path.join(out_dir, "curves_mean.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "query", "mean_best", "min_best", "max_best", "n_seeds"])
        w.writerows(curve_rows)

    # summary table
    table = []
    for m in methods:
        ss = [x for x in summaries if x["method"] == m and x["status"] in ("ok", "timeout")]
        if not ss:
            continue
        bests = [x["best_obj"] for x in ss]
        qts = [x["queries_to_target"] for x in ss if x["queries_to_target"] is not None]
        firsts = [x["first_support_query"] for x in ss if x["first_support_query"] is not None]
        walls = [x["wall_seconds"] for x in ss]
        succ = [1.0 if x["best_obj"] <= target_obj else 0.0 for x in ss]
        table.append({
            "method": m,
            "n_seeds": len(ss),
            "mean_best_obj": float(np.mean(bests)),
            "sd_best_obj": float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0,
            "success_rate": float(np.mean(succ)),
            "mean_queries_to_target": float(np.mean(qts)) if qts else None,
            "mean_first_support_query": float(np.mean(firsts)) if firsts else None,
            "mean_wall_seconds": float(np.mean(walls)),
            "mean_suggest_seconds": float(np.mean([x["mean_suggest_seconds"] for x in ss
                                                   if x["mean_suggest_seconds"] is not None])) if ss else None,
        })
    with open(os.path.join(out_dir, "summary_table.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_seeds", "mean_best_obj", "sd_best_obj", "success_rate",
                    "mean_queries_to_target", "mean_first_support_query",
                    "mean_wall_seconds", "mean_suggest_seconds"])
        for r in table:
            w.writerow([r[k] for k in ["method", "n_seeds", "mean_best_obj", "sd_best_obj",
                                       "success_rate", "mean_queries_to_target",
                                       "mean_first_support_query", "mean_wall_seconds",
                                       "mean_suggest_seconds"]])

    # Figure 1: convergence curves
    plt.figure(figsize=(8.2, 4.6), dpi=160)
    for m in methods:
        arrs = curves[m]
        if not arrs:
            continue
        L = min(len(a) for a in arrs)
        C = np.vstack([a[:L] for a in arrs])
        x = np.arange(L)
        mean = C.mean(axis=0)
        lo = C.min(axis=0)
        hi = C.max(axis=0)
        plt.plot(x, mean, label=METHOD_LABEL.get(m, m), color=COLORS.get(m), lw=2)
        plt.fill_between(x, lo, hi, color=COLORS.get(m), alpha=0.12)
    plt.axhline(target_obj, color="k", ls="--", lw=1, label="target = %.2f" % target_obj)
    plt.yscale("log")
    plt.xlabel("BO query (after shared warm-up)")
    plt.ylabel("best objective so far (lower = closer to paper)")
    plt.title("%s — reverse-engineering convergence" % exp)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_best_so_far.png"))
    plt.close()

    # Figure 2: queries-to-target / final best
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    labels = [METHOD_LABEL.get(r["method"], r["method"]) for r in table]
    xs = np.arange(len(table))
    axes[0].bar(xs, [r["mean_best_obj"] for r in table],
                color=[COLORS.get(r["method"], "#555") for r in table])
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("final best objective (lower better)")
    axes[0].set_title("Final best")
    axes[0].axhline(target_obj, color="k", ls="--", lw=1)
    qvals = [r["mean_queries_to_target"] if r["mean_queries_to_target"] is not None else np.nan
             for r in table]
    axes[1].bar(xs, qvals, color=[COLORS.get(r["method"], "#555") for r in table])
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("mean BO queries to target")
    axes[1].set_title("Queries to target (lower better)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "queries_to_target.png"))
    plt.close()

    notes = {
        "exp": exp,
        "target_obj": target_obj,
        "n_runs": len(summaries),
        "table": table,
        "curve_rows": curve_rows,
        "summaries": summaries,
    }
    _atomic_json(os.path.join(out_dir, "aggregate.json"), notes)
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--methods", default="BOCS-SA,COMBO,RandomSearch")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--warmup-seed", type=int, default=20261020)
    ap.add_argument("--target-obj", type=float, default=0.05)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--random-warmup", action="store_true",
                    help="pure random warm-up (no per-choice coverage pass)")
    ap.add_argument("--exclude-warmup-hits", action="store_true",
                    help="resample warm-up until it does not already hit target")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if not args.aggregate_only:
        if args.method and args.seed is not None:
            run_one(args.space, args.targets, args.exp, args.out, args.method,
                    args.seed, args.warmup, args.queries, args.target_obj,
                    args.warmup_seed, args.max_seconds,
                    warmup_coverage=not args.random_warmup,
                    exclude_warmup_hits=args.exclude_warmup_hits)
        else:
            for m in methods:
                for s in seeds:
                    try:
                        run_one(args.space, args.targets, args.exp, args.out, m, s,
                                args.warmup, args.queries, args.target_obj,
                                args.warmup_seed, args.max_seconds,
                                warmup_coverage=not args.random_warmup,
                                exclude_warmup_hits=args.exclude_warmup_hits)
                    except Exception as e:
                        print("[%s seed%d] FATAL %s: %s" % (m, s, type(e).__name__, e),
                              flush=True)
                        traceback.print_exc()

    if not args.no_plot:
        aggregate(args.out, args.exp, methods, seeds, args.target_obj)


if __name__ == "__main__":
    main()
