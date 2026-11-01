# -*- coding: utf-8 -*-
"""attack_runner.py — attack search: maximize/minimize the estimate d directly.

direction = +1 : maximize d (loss = -d)
direction = -1 : minimize d (loss = +d)
Invalid assignments get INVALID_PENALTY. No tolerance stop: full budget.
Logs the same trace.csv format as runner.py plus reversal markers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.algorithms import make_optimizer  # noqa: E402
from core.codec import SpaceCodec, load_pool  # noqa: E402
from core.evaluator import Evaluator, INVALID_PENALTY  # noqa: E402


def _json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _sample_feasible(evaluator, codec, pool_X, rng, exclude=(), max_tries=200):
    seen = set(exclude)
    for _ in range(max_tries):
        idx = int(rng.integers(len(pool_X)))
        if idx in seen:
            continue
        seen.add(idx)
        asg = codec.to_string(pool_X[idx])
        ok, _reason = evaluator.feasible(asg)
        if ok:
            return idx, asg
    start = int(rng.integers(len(pool_X)))
    for j in range(len(pool_X)):
        idx = (start + j) % len(pool_X)
        asg = codec.to_string(pool_X[idx])
        ok, _reason = evaluator.feasible(asg)
        if ok:
            return idx, asg
    raise RuntimeError("no_feasible_candidate_in_pool")


def run(space: str, algo: str, seed: int, out_root: str,
        direction: float = 1.0, warmup: int = 20, budget: int = 100,
        max_seconds: float = 10800.0, max_proposals: int = 500,
        pool_n: int = 200000, verbose: bool = True,
        pool_file: str = None) -> Dict[str, Any]:
    t_start = time.time()
    space = os.path.abspath(space)
    slug = os.path.basename(os.path.dirname(space))
    evaluator = Evaluator(space)
    codec = SpaceCodec(evaluator.loader)
    pool_X = load_pool(space, pool_file)
    if len(pool_X) > pool_n:
        pool_X = pool_X[:pool_n]
    rng = np.random.default_rng(seed + 20261024)

    out_dir = os.path.join(out_root, slug, algo, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "trace.csv")
    sum_path = os.path.join(out_dir, "summary.json")
    log_path = os.path.join(out_dir, "run.log")
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg):
        line = f"[{algo} {slug} seed{seed} dir{direction:+g}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    warm_rows: List[Dict[str, Any]] = []
    warm_asgs: List[Dict[str, str]] = []
    warm_idx: List[int] = []
    warm_y: List[float] = []
    tries = 0
    while len(warm_asgs) < warmup and tries < 5000:
        tries += 1
        idx, asg = _sample_feasible(evaluator, codec, pool_X, rng,
                                    exclude=set(warm_idx), max_tries=30)
        rec = evaluator.evaluate(asg)
        y = -direction * rec["d"] if rec.get("d") is not None else INVALID_PENALTY
        warm_idx.append(idx)
        warm_asgs.append(asg)
        warm_y.append(float(y))
        warm_rows.append({"attempt": len(warm_rows), "phase": "warmup", "pool_index": idx,
                          "assignment": asg, "rec": rec, "suggest_s": 0.0,
                          "eval_s": rec["runtime_s"]})
    if len(warm_asgs) < warmup:
        raise RuntimeError(f"warmup_failed:{len(warm_asgs)}/{warmup}")

    obs_X = np.asarray([codec.to_index(a) for a in warm_asgs], dtype=int)
    obs_y = np.asarray(warm_y, dtype=float)
    best = float(np.min(obs_y))
    best_d = None
    best_asg = None
    for i, y in enumerate(obs_y):
        if y == best:
            best_asg = warm_asgs[i]
            best_d = warm_rows[i]["rec"].get("d")
            break

    status = "cap"
    error = None
    try:
        opt = make_optimizer(algo, codec, pool_X, seed)
    except Exception as exc:
        log(f"optimizer unavailable: {type(exc).__name__}: {exc}")
        opt = None
        status = "not_run"
        error = f"{type(exc).__name__}: {exc}"

    bo_rows: List[Dict[str, Any]] = []
    suggest_times: List[float] = []
    eval_times: List[float] = []
    n_proposals = 0
    n_feasible = 0
    # reversal: first attempt where d crosses 0 in the attack direction
    reversal_attempt = None
    reversal_d = None
    # direction==+1: reversal when d >= 0 ; direction==-1: reversal when d <= 0
    def is_reversal(d):
        if d is None:
            return False
        return (direction > 0 and d >= 0.0) or (direction < 0 and d <= 0.0)

    if opt is not None:
        for q in range(budget):
            if time.time() - t_start > max_seconds:
                status = "timeout"
                break
            if n_proposals >= max_proposals:
                status = "cap"
                break
            t_s = time.time()
            try:
                sug = opt.suggest(obs_X, obs_y, rng)
                sug = np.asarray(sug, dtype=int).reshape(-1)
                if sug.shape[0] != codec.dim:
                    raise ValueError(f"bad_suggestion_shape:{sug.shape}")
                sug = np.clip(sug, 0, codec.n_vertices - 1)
            except Exception as exc:
                log(f"q={q} suggest failed: {type(exc).__name__}: {exc}")
                status = "error"
                error = f"suggest:{type(exc).__name__}: {exc}"
                break
            t_suggest = time.time() - t_s
            suggest_times.append(t_suggest)
            n_proposals += 1
            asg = codec.to_string(sug)
            ok, reason = evaluator.feasible(asg)
            if not ok:
                idx, asg = _sample_feasible(evaluator, codec, pool_X, rng)
                sug = codec.to_index(asg)
            t_e = time.time()
            rec = evaluator.evaluate(asg)
            t_eval = time.time() - t_e
            eval_times.append(t_eval)
            y = -direction * rec["d"] if rec.get("d") is not None else INVALID_PENALTY
            if rec.get("d") is not None:
                n_feasible += 1
            obs_X = np.vstack([obs_X, sug.reshape(1, -1)])
            obs_y = np.append(obs_y, y)
            attempt = len(warm_rows) + len(bo_rows)
            bo_rows.append({"attempt": attempt, "phase": "bo", "pool_index": idx,
                            "assignment": asg, "rec": rec,
                            "suggest_s": t_suggest, "eval_s": t_eval})
            if y < best:
                best = float(y)
                best_d = rec.get("d")
                best_asg = asg
            if reversal_attempt is None and is_reversal(rec.get("d")):
                reversal_attempt = attempt
                reversal_d = rec.get("d")
            if verbose and (q + 1) % 10 == 0:
                log(f"q={q+1} d={rec.get('d')} loss={y:.6g} best_d={best_d} "
                    f"suggest={t_suggest:.2f}s eval={t_eval:.2f}s")
        else:
            status = "cap"

    wall = time.time() - t_start
    all_rows = warm_rows + bo_rows
    all_rows.sort(key=lambda r: r["attempt"])
    best_so_far = float("inf")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["attempt", "phase", "pool_index", "assignment", "d", "loss",
                    "t", "df", "n", "invalid", "invalid_reason", "suggest_s",
                    "eval_s", "wall_s", "best_loss", "reversal"])
        for r in all_rows:
            rec = r["rec"]
            best_so_far = min(best_so_far, rec.get("loss") if rec.get("d") is not None else INVALID_PENALTY)
            w.writerow([r["attempt"], r["phase"], r["pool_index"],
                        _json(r["assignment"]), rec.get("d"),
                        rec.get("loss"), rec.get("t"), rec.get("df"), rec.get("n"),
                        rec.get("invalid", 0), rec.get("invalid_reason", ""),
                        round(r["suggest_s"], 5), round(r["eval_s"], 5),
                        round(wall, 3), best_so_far,
                        int(is_reversal(rec.get("d")))])

    ds = [r["rec"].get("d") for r in all_rows if r["rec"].get("d") is not None]
    summary = {
        "paper": slug, "algo": algo, "seed": seed, "direction": direction,
        "space": space, "universe_count": int(evaluator.loader.universe_count),
        "target_d": evaluator.target_d, "tol": evaluator.tol,
        "status": status, "error": error,
        "n_warmup": len(warm_rows), "n_bo": len(bo_rows),
        "n_feasible_evals": n_feasible, "n_proposals": n_proposals,
        "best_d": best_d, "best_assignment": best_asg,
        "min_d": float(np.min(ds)) if ds else None,
        "max_d": float(np.max(ds)) if ds else None,
        "reversal_attempt": reversal_attempt, "reversal_d": reversal_d,
        "n_reversals": int(sum(1 for r in all_rows if is_reversal(r["rec"].get("d")))),
        "wall_seconds": round(wall, 3),
        "mean_suggest_s": float(np.mean(suggest_times)) if suggest_times else None,
        "mean_eval_s": float(np.mean(eval_times)) if eval_times else None,
    }
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log(f"DONE status={status} best_d={best_d} min={summary['min_d']} max={summary['max_d']} "
        f"reversal_attempt={reversal_attempt} wall={wall:.1f}s")
    log_fh.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction", type=float, default=1.0,
                    help="+1 maximize d, -1 minimize d")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=10800.0)
    ap.add_argument("--max-proposals", type=int, default=500)
    ap.add_argument("--pool-n", type=int, default=200000)
    ap.add_argument("--pool-file", default=None)
    args = ap.parse_args()
    run(args.space, args.algo, args.seed, args.out, direction=args.direction,
        warmup=args.warmup, budget=args.budget, max_seconds=args.max_seconds,
        max_proposals=args.max_proposals, pool_n=args.pool_n,
        pool_file=args.pool_file)


if __name__ == "__main__":
    main()
