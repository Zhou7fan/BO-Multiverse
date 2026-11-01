# -*- coding: utf-8 -*-
"""Run one (paper, algorithm, seed) experiment with full per-proposal trace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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

TOL_MULTS = [1, 2, 5, 10, 50, 100]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


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
    # last resort: scan from a random offset
    start = int(rng.integers(len(pool_X)))
    for j in range(len(pool_X)):
        idx = (start + j) % len(pool_X)
        asg = codec.to_string(pool_X[idx])
        ok, _reason = evaluator.feasible(asg)
        if ok:
            return idx, asg
    raise RuntimeError("no_feasible_candidate_in_pool")


def run(space: str, algo: str, seed: int, out_root: str,
        warmup: int = 20, budget: int = 100, max_seconds: float = 7200.0,
        max_proposals: int = 500, pool_n: int = 200000,
        exclude_warmup_hits: bool = True, verbose: bool = True,
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
    meta_path = os.path.join(out_dir, "run.json")
    log_path = os.path.join(out_dir, "run.log")
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg):
        line = f"[{algo} {slug} seed{seed}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    # ---------------- warm-up ----------------
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
        if exclude_warmup_hits and rec["loss"] <= evaluator.tol:
            continue
        warm_idx.append(idx)
        warm_asgs.append(asg)
        warm_y.append(rec["loss"])
        warm_rows.append({"attempt": len(warm_rows), "phase": "warmup", "pool_index": idx,
                          "assignment": asg, "rec": rec, "suggest_s": 0.0, "eval_s": rec["runtime_s"]})
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
            best_d = evaluator.evaluate(warm_asgs[i])["d"]
            break

    # ---------------- optimizer ----------------
    status = "cap"
    error = None
    try:
        opt = make_optimizer(algo, codec, pool_X, seed)
    except Exception as exc:
        log(f"optimizer unavailable: {type(exc).__name__}: {exc}")
        rows = warm_rows
        status = "not_run"
        error = f"{type(exc).__name__}: {exc}"
        opt = None

    bo_rows: List[Dict[str, Any]] = []
    reject_rows: List[Dict[str, Any]] = []
    suggest_times: List[float] = []
    eval_times: List[float] = []
    n_proposals = 0
    n_feasible = 0
    n_rejected = 0

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
                n_rejected += 1
                reject_rows.append({"attempt": len(warm_rows) + len(reject_rows) + len(bo_rows),
                                    "phase": "rejected", "pool_index": -1,
                                    "assignment": asg, "rec": {"d": None, "loss": INVALID_PENALTY,
                                                               "t": None, "df": None, "n": None,
                                                               "invalid": 1, "invalid_reason": reason,
                                                               "runtime_s": 0.0},
                                    "suggest_s": t_suggest, "eval_s": 0.0})
                # constrained-BO fallback: resample a feasible point
                idx, asg = _sample_feasible(evaluator, codec, pool_X, rng)
                sug = codec.to_index(asg)
            else:
                idx = -1
            t_e = time.time()
            rec = evaluator.evaluate(asg)
            t_eval = time.time() - t_e
            eval_times.append(t_eval)
            if rec["loss"] < INVALID_PENALTY:
                n_feasible += 1
            obs_X = np.vstack([obs_X, sug.reshape(1, -1)])
            obs_y = np.append(obs_y, rec["loss"])
            bo_rows.append({"attempt": len(warm_rows) + len(reject_rows) + len(bo_rows),
                            "phase": "bo" if idx == -1 else "fallback",
                            "pool_index": idx, "assignment": asg, "rec": rec,
                            "suggest_s": t_suggest, "eval_s": t_eval})
            if rec["loss"] < best:
                best = rec["loss"]
                best_d = rec["d"]
                best_asg = asg
            if verbose and (q + 1) % 5 == 0:
                log(f"q={q+1} d={rec['d']} loss={rec['loss']:.6g} best={best:.6g} "
                    f"suggest={t_suggest:.2f}s eval={t_eval:.2f}s")
            if rec["loss"] <= evaluator.tol:
                status = "success"
                break
        else:
            status = "cap"

    wall = time.time() - t_start
    # ---------------- write trace ----------------
    all_rows = warm_rows + reject_rows + bo_rows
    all_rows.sort(key=lambda r: r["attempt"])
    best_so_far = float("inf")
    success_attempt = None
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["attempt", "phase", "pool_index", "assignment", "d", "loss",
                    "t", "df", "n", "invalid", "invalid_reason", "suggest_s",
                    "eval_s", "wall_s", "best_loss", "success"])
        for r in all_rows:
            rec = r["rec"]
            best_so_far = min(best_so_far, rec["loss"])
            succ = int(rec["loss"] <= evaluator.tol)
            if succ and success_attempt is None:
                success_attempt = r["attempt"]
            w.writerow([r["attempt"], r["phase"], r["pool_index"],
                        _json(r["assignment"]), rec.get("d"), rec.get("loss"),
                        rec.get("t"), rec.get("df"), rec.get("n"), rec.get("invalid", 0),
                        rec.get("invalid_reason", ""), round(r["suggest_s"], 5),
                        round(r["eval_s"], 5), round(wall, 3), best_so_far, succ])

    # ---------------- q2t at multiple tolerances ----------------
    losses = [r["rec"]["loss"] for r in all_rows]
    best_curve = np.minimum.accumulate(np.asarray(losses, dtype=float))
    q2t = {}
    for m in TOL_MULTS:
        t = evaluator.tol * m
        idx = np.where(best_curve <= t)[0]
        q2t[str(m)] = int(idx[0]) if len(idx) else None
    first_success = q2t.get("1")
    summary = {
        "paper": slug, "algo": algo, "seed": seed, "space": space,
        "universe_count": int(evaluator.loader.universe_count),
        "target_d": evaluator.target_d, "tol": evaluator.tol,
        "status": status, "error": error,
        "n_warmup": len(warm_rows), "n_bo": len(bo_rows), "n_rejected": len(reject_rows),
        "n_feasible_evals": n_feasible, "n_proposals": n_proposals,
        "q2t": first_success, "q2t_by_mult": q2t,
        "success_attempt": success_attempt,
        "best_loss": float(best), "best_d": best_d, "best_assignment": best_asg,
        "wall_seconds": round(wall, 3),
        "mean_suggest_s": float(np.mean(suggest_times)) if suggest_times else None,
        "mean_eval_s": float(np.mean(eval_times)) if eval_times else None,
        "warmup_losses": [float(x) for x in warm_y],
    }
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    meta = {
        "space_sha256": _sha256(space), "pool_sha256": _sha256(os.path.join(os.path.dirname(space), "pool.npz")),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "targets": evaluator.targets,
        "cmd": sys.argv, "started": t_start, "ended": time.time(),
        "pool_size": int(len(pool_X)), "warmup": warmup, "budget": budget,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    log(f"DONE status={status} best={best:.6g} q2t={first_success} wall={wall:.1f}s")
    log_fh.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--max-proposals", type=int, default=500)
    ap.add_argument("--pool-n", type=int, default=200000)
    ap.add_argument("--pool-file", default=None)
    ap.add_argument("--allow-warmup-hits", action="store_true")
    args = ap.parse_args()
    run(args.space, args.algo, args.seed, args.out,
        warmup=args.warmup, budget=args.budget, max_seconds=args.max_seconds,
        max_proposals=args.max_proposals, pool_n=args.pool_n,
        exclude_warmup_hits=not args.allow_warmup_hits,
        pool_file=args.pool_file)


if __name__ == "__main__":
    main()
