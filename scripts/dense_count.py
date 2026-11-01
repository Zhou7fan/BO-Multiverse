# -*- coding: utf-8 -*-
"""dense_count.py — parallel uniform dense sampling: direct counts within
tolerance and within reporting precision. Saves every feasible draw to draws.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.codec import SpaceCodec, load_pool  # noqa: E402
from core.evaluator import Evaluator, INVALID_PENALTY  # noqa: E402

_G = {}


def _init(space):
    _G["evaluator"] = Evaluator(space)
    _G["codec"] = SpaceCodec(_G["evaluator"].loader)


def _eval_chunk(args):
    idxs = args
    ev = _G["evaluator"]
    codec = _G["codec"]
    rows = []
    for idx in idxs:
        asg = codec.to_string(np.asarray(_G["pool"][idx], dtype=int))
        ok, reason = ev.feasible(asg)
        if not ok:
            rows.append({"pool_index": int(idx), "feasible": 0, "d": None,
                         "reason": reason, "assignment": asg})
            continue
        rec = ev.evaluate(asg)
        rows.append({"pool_index": int(idx), "feasible": 1, "d": rec.get("d"),
                     "reason": "", "assignment": asg})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--precision", type=float, required=True,
                    help="reporting precision, e.g. 0.0005 for 0.336")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    space = os.path.abspath(args.space)
    slug = os.path.basename(os.path.dirname(space))
    os.makedirs(args.out, exist_ok=True)

    pool_X = load_pool(space)
    n = min(args.n, len(pool_X))
    rng = np.random.default_rng(20261101)
    sel = rng.choice(len(pool_X), size=n, replace=False)
    sel = np.sort(sel)

    ev0 = Evaluator(space)
    target = ev0.target_d
    tol = ev0.tol
    _G["pool"] = pool_X

    chunks = np.array_split(sel, args.workers * 2)
    chunks = [c for c in chunks if len(c) > 0]

    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(space,)) as pool:
        results = pool.map(_eval_chunk, chunks)
    wall = time.time() - t0

    rows = [r for chunk in results for r in chunk]
    feasible = [r for r in rows if r["feasible"] == 1 and r["d"] is not None]
    ds = np.asarray([r["d"] for r in feasible], dtype=float)
    within_tol = int(np.sum(np.abs(ds - target) <= tol))
    within_prec = int(np.sum(np.abs(ds - target) <= args.precision))
    summary = {
        "paper": slug, "space": space, "n_draws": n,
        "n_feasible": len(feasible), "n_invalid": len(rows) - len(feasible),
        "target_d": target, "tol": tol, "precision": args.precision,
        "within_tol": within_tol, "within_prec": within_prec,
        "share_tol": within_tol / max(len(feasible), 1),
        "share_prec": within_prec / max(len(feasible), 1),
        "min_d": float(ds.min()) if len(ds) else None,
        "max_d": float(ds.max()) if len(ds) else None,
        "wall_seconds": round(wall, 3),
    }
    with open(os.path.join(args.out, f"{slug}_dense_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, f"{slug}_dense_draws.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["pool_index", "feasible", "d", "reason", "assignment"])
        for r in rows:
            w.writerow([r["pool_index"], r["feasible"], r["d"], r["reason"],
                        json.dumps(r["assignment"], ensure_ascii=False,
                                   sort_keys=True)])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
