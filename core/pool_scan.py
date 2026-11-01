# -*- coding: utf-8 -*-
"""Post-hoc scan of the candidate pool: d distribution and target density."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.codec import SpaceCodec, load_pool  # noqa: E402
from core.evaluator import Evaluator  # noqa: E402

_G = {}


def _init(space):
    _G["ev"] = Evaluator(space)
    _G["codec"] = SpaceCodec(_G["ev"].loader)


def _work(chunk):
    ev = _G["ev"]
    codec = _G["codec"]
    out = []
    for row in chunk:
        asg = codec.to_string(row)
        rec = ev.evaluate(asg)
        out.append((rec["d"] if rec["d"] is not None else np.nan, rec["loss"]))
    return out


def scan(space, workers=80, chunk=2000):
    ev = Evaluator(space)
    codec = SpaceCodec(ev.loader)
    X = load_pool(space)
    chunks = [X[i:i + chunk] for i in range(0, len(X), chunk)]
    with mp.Pool(workers, initializer=_init, initargs=(space,)) as pool:
        results = pool.map(_work, chunks)
    d = np.asarray([x[0] for r in results for x in r], dtype=float)
    loss = np.asarray([x[1] for r in results for x in r], dtype=float)
    tol = ev.tol
    summary = {
        "paper": os.path.basename(os.path.dirname(space)),
        "pool_size": int(len(X)), "target_d": ev.target_d, "tol": tol,
        "n_finite": int(np.isfinite(d).sum()),
        "n_within_tol": int((loss <= tol).sum()),
        "n_within_2tol": int((loss <= 2 * tol).sum()),
        "n_within_5tol": int((loss <= 5 * tol).sum()),
        "n_within_10tol": int((loss <= 10 * tol).sum()),
        "density_tol": float((loss <= tol).mean()),
        "best_loss": float(np.nanmin(loss)),
        "best_d": float(d[np.nanargmin(loss)]) if np.isfinite(loss).any() else None,
        "quantiles_loss": {str(q): float(np.nanquantile(loss, q))
                           for q in (0.001, 0.01, 0.1, 0.5, 0.9)} if np.isfinite(loss).any() else {},
    }
    base = os.path.dirname(space)
    np.savez_compressed(os.path.join(base, "pool_scan.npz"), d=d, loss=loss)
    with open(os.path.join(base, "pool_density.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--workers", type=int, default=80)
    args = ap.parse_args()
    print(json.dumps(scan(os.path.abspath(args.space), workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
