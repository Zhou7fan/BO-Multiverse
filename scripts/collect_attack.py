# -*- coding: utf-8 -*-
"""collect_attack.py — aggregate attack summaries per paper.

Reports, per paper: attack-direction extremes (filtered |d| <= 1000),
largest/smallest across all attack evaluations, and sign-reversal facts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

FILTER = 1000.0  # treat |d| beyond this as degenerate fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    papers = {}
    for f in glob.glob(os.path.join(args.root, "*", "*", "seed*", "summary.json")):
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        p = s.get("paper")
        if p not in papers:
            papers[p] = []
        papers[p].append(s)

    out = {}
    for p, ss in papers.items():
        maximizers = [s for s in ss if s.get("direction", 1.0) > 0]
        minimizers = [s for s in ss if s.get("direction", 1.0) < 0]
        max_ds = [s["best_d"] for s in maximizers
                  if s.get("best_d") is not None and abs(s["best_d"]) <= FILTER]
        min_ds = [s["best_d"] for s in minimizers
                  if s.get("best_d") is not None and abs(s["best_d"]) <= FILTER]
        revs = [s for s in ss if s.get("reversal_attempt") is not None]
        # reversal direction depends on the claim sign; report per direction:
        rev_max = [s for s in maximizers if s.get("reversal_attempt") is not None]
        rev_min = [s for s in minimizers if s.get("reversal_attempt") is not None]
        out[p] = {
            "n_runs": len(ss),
            "n_max_runs": len(maximizers),
            "n_min_runs": len(minimizers),
            "max_d_attacked": float(np.max(max_ds)) if max_ds else None,
            "min_d_attacked": float(np.min(min_ds)) if min_ds else None,
            "max_d_attacked_by": [s["algo"] for s in maximizers
                                  if s.get("best_d") is not None
                                  and abs(s["best_d"]) <= FILTER
                                  and s["best_d"] == max(max_ds)]
                if max_ds else None,
            "min_d_attacked_by": [s["algo"] for s in minimizers
                                  if s.get("best_d") is not None
                                  and abs(s["best_d"]) <= FILTER
                                  and s["best_d"] == min(min_ds)]
                if min_ds else None,
            "reversal_found_maximize": bool(rev_max),
            "reversal_found_minimize": bool(rev_min),
            "reversal_d_maximize": min((s["reversal_d"] for s in rev_max),
                                       default=None),
            "reversal_d_minimize": max((s["reversal_d"] for s in rev_min),
                                       default=None),
            "evals_to_reversal_min": min((s["reversal_attempt"] + 1
                                          for s in revs), default=None),
            "best_wall_s": float(np.min([s.get("wall_seconds") or 1e9
                                         for s in ss])),
        }
        print(json.dumps({p: out[p]}, ensure_ascii=False, indent=2))

    with open(os.path.join(args.out, "attack_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("WROTE", os.path.join(args.out, "attack_summary.json"))


if __name__ == "__main__":
    main()
