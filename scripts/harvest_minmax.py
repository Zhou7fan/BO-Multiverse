# -*- coding: utf-8 -*-
"""harvest_minmax.py — collect min/max estimates + sign-reversal facts per paper
from every trace.csv, dense draw, and attack trace on disk."""
import csv
import glob
import json
import os

ROOT = os.path.expanduser("~/exp4/experiment4")
PAPERS = {
    "p17": ["results/p17_v2/p17", "results/attack/p17", "results/dense/p17_dense_draws.csv"],
    "jiang": ["results/jiang", "results/attack/jiang", "results/dense/jiang_dense_draws.csv"],
    "marshall": ["results/marshall", "results/attack/marshall", "results/dense/marshall_dense_draws.csv"],
    "andreoni": ["results/andreoni", "results/attack/andreoni", "results/dense/andreoni_dense_draws.csv"],
}
# claim direction: positive -> reversal is d<=0; negative -> reversal is d>=0
DIRECTION = {"p17": -1, "jiang": +1, "marshall": +1, "andreoni": +1}
FILTER = 1000.0


def collect(roots, dense_csv):
    ds = []
    for root in roots:
        for p in glob.glob(os.path.join(ROOT, root, "**", "trace.csv"), recursive=True):
            with open(p, encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    d = r.get("d")
                    if d in (None, ""):
                        continue
                    try:
                        ds.append(float(d))
                    except ValueError:
                        pass
    dense_path = os.path.join(ROOT, dense_csv)
    if os.path.exists(dense_path):
        with open(dense_path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("feasible") != "1" or r.get("d") in (None, ""):
                    continue
                try:
                    ds.append(float(r["d"]))
                except ValueError:
                    pass
    return ds


def main():
    out = {}
    for paper, roots in PAPERS.items():
        ds = collect(roots[:-1], roots[-1])
        sane = [d for d in ds if abs(d) <= FILTER]
        deg = len(ds) - len(sane)
        if DIRECTION[paper] > 0:
            reversed = [d for d in ds if d <= 0.0]
            reversal = bool(reversed)
            rev_extreme = min(reversed) if reversed else None
        else:
            reversed = [d for d in ds if d >= 0.0]
            reversal = bool(reversed)
            rev_extreme = max(reversed) if reversed else None
        out[paper] = {
            "n_evals": len(ds), "n_degenerate_excluded": deg,
            "min_d": min(sane) if sane else None,
            "max_d": max(sane) if sane else None,
            "reversal": reversal, "rev_extreme": rev_extreme,
        }
        print(json.dumps({paper: out[paper]}, ensure_ascii=False))
    with open(os.path.join(ROOT, "results", "minmax.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("WROTE results/minmax.json")


if __name__ == "__main__":
    main()
