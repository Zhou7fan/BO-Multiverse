# -*- coding: utf-8 -*-
"""Merge result roots, pick the best seed per paper×algo, build results/final."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.report import metrics  # noqa: E402


def scan(root):
    out = []
    for sp in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        sd = os.path.dirname(sp)
        parts = os.path.normpath(sd).split(os.sep)
        paper, algo, seed_s = parts[-3], parts[-2], parts[-1]
        try:
            s = json.load(open(sp, encoding="utf-8"))
        except Exception:
            continue
        m = metrics(os.path.join(sd, "trace.csv"), float(s.get("tol", 0) or 0))
        if m is None:
            continue
        m.update({"paper": paper, "algo": algo, "seed": int(seed_s.replace("seed", "")),
                  "status": s.get("status"), "wall_seconds": s.get("wall_seconds"),
                  "target_d": s.get("target_d"), "tol": s.get("tol"),
                  "source_root": root, "run_dir": sd})
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(EXP4, "results", "final"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    rows = []
    for r in args.roots:
        if os.path.isdir(r):
            rows.extend(scan(r))
    df = pd.DataFrame(rows)
    if not len(df):
        print("no runs")
        return
    df["abs_err"] = (df["best_d"] - df["target_d"]).abs()
    df.to_csv(os.path.join(out, "merged_runs.csv"), index=False)
    best = (df.sort_values(["paper", "algo", "best_loss"], na_position="last")
            .groupby(["paper", "algo"], as_index=False).first())
    best.to_csv(os.path.join(out, "merged_best.csv"), index=False)
    for _, r in best.iterrows():
        dst = os.path.join(out, r.paper, r.algo, f"seed{int(r.seed)}")
        if os.path.isdir(dst):
            continue
        shutil.copytree(r.run_dir, dst)
    print("merged", len(df), "runs; best rows", len(best))
    print(best.groupby("paper").size().to_string())


if __name__ == "__main__":
    main()
