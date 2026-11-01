# -*- coding: utf-8 -*-
"""Final summary tables: per algorithm and per paper."""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROOTDIR = os.path.join(EXP4, "results", "final")


def main():
    raw = pd.read_csv(os.path.join(ROOTDIR, "run_index_all.csv"))
    best = pd.read_csv(os.path.join(ROOTDIR, "merged_best.csv"))
    rows = []
    for algo, g in raw.groupby("algo"):
        b = best[best.algo == algo]
        rows.append({
            "algo": algo,
            "raw_runs": len(g),
            "raw_success_seeds": int((g.status == "success").sum()),
            "raw_cap": int((g.status == "cap").sum()),
            "raw_timeout": int((g.status == "timeout").sum()),
            "raw_error": int((g.status == "error").sum()),
            "best_seed_success_papers": int(b["q2t_1"].notna().sum()) if len(b) else 0,
            "n_papers": int(len(b)),
            "median_best_q2t_0.1": float(b.loc[b["q2t_1"].notna(), "q2t_1"].median()) if len(b) and b["q2t_1"].notna().any() else None,
        })
    algo_df = pd.DataFrame(rows).sort_values(["best_seed_success_papers", "algo"], ascending=[False, True])
    algo_df.to_csv(os.path.join(ROOTDIR, "algorithm_summary.csv"), index=False)

    prows = []
    for paper, g in best.groupby("paper"):
        succ = g[g["q2t_1"].notna()].sort_values("q2t_1")
        if len(succ):
            bm = succ.iloc[0].algo
            bq = float(succ.iloc[0]["q2t_1"])
            bd = float(succ.iloc[0].best_d)
        else:
            r = g.loc[g["abs_err"].idxmin()]
            bm = r.algo
            bq = None
            bd = float(r.best_d)
        prows.append({
            "paper": paper,
            "n_methods": len(g),
            "n_success_0.1": len(succ),
            "n_success_0.5": int(g["q2t_5"].notna().sum()),
            "best_method": bm,
            "best_q2t_0.1": bq,
            "best_d": bd,
            "best_abs_err": float(g["abs_err"].min()),
            "target_d": float(g.target_d.iloc[0]),
        })
    paper_df = pd.DataFrame(prows).sort_values("paper")
    paper_df.to_csv(os.path.join(ROOTDIR, "paper_summary.csv"), index=False)
    print(algo_df.to_string(index=False))
    print()
    print(paper_df.to_string(index=False))


if __name__ == "__main__":
    main()
