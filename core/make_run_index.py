# -*- coding: utf-8 -*-
"""Index every raw exp4 run across all result roots."""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.report import metrics  # noqa: E402


def main():
    roots = ["core_v1", "core_v2", "core_hard", "agent_a", "agent_b", "agent_c"]
    rows = []
    for r in roots:
        base = os.path.join(EXP4, "results", r)
        for sd in sorted(glob.glob(os.path.join(base, "*", "*", "seed*"))):
            parts = os.path.normpath(sd).split(os.sep)
            paper, algo, seed_s = parts[-3], parts[-2], parts[-1]
            seed = int(seed_s.replace("seed", ""))
            sp = os.path.join(sd, "summary.json")
            s = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
            m = metrics(os.path.join(sd, "trace.csv"), float(s.get("tol", 0) or 0)) or {}
            rows.append({
                "root": r, "paper": paper, "algo": algo, "seed": seed,
                "status": s.get("status"), "best_d": m.get("best_d"),
                "best_loss": m.get("best_loss"),
                "q2t_0.1": m.get("q2t_1"), "q2t_0.5": m.get("q2t_5"), "q2t_1": m.get("q2t_10"),
                "n_feasible": m.get("n_feasible"), "wall_seconds": s.get("wall_seconds"),
                "trace": os.path.join(sd, "trace.csv") if os.path.exists(os.path.join(sd, "trace.csv")) else "",
                "summary": sp if os.path.exists(sp) else "",
                "run_json": os.path.join(sd, "run.json") if os.path.exists(os.path.join(sd, "run.json")) else "",
                "log": os.path.join(sd, "run.log") if os.path.exists(os.path.join(sd, "run.log")) else "",
            })
    df = pd.DataFrame(rows)
    out = os.path.join(EXP4, "results", "final", "run_index_all.csv")
    df.to_csv(out, index=False)
    print("indexed", len(df), "raw runs ->", out)
    print(df.groupby("root").size().to_string())


if __name__ == "__main__":
    main()
