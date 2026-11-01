# -*- coding: utf-8 -*-
"""Build REPORT.md / report.json and plots from exp4 results."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.plot import METHOD_ORDER, PAPERS, collect, plot_paper  # noqa: E402

PLANNED = ["COMBO", "CASMOPOLITAN", "CoCaBO", "MerCBO", "LADDER", "BODi", "Bounce",
           "MOCA-HESP", "MixedTypeBO", "BO-MINLP", "HEBO", "bo_pr", "Gryffin",
           "Summit", "BayBE", "bayes_optim", "BoTorch", "Ax"]
RAN = {"COMBO", "GP-EI", "GP-UCB", "GP-TS", "GP-PI", "Random",
       "CASMOPOLITAN", "Bounce", "HEBO", "Summit", "bayes_optim", "MerCBO", "BODi",
       "BoTorch", "bo_pr", "BayBE", "Ax"}
NOT_RUN_REASON = {
    "CoCaBO": "API mapping failure: CoCaBO's BaseBO concatenates continuous x_bounds; this space has 0 continuous dims (structural mismatch).",
    "MOCA-HESP": "no suggest/observe API (monolithic optimize() owns evaluation); cma/category_encoders missing; the only incremental API is the bundled CASMOPOLITAN baseline, which would mislabel MOCA-HESP.",
    "LADDER": "requires pytorch_lightning/tensorflow/rdkit + pretrained VAE checkpoints; no string/latent input for 15 choice indices.",
    "MixedTypeBO": "MATLAB-only repo (24 .m files, no Python implementation).",
    "BO-MINLP": "only BO_GPy.fit() is public; EI/UCB/TS are closures inside fit(); continuous-x + binary-y MINLP shape.",
    "Gryffin": "PyPI sdist needs Cython/C compiler and `resource` (Unix-only); no Windows wheel.",
}


def metrics(trace_path, tol):
    if not os.path.exists(trace_path):
        return None
    df = pd.read_csv(trace_path)
    if not len(df):
        return None
    feas = df[df["phase"].astype(str) != "rejected"].copy().reset_index(drop=True)
    if not len(feas):
        return None
    feas["best_loss"] = feas["loss"].cummin()
    out = {"n_feasible": int(len(feas)), "n_total": int(len(df))}
    for m in (1, 2, 5, 10, 50, 100):
        t = tol * m
        idx = np.where(feas["best_loss"].to_numpy(float) <= t)[0]
        out[f"q2t_{m}"] = int(idx[0] + 1) if len(idx) else None
    j = int(feas["loss"].idxmin())
    out["best_d"] = float(feas.loc[j, "d"]) if pd.notna(feas.loc[j, "d"]) else None
    out["best_loss"] = float(feas.loc[j, "loss"])
    out["best_attempt"] = int(feas.loc[j, "attempt"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    out_dir = args.out or root
    os.makedirs(out_dir, exist_ok=True)
    df, best_summary, traces = collect(root)
    # recompute metrics from traces
    rows = []
    for paper in PAPERS:
        for algo in METHOD_ORDER:
            base = os.path.join(root, paper, algo)
            if not os.path.isdir(base):
                continue
            for sd in sorted(glob.glob(os.path.join(base, "seed*"))):
                seed = int(os.path.basename(sd).replace("seed", ""))
                s = {}
                sp = os.path.join(sd, "summary.json")
                if os.path.exists(sp):
                    s = json.load(open(sp, encoding="utf-8"))
                m = metrics(os.path.join(sd, "trace.csv"), float(s.get("tol", 0.0)))
                if m is None:
                    continue
                m.update({"paper": paper, "algo": algo, "seed": seed,
                          "status": s.get("status"), "wall_seconds": s.get("wall_seconds"),
                          "target_d": s.get("target_d"), "tol": s.get("tol"),
                          "universe_count": s.get("universe_count"),
                          "mean_suggest_s": s.get("mean_suggest_s")})
                rows.append(m)
    runs = pd.DataFrame(rows)
    if not len(runs):
        print("no runs")
        return
    runs["abs_err"] = (runs["best_d"] - runs["target_d"]).abs()
    runs.to_csv(os.path.join(out_dir, "runs_all_recomputed.csv"), index=False)
    best = (runs.sort_values(["paper", "algo", "best_loss"], na_position="last")
            .groupby(["paper", "algo"], as_index=False).first())
    best.to_csv(os.path.join(out_dir, "runs_best_recomputed.csv"), index=False)

    # plots
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_files = {}
    for paper in PAPERS:
        p = plot_paper(paper, best, traces, plot_dir)
        if p:
            plot_files[paper] = p
    # report
    lines = []
    lines.append("# Experiment 4 — GP-family CBO vs random on 5 papers\n")
    lines.append(f"- Results root: `{root}`\n")
    lines.append(f"- Runs found: **{len(runs)}** (paper × method × seed)\n")
    lines.append("- Target: `d_paper` from Multi100 `original_cohens_d`; success = "
                 "`|d_hat - d_paper| <= max(0.0005, 0.001*|d_paper|)` (≈0.1% relative).\n")
    lines.append("- Every proposal is in `trace.csv`; rejected/infeasible proposals are kept "
                 "with `phase=rejected`.\n")

    lines.append("\n## Spaces\n")
    lines.append("| paper | universe_count | target d | tol |")
    lines.append("|---|---:|---:|---:|")
    for paper in PAPERS:
        s = runs[runs.paper == paper]
        if len(s):
            lines.append(f"| {paper} | {int(s.universe_count.iloc[0]):,} | "
                         f"{s.target_d.iloc[0]:.4f} | {s.tol.iloc[0]:.6g} |")

    lines.append("\n## Best seed per paper × method (recomputed from trace)\n")
    for paper in PAPERS:
        sub = best[best.paper == paper]
        if not len(sub):
            continue
        lines.append(f"\n### {paper}\n")
        lines.append("| method | seed | status | best d | \\|err\\| | q2t@0.1% | q2t@0.5% | q2t@1% | feasible evals | wall s |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.sort_values("best_loss").iterrows():
            def q(k):
                v = r.get(k)
                return "—" if pd.isna(v) else int(v)
            lines.append(f"| {r.algo} | {int(r.seed)} | {r.status} | {r.best_d:.4f} | "
                         f"{r.abs_err:.5f} | {q('q2t_1')} | {q('q2t_5')} | {q('q2t_10')} | "
                         f"{int(r.n_feasible)} | {r.wall_seconds:.0f} |")

    lines.append("\n## Success counts (best seed per paper × method)\n")
    for m, label in [(1, "0.1%"), (2, "0.2%"), (5, "0.5%"), (10, "1%")]:
        col = f"q2t_{m}"
        succ = best[best[col].notna()]
        lines.append(f"- **{label}**: {len(succ)} / {len(best)} paper×method pairs succeeded")
        if len(succ):
            med = succ[col].median()
            lines.append(f"  - median q2t = {med:.1f} feasible evaluations")

    lines.append("\n## Algorithm status (18 planned GP algorithms)\n")
    lines.append("| algorithm | status | note |")
    lines.append("|---|---|---|")
    for a in PLANNED:
        if a in RAN:
            lines.append(f"| {a} | ran | — |")
        else:
            lines.append(f"| {a} | not run | {NOT_RUN_REASON.get(a, 'not available')} |")
    lines.append(f"| Random (baseline) | ran | uniform random from the same candidate pool |\n")
    lines.append("Additional GP baselines (not in the 18-name list): **GP-EI, GP-UCB, GP-TS, GP-PI** "
                 "(sklearn Matérn GP on ordinal-coded choices; EI / UCB / Thompson / PI).\n")

    lines.append("\n## Plots\n")
    lines.append("- [overview: best-so-far |d-target|](plots/overview_best_so_far.png)")
    for paper, p in plot_files.items():
        rel = os.path.relpath(p, out_dir).replace("\\\\", "/")
        lines.append(f"- [{paper}]({rel})")

    lines.append("\n## Caveats\n")
    lines.append("- The 10^7-scale spaces are real (15 forks, 3–90M assignments), but BO/random "
                 "search a finite candidate pool: 500k uniform + ~30k reference-neighbourhood "
                 "points (paper-code reference and 1–3 fork perturbations). The same pool is "
                 "used by every method, so the comparison is conditional on it.\n")
    lines.append("- The reference path was pinned from the paper / Multi100 analysis code; "
                 "it is included in the pool. The candidate pool is therefore not a pure "
                 "uniform sample of the space.\n")
    lines.append("- At 0.1% relative tolerance the target basin is tiny (pool density ~1e-4 "
                 "or less). Some methods/runs will not hit; all traces are kept, and q2t at "
                 "0.5%/1% is reported for post-hoc analysis.\n")
    lines.append("- Bigoni's Multi100 F(1,7)=82.3 is not reproduced by the paper-code "
                 "reanalyses (closest code-consistent d≈3.65); the pool nevertheless contains "
                 "near-target points found by perturbation.\n")

    report = "\n".join(lines)
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"n_runs": len(runs), "best": best.to_dict(orient="records"),
                   "plots": plot_files, "planned": PLANNED,
                   "not_run": {a: NOT_RUN_REASON.get(a, "") for a in PLANNED if a not in RAN}},
                  f, ensure_ascii=False, indent=2)
    print(report[:4000])
    print("written", os.path.join(out_dir, "REPORT.md"))


if __name__ == "__main__":
    main()
