# -*- coding: utf-8 -*-
"""Collect exp4 results and draw per-paper d-vs-attempt plots."""
from __future__ import annotations

import argparse
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

METHOD_ORDER = ["RandomUniform", "Random", "COMBO", "CASMOPOLITAN", "BoTorch", "HEBO",
                "Bounce", "Summit", "bayes_optim", "MerCBO", "BODi", "bo_pr",
                "BayBE", "Ax", "GP-EI", "GP-UCB", "GP-TS", "GP-PI",
                "CoCaBO", "LADDER", "MOCA-HESP", "MixedTypeBO", "BO-MINLP",
                "Gryffin", "SMAC3"]
COLORS = {
    "RandomUniform": "#bdc3c7", "Random": "#7f8c8d", "COMBO": "#2471a3",
    "BoTorch": "#8e44ad", "GP-EI": "#c0392b", "GP-UCB": "#e67e22",
    "GP-TS": "#27ae60", "GP-PI": "#d35400", "HEBO": "#16a085",
    "Gryffin": "#2c3e50", "BayBE": "#7d3c98", "Ax": "#b03a2e",
}
PAPERS = ["bigoni", "marshall", "anderson", "jiang", "andreoni"]
YLIM = {
    "bigoni": (-0.5, 10.0),
    "marshall": (-0.1, 1.15),
    "anderson": (0.0, 1.0),
    "jiang": (-0.1, 1.9),
    "andreoni": (0.0, 3.0),
}


def _read_run(run_dir):
    sp = os.path.join(run_dir, "summary.json")
    tp = os.path.join(run_dir, "trace.csv")
    if not os.path.exists(sp):
        return None, None
    s = json.load(open(sp, encoding="utf-8"))
    t = pd.read_csv(tp) if os.path.exists(tp) else pd.DataFrame()
    return s, t


def collect(root):
    rows = []
    traces = {}
    for paper in PAPERS:
        for algo in METHOD_ORDER:
            base = os.path.join(root, paper, algo)
            if not os.path.isdir(base):
                continue
            for seed_dir in sorted(os.listdir(base)):
                if not seed_dir.startswith("seed"):
                    continue
                d = os.path.join(base, seed_dir)
                s, t = _read_run(d)
                if s is None:
                    continue
                seed = int(seed_dir.replace("seed", ""))
                rows.append({
                    "paper": paper, "algo": algo, "seed": seed,
                    "status": s.get("status"), "best_d": s.get("best_d"),
                    "best_loss": s.get("best_loss"), "q2t": s.get("q2t"),
                    "q2t_2": (s.get("q2t_by_mult") or {}).get("2"),
                    "q2t_5": (s.get("q2t_by_mult") or {}).get("5"),
                    "q2t_10": (s.get("q2t_by_mult") or {}).get("10"),
                    "q2t_50": (s.get("q2t_by_mult") or {}).get("50"),
                    "q2t_100": (s.get("q2t_by_mult") or {}).get("100"),
                    "n_warmup": s.get("n_warmup"), "n_bo": s.get("n_bo"),
                    "n_rejected": s.get("n_rejected"),
                    "wall_seconds": s.get("wall_seconds"),
                    "mean_suggest_s": s.get("mean_suggest_s"),
                    "target_d": s.get("target_d"), "tol": s.get("tol"),
                    "universe_count": s.get("universe_count"),
                })
                traces[(paper, algo, seed)] = (s, t)
    df = pd.DataFrame(rows)
    if len(df):
        df["abs_err"] = (df["best_d"] - df["target_d"]).abs()
        df.to_csv(os.path.join(root, "runs_all.csv"), index=False)
        best = (df.sort_values(["paper", "algo", "best_loss", "q2t"],
                               na_position="last")
                .groupby(["paper", "algo"], as_index=False).first())
        best.to_csv(os.path.join(root, "runs_best.csv"), index=False)
    else:
        best = pd.DataFrame()
    return df, best, traces


def plot_paper(paper, best, traces, out_dir, dpi=180):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = best[best["paper"] == paper] if len(best) else pd.DataFrame()
    if not len(sub):
        return None
    target = float(sub["target_d"].iloc[0])
    tol = float(sub["tol"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax, ax2 = axes
    fig2, axb = plt.subplots(figsize=(11, 6))
    any_line = False
    n_warm = 0
    for algo in METHOD_ORDER:
        row = sub[sub["algo"] == algo]
        if not len(row):
            continue
        seed = int(row["seed"].iloc[0])
        s, t = traces.get((paper, algo, seed), (None, None))
        if t is None or not len(t):
            continue
        n_warm = max(n_warm, int((t["phase"].astype(str) == "warmup").sum()))
        tt = t.dropna(subset=["d"]).sort_values("attempt")
        if not len(tt):
            continue
        color = COLORS.get(algo)
        if color is None:
            idx = METHOD_ORDER.index(algo) if algo in METHOD_ORDER else 0
            color = plt.cm.tab20((idx % 20) / 20.0)
        ylim = YLIM.get(paper, None)
        if ylim:
            tt = tt.copy()
            tt["d"] = tt["d"].clip(lower=ylim[0], upper=ylim[1])
        ax.plot(tt["attempt"], tt["d"], label=algo, color=color, lw=1.2, alpha=0.8)
        err = (tt["d"] - target).abs()
        ax2.plot(tt["attempt"], err.cummin(), color=color, lw=1.2, alpha=0.9)
        axb.plot(tt["attempt"], err.cummin(), label=algo, color=color, lw=1.3, alpha=0.9)
        succ = tt[tt["success"] == 1]
        if len(succ):
            ax.scatter(succ["attempt"].iloc[0], succ["d"].iloc[0], marker="*",
                       s=180, color=color, edgecolor="black", zorder=5)
        any_line = True

    if n_warm > 0:
        ax.axvspan(-0.5, n_warm - 0.5, color="#95a5a6", alpha=0.12, lw=0)
        ax.text(n_warm / 2, ax.get_ylim()[1], "warm-up", ha="center", va="top",
                fontsize=8, color="#555555")
    ax.axhline(target, color="black", lw=1.2, ls="--", label=f"target d={target:.4f}")
    ax.axhspan(target - tol, target + tol, color="black", alpha=0.25, lw=0)
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"{paper}: d vs attempt (best seed per method; target ± {tol:.2g})")
    if YLIM.get(paper):
        ax.set_ylim(*YLIM[paper])
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="best")

    ax2.axhline(tol, color="black", lw=1.0, ls=":")
    ax2.set_yscale("log")
    ax2.set_ylabel("best-so-far |d - target| (log)")
    ax2.set_xlabel("attempt (warm-up + proposals)")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{paper}_d_vs_attempt.png")
    pdf = os.path.join(out_dir, f"{paper}_d_vs_attempt.pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)

    axb.axhline(tol, color="black", lw=1.0, ls=":")
    axb.set_yscale("log")
    if YLIM.get(paper):
        lo = max(1e-5, tol / 20.0)
        axb.set_ylim(lo, YLIM[paper][1])
    axb.set_xlabel("attempt (warm-up + proposals)")
    axb.set_ylabel("best-so-far |d - target| (log)")
    axb.set_title(f"{paper}: best-so-far |d-target| (best seed per method)")
    axb.grid(alpha=0.3, which="both")
    axb.legend(ncol=3, fontsize=7, loc="best")
    fig2.tight_layout()
    png2 = os.path.join(out_dir, f"{paper}_best_so_far.png")
    fig2.savefig(png2, dpi=dpi)
    fig2.savefig(png2.replace(".png", ".pdf"))
    plt.close(fig2)
    return png if any_line else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    out = args.out or os.path.join(root, "plots")
    os.makedirs(out, exist_ok=True)
    df, best, traces = collect(root)
    print("runs", len(df), "best rows", len(best))
    for paper in PAPERS:
        p = plot_paper(paper, best, traces, out)
        print("plot", p)


if __name__ == "__main__":
    main()
