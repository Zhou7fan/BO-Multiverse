# -*- coding: utf-8 -*-
"""One overview figure: best-so-far |d-target| for all papers/methods."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.plot import COLORS, METHOD_ORDER, PAPERS, collect  # noqa: E402

ROOTDIR = os.path.join(EXP4, "results", "final")
OUT = os.path.join(ROOTDIR, "plots")
YLIM = {
    "bigoni": (8e-3, 5.0),
    "marshall": (5e-5, 1.0),
    "anderson": (5e-4, 1.0),
    "jiang": (5e-5, 1.0),
    "andreoni": (5e-3, 3.0),
}


def main():
    df, best, traces = collect(ROOTDIR)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.ravel()
    for ax, paper in zip(axes, PAPERS):
        sub = best[best.paper == paper]
        target = float(sub.target_d.iloc[0])
        tol = float(sub.tol.iloc[0])
        for algo in METHOD_ORDER:
            row = sub[sub.algo == algo]
            if not len(row):
                continue
            seed = int(row.seed.iloc[0])
            s, t = traces.get((paper, algo, seed), (None, None))
            if t is None or not len(t):
                continue
            tt = t.dropna(subset=["d"]).sort_values("attempt")
            err = (tt["d"] - target).abs().cummin()
            color = COLORS.get(algo)
            if color is None:
                idx = METHOD_ORDER.index(algo) if algo in METHOD_ORDER else 0
                color = plt.cm.tab20((idx % 20) / 20.0)
            ax.plot(tt["attempt"], err, label=algo, color=color, lw=1.5)
        ax.axhline(tol, color="black", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(*YLIM[paper])
        ax.set_title(f"{paper} (target {target:.4f})")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("attempt")
        ax.set_ylabel("best-so-far |d-target|")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    p = os.path.join(OUT, "overview_best_so_far.png")
    fig.savefig(p, dpi=180)
    fig.savefig(p.replace(".png", ".pdf"))
    print(p)


if __name__ == "__main__":
    main()
