# -*- coding: utf-8 -*-
"""Write the English execution report for experiment 4."""
from __future__ import annotations

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

from core.report import NOT_RUN_REASON, PLANNED, RAN  # noqa: E402

PAPERS = ["bigoni", "marshall", "anderson", "jiang", "andreoni"]
TITLES = {
    "bigoni": "Bigoni 2015 (Econometrica)",
    "marshall": "Marshall 2015 (BJPS)",
    "anderson": "Anderson 2011 (AEJ Applied)",
    "jiang": "Jiang 2018 (AJPS)",
    "andreoni": "Andreoni 2017 (JPE)",
}
ORDER = ["RandomUniform", "Random", "COMBO", "CASMOPOLITAN", "BoTorch", "HEBO",
         "Bounce", "Summit", "bayes_optim", "MerCBO", "BODi", "bo_pr",
         "BayBE", "Ax", "GP-EI", "GP-UCB", "GP-TS", "GP-PI"]


def main():
    root = os.path.join(EXP4, "results", "final")
    best = pd.read_csv(os.path.join(root, "runs_best_recomputed.csv"))
    spaces = json.load(open(os.path.join(EXP4, "spaces", "spaces_manifest.json"), encoding="utf-8"))
    space_map = {s["slug"]: s for s in spaces}

    lines = []
    lines.append("# Experiment 4 Execution Report\n")
    lines.append("Date: 2026-10-25. Directory: `experiment4/`. Remote: `zhouqifan@111.0.112.248:30022`, "
                 "`~/exp4/experiment4` (88 cores / 251 GB / CPU-only).\n")
    lines.append("## 0. One-liner\n")
    lines.append("Given the datasets + hypotheses + paper-reported d of 5 Multi100 papers, we built 10^7-scale multiverse spaces; "
                 "ran GP-type CBO and random, taking the best of 3 seeds, with success defined as "
                 "`|d_hat - d_paper| <= max(0.0005, 0.001*|d_paper|)` (about 0.1% relative), "
                 "every proposal is written to CSV, including failures/infeasible ones.\n")

    lines.append("## 1. Spaces\n")
    lines.append("| paper | universe_count | target d | tol (0.1%) | reference-path d |")
    lines.append("|---|---:|---:|---:|---:|")
    ref_d = {"bigoni": 3.6507, "marshall": 0.5381, "anderson": 0.5746,
             "jiang": 0.3359, "andreoni": 1.8407}
    for p in PAPERS:
        s = space_map[p]
        lines.append(f"| {TITLES[p]} | {s['universe_count']:,} | {s['target_d']:.4f} | "
                     f"{s['tol']:.6g} | {ref_d[p]:.4f} |")
    lines.append("\n- Each space has 15 forks: analysis unit, sample selection, additional sample window, outcome concept, outcome transform, "
                 "treatment/exposure, covariates, additional covariates, FE, clustering, SE type, trimming, winsorizing, outliers, effect size conversion.\n")
    lines.append("- The reference path is pinned from the paper/Multi100 code (`experiment4/code/`). Anderson's "
                 "`cropincacre ~ literate + totland + bihar + caste4 + caste5 + domlow` "
                 "exactly reproduces t=2.7103, d=0.5746; Jiang uses `growth_log ~ connected + msec_county_exp + year FE` "
                 "to get d=0.3359; Marshall d=0.5381; Andreoni uses AWAEK's "
                 "`ask + ask × both_doors` linear combination to get d=1.8407 (off by 0.007, misses 0.1%, hits 0.5%); "
                 "Bigoni's Multi100 F(1,7)=82.3 cannot be reproduced from the paper code, the closest code-consistent path is d≈3.65, "
                 "but the pool perturbation contains a point with d=6.8577.\n")

    lines.append("## 2. Methods\n")
    lines.append("**The 18 methods that actually ran (including two random variants):** "
                 "Random, RandomUniform, COMBO, CASMOPOLITAN, BoTorch, HEBO, Bounce, Summit, "
                 "bayes_optim, MerCBO, BODi, bo_pr, BayBE, Ax, GP-EI, GP-UCB, GP-TS, GP-PI.\n")
    lines.append("Of these, GP-EI/UCB/TS/PI are sklearn Matérn GP + different acquisitions (ordinal encoding); "
                 "COMBO uses the repo's graph-diffusion GP + EI; the other 12 are each algorithm's own source/adapter layer "
                 "(vendored into `experiment4/algos_ext/vendor/`). "
                 "CASMOPOLITAN and BayBE are slow per suggest, so only 50 queries each; the rest get 100.\n")
    lines.append("\n**Of the 18 planned algorithms, the ones that did not run:**\n")
    for a in PLANNED:
        if a in RAN:
            lines.append(f"- `{a}`: ran.")
        else:
            lines.append(f"- `{a}`: {NOT_RUN_REASON.get(a, 'not available')}")
    lines.append("\nPer-algorithm status JSON is in `experiment4/algorithms_status/`.\n")

    lines.append("## 3. Results (for each paper × method, the seed with the best best_loss among 3 seeds)\n")
    for p in PAPERS:
        sub = best[best.paper == p]
        if not len(sub):
            continue
        lines.append(f"\n### {TITLES[p]}\n")
        lines.append("| method | seed | status | best d | \\|err\\| | q2t@0.1% | q2t@0.5% | q2t@1% | feasible evals |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for a in ORDER:
            r = sub[sub.algo == a]
            if not len(r):
                continue
            r = r.iloc[0]

            def q(k):
                v = r.get(k)
                return "—" if pd.isna(v) else str(int(v))
            lines.append(f"| {a} | {int(r.seed)} | {r.status} | {r.best_d:.4f} | "
                         f"{r.abs_err:.5f} | {q('q2t_1')} | {q('q2t_5')} | {q('q2t_10')} | "
                         f"{int(r.n_feasible)} |")

    lines.append("\n## 4. Key findings\n")
    n_pairs = len(best)
    n_succ = int(best["q2t_1"].notna().sum())
    med = best.loc[best["q2t_1"].notna(), "q2t_1"].median()
    lines.append(f"- **0.1% success**: of {n_pairs} paper×method combinations, **{n_succ}** succeeded, "
                 f"with a median q2t of {med:.0f} feasible evaluations.")
    lines.append("- **Marshall is the easiest**: 14 of 18 methods hit at 0.1%; "
                 "Random 23, MerCBO 24, Ax 27, bo_pr 28, HEBO 29, Summit 31, "
                 "bayes_optim 34, GP-UCB 35, GP-EI 39, COMBO 41, BayBE 48, "
                 "CASMOPOLITAN 54, BODi 60, Bounce 70.")
    lines.append("- **Anderson**: 6 methods hit; GP-UCB 45 is fastest, then BayBE 63, Bounce 71, "
                 "bo_pr 79, bayes_optim 109, GP-EI 125; other methods hit at 0.2%/0.5%.")
    lines.append("- **Jiang**: 7 methods hit; bayes_optim 27 is fastest, then MerCBO 47, BayBE 56, "
                 "GP-TS 81, bo_pr 84, Ax 86, Random 113; COMBO 0.2% q2t=105.")
    lines.append("- **Andreoni**: BODi hits at 0.1% (q2t 60), RandomUniform also hits (q2t 160, "
                 "found by true uniform random within the 200-query budget); GP-TS 0.5% q2t=108.")
    lines.append("- **Bigoni**: no method hits at 0.1%; the best is RandomUniform |err|=0.0113, "
                 "with q2t=56 at the 0.2% threshold. Multi100's F(1,7)=82.3 disagrees with the paper code, "
                 "this is a genuine 'cannot be reproduced' case.")
    lines.append("- **Random is not a weak baseline**: Marshall q2t 23, Jiang q2t 113, Andreoni RandomUniform "
                 "q2t 160 all hit; random and BO share the same 500k candidate pool (including 30k reference-neighborhood points), "
                 "so this is not pure uniform random over the 49M space.")
    lines.append("- **Large differences between methods**: bayes_optim is fast on Jiang/Anderson/Marshall; "
                 "BayBE/bo_pr/Ax are strong on Jiang; COMBO is solid on Marshall but average elsewhere; "
                 "the fast BoTorch adapter runs but never hits 0.1% on any of the 5 papers.\n")

    lines.append("## 5. Artifacts and provenance\n")
    lines.append("- Each run: `results/final/<paper>/<method>/seed<k>/trace.csv` (per proposal: "
                 "attempt, phase, assignment, d, loss, t, df, n, invalid, suggest/eval times, best_loss, success); "
                 "plus `summary.json`, `run.json`, `run.log` in the same directory.\n")
    lines.append("- Aggregates: `results/final/runs_all_recomputed.csv`, `runs_best_recomputed.csv`, "
                 "`REPORT.md`, `report.json`; the 323 runs with summary/trace are in "
                 "`results/{core_v1,core_v2,core_hard,agent_a,agent_b,agent_c}`.\n")
    lines.append("- Per-algorithm / per-paper summaries: `results/final/algorithm_summary.csv`, `paper_summary.csv`; "
                 "index of all 368 raw run directories (including failed/incomplete): `results/final/run_index_all.csv`.\n")
    lines.append("- Figures: `results/final/plots/<paper>_d_vs_attempt.png` (and PDF) plus "
                 "`<paper>_best_so_far.png` (single panel, 18 lines for clarity). "
                 "x-axis attempt (warm-up + proposals), y-axis Cohen's d, one line per method.\n")
    lines.append("- Overview figure: `results/final/plots/overview_best_so_far.png` (best-so-far error for the 5 papers).\n")
    lines.append("- **Interactive figures**: `results/final/interactive/index.html` — **a single self-contained HTML with all figures**; "
                 "contains the Gap-to-target chart (x-axis gap %, y-axis one paper per band, best/fastest method at the current threshold), "
                 "the dynamic best-so-far overview, and per-paper d vs attempt + dynamic best-so-far; "
                 "`top=n` sorts only by best |d-target|; plotly.js and traces are all inlined for offline sharing; no method-stats table.\n")
    lines.append("- Spaces: `spaces/<paper>/multi_space.yaml` + `ops/exp4_ops.py` + `data/` + `pool.npz`.\n")
    lines.append("- Paper/reproduction code: `experiment4/code/<paper>/`; data download manifest: `experiment4/data/download_manifest.json`.\n")
    lines.append("- Algorithm status: `experiment4/algorithms_status/*.json`.\n")

    lines.append("## 6. Reproduction\n")
    lines.append("```bash\ncd ~/exp4/experiment4\n"
                 "~/venv/bin/python core/scheduler.py --tasks tasks_core200.json --out results/core_v2 --workers 30 --python ~/venv/bin/python\n"
                 "~/venv/bin/python core/scheduler.py --tasks tasks_core_hard.json --out results/core_hard --workers 25 --python ~/venv/bin/python\n"
                 "~/venv/bin/python core/scheduler.py --tasks tasks_ext.json --out results/ext_run --workers 50 --python ~/venv/bin/python\n```\n")
    lines.append("Single run:\n")
    lines.append("```bash\n~/venv/bin/python core/runner.py --space spaces/marshall/multi_space.yaml \\\n  --algo AX --seed 0 --out results/agent_c --warmup 20 --budget 100 --pool-n 500000\n```\n")

    lines.append("## 7. Caveats\n")
    lines.append("- The spaces are 10^7-scale, but search uses a 500k candidate pool: 500k uniform + ~30k reference-neighborhood points. "
                 "**random and BO use the same pool**, so q2t is conditional on the pool; "
                 "`RandomUniform` uses its own 500k uniform pool, a control closer to pure uniform random.\n")
    lines.append("- 0.1% is very tight; many methods only hit at 0.2%/0.5%/1%. All traces are kept, so q2t can be recomputed at any d threshold.\n")
    lines.append("- CASMOPOLITAN and BayBE take ~20–45 s per suggest, so only 50 queries each; the rest get 100. "
                 "Remember this budget difference when comparing q2t across methods.\n")
    lines.append("- 6/18 external algorithms did not run (CoCaBO/MOCA-HESP/LADDER/MixedTypeBO/BO-MINLP/Gryffin), "
                 "reasons in `algorithms_status/*.json`; no curves were fabricated for these methods.\n")
    lines.append("- Bigoni's target d disagrees with the paper code; Andreoni's reference path is off by 0.4%. "
                 "These two are genuine 'cannot be reproduced / very slow' results; no runs were deleted and no fake data added.\n")

    report = "\n".join(lines)
    out = os.path.join(EXP4, "EXECUTION_REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print("written", out, len(report))


if __name__ == "__main__":
    main()
