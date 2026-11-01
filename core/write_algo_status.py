# -*- coding: utf-8 -*-
"""Write per-algorithm status JSONs for experiment 4 (ran and not-run)."""
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

from core.report import NOT_RUN_REASON, PLANNED, RAN  # noqa: E402

SRC = {
    "COMBO": "engine/bo/combo/combo_optimizer.py (graph-diffusion GP + EI)",
    "BoTorch": "core/algorithms.py::BoTorchGP (botorch SingleTaskGP + LogEI/EI)",
    "GP-EI": "core/algorithms.py::GPEI (sklearn Matérn GP + EI, ordinal encoding)",
    "GP-UCB": "core/algorithms.py::GPUCB (sklearn Matérn GP + UCB)",
    "GP-TS": "core/algorithms.py::GPTS (sklearn Matérn GP + Thompson sampling)",
    "GP-PI": "core/algorithms.py::GPPI (sklearn Matérn GP + PI)",
    "CASMOPOLITAN": "experiment3/cbo_survey/repos/casmopolitan",
    "CoCaBO": "experiment3/cbo_survey/repos/cocabo",
    "MerCBO": "experiment3/cbo_survey/repos/mercbo",
    "LADDER": "experiment3/cbo_survey/repos/ladder",
    "BODi": "experiment3/cbo_survey/repos/bodi",
    "Bounce": "experiment3/cbo_survey/repos/bounce",
    "MOCA-HESP": "experiment3/cbo_survey/repos/moca_hesp",
    "MixedTypeBO": "experiment3/cbo_survey/repos/mixedtypebo",
    "BO-MINLP": "experiment3/cbo_survey/repos/bo_minlp",
    "HEBO": "experiment3/cbo_survey/repos/hebo; pip HEBO 0.3.6",
    "bo_pr": "experiment3/cbo_survey/repos/bo_pr",
    "Gryffin": "experiment3/cbo_survey/repos/gryffin; pip gryffin (build failed)",
    "Summit": "experiment3/cbo_survey/repos/summit",
    "BayBE": "experiment3/cbo_survey/repos/baybe",
    "bayes_optim": "experiment3/cbo_survey/repos/bayes_optim",
    "Ax": "experiment3/cbo_survey/repos/ax",
}


def main():
    root = os.path.join(EXP4, "results", "final")
    raw_index = os.path.join(root, "run_index_all.csv")
    raw_df = pd.read_csv(raw_index) if os.path.exists(raw_index) else pd.DataFrame()
    out = os.path.join(EXP4, "algorithms_status")
    os.makedirs(out, exist_ok=True)
    summary = {}
    for algo in PLANNED:
        runs = glob.glob(os.path.join(root, "*", algo, "seed*", "summary.json"))
        recs = []
        for f in runs:
            try:
                recs.append(json.load(open(f, encoding="utf-8")))
            except Exception:
                pass
        succ = [r for r in recs if r.get("status") == "success"]
        raw = raw_df[raw_df["algo"] == algo] if len(raw_df) else pd.DataFrame()
        status = "ran" if algo in RAN else "not_run"
        if algo == "BoTorch" and not recs:
            status = "partial_timeout"
        rec = {
            "algorithm": algo,
            "status": status,
            "reason": "" if status == "ran" else NOT_RUN_REASON.get(algo, "not available"),
            "source": SRC.get(algo, ""),
            "n_runs": len(recs),
            "n_success": len(succ),
            "n_raw_runs": int(len(raw)),
            "n_raw_success": int((raw["status"] == "success").sum()) if len(raw) else 0,
            "papers": sorted({r.get("paper") for r in recs if r.get("paper")}),
            "seeds": sorted({r.get("seed") for r in recs if r.get("seed") is not None}),
            "result_dir": os.path.join(root, "*", algo) if status in ("ran", "partial_timeout") else None,
            "command": ("cd experiment4 && python core/runner.py --space spaces/<paper>/multi_space.yaml "
                        f"--algo {algo} --seed <k> --out results/core_v1 --warmup 20 --budget 100")
                        if status in ("ran", "partial_timeout") else None,
            "environment": "remote ~/venv: Python 3.12, numpy 2.5.4, pandas 3.0.6, sklearn 1.9.1, "
                           "torch 2.14.1+cpu, botorch 0.18.1, gpytorch 1.15.2",
        }
        if algo == "BoTorch":
            dirs = glob.glob(os.path.join(root, "*", "BoTorch", "seed*"))
            rec["n_attempted_dirs"] = len(dirs)
            rec["n_with_run_log"] = len([d for d in dirs if os.path.exists(os.path.join(d, "run.log"))])
            rec["partial_note"] = "30/100 queries per run before kill; run.log retained."
        summary[algo] = rec
        with open(os.path.join(out, f"{algo}.json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: (v["status"], v["n_runs"], v["n_success"]) for k, v in summary.items()},
                     indent=2))


if __name__ == "__main__":
    main()
