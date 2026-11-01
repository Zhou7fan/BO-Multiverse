# -*- coding: utf-8 -*-
"""Verify the pinned reference path in each expanded space."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
sys.path.insert(0, ROOT)

from engine.choice_runtime import ChoiceExecutor  # noqa: E402
from engine.space_loader import SpaceLoader  # noqa: E402

REF = {
    "bigoni": dict(f_unit="subject", f_sample="cond3_4", f_sample2="all", f_y="coop_frequency",
                   f_transform="raw", f_treat="ending", f_controls="none", f_controls2="none",
                   f_fe="none", f_cluster="session", f_se="cluster", f_trim="none",
                   f_winsor="none", f_outlier="none", f_estimator="cluster_d"),
    "marshall": dict(f_unit="country_year", f_sample="election_years", f_sample2="all", f_y="turnout",
                     f_transform="raw", f_treat="logfdistock", f_controls="logpopden_fed",
                     f_controls2="none", f_fe="none", f_cluster="country", f_se="cluster",
                     f_trim="none", f_winsor="none", f_outlier="none", f_estimator="cluster_d"),
    "anderson": dict(f_unit="household", f_sample="domlow", f_sample2="all", f_y="cropincacre",
                     f_transform="raw", f_treat="domlow", f_controls="paper_table3",
                     f_controls2="none", f_fe="none", f_cluster="village", f_se="cluster",
                     f_trim="none", f_winsor="none", f_outlier="none", f_estimator="cluster_d"),
    "jiang": dict(f_unit="city_year", f_sample="all", f_sample2="all", f_y="growth_log",
                  f_transform="raw", f_treat="connected_sec", f_controls="county_exp",
                  f_controls2="none", f_fe="year", f_cluster="city", f_se="cluster",
                  f_trim="none", f_winsor="none", f_outlier="none", f_estimator="cluster_d"),
    "andreoni": dict(f_unit="session_wide", f_sample="all", f_sample2="all", f_y="givers_sum",
                     f_transform="raw", f_treat="ask", f_controls="awae_terms",
                     f_controls2="none", f_fe="none", f_cluster="block", f_se="cluster",
                     f_trim="none", f_winsor="none", f_outlier="none", f_estimator="combo_d"),
}


def main():
    for slug, asg in REF.items():
        space = os.path.join(EXP4, "spaces", slug, "multi_space.yaml")
        loader = SpaceLoader(space)
        ex = ChoiceExecutor(loader)
        res = ex.execute(asg)
        print("\n" + "=" * 100)
        print(slug, "universe_count", loader.universe_count)
        if not res["ok"]:
            print("  EXEC FAILED:", res.get("invalid_reason"))
            continue
        spec = res["spec"]
        func = spec.get("custom_func", "")
        mod_name, fn_name = func.split(":", 1)
        import importlib
        if ex.exp_dir not in sys.path:
            sys.path.insert(0, ex.exp_dir)
        # ops package may be cached from another space
        for m in [m for m in list(sys.modules) if m == "ops" or m.startswith("ops.")]:
            del sys.modules[m]
        mod = importlib.import_module(mod_name)
        out = getattr(mod, fn_name)(res["df"], spec, {})
        target = __import__("json").load(open(os.path.join(EXP4, "spaces", slug, "targets.json"), encoding="utf-8"))
        d = out.get("estimate")
        err = abs(d - target["target_d"]) if d is not None else None
        print(f"  d={d} target={target['target_d']} err={err} tol={target['tol']} "
              f"success={err is not None and err <= target['tol']}")
        print(f"  t={out.get('t_stat')} df={out.get('df')} n={out.get('n_obs')} note={out.get('note')}")
        if out.get("invalid"):
            print("  INVALID:", out.get("invalid_reason"))
        rec = {
            "paper": slug, "assignment": asg, "target_d": target["target_d"],
            "tol": target["tol"], "d": d, "abs_err": err,
            "success": bool(err is not None and err <= target["tol"]),
            "t_stat": out.get("t_stat"), "df": out.get("df"), "n_obs": out.get("n_obs"),
            "note": out.get("note"), "invalid": out.get("invalid"),
            "invalid_reason": out.get("invalid_reason", ""),
        }
        with open(os.path.join(EXP4, "spaces", slug, "reference_verification.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
