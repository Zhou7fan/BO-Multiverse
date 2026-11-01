# -*- coding: utf-8 -*-
"""Evaluate all one-fork perturbations of the pinned reference path."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.evaluator import Evaluator  # noqa: E402

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
    for slug, ref in REF.items():
        space = os.path.join(EXP4, "spaces", slug, "multi_space.yaml")
        ev = Evaluator(space)
        base = ev.evaluate(ref)
        print("\n" + "=" * 110)
        print(f"{slug}: ref d={base['d']} loss={base['loss']:.6g} tol={ev.tol:.6g}")
        for fk in ev.loader.forks:
            for c in fk.choices:
                if c.choice_id == ref[fk.fork_id]:
                    continue
                asg = dict(ref)
                asg[fk.fork_id] = c.choice_id
                rec = ev.evaluate(asg)
                if rec["d"] is not None:
                    flag = "OK " if rec["loss"] <= ev.tol else "   "
                    print(f"  {flag}{fk.fork_id}={c.choice_id:28s} d={rec['d']:10.4f} loss={rec['loss']:.6g}")


if __name__ == "__main__":
    main()
