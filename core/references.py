# -*- coding: utf-8 -*-
"""Pinned reference paths (from paper / Multi100 code)."""
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
