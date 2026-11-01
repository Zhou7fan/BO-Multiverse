# -*- coding: utf-8 -*-
"""Build the five experiment-4 multiverse spaces.

Reference paths are pinned from the paper / Multi100 analysis code (see
experiment4/code/); the rest of each space instantiates many other menu options
(sample windows, outcome transforms, controls, FE, inference, robustness) so the
space is a genuine 10^7-scale multiverse rather than a small neighbourhood.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
sys.path.insert(0, ROOT)

from engine.space_loader import SpaceLoader  # noqa: E402

DATA = os.path.join(EXP4, "data")
SPACES = os.path.join(EXP4, "spaces")
OPS_SRC = os.path.join(HERE, "exp4_ops.py")

PAPERS = {
    "bigoni": {
        "paper_id": "Bigoni_Econometrica_2015_VBx1",
        "title": "Time horizon and cooperation in continuous time",
        "osf": "osf.io/85vpx",
        "claim": "Short-duration treatments: cooperation rates are higher with a deterministic horizon than with a stochastic horizon.",
        "target_d": 6.8581,
        "data": ["bigoni/subjects_data_ready.dta"],
        "read_code": """import pandas as pd
df = pd.read_stata(exp_dir + '/data/subjects_data_ready.dta', convert_categoricals=False)
""",
        "unit": [
            ("period", "subject-period row", "noop", {}),
            ("subject", "one row per session x subject", "collapse",
             {"by": ["session", "Subject"],
              "keep_first": ["condition", "short", "ending", "tmnt", "risk", "male", "age"]}),
            ("group", "one row per session x matching group", "collapse",
             {"by": ["session", "Group"], "keep_first": ["condition", "short", "ending", "tmnt"]}),
            ("session", "one row per session", "collapse",
             {"by": ["session"], "keep_first": ["condition", "short", "ending"]}),
        ],
        "sample": [
            ("all", "all sessions", "noop", {}),
            ("s1_8", "sessions 1-8", "filter_rows", {"col": "session", "values": [1, 2, 3, 4, 5, 6, 7, 8]}),
            ("s5_8", "short treatments (sessions 5-8)", "filter_rows", {"col": "session", "values": [5, 6, 7, 8]}),
            ("cond3_4", "conditions 3/4 (QPSID reference)", "filter_rows", {"col": "condition", "values": [3, 4]}),
            ("cond1_4", "conditions 1-4", "filter_rows", {"col": "condition", "values": [1, 2, 3, 4]}),
            ("short1", "short=1", "filter_rows", {"col": "short", "values": [1]}),
            ("short12", "short=1 or 2", "filter_rows", {"col": "short", "values": [1, 2]}),
        ],
        "sample2": [
            ("all", "no extra row filter", "noop", {}),
            ("period_le13", "Period <= 13", "filter_rows", {"expr": "Period <= 13"}),
            ("period_ge11", "Period >= 11", "filter_rows", {"expr": "Period >= 11"}),
            ("period_le8", "Period <= 8", "filter_rows", {"expr": "Period <= 8"}),
            ("period_9_16", "Period 9-16", "filter_rows", {"expr": "9 <= Period <= 16"}),
        ],
        "y": [
            ("final_coop", "final cooperation", "set_y", {"y": "final_coop"}),
            ("coop_frequency", "cooperation frequency", "set_y", {"y": "coop_frequency"}),
            ("zero_final_C", "final cooperation collapsed to zero", "set_y", {"y": "zero_final_C"}),
            ("initial_coop", "initial cooperation", "set_y", {"y": "initial_coop"}),
            ("num_switches", "number of switches", "set_y", {"y": "num_switches"}),
            ("payoff", "payoff", "set_y", {"y": "payoff"}),
        ],
        "transform": [
            ("raw", "raw", "transform_y", {"kind": "raw"}),
            ("log", "log", "transform_y", {"kind": "log"}),
            ("sqrt", "sqrt", "transform_y", {"kind": "sqrt"}),
            ("rank", "rank", "transform_y", {"kind": "rank"}),
        ],
        "treat": [
            ("ending", "deterministic vs stochastic ending", "set_treatment_binary", {"col": "ending", "value": 1}),
            ("short", "long vs short horizon", "set_treatment_cont", {"col": "short"}),
            ("tmnt", "treatment index", "set_treatment_cont", {"col": "tmnt"}),
        ],
        "controls": [
            ("none", "no controls", []),
            ("risk", "risk", ["risk"]),
            ("risk_male", "risk + male", ["risk", "male"]),
            ("risk_male_age", "risk + male + age", ["risk", "male", "age"]),
            ("demographics", "male + age + italian + degree", ["male", "age", "italian", "degree"]),
            ("all_controls", "risk + demographics + cognitive", ["risk", "male", "age", "italian", "degree", "trust", "logic1", "logic2", "economics", "statistics", "gametheory", "profit"]),
            ("qpsid_controls", "QPSID control set", ["age", "male", "field", "degree", "job", "italian", "wronganswer", "timeCQ", "economics", "statistics", "gametheory", "risk", "trust", "logic1", "logic2"]),
        ],
        "controls2": [
            ("none", "no extra controls", []),
            ("trust_logic", "trust + logic", ["trust", "logic1", "logic2"]),
            ("econ", "economics + statistics + game theory", ["economics", "statistics", "gametheory"]),
            ("field", "field + job", ["field", "job"]),
        ],
        "fe": [("none", "no fixed effects", [])],
        "cluster": [("session", "cluster by session"), ("subject", "cluster by subject")],
        "trim": [
            ("none", "no trim", "noop", {}),
            ("p1", "trim y at 1%", "trim", {"y": None, "p": 0.01}),
            ("p5", "trim y at 5%", "trim", {"y": None, "p": 0.05}),
        ],
        "winsor": [
            ("none", "no winsor", "noop", {}),
            ("p1", "winsor y at 1%", "winsorize", {"y": None, "p": 0.01}),
            ("p5", "winsor y at 5%", "winsorize", {"y": None, "p": 0.05}),
        ],
        "outlier": [
            ("none", "no outlier drop", "noop", {}),
            ("sd2", "drop |y-mean|>2sd", "filter_outlier", {"y": None, "k": 2.0}),
            ("sd3", "drop |y-mean|>3sd", "filter_outlier", {"y": None, "k": 3.0}),
        ],
        "se": [("cluster", "cluster-robust"), ("classical", "classical"), ("hc1", "HC1")],
        "estimator": [("cluster_d", "OLS t -> d", "use_cluster_d", {"sign": "absolute"})],
    },
    "marshall": {
        "paper_id": "Marshall_BritJournPoliSci_2015_GOYb",
        "title": "Compensation or Constraint? Economic Globalization and Electoral Turnout",
        "osf": "osf.io/qfm2g",
        "claim": "Foreign ownership reduces turnout (ownership-constraint hypothesis), all OECD countries in the data.",
        "target_d": 0.5384,
        "data": ["marshall/Turnout.dta"],
        "read_code": """import numpy as np, pandas as pd
df = pd.read_stata(exp_dir + '/data/Turnout.dta', convert_categoricals=False)
df['year2'] = np.where(df['year'] > 2010, np.floor(df['year'] / 10), df['year'])
df['logfdistock'] = np.log(df['fdistock'].clip(lower=1e-9))
df['logpopden'] = np.log(df['popden'].clip(lower=1e-9))
df = df.sort_values(['cowcode', 'year2']).reset_index(drop=True)
df['lag_logfdistock'] = df.groupby('cowcode')['logfdistock'].shift(1)
df['lag_logportequ'] = df.groupby('cowcode')['logportequ'].shift(1)
df['lag_logtrade'] = df.groupby('cowcode')['logtrade'].shift(1)
""",
        "unit": [("country_year", "country-year", "noop", {})],
        "sample": [
            ("all", "all country-years", "noop", {}),
            ("election_years", "country-election years", "filter_rows", {"notna": "t"}),
            ("y1980", "1980-2008", "filter_rows", {"expr": "1980 <= year2 <= 2008"}),
            ("y1990", "1990-2008", "filter_rows", {"expr": "1990 <= year2 <= 2008"}),
            ("vap", "VAP turnout sample", "filter_rows", {"notna": "turnoutvap"}),
            ("oecd", "OECD countries", "filter_rows", {"notna": "turnout"}),
        ],
        "sample2": [
            ("all", "no extra row filter", "noop", {}),
            ("year_le2008", "year <= 2008", "filter_rows", {"expr": "year2 <= 2008"}),
            ("year_ge1970", "year >= 1970", "filter_rows", {"expr": "year2 >= 1970"}),
            ("drop_us", "drop United States", "filter_rows", {"expr": "country != 'United States'"}),
            ("year_ge1990", "year >= 1990", "filter_rows", {"expr": "year2 >= 1990"}),
        ],
        "y": [
            ("turnout", "turnout", "set_y", {"y": "turnout"}),
            ("turnoutvap", "turnout (% VAP)", "set_y", {"y": "turnoutvap"}),
            ("log_turnout", "log turnout", "set_y", {"y": "turnout", "transform": "log"}),
        ],
        "transform": [
            ("raw", "raw", "transform_y", {"kind": "raw"}),
            ("log", "log", "transform_y", {"kind": "log"}),
            ("sqrt", "sqrt", "transform_y", {"kind": "sqrt"}),
            ("rank", "rank", "transform_y", {"kind": "rank"}),
        ],
        "treat": [
            ("logfdistock", "log FDI stock", "set_treatment_cont", {"col": "logfdistock"}),
            ("fdistock", "FDI stock", "set_treatment_cont", {"col": "fdistock"}),
            ("fdiin", "FDI inward", "set_treatment_cont", {"col": "fdiin"}),
            ("fdiout", "FDI outward", "set_treatment_cont", {"col": "fdiout"}),
            ("lag_logfdistock", "lagged log FDI stock", "set_treatment_cont", {"col": "lag_logfdistock"}),
        ],
        "controls": [
            ("none", "no controls", []),
            ("loggdppc", "log GDP pc", ["loggdppc"]),
            ("loggdppc_growth", "log GDP pc + growth", ["loggdppc", "growthpc"]),
            ("logpopden_fed", "log population density + federalism", ["logpopden", "fed"]),
            ("macro", "GDP pc + growth + unemployment + deindustrialization", ["loggdppc", "growthpc", "u", "deindust"]),
            ("political", "polity + PR + compulsory voting", ["polity", "pr", "cvvfranklin"]),
            ("code_controls", "paper control set", ["logreg", "vap3069", "timesince", "nopres", "cvvfranklin", "mixed", "pr", "disrel", "enps", "closeness"]),
        ],
        "controls2": [
            ("none", "no extra controls", []),
            ("fed", "federalism", ["fed"]),
            ("pr", "PR system", ["pr"]),
            ("u", "unemployment", ["u"]),
        ],
        "fe": [
            ("none", "no fixed effects", []),
            ("year", "year FE", ["year2"]),
            ("country", "country FE", ["country"]),
        ],
        "cluster": [("country", "cluster by country"), ("year2", "cluster by year")],
        "trim": [
            ("none", "no trim", "noop", {}),
            ("p1", "trim y at 1%", "trim", {"y": None, "p": 0.01}),
            ("p5", "trim y at 5%", "trim", {"y": None, "p": 0.05}),
        ],
        "winsor": [
            ("none", "no winsor", "noop", {}),
            ("p1", "winsor y at 1%", "winsorize", {"y": None, "p": 0.01}),
            ("p5", "winsor y at 5%", "winsorize", {"y": None, "p": 0.05}),
        ],
        "outlier": [
            ("none", "no outlier drop", "noop", {}),
            ("sd2", "drop |y-mean|>2sd", "filter_outlier", {"y": None, "k": 2.0}),
            ("sd3", "drop |y-mean|>3sd", "filter_outlier", {"y": None, "k": 3.0}),
        ],
        "se": [("cluster", "cluster-robust"), ("classical", "classical"), ("hc1", "HC1")],
        "estimator": [
            ("cluster_d", "OLS t -> d", "use_cluster_d", {"sign": "absolute"}),
            ("corr_d", "correlation -> d", "use_corr_d", {"sign": "absolute"}),
        ],
    },
    "anderson": {
        "paper_id": "Anderson_AmEcoJourn_2011_bLe8",
        "title": "Caste as an Impediment to Trade",
        "osf": "osf.io/j7etu",
        "claim": "Low-caste households in villages dominated by BACs have substantially higher income than in villages dominated by upper castes.",
        "target_d": 0.5746,
        "data": ["anderson/household.dta", "anderson/village.dta"],
        "read_code": """import numpy as np, pandas as pd
hh = pd.read_stata(exp_dir + '/data/household.dta', convert_categoricals=False)
vv = pd.read_stata(exp_dir + '/data/village.dta', convert_categoricals=False)
df = hh.merge(vv, on='village', how='left', suffixes=('', '_v'))
df['logcropinc'] = np.log(df['cropinc'].clip(lower=1))
df['logtotinc'] = np.log(df['totinc'].clip(lower=1))
df['logland'] = np.log1p(df['totland'].clip(lower=0))
df['anybank'] = (df['bank3'] > 0).astype(float)
df['anybus'] = (df['bus3'] > 0).astype(float)
df['anymarket'] = (df['ms3'] > 0).astype(float)
df['domlow_x_bfh'] = df['domlow'] * df['bfhcast']
df['domlow_x_caste4'] = df['domlow'] * df['caste4']
df['domlow_x_caste5'] = df['domlow'] * df['caste5']
""",
        "unit": [("household", "household", "noop", {})],
        "sample": [
            ("domlow", "villages with caste-dominance data", "filter_rows", {"notna": "domlow"}),
            ("all", "all households", "noop", {}),
            ("non_bihar", "non-Bihar", "filter_rows", {"col": "bihar", "values": [0]}),
            ("low_caste", "low-caste households", "filter_rows", {"col": "bfhcast", "values": [1]}),
        ],
        "sample2": [
            ("all", "no extra row filter", "noop", {}),
            ("land_gt0", "totland > 0", "filter_rows", {"expr": "totland > 0"}),
            ("cropinc_gt0", "cropinc > 0", "filter_rows", {"expr": "cropinc > 0"}),
            ("non_tenant", "non-tenant", "filter_rows", {"col": "tenant", "values": [0]}),
            ("borrow", "borrowed in past 12 months", "filter_rows", {"col": "borrow", "values": [1]}),
        ],
        "y": [
            ("cropinc", "crop income", "set_y", {"y": "cropinc"}),
            ("logcropinc", "log crop income", "set_y", {"y": "cropinc", "transform": "log"}),
            ("totinc", "total income", "set_y", {"y": "totinc"}),
            ("logtotinc", "log total income", "set_y", {"y": "totinc", "transform": "log"}),
            ("cropincacre", "crop income per acre", "set_y", {"y": "cropincacre"}),
        ],
        "transform": [
            ("raw", "raw", "transform_y", {"kind": "raw"}),
            ("log", "log", "transform_y", {"kind": "log"}),
            ("sqrt", "sqrt", "transform_y", {"kind": "sqrt"}),
            ("rank", "rank", "transform_y", {"kind": "rank"}),
        ],
        "treat": [
            ("domlow", "village dominated by low caste", "set_treatment_cont", {"col": "domlow"}),
            ("domlow_x_bfh", "low-caste x low-caste-dominated village", "set_treatment_cont", {"col": "domlow_x_bfh"}),
            ("domlow_x_caste4", "caste4 x low-caste-dominated village", "set_treatment_cont", {"col": "domlow_x_caste4"}),
            ("domlow_x_caste5", "caste5 x low-caste-dominated village", "set_treatment_cont", {"col": "domlow_x_caste5"}),
            ("bfhcast", "low-caste household", "set_treatment_cont", {"col": "bfhcast"}),
            ("pmixdominant", "mixed-caste dominance share", "set_treatment_cont", {"col": "pmixdominant"}),
        ],
        "controls": [
            ("none", "no controls", []),
            ("paper_table3", "paper Table 3 controls (reference)", ["literate", "totland", "bihar", "caste4", "caste5"]),
            ("hhelec_anymarket_bfhcast", "electricity + market + low caste", ["hhelec", "anymarket", "bfhcast"]),
            ("literate_anymarket_caste5", "literacy + market + caste5", ["literate", "anymarket", "caste5"]),
            ("logland_ownpump_anybus", "land + pump + bus", ["logland", "ownpump", "anybus"]),
            ("basic", "literacy + land + electricity", ["literate", "logland", "hhelec"]),
            ("village", "rainfall + groundwater development", ["rainfall", "gwdevelopment"]),
        ],
        "controls2": [
            ("none", "no extra controls", []),
            ("caste", "caste dummies", ["caste4", "caste5"]),
            ("village", "village infrastructure", ["hhelec", "anymarket", "anybank"]),
            ("agri", "agricultural controls", ["landirr", "ownpump", "totland"]),
        ],
        "fe": [("none", "no fixed effects", []), ("bihar", "Bihar FE", ["bihar"])],
        "cluster": [("village", "cluster by village")],
        "trim": [
            ("none", "no trim", "noop", {}),
            ("p1", "trim y at 1%", "trim", {"y": None, "p": 0.01}),
            ("p5", "trim y at 5%", "trim", {"y": None, "p": 0.05}),
        ],
        "winsor": [
            ("none", "no winsor", "noop", {}),
            ("p1", "winsor y at 1%", "winsorize", {"y": None, "p": 0.01}),
            ("p5", "winsor y at 5%", "winsorize", {"y": None, "p": 0.05}),
        ],
        "outlier": [
            ("none", "no outlier drop", "noop", {}),
            ("sd2", "drop |y-mean|>2sd", "filter_outlier", {"y": None, "k": 2.0}),
            ("sd3", "drop |y-mean|>3sd", "filter_outlier", {"y": None, "k": 3.0}),
        ],
        "se": [("cluster", "cluster-robust"), ("classical", "classical"), ("hc1", "HC1")],
        "estimator": [("cluster_d", "OLS t -> d", "use_cluster_d", {"sign": "absolute"})],
    },
    "jiang": {
        "paper_id": "Jiang_AmJourPoliSci_2018_Rjp9",
        "title": "Making Bureaucracy Work: Patronage Networks and Economic Development in China",
        "osf": "osf.io/8w4se",
        "claim": "City leaders with informal ties to incumbent provincial leaders deliver faster economic growth than those without.",
        "target_d": 0.3360,
        "data": ["jiang/replication.dta"],
        "read_code": """import numpy as np, pandas as pd
df = pd.read_stata(exp_dir + '/data/replication.dta', convert_categoricals=False)
df = df.sort_values(['cityid', 'year']).reset_index(drop=True)
df['lgdpidx'] = np.log(df['gdpidx'].clip(lower=1e-9))
df['gdp_lead2'] = df.groupby('cityid')['gdpidx'].shift(-2)
df['lgdp_lead2'] = df.groupby('cityid')['lgdpidx'].shift(-2)
df['growth_lead2'] = (df['gdp_lead2'] / df['gdpidx'] - 1) * 100
df['growth_log'] = (df['lgdp_lead2'] - df['lgdpidx']) * 100
df['growth_f2'] = (df['f2gdpidx'] / df['gdpidx'] - 1) * 100
df['n_obs_city'] = df.groupby('cityid')['gdpidx'].transform('count')
""",
        "unit": [("city_year", "city-year", "noop", {})],
        "sample": [
            ("all", "all city-years", "noop", {}),
            ("y2000", "2000 onwards", "filter_rows", {"expr": "year >= 2000"}),
            ("y2001", "2001 onwards", "filter_rows", {"expr": "year >= 2001"}),
            ("non_minor", "non-minority regions", "filter_rows", {"col": "minor_region", "values": [0]}),
            ("east", "eastern region", "filter_rows", {"col": "east_region", "values": [1]}),
        ],
        "sample2": [
            ("all", "no extra row filter", "noop", {}),
            ("year_le2011", "year <= 2011", "filter_rows", {"expr": "year <= 2011"}),
            ("year_ge2005", "year >= 2005", "filter_rows", {"expr": "year >= 2005"}),
            ("drop_farwest", "drop far-west", "filter_rows", {"col": "farwest", "values": [0]}),
            ("balanced_ge12", "cities with >= 12 obs", "filter_rows", {"expr": "n_obs_city >= 12"}),
        ],
        "y": [
            ("growth_lead2", "GDP growth at t+2", "set_y", {"y": "growth_lead2"}),
            ("growth_log", "log GDP growth at t+2", "set_y", {"y": "growth_log"}),
            ("growth_f2", "GDP index at t+2 over t", "set_y", {"y": "growth_f2"}),
            ("f2gdpidx", "GDP index at t+2", "set_y", {"y": "f2gdpidx"}),
            ("gdpidx", "GDP index level", "set_y", {"y": "gdpidx"}),
            ("lgdpidx", "log GDP index level", "set_y", {"y": "lgdpidx"}),
        ],
        "transform": [
            ("raw", "raw", "transform_y", {"kind": "raw"}),
            ("log", "log", "transform_y", {"kind": "log"}),
            ("sqrt", "sqrt", "transform_y", {"kind": "sqrt"}),
            ("rank", "rank", "transform_y", {"kind": "rank"}),
        ],
        "treat": [
            ("connected_sec", "connected to provincial secretary", "set_treatment_cont", {"col": "bin_mleader2currentsec"}),
            ("connected_sec2", "connected incl. promotions", "set_treatment_cont", {"col": "bin_mleader2currentsec2"}),
            ("connected_pleader", "connected to secretary or governor", "set_treatment_cont", {"col": "bin_mleader2pleader"}),
            ("connected_gvn", "connected to governor", "set_treatment_cont", {"col": "bin_mleader2currentgvn"}),
            ("mayor2currentsec", "mayor connected to provincial secretary", "set_treatment_cont", {"col": "mayor2currentsec"}),
        ],
        "controls": [
            ("none", "no controls", []),
            ("econ2", "log GDP + log population", ["loggdp", "logpop"]),
            ("econ4", "econ + gov size + night lights", ["loggdp", "logpop", "govsize", "logltavg"]),
            ("econ4_leader", "econ + leader tenure/age/education", ["loggdp", "logpop", "govsize", "logltavg", "msec_tenure", "mayor_tenure", "msec_age", "mayor_age"]),
            ("leader", "leader tenure/age/education", ["msec_tenure", "mayor_tenure", "msec_age", "mayor_age", "msec_edu", "mayor_edu"]),
            ("region", "region dummies", ["minor_region", "western", "east_region"]),
            ("county_exp", "county experience (reference proxy)", ["msec_county_exp"]),
        ],
        "controls2": [
            ("none", "no extra controls", []),
            ("leader", "leader background", ["msec_soe_exp", "mayor_soe_exp", "msec_univ_exp", "mayor_univ_exp"]),
            ("region", "regional dummies", ["minor_region", "western", "east_region"]),
            ("laws", "legislation count", ["laws"]),
        ],
        "fe": [
            ("none", "no fixed effects", []),
            ("year", "year FE", ["year"]),
            ("city", "city FE", ["cityid"]),
            ("year_city", "year + city FE", ["year", "cityid"]),
            ("province", "province FE", ["province"]),
        ],
        "cluster": [("city", "cluster by city", "cityid"), ("province", "cluster by province", "province")],
        "trim": [
            ("none", "no trim", "noop", {}),
            ("p1", "trim y at 1%", "trim", {"y": None, "p": 0.01}),
            ("p5", "trim y at 5%", "trim", {"y": None, "p": 0.05}),
        ],
        "winsor": [
            ("none", "no winsor", "noop", {}),
            ("p1", "winsor y at 1%", "winsorize", {"y": None, "p": 0.01}),
            ("p5", "winsor y at 5%", "winsorize", {"y": None, "p": 0.05}),
        ],
        "outlier": [
            ("none", "no outlier drop", "noop", {}),
            ("sd2", "drop |y-mean|>2sd", "filter_outlier", {"y": None, "k": 2.0}),
            ("sd3", "drop |y-mean|>3sd", "filter_outlier", {"y": None, "k": 3.0}),
        ],
        "se": [("cluster", "cluster-robust"), ("classical", "classical"), ("hc1", "HC1")],
        "estimator": [("cluster_d", "OLS t -> d", "use_cluster_d", {"sign": "absolute"})],
    },
    "andreoni": {
        "paper_id": "Andreoni_JournPoliEco_2017_La9x",
        "title": "Avoiding the Ask: A Field Experiment on Altruism and Charitable Giving",
        "osf": "osf.io/hd8e6",
        "claim": "The verbal ask increased giving (number of givers per 23-minute session).",
        "target_d": 1.8477,
        "data": ["andreoni/workingdata2009.csv"],
        "read_code": """import numpy as np, pandas as pd
df = pd.read_csv(exp_dir + '/data/workingdata2009.csv')
df['session_id'] = df['date'].astype(str) + '_' + df['session'].astype(str) + '_' + df['session_quarter'].astype(str)
df['block_id'] = df['date'].astype(str) + '_' + df['session'].astype(str)
df['ask'] = df['treatment'].astype(str).str.startswith('A').astype(float)
df['both_doors'] = df['treatment'].astype(str).str.endswith('2').astype(float)
df['ask_x_both'] = df['ask'] * df['both_doors']
""",
        "unit": [
            ("session_door", "session x door rows", "noop", {}),
            ("session_wide", "one row per session, sum both doors (AWAEK reference)", "collapse_sum",
             {"by": ["session_id"],
              "keep_first": ["date", "session", "session_quarter", "block_id", "ask", "both_doors", "ask_x_both"],
              "sum_cols": ["givers_total", "givers_in", "givers_out", "money", "traffic_total"]}),
        ],
        "sample": [
            ("all", "all sessions", "noop", {}),
            ("door1", "door 1", "filter_rows", {"col": "door_num", "values": [1]}),
            ("door2", "door 2", "filter_rows", {"col": "door_num", "values": [2]}),
        ],
        "sample2": [
            ("all", "no extra row filter", "noop", {}),
            ("quarter1", "first quarter of block", "filter_rows", {"expr": "session_quarter == 1"}),
            ("day_le2", "first two days", "filter_rows", {"expr": "date in ['6-Dec-09', '7-Dec-09']"}),
            ("session_le2", "sessions 1-2", "filter_rows", {"expr": "session <= 2"}),
        ],
        "y": [
            ("givers_total", "givers (door row)", "set_y", {"y": "givers_total"}),
            ("givers_in", "givers (in)", "set_y", {"y": "givers_in"}),
            ("givers_out", "givers (out)", "set_y", {"y": "givers_out"}),
            ("money", "money donated (door row)", "set_y", {"y": "money"}),
            ("givers_sum", "total givers (both doors)", "set_y", {"y": "givers_total"}),
            ("money_sum", "total money (both doors)", "set_y", {"y": "money"}),
        ],
        "transform": [
            ("raw", "raw", "transform_y", {"kind": "raw"}),
            ("log1p", "log1p", "transform_y", {"kind": "log1p"}),
            ("sqrt", "sqrt", "transform_y", {"kind": "sqrt"}),
            ("rank", "rank", "transform_y", {"kind": "rank"}),
        ],
        "treat": [
            ("ask", "verbal ask vs no ask", "set_treatment_cont", {"col": "ask"}),
            ("both_doors", "both doors occupied", "set_treatment_cont", {"col": "both_doors"}),
            ("ask_x_both", "ask x both doors", "set_treatment_cont", {"col": "ask_x_both"}),
        ],
        "controls": [
            ("none", "no controls", []),
            ("traffic_in", "traffic in", ["traffic_in"]),
            ("traffic_out", "traffic out", ["traffic_out"]),
            ("traffic_both", "traffic in + out", ["traffic_in", "traffic_out"]),
            ("traffic_total", "total traffic", ["traffic_total"]),
            ("awae_terms", "AWAEK interaction terms (reference)", ["both_doors", "ask_x_both"]),
            ("both_doors", "both doors", ["both_doors"]),
            ("traffic_awae", "traffic + interaction terms", ["traffic_total", "both_doors", "ask_x_both"]),
        ],
        "controls2": [
            ("none", "no extra controls", []),
            ("day", "day dummies", []),
            ("session", "session dummies", []),
            ("traffic", "traffic total", ["traffic_total"]),
        ],
        "fe": [
            ("none", "no fixed effects", []),
            ("date", "date FE", ["date"]),
            ("session", "session FE", ["session"]),
            ("block", "date x session FE", ["block_id"]),
            ("date_quarter", "date + session-quarter FE", ["date", "session_quarter"]),
        ],
        "cluster": [("block", "cluster by block", "block_id"), ("date", "cluster by date", "date")],
        "trim": [
            ("none", "no trim", "noop", {}),
            ("p1", "trim y at 1%", "trim", {"y": None, "p": 0.01}),
            ("p5", "trim y at 5%", "trim", {"y": None, "p": 0.05}),
        ],
        "winsor": [
            ("none", "no winsor", "noop", {}),
            ("p1", "winsor y at 1%", "winsorize", {"y": None, "p": 0.01}),
            ("p5", "winsor y at 5%", "winsorize", {"y": None, "p": 0.05}),
        ],
        "outlier": [
            ("none", "no outlier drop", "noop", {}),
            ("sd2", "drop |y-mean|>2sd", "filter_outlier", {"y": None, "k": 2.0}),
            ("sd3", "drop |y-mean|>3sd", "filter_outlier", {"y": None, "k": 3.0}),
        ],
        "se": [("cluster", "cluster-robust"), ("classical", "classical"), ("hc1", "HC1")],
        "estimator": [
            ("combo_d", "AWAEK ask + ask x both doors", "use_cluster_combo_d",
             {"terms": {"ask": 1.0, "ask_x_both": 1.0}}),
            ("cluster_d", "OLS t -> d", "use_cluster_d", {"sign": "absolute"}),
        ],
    },
}


def _choice(cid, label, func, params):
    return {"choice_id": cid, "label": label,
            "ops_module": "ops.exp4_ops", "func": func, "params": params}


def _fork(fid, step, option, label, choices):
    return {"fork_id": fid, "step": step, "option": option, "label": label,
            "optional": False, "ordered": False, "choices": choices}


def _simple_choices(spec_list, func_default=None):
    out = []
    for item in spec_list:
        cid, label, *rest = item
        if len(rest) == 2:
            func, params = rest
        elif func_default:
            func, params = func_default, rest[0]
        else:
            raise ValueError(item)
        out.append(_choice(cid, label, func, params))
    return out


def build(slug, spec):
    d = os.path.join(SPACES, slug)
    os.makedirs(os.path.join(d, "ops"), exist_ok=True)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    for rel in spec["data"]:
        shutil.copy2(os.path.join(DATA, rel), os.path.join(d, "data", os.path.basename(rel)))
    shutil.copy2(OPS_SRC, os.path.join(d, "ops", "exp4_ops.py"))
    open(os.path.join(d, "ops", "__init__.py"), "w").write("")

    forks = []
    forks.append(_fork("f_unit", "data_ingestion_assembly", "set_analysis_unit", "Analysis unit",
                       [_choice(c, l, f, p) for c, l, f, p in spec["unit"]]))
    forks.append(_fork("f_sample", "sample_selection", "filter_rows", "Sample selection",
                       [_choice(c, l, f, p) for c, l, f, p in spec["sample"]]))
    forks.append(_fork("f_sample2", "sample_selection", "filter_rows", "Additional sample window",
                       [_choice(c, l, f, p) for c, l, f, p in spec["sample2"]]))
    forks.append(_fork("f_y", "concept_operationalization", "map_hypothesis_concept", "Outcome concept",
                       [_choice(c, l, f, p) for c, l, f, p in spec["y"]]))
    forks.append(_fork("f_transform", "variable_construction", "monotonic_transform", "Outcome transform",
                       [_choice(c, l, f, p) for c, l, f, p in spec["transform"]]))
    forks.append(_fork("f_treat", "concept_operationalization", "treatment_exposure_mapping", "Treatment / exposure",
                       [_choice(c, l, f, p) for c, l, f, p in spec["treat"]]))
    forks.append(_fork("f_controls", "model_specification", "covariates_set", "Covariate set",
                       [_choice(c, l, "set_controls", {"controls": p}) for c, l, p in spec["controls"]]))
    forks.append(_fork("f_controls2", "model_specification", "covariates_set", "Additional covariates",
                       [_choice(c, l, "add_controls", {"controls": p}) for c, l, p in spec["controls2"]]))
    forks.append(_fork("f_fe", "model_specification", "fixed_effects", "Fixed effects",
                       [_choice(c, l, "set_fe", {"cols": p}) for c, l, p in spec["fe"]]))
    forks.append(_fork("f_cluster", "inference", "standard_errors", "Clustering dimension",
                       [_choice(it[0], it[1], "set_cluster", {"col": it[2] if len(it) > 2 else it[0]})
                        for it in spec["cluster"]]))
    forks.append(_fork("f_se", "inference", "standard_errors", "Standard error type",
                       [_choice(c, l, "set_se", {"se": c}) for c, l in spec["se"]]))
    forks.append(_fork("f_trim", "variable_construction", "winsorize_trim", "Trimming",
                       [_choice(c, l, f, p) for c, l, f, p in spec["trim"]]))
    forks.append(_fork("f_winsor", "variable_construction", "winsorize_trim", "Winsorizing",
                       [_choice(c, l, f, p) for c, l, f, p in spec["winsor"]]))
    forks.append(_fork("f_outlier", "diagnostic_sensitivity", "expand_existing_fork", "Outliers",
                       [_choice(c, l, f, p) for c, l, f, p in spec["outlier"]]))
    forks.append(_fork("f_estimator", "effect_size_reporting", "effect_size_conversion", "Effect size conversion",
                       [_choice(c, l, f, p) for c, l, f, p in spec["estimator"]]))

    tol = max(0.0005, 0.001 * abs(spec["target_d"]))
    raw = {
        "schema_version": "0.2.0",
        "paper": {"id": spec["paper_id"], "title": spec["title"], "osf": spec["osf"]},
        "hypothesis": {"text": spec["claim"], "direction": "none", "null_value": 0},
        "support_decision_rule": {
            "direction": "none", "null_value": 0, "test_type": "two_sided",
            "alpha": 0.05, "close_tol": tol, "require_significance": False,
            "success_metric": "tolerance", "objective_mode": "normalized_distance",
            "targets_by_estimator": {"custom": spec["target_d"]},
            "effect_measure": "cohens_d", "tolerance_relative": 0.001},
        "dataset": {"data_paths": [os.path.basename(x) for x in spec["data"]],
                    "read_code": spec["read_code"]},
        "experiment": {
            "mode": "blind",
            "agent_can_see_paper_text": False,
            "agent_can_see_paper_code": False,
            "success_definition": f"|d - {spec['target_d']}| <= {tol}",
            "instantiation_notes": "Reference path pinned from paper/Multi100 code; other menu options expanded to a 10^7-scale multiverse.",
        },
        "meta": {"strict_roles": False, "selection_seed": 20261024,
                 "source": f"Multi100 OSF {spec['osf']}"},
        "forks": forks,
    }
    path = os.path.join(d, "multi_space.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    loader = SpaceLoader(path)
    raw["meta"]["universe_count"] = int(loader.universe_count)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    with open(os.path.join(d, "targets.json"), "w", encoding="utf-8") as f:
        json.dump({"target_d": spec["target_d"], "tol": tol,
                   "tol_relative": 0.001, "effect_measure": "cohens_d",
                   "paper_id": spec["paper_id"]}, f, indent=2)
    return {"slug": slug, "universe_count": int(loader.universe_count),
            "forks": len(forks), "target_d": spec["target_d"], "tol": tol,
            "n_choices": [len(fk.choices) for fk in loader.forks]}


if __name__ == "__main__":
    rows = []
    for slug, spec in PAPERS.items():
        r = build(slug, spec)
        rows.append(r)
        print(r)
    with open(os.path.join(SPACES, "spaces_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
