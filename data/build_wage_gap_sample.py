#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the wage-gap sample for the paper "Estimating wage disparities
using foundation models" (Vafa, Athey and Blei 2025).

The paper's analysis file is the PSID 1990-2019 extract of Blau and Kahn
(2017), "The Gender Wage Gap: Extent, Trends, and Explanations", prepared by
the GSB DBI career-wage-gaps replication package
(https://github.com/gsbdbi/career-wage-gaps). The official PSID files cannot
be redistributed, so this script documents the construction, points at the
canonical extract code, and verifies that the resulting CSV carries every
column the search space (`spaces/wage_gap/ops/wage_ops.py`) consumes.

Usage:
    python data/build_wage_gap_sample.py --psid-dir /path/to/official/PSID \
        --out data/psid_blau_kahn_2017.csv
    python data/build_wage_gap_sample.py --verify data/psid_blau_kahn_2017.csv

Steps performed (matching the DBI package):
    1. Merge the PSID family files (FAM1981-FAM2019) into a person-year panel
       with the paper's sample flags: non-farm, non-military, non-SMSA
       exclusions as in Blau-Kahn, wage workers aged 18-64, 26+ weeks and
       full-time hours restrictions, and the hourly-wage construction
       (annual labor earnings / annual hours, top-coded as in the paper).
    2. Build the covariates the space uses: female, age, years of schooling
       (sch), potential experience (potexp), full-time experience (yrsftexp),
       BA / advanced-degree indicators, race and region dummies, SMSA, the
       PSID family weight (famwgt), and the industry / occupation dummies.
    3. Validate the extract against the required schema and print a report.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

# Schema consumed by spaces/wage_gap/ops/wage_ops.py (core columns and the
# dummy blocks it references). The extract must carry all of them.
REQUIRED_COLUMNS = [
    "female", "age", "sch", "potexp", "yrsftexp", "wave", "famwgt",
    "ba", "adv", "black", "hisp", "smsa", "south", "northeast",
    "northcentral", "hrwage", "realhrwage", "lnwage", "othrace", "unjob",
    "wtrgov",
]

INDUSTRY_DUMMIES = [
    "miningconstruction", "durables", "nondurables", "Transport", "Utilities",
    "Communications", "retailtrade", "wholesaletrade", "finance", "SocArtOther",
    "hotelsrestaurants", "Medical", "Education", "professional", "publicadmin",
]

OCCUPATION_DUMMIES = [
    "business", "financialop", "computer", "architect", "scientist",
    "socialworker", "postseceduc", "legaleduc", "artist", "lawyerphysician",
    "healthcare", "healthsupport", "protective", "foodcare", "building",
    "sales", "officeadmin", "constructextractinstall", "production", "transport",
]

DBI_PACKAGE_URL = "https://github.com/gsbdbi/career-wage-gaps"


def build(psid_dir: str, out: str) -> None:
    """Run the canonical extract and validate the output.

    The canonical construction is the DBI replication package; clone it and
    run its `psid_blau_kahn_2017` build target, which produces
    `data/psid_blau_kahn_2017.csv` from the official PSID family files.
    """
    if not os.path.isdir(psid_dir):
        sys.exit(f"PSID directory not found: {psid_dir}")
    fam = [f for f in os.listdir(psid_dir)
           if f.startswith("FAM") and f.endswith((".dta", ".csv"))]
    if not fam:
        sys.exit(
            "No PSID family files (FAM*.dta/FAM*.csv) found in the given "
            "directory. Download them from https://psidonline.isr.umich.edu "
            "(registration required)."
        )
    print(f"Found {len(fam)} PSID family files, e.g. {sorted(fam)[0]}.")
    print(
        "The canonical extract is the GSB DBI career-wage-gaps replication "
        f"package ({DBI_PACKAGE_URL}). Run its Blau-Kahn build script against "
        "the files above; it writes data/psid_blau_kahn_2017.csv."
    )
    if not os.path.exists(out):
        sys.exit(
            f"Extract not found at {out}. Run the DBI package build first, "
            "then re-run this script with --verify."
        )
    verify(out)


def verify(path: str) -> None:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    missing_dummies = ([c for c in INDUSTRY_DUMMIES if c not in df.columns]
                       + [c for c in OCCUPATION_DUMMIES if c not in df.columns])
    print(f"rows: {len(df):,}; columns: {df.shape[1]}")
    print(f"waves: {sorted(df['wave'].dropna().unique().tolist())}"
          if "wave" in df.columns else "waves: (no wave column)")
    if missing or missing_dummies:
        print("SCHEMA CHECK FAILED")
        if missing:
            print("  missing core columns:", ", ".join(missing))
        if missing_dummies:
            print("  missing dummy columns:", ", ".join(sorted(set(missing_dummies))))
        sys.exit(1)
    print("SCHEMA CHECK PASSED: all columns required by spaces/wage_gap/ops are present.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psid-dir", default=None,
                    help="directory with the official PSID family files")
    ap.add_argument("--out", default="data/psid_blau_kahn_2017.csv")
    ap.add_argument("--verify", default=None, help="only verify an existing extract")
    args = ap.parse_args()
    if args.verify:
        verify(args.verify)
    elif args.psid_dir:
        build(args.psid_dir, args.out)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
