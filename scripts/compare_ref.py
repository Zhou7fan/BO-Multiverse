# -*- coding: utf-8 -*-
"""compare_ref.py — find the closest match to the 4-digit wage-gap analysis
(-0.1504716) in the p17 traces and compare it with the reference (authors')
assignment decision by decision."""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.space_loader import SpaceLoader

loader = SpaceLoader("spaces/p17/multi_space.yaml")
ref = loader.reference_assignment() or {}
print("reference assignment:", json.dumps(ref, ensure_ascii=False))

target = -0.1504716
best = None
for p in glob.glob("results/p17_v2/p17/*/seed*/trace.csv"):
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            d = r.get("d")
            if d in (None, ""):
                continue
            d = float(d)
            dist = abs(d - target)
            if best is None or dist < best[1]:
                best = (r.get("assignment"), dist, d, p)
asg = json.loads(best[0])
print("closest found: d =", best[2], "dist =", best[1], "in", best[3])
print("found assignment:", json.dumps(asg, ensure_ascii=False))
agree = [k for k in ref if k in asg and asg.get(k) == ref[k]]
diff = [k for k in ref if k in asg and asg.get(k) != ref[k]]
missing = [k for k in ref if k not in asg]
print("reference decisions:", len(ref))
print("agree:", len(agree), agree)
print("differ:", len(diff), {k: (asg.get(k), ref[k]) for k in diff})
print("missing from found:", missing)
