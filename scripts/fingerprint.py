# -*- coding: utf-8 -*-
"""fingerprint.py — decision fingerprints from all traces + dense draws.

Per paper: consistent set S at search tolerance and at reporting precision,
per-fork option shares in S vs the uniform pool (the space as a whole),
classified fixed / constrained / free. Outputs JSON + a tex-style table CSV.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.space_loader import SpaceLoader  # noqa: E402
from core.codec import SpaceCodec, load_pool  # noqa: E402


def _chi2_p(observed, expected):
    from scipy.stats import chisquare
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    keep = exp >= 5.0
    if keep.sum() < 2:
        return None
    obs_k = np.concatenate([obs[keep], [obs[~keep].sum()]])
    exp_k = np.concatenate([exp[keep], [exp[~keep].sum()]])
    if exp_k[-1] < 5.0 and (~keep).any():
        return None
    if exp_k[-1] < 5.0:
        obs_k, exp_k = obs[keep], exp[keep]
    try:
        _, p = chisquare(obs_k, f_exp=exp_k)
        return float(p)
    except Exception:
        return None


def classify(loader, codec, pool_X, assignments):
    """assignments: list of assignment dicts (all within tolerance)."""
    n = len(assignments)
    pool_shares = []
    for f in range(codec.dim):
        counts = np.bincount(pool_X[:, f], minlength=int(codec.n_vertices[f]))
        pool_shares.append(counts.astype(float) / max(counts.sum(), 1))
    out = []
    for f, fk in enumerate(loader.forks):
        idx = codec.index_of[f]
        s_counts = np.zeros(int(codec.n_vertices[f]), dtype=float)
        for a in assignments:
            cid = a.get(fk.fork_id)
            if cid in idx:
                s_counts[idx[cid]] += 1
        used = np.flatnonzero(s_counts > 0).tolist()
        used_cids = [codec.choices[f][u].choice_id for u in used]
        if n == 0:
            status = "no_data"
            required = None
        elif len(used) == 1:
            status = "fixed"
            required = used_cids[0]
        else:
            p = _chi2_p(s_counts, pool_shares[f] * n)
            if p is None:
                # fallback: any option present in the space but absent in S?
                status = "constrained" if len(used) < int(codec.n_vertices[f]) else "free"
            else:
                status = "constrained" if p < 0.05 else "free"
            required = None
        out.append({
            "fork": fk.fork_id,
            "label": fk.label,
            "status": status,
            "required": required,
            "options_in_S": used_cids,
            "n_options": int(codec.n_vertices[f]),
            "share_in_S": {codec.choices[f][u].choice_id: float(s_counts[u] / n)
                           for u in used} if n else {},
            "share_in_space": {codec.choices[f][u].choice_id: float(pool_shares[f][u])
                               for u in range(int(codec.n_vertices[f]))},
        })
    return out


def collect(space, trace_roots, dense_csvs, prec):
    loader = SpaceLoader(space)
    codec = SpaceCodec(loader)
    pool_X = load_pool(space)
    tpath = os.path.join(os.path.dirname(os.path.abspath(space)), "targets.json")
    with open(tpath, encoding="utf-8") as fh:
        targets = json.load(fh)
    tgt = float(targets["target_d"])
    tol = float(targets["tol"])

    seen = {}
    n_rows = 0
    for root in trace_roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "trace.csv"), recursive=True)):
            with open(p, encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    d = r.get("d")
                    if d in (None, ""):
                        continue
                    d = float(d)
                    n_rows += 1
                    a = r.get("assignment") or ""
                    if not a:
                        continue
                    try:
                        asg = json.loads(a)
                    except Exception:
                        continue
                    uid = loader.universe_id(asg)
                    if uid not in seen:
                        seen[uid] = (d, asg)
    for p in dense_csvs:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("feasible") != "1" or r.get("d") in (None, ""):
                    continue
                d = float(r["d"])
                n_rows += 1
                try:
                    asg = json.loads(r["assignment"])
                except Exception:
                    continue
                uid = loader.universe_id(asg)
                if uid not in seen:
                    seen[uid] = (d, asg)

    s_tol = {u: (d, a) for u, (d, a) in seen.items() if abs(d - tgt) <= tol}
    s_prec = {u: (d, a) for u, (d, a) in seen.items() if abs(d - tgt) <= prec}
    f_tol = classify(loader, codec, pool_X, [a for (d, a) in s_tol.values()])
    f_prec = classify(loader, codec, pool_X, [a for (d, a) in s_prec.values()])
    return {
        "paper": os.path.basename(os.path.dirname(os.path.abspath(space))),
        "target": tgt, "tol": tol, "precision": prec,
        "n_rows": n_rows, "n_distinct": len(seen),
        "n_tol": len(s_tol), "n_prec": len(s_prec),
        "forks_tol": f_tol, "forks_prec": f_prec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--precision", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", default="", help="comma-separated trace dirs")
    ap.add_argument("--dense", default="", help="comma-separated dense draws csv")
    args = ap.parse_args()
    traces = [t for t in args.traces.split(",") if t]
    dense = [d for d in args.dense.split(",") if d]
    res = collect(args.space, traces, dense, args.precision)
    os.makedirs(args.out, exist_ok=True)
    slug = res["paper"]
    with open(os.path.join(args.out, f"{slug}_fingerprint.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in
                      ("forks_tol", "forks_prec")}, ensure_ascii=False, indent=2))
    print("--- forks at tol ---")
    for f in res["forks_tol"]:
        if f["status"] != "free":
            print(f"{f['status']:11s} {f['fork']:20s} required={f['required']} "
                  f"in_S={f['options_in_S']}")


if __name__ == "__main__":
    main()
