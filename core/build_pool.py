# -*- coding: utf-8 -*-
"""Build a candidate pool: uniform sample + reference neighbourhood.

The multiverse space itself is the full 10^7-scale product.  For BO/random we
need a finite candidate set; we use a uniform sample plus a local sample around
the code-pinned reference so the known paper path is reachable by every method.
Both baselines draw from the same pool.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.codec import SpaceCodec, pool_path  # noqa: E402
from core.references import REF  # noqa: E402
from engine.space_loader import SpaceLoader  # noqa: E402


def build(slug: str, space: str, n_uniform=500000, n_one=0, n_two=20000,
          n_three=10000, seed=20261024, overwrite=True):
    loader = SpaceLoader(space)
    codec = SpaceCodec(loader)
    rng = np.random.default_rng(seed)
    ref = REF[slug]
    ref_vec = codec.to_index(ref)
    parts = [ref_vec.reshape(1, -1)]
    # one-fork perturbations
    for f, k in enumerate(codec.n_vertices):
        for c in range(int(k)):
            if c == ref_vec[f]:
                continue
            v = ref_vec.copy()
            v[f] = c
            parts.append(v.reshape(1, -1))
    # two-fork perturbations
    for _ in range(n_two):
        v = ref_vec.copy()
        f1, f2 = rng.choice(codec.dim, size=2, replace=False)
        v[f1] = rng.integers(0, int(codec.n_vertices[f1]))
        v[f2] = rng.integers(0, int(codec.n_vertices[f2]))
        parts.append(v.reshape(1, -1))
    # three-fork perturbations
    for _ in range(n_three):
        v = ref_vec.copy()
        fs = rng.choice(codec.dim, size=3, replace=False)
        for f in fs:
            v[f] = rng.integers(0, int(codec.n_vertices[f]))
        parts.append(v.reshape(1, -1))
    local = np.vstack(parts).astype(np.int8)

    uniform = np.empty((n_uniform, codec.dim), dtype=np.int8)
    for f, k in enumerate(codec.n_vertices):
        uniform[:, f] = rng.integers(0, int(k), size=n_uniform, dtype=np.int8)

    X = np.vstack([local, uniform])
    X = np.unique(X, axis=0)
    # ensure reference is present
    if not np.any(np.all(X == ref_vec.astype(np.int8), axis=1)):
        X = np.vstack([ref_vec.astype(np.int8), X])
    np.savez_compressed(pool_path(space), X=X, n_uniform=np.asarray([n_uniform]),
                        n_local=np.asarray([len(local)]), seed=np.asarray([seed]),
                        ref=np.asarray(ref_vec, dtype=np.int16))
    return {"slug": slug, "pool_size": int(len(X)), "local": int(len(local)),
            "uniform": int(n_uniform), "path": pool_path(space)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", default=os.path.join(EXP4, "spaces"))
    ap.add_argument("--uniform", type=int, default=500000)
    ap.add_argument("--two", type=int, default=20000)
    ap.add_argument("--three", type=int, default=10000)
    args = ap.parse_args()
    for slug in REF:
        space = os.path.join(args.spaces, slug, "multi_space.yaml")
        print(build(slug, space, n_uniform=args.uniform, n_two=args.two,
                    n_three=args.three, overwrite=True))


if __name__ == "__main__":
    main()
