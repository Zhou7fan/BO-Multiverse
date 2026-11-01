# -*- coding: utf-8 -*-
"""Build a pure-uniform candidate pool (no reference neighbourhood)."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.codec import SpaceCodec  # noqa: E402
from core.references import REF  # noqa: E402
from engine.space_loader import SpaceLoader  # noqa: E402


def build(slug, space, n=500000, seed=20261024):
    loader = SpaceLoader(space)
    codec = SpaceCodec(loader)
    rng = np.random.default_rng(seed + 7)
    X = np.empty((n, codec.dim), dtype=np.int8)
    for f, k in enumerate(codec.n_vertices):
        X[:, f] = rng.integers(0, int(k), size=n, dtype=np.int8)
    path = os.path.join(os.path.dirname(space), "pool_uniform.npz")
    np.savez_compressed(path, X=X, seed=np.asarray([seed]), n=np.asarray([n]))
    return path


if __name__ == "__main__":
    for slug in REF:
        p = build(slug, os.path.join(EXP4, "spaces", slug, "multi_space.yaml"))
        print(slug, p)
