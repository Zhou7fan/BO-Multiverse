# -*- coding: utf-8 -*-
"""Choice-index codec and candidate pool for experiment 4."""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.space_loader import SpaceLoader


class SpaceCodec:
    def __init__(self, loader: SpaceLoader):
        self.loader = loader
        self.forks = loader.forks
        self.choices = [[c for c in fk.choices if not c.is_skip] for fk in self.forks]
        self.index_of = [{c.choice_id: i for i, c in enumerate(cs)} for cs in self.choices]
        self.n_vertices = np.asarray([len(cs) for cs in self.choices], dtype=int)

    @property
    def dim(self):
        return len(self.forks)

    def to_index(self, assignment: Dict[str, str]) -> np.ndarray:
        out = np.zeros(self.dim, dtype=int)
        for f, fk in enumerate(self.forks):
            cid = assignment.get(fk.fork_id)
            if cid not in self.index_of[f]:
                raise KeyError(f"{fk.fork_id}:{cid}")
            out[f] = self.index_of[f][cid]
        return out

    def to_string(self, vec) -> Dict[str, str]:
        vec = np.asarray(vec, dtype=int).reshape(-1)
        return {fk.fork_id: self.choices[f][int(vec[f])].choice_id
                for f, fk in enumerate(self.forks)}

    decode = to_string

    def onehot(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=int)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        out = np.zeros((n, int(self.n_vertices.sum())), dtype=float)
        off = 0
        for f, k in enumerate(self.n_vertices):
            out[np.arange(n), off + X[:, f]] = 1.0
            off += int(k)
        return out

    def ordinal(self, X) -> np.ndarray:
        """Scaled ordinal encoding: choice_index / (n_choices - 1)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        denom = np.maximum(self.n_vertices - 1, 1).astype(float)
        return X / denom


def pool_path(space_yaml: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(space_yaml)), "pool.npz")


def build_pool(space_yaml: str, n: int = 200000, seed: int = 20261024,
               overwrite: bool = False) -> str:
    path = pool_path(space_yaml)
    if os.path.exists(path) and not overwrite:
        return path
    loader = SpaceLoader(space_yaml)
    codec = SpaceCodec(loader)
    rng = np.random.default_rng(seed)
    X = np.empty((n, codec.dim), dtype=np.int8)
    for f, k in enumerate(codec.n_vertices):
        X[:, f] = rng.integers(0, int(k), size=n, dtype=np.int8)
    np.savez_compressed(path, X=X, seed=np.asarray([seed]), n=np.asarray([n]))
    return path


def load_pool(space_yaml: str, path: str = None):
    path = path or pool_path(space_yaml)
    if not os.path.exists(path):
        path = build_pool(space_yaml)
    z = np.load(path)
    return z["X"].astype(np.int16)
