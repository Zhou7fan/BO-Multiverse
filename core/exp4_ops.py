# -*- coding: utf-8 -*-
"""Experiment-4 generic operations.

Every function is paper-agnostic: column names, filters, FE and cluster keys all
come from the generated multi_space.yaml.  The terminal estimator is a
cluster-robust OLS t -> Cohen's d conversion following Multi100:
    d = 2 * t / sqrt(df),  df = G - 1 for cluster-robust inference.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# basic data operations
# --------------------------------------------------------------------------- #
def noop(df, ctx, **kwargs):
    return df


def filter_rows(df, ctx, col=None, values=None, expr=None, notna=None, dropna=None):
    out = df
    if col is not None and values is not None:
        vals = values if isinstance(values, (list, tuple, set)) else [values]
        out = out[out[col].isin(vals)]
    if expr:
        out = out.query(expr)
    if notna is not None:
        cols = notna if isinstance(notna, (list, tuple)) else [notna]
        out = out.dropna(subset=[c for c in cols if c in out.columns])
    if dropna is not None:
        cols = dropna if isinstance(dropna, (list, tuple)) else [dropna]
        out = out.dropna(subset=[c for c in cols if c in out.columns])
    return out.copy()


def drop_missing(df, ctx, cols, how="drop"):
    cols = [c for c in cols if c in df.columns]
    if how == "drop":
        return df.dropna(subset=cols).copy()
    out = df.copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        if how == "mean":
            out[c] = out[c].fillna(out[c].mean())
        elif how == "median":
            out[c] = out[c].fillna(out[c].median())
        elif how == "zero":
            out[c] = out[c].fillna(0.0)
    return out


def set_y(df, ctx, y, transform="raw", new=None):
    out = df
    col = y
    if transform and transform != "raw":
        col = new or f"__y_{transform}"
        out = out.copy()
        v = out[y].astype(float)
        if transform == "log":
            out[col] = np.log(v.where(v > 0))
        elif transform == "log1p":
            out[col] = np.log1p(v)
        elif transform == "sqrt":
            out[col] = np.sqrt(v.clip(lower=0))
        elif transform == "z":
            sd = v.std(ddof=1)
            out[col] = (v - v.mean()) / sd if np.isfinite(sd) and sd > 0 else v
        elif transform == "rank":
            out[col] = v.rank(pct=True)
        elif transform == "neg":
            out[col] = -v
        else:
            raise ValueError(f"unknown transform {transform}")
    ctx.spec["y"] = col
    return out


def set_treatment_binary(df, ctx, col, value, new="_treat"):
    out = df.copy()
    out[new] = (out[col] == value).astype(float)
    ctx.spec["treatment"] = new
    return out


def set_treatment_cont(df, ctx, col, new=None):
    ctx.spec["treatment"] = new or col
    return df


def set_controls(df, ctx, controls):
    ctx.spec["controls"] = list(controls or [])
    return df


def set_fe(df, ctx, cols):
    ctx.spec["fe_dummies"] = list(cols or [])
    return df


def set_cluster(df, ctx, col):
    ctx.spec["cluster"] = col
    ctx.spec["se"] = "cluster" if col else "classical"
    return df


def set_se(df, ctx, se, cluster=None):
    ctx.spec["se"] = se
    if cluster is not None:
        ctx.spec["cluster"] = cluster
    return df


def set_min_n(df, ctx, n=30):
    ctx.spec["min_n"] = int(n)
    return df


def transform(df, ctx, cols, kind="raw"):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        v = out[c].astype(float)
        if kind == "raw":
            pass
        elif kind == "log":
            out[c] = np.log(v.where(v > 0))
        elif kind == "log1p":
            out[c] = np.log1p(v)
        elif kind == "sqrt":
            out[c] = np.sqrt(v.clip(lower=0))
        elif kind == "z":
            sd = v.std(ddof=1)
            out[c] = (v - v.mean()) / sd if np.isfinite(sd) and sd > 0 else v
        elif kind == "rank":
            out[c] = v.rank(pct=True)
        else:
            raise ValueError(f"unknown transform {kind}")
    return out


def winsorize(df, ctx, y=None, p=0.01):
    y = y or ctx.spec.get("y")
    out = df.copy()
    if y in out.columns and p and p > 0:
        lo, hi = out[y].quantile(p), out[y].quantile(1 - p)
        out[y] = out[y].clip(lo, hi)
    return out


def trim(df, ctx, y=None, p=0.0):
    y = y or ctx.spec.get("y")
    out = df.copy()
    if y in out.columns and p and p > 0:
        lo, hi = out[y].quantile(p), out[y].quantile(1 - p)
        out = out[(out[y] >= lo) & (out[y] <= hi)].copy()
    return out


def filter_outlier(df, ctx, y=None, group=None, k=3.0):
    y = y or ctx.spec.get("y")
    out = df.copy()
    if y not in out.columns:
        return out
    if group and group in out.columns:
        keep = pd.Series(True, index=out.index)
        for _, idx in out.groupby(group).groups.items():
            v = out.loc[idx, y]
            mu, sd = v.mean(), v.std(ddof=1)
            if np.isfinite(sd) and sd > 0:
                keep.loc[idx] = (v - mu).abs() <= k * sd
        return out.loc[keep].copy()
    v = out[y]
    mu, sd = v.mean(), v.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return out
    return out.loc[(v - mu).abs() <= k * sd].copy()


def derive_interaction(df, ctx, a, b, new="_inter"):
    out = df.copy()
    out[new] = out[a].astype(float) * out[b].astype(float)
    return out


def derive_dummy(df, ctx, col, value, new):
    out = df.copy()
    out[new] = (out[col] == value).astype(float)
    return out


def derive_log(df, ctx, col, new=None, shift=0.0):
    out = df.copy()
    out[new or f"log_{col}"] = np.log(out[col].astype(float) + shift)
    return out


def derive_group_mean(df, ctx, col, group, new=None):
    out = df.copy()
    out[new or f"{col}_gm"] = out.groupby(group)[col].transform("mean")
    return out


def collapse(df, ctx, by, keep_first=None, mean_cols=None):
    """Collapse to one row per `by` group; numeric columns are averaged."""
    by = list(by)
    keep_first = list(keep_first or [])
    out = df.copy()
    num = [c for c in out.columns if c not in by and pd.api.types.is_numeric_dtype(out[c])]
    if mean_cols is not None:
        num = [c for c in num if c in mean_cols]
    agg = {c: "mean" for c in num}
    for c in keep_first:
        if c not in by:
            agg[c] = "first"
    grouped = out.groupby(by, as_index=False, dropna=False).agg(agg)
    for c in by:
        if c not in grouped.columns:
            grouped[c] = out[c].iloc[0] if len(out) else np.nan
    return grouped


def transform_y(df, ctx, kind="raw"):
    y = ctx.spec.get("y")
    if not y or y not in df.columns:
        return df
    out = df.copy()
    new = f"__y_{kind}"
    v = out[y].astype(float)
    if kind == "raw":
        return out
    if kind == "log":
        out[new] = np.log(v.where(v > 0))
    elif kind == "log1p":
        out[new] = np.log1p(v)
    elif kind == "sqrt":
        out[new] = np.sqrt(v.clip(lower=0))
    elif kind == "z":
        sd = v.std(ddof=1)
        out[new] = (v - v.mean()) / sd if np.isfinite(sd) and sd > 0 else v
    elif kind == "rank":
        out[new] = v.rank(pct=True)
    else:
        raise ValueError(f"unknown transform {kind}")
    ctx.spec["y"] = new
    return out


def add_controls(df, ctx, controls):
    cur = list(ctx.spec.get("controls") or [])
    for c in controls or []:
        if c not in cur:
            cur.append(c)
    ctx.spec["controls"] = cur
    return df


def lag_by(df, ctx, col, group, lag=1, new=None):
    out = df.copy()
    out[new or f"{col}_lag{lag}"] = out.sort_values(group).groupby(group)[col].shift(lag)
    return out


def collapse_sum(df, ctx, by, keep_first=None, sum_cols=None):
    """Collapse to one row per `by` group, summing `sum_cols` (e.g. both doors)."""
    by = list(by)
    keep_first = list(keep_first or [])
    sum_cols = list(sum_cols or [])
    out = df.copy()
    agg = {}
    for c in sum_cols:
        if c in out.columns:
            agg[c] = "sum"
    for c in keep_first:
        if c not in by and c in out.columns:
            agg[c] = "first"
    if not agg:
        return out.groupby(by, as_index=False, dropna=False).first()
    grouped = out.groupby(by, as_index=False, dropna=False).agg(agg)
    return grouped


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def _cluster_meat(X, e, groups):
    k = X.shape[1]
    meat = np.zeros((k, k), dtype=float)
    for g in pd.unique(groups):
        m = groups == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    return meat


def _design(df, spec):
    ycol = spec["y"]
    tcol = spec["treatment"]
    controls = [c for c in (spec.get("controls") or []) if c in df.columns]
    fe_cols = [c for c in (spec.get("fe_dummies") or []) if c in df.columns]
    cols = [ycol, tcol] + controls + fe_cols
    cluster = spec.get("cluster")
    if cluster:
        cols.append(cluster)
    cols = list(dict.fromkeys(cols))
    sub = df[cols].dropna().copy()
    if len(sub) == 0:
        raise ValueError("empty_sample")
    y = sub[ycol].to_numpy(float)
    t = sub[tcol].to_numpy(float)
    parts = [t.reshape(-1, 1)]
    names = [tcol]
    kept_controls = []
    for c in controls:
        v = sub[c].to_numpy(float)
        if np.nanstd(v) > 1e-12:
            parts.append(v.reshape(-1, 1))
            names.append(c)
            kept_controls.append(c)
    for c in fe_cols:
        d = pd.get_dummies(sub[c], prefix=c, drop_first=True).astype(float)
        if d.shape[1]:
            d = d.loc[:, d.std(axis=0).to_numpy() > 1e-12]
            if d.shape[1]:
                parts.append(d.to_numpy(float))
                names.extend(list(d.columns))
    parts.append(np.ones((len(sub), 1)))
    names.append("_const")
    X = np.hstack(parts)
    return X, y, sub, kept_controls, names


def _fit_cluster(df, spec):
    X, y, sub, kept, names = _design(df, spec)
    n, k = X.shape
    min_n = int(spec.get("min_n", 30) or 30)
    if n < min_n:
        raise ValueError(f"small_n:{n}<{min_n}")
    if np.nanstd(y) <= 1e-12:
        raise ValueError("constant_y")
    tcol = spec["treatment"]
    if sub[tcol].nunique() < 2:
        raise ValueError("treatment_no_variation")
    rank = int(np.linalg.matrix_rank(X))
    if rank < 2:
        raise ValueError("rank_lt_2")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    se_type = (spec.get("se") or "classical").lower()
    cluster = spec.get("cluster")
    if se_type == "cluster":
        if not cluster or cluster not in sub.columns:
            raise ValueError("cluster_missing")
        groups = sub[cluster].astype(str).to_numpy()
        G = len(pd.unique(groups))
        if G < 2:
            raise ValueError(f"cluster_G:{G}")
        meat = _cluster_meat(X, e, groups)
        c = G / (G - 1) * (n - 1) / max(n - rank, 1)
        V = c * XtX_inv @ meat @ XtX_inv
        dfree = float(G - 1)
        note = f"exp4_cluster_d:G={G},k={rank}"
    elif se_type in ("hc0", "hc1", "hc2", "hc3", "robust"):
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        h = np.clip(h, 0.0, 1.0 - 1e-12)
        if se_type == "hc2":
            e2 = e * e / (1.0 - h)
        elif se_type == "hc3":
            e2 = e * e / ((1.0 - h) ** 2)
        else:
            e2 = e * e
        meat = (X * e2[:, None]).T @ X
        if se_type == "hc1":
            meat = meat * (n / max(n - rank, 1))
        V = XtX_inv @ meat @ XtX_inv
        dfree = float(n - rank)
        note = f"exp4_{se_type}_d:k={rank}"
    else:
        sigma2 = float(e @ e) / max(n - rank, 1)
        V = sigma2 * XtX_inv
        dfree = float(n - rank)
        note = f"exp4_ols_d:k={rank}"
    return dict(beta=beta, V=V, names=names, sub=sub, n=n, rank=rank,
                dfree=dfree, note=note, kept=kept)


def _pack_d(est, se, dfree, n, note, spec):
    if not np.isfinite(se) or se <= 0:
        raise ValueError("non_convergent_or_singular")
    t = float(est / se)
    d = float(2.0 * t / math.sqrt(dfree)) if dfree > 1 else float("nan")
    if not np.isfinite(d):
        raise ValueError("non_finite_d")
    if (spec.get("sign") or "absolute").lower() in ("absolute", "abs", "magnitude"):
        d = abs(d)
        t_rep = abs(t)
    else:
        t_rep = t
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t_rep), dfree))
    except Exception:
        p = None
    se_d = float(2.0 / math.sqrt(dfree)) if dfree > 2 else None
    return {"estimate": float(d), "std_error": se_d, "p_value": p,
            "ci_low": (d - 1.96 * se_d) if se_d else None,
            "ci_high": (d + 1.96 * se_d) if se_d else None,
            "n_obs": int(n), "effect_measure": "cohens_d", "invalid": 0,
            "invalid_reason": "", "note": note, "model_size": 0,
            "t_stat": float(t_rep), "df": float(dfree)}


def estimate_cluster_d(df, spec, ctx=None):
    out = {"estimate": None, "std_error": None, "p_value": None,
           "ci_low": None, "ci_high": None, "n_obs": None, "effect_measure": "cohens_d",
           "invalid": 0, "invalid_reason": "", "note": "exp4_cluster_d", "model_size": 0,
           "t_stat": None, "df": None}
    try:
        fit = _fit_cluster(df, spec)
        out.update(_pack_d(float(fit["beta"][0]), float(np.sqrt(max(fit["V"][0, 0], 0))),
                           fit["dfree"], fit["n"], fit["note"], spec))
        out["model_size"] = len(fit["kept"])
        return out
    except Exception as exc:
        out.update(invalid=1, invalid_reason=f"estimate_exception:{type(exc).__name__}:{exc}")
        return out


def estimate_cluster_combo_d(df, spec, ctx=None):
    out = {"estimate": None, "std_error": None, "p_value": None,
           "ci_low": None, "ci_high": None, "n_obs": None, "effect_measure": "cohens_d",
           "invalid": 0, "invalid_reason": "", "note": "exp4_cluster_combo_d",
           "model_size": 0, "t_stat": None, "df": None}
    try:
        fit = _fit_cluster(df, spec)
        terms = spec.get("linear_combo") or {}
        L = np.zeros(fit["beta"].shape[0], dtype=float)
        for name, w in terms.items():
            if name not in fit["names"]:
                raise ValueError(f"combo_term_missing:{name}")
            L[fit["names"].index(name)] = float(w)
        est = float(L @ fit["beta"])
        se = float(np.sqrt(max(L @ fit["V"] @ L, 0)))
        out.update(_pack_d(est, se, fit["dfree"], fit["n"], fit["note"], spec))
        out["model_size"] = len(fit["kept"])
        return out
    except Exception as exc:
        out.update(invalid=1, invalid_reason=f"combo_exception:{type(exc).__name__}:{exc}")
        return out


def use_cluster_combo_d(df, ctx, terms):
    ctx.spec.update({
        "estimator": "custom",
        "custom_func": "ops.exp4_ops:estimate_cluster_combo_d",
        "linear_combo": dict(terms or {}),
        "sign": "absolute",
        "effect_measure": "cohens_d",
        "min_n": int(ctx.spec.get("min_n", 30) or 30),
    })
    return df


def estimate_corr_d(df, spec, ctx=None):
    out = {"estimate": None, "std_error": None, "p_value": None,
           "ci_low": None, "ci_high": None, "n_obs": None, "effect_measure": "cohens_d",
           "invalid": 0, "invalid_reason": "", "note": "exp4_corr_d", "model_size": 0}
    try:
        ycol = spec["y"]
        xcol = spec.get("x") or spec["treatment"]
        sub = df[[xcol, ycol]].dropna()
        n = len(sub)
        out["n_obs"] = int(n)
        if n < int(spec.get("min_n", 10) or 10):
            out.update(invalid=1, invalid_reason=f"small_n:{n}")
            return out
        r = float(np.corrcoef(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float))[0, 1])
        if not np.isfinite(r) or abs(r) >= 1:
            out.update(invalid=1, invalid_reason="bad_correlation")
            return out
        d = float(2.0 * r / math.sqrt(1.0 - r * r))
        dfree = float(n - 2)
        t = float(r * math.sqrt(dfree / (1.0 - r * r)))
        if (spec.get("sign") or "absolute").lower() in ("absolute", "abs", "magnitude"):
            d = abs(d)
        out.update(estimate=d, std_error=float(2.0 / math.sqrt(dfree)),
                   p_value=float(2.0 * (1.0 - __import__("scipy").stats.norm.cdf(abs(t)))),
                   note="exp4_corr_d")
        return out
    except Exception as exc:
        out.update(invalid=1, invalid_reason=f"corr_exception:{type(exc).__name__}:{exc}")
        return out


def estimate_two_group_d(df, spec, ctx=None):
    out = {"estimate": None, "std_error": None, "p_value": None,
           "ci_low": None, "ci_high": None, "n_obs": None, "effect_measure": "cohens_d",
           "invalid": 0, "invalid_reason": "", "note": "exp4_two_group_d", "model_size": 0}
    try:
        ycol = spec["y"]
        gcol = spec.get("group") or spec["treatment"]
        vals = spec.get("group_values")
        sub = df[[ycol, gcol]].dropna()
        levels = list(vals) if vals else list(pd.unique(sub[gcol]))
        if len(levels) != 2:
            out.update(invalid=1, invalid_reason=f"need_2_groups:{levels}")
            return out
        y0 = sub.loc[sub[gcol] == levels[0], ycol].to_numpy(float)
        y1 = sub.loc[sub[gcol] == levels[1], ycol].to_numpy(float)
        n1, n0 = len(y1), len(y0)
        out["n_obs"] = int(n1 + n0)
        if n1 < 3 or n0 < 3:
            out.update(invalid=1, invalid_reason=f"small_group:{n1}/{n0}")
            return out
        m1, m0 = y1.mean(), y0.mean()
        v1, v0 = y1.var(ddof=1), y0.var(ddof=1)
        se = math.sqrt(v1 / n1 + v0 / n0)
        if not np.isfinite(se) or se <= 0:
            out.update(invalid=1, invalid_reason="zero_se")
            return out
        t = (m1 - m0) / se
        dfree = n1 + n0 - 2
        d = 2.0 * t / math.sqrt(dfree)
        if (spec.get("sign") or "absolute").lower() in ("absolute", "abs", "magnitude"):
            d = abs(d)
        out.update(estimate=float(d), std_error=float(2.0 / math.sqrt(dfree)),
                   p_value=None, note="exp4_two_group_d")
        return out
    except Exception as exc:
        out.update(invalid=1, invalid_reason=f"two_group_exception:{type(exc).__name__}:{exc}")
        return out


def use_cluster_d(df, ctx, sign="absolute"):
    ctx.spec.update({
        "estimator": "custom",
        "custom_func": "ops.exp4_ops:estimate_cluster_d",
        "sign": sign,
        "effect_measure": "cohens_d",
        "min_n": int(ctx.spec.get("min_n", 30) or 30),
    })
    return df


def use_corr_d(df, ctx, sign="absolute"):
    ctx.spec.update({
        "estimator": "custom",
        "custom_func": "ops.exp4_ops:estimate_corr_d",
        "sign": sign,
        "effect_measure": "cohens_d",
    })
    return df


def use_two_group_d(df, ctx, sign="absolute"):
    ctx.spec.update({
        "estimator": "custom",
        "custom_func": "ops.exp4_ops:estimate_two_group_d",
        "sign": sign,
        "effect_measure": "cohens_d",
    })
    return df
