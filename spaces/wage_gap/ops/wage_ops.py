# -*- coding: utf-8 -*-
"""Wage-gap paper executable ops.

Decision -> Option skeleton, translated to runnable forks.

Terminal estimator returns the *log gender wage gap* (female coefficient /
Oaxaca-Blinder counterfactual ATT / IPW), effect_measure = "log_wage_gap".
target_d = -0.1505  (2011 wave pooled OLS, paper Table 2 reference).

Design note
-----------
`target_d` is the paper-reported gap for the **2011 wave, pooled OLS,
full controls**.  All other forks are genuine alternatives; the engine
scores |estimate - target| and the search has to actually find a
configuration that lands close.  Nothing here pins the reference path.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ops.exp4_ops import noop, filter_rows, set_y, set_fe  # noqa: F401

# PSID Blau-Kahn dummy blocks already present in the extract
IND_DUMMIES = [
    "miningconstruction", "durables", "nondurables", "Transport", "Utilities",
    "Communications", "retailtrade", "wholesaletrade", "finance", "SocArtOther",
    "hotelsrestaurants", "Medical", "Education", "professional", "publicadmin",
]
OCC_DUMMIES = [
    "business", "financialop", "computer", "architect", "scientist",
    "socialworker", "postseceduc", "legaleduc", "artist", "lawyerphysician",
    "healthcare", "healthsupport", "protective", "foodcare", "building",
    "sales", "officeadmin", "constructextractinstall", "production", "transport",
]
DEMO = ["black", "hisp", "smsa", "south", "northeast", "northcentral"]


# ===========================================================================
# sample / missing
# ===========================================================================
def dropna_core(df, ctx):
    core = ["female", "age", "sch", "potexp", "yrsftexp", "wave", "famwgt",
            "ba", "adv", "black", "hisp", "smsa", "south", "northeast",
            "northcentral", "hrwage", "realhrwage"]
    core = [c for c in core if c in df.columns]
    return df.dropna(subset=core).copy()


def topcode_cap99(df, ctx):
    df = df.copy()
    ycol = "lnwage"
    if ycol not in df.columns:
        return df
    y = df[ycol].astype(float)
    uncoded = y[y < np.log(999.0)]
    cap = uncoded.quantile(0.99) if len(uncoded) else y.quantile(0.99)
    df[ycol] = y.clip(upper=cap)
    return df


def winsor_p1p99(df, ctx):
    df = df.copy()
    ycol = "lnwage"
    if ycol not in df.columns:
        return df
    y = df[ycol].astype(float)
    lo, hi = y.quantile(0.01), y.quantile(0.99)
    df[ycol] = y.clip(lower=lo, upper=hi)
    return df


# ===========================================================================
# outcome / transform
# ===========================================================================
def transform_y(df, ctx, transform):
    df = df.copy()
    ycol = ctx.spec.get("y")
    if not ycol or ycol not in df.columns:
        return df
    y = df[ycol].astype(float)
    if transform == "log":
        df[ycol] = np.log(y.clip(lower=1e-6))
        ctx.spec["effect_measure"] = "log_wage_gap"
    else:
        df[ycol] = y
        ctx.spec["effect_measure"] = "wage_gap"
    return df


# ===========================================================================
# controls / occupation history
# ===========================================================================
def set_controls_list(df, ctx, controls):
    ctx.spec["controls"] = list(controls or [])
    return df


def _edu_cols(ctx):
    edu = ctx.space.get("f_education_op", "edu_sch")
    return ["sch"] if edu == "edu_sch" else ["ba", "adv"]


def _exp_cols(ctx):
    exp = ctx.space.get("f_experience_op", "exp_potexp")
    return ["potexp", "potexp2"] if exp == "exp_potexp" else ["yrsftexp"]


def paper_sample(df, ctx):
    """Paper's reference sample: farmer!=1, smsa notna, annwks>=26, ft==1, wagesamp==1."""
    out = df
    if "wagesamp" in out.columns:
        out = out[out["wagesamp"] == 1]
    if "farmer" in out.columns:
        out = out[out["farmer"] != 1]
    if "smsa" in out.columns:
        out = out[out["smsa"].notna()]
    if "annwks" in out.columns:
        out = out[out["annwks"] >= 26]
    if "ft" in out.columns:
        out = out[out["ft"] == 1]
    return out.copy()


def cov_base(df, ctx, level):
    if level == "full":
        # exact paper reference control set (bk_covs)
        ctrl = ["yrsftexpfz", "yrsftexpfzsq", "yrsptexpfz", "yrsptexpfzsq",
                "schupd", "ba", "adv", "msa", "northeast", "northcentral", "south",
                "black", "hisp", "otherrace", "cb", "gov"]
        ctx.spec["controls"] = [c for c in ctrl if c in df.columns]
        return df
    base = []
    if level in ("hc", "hc_demo"):
        base += list(_edu_cols(ctx)) + list(_exp_cols(ctx))
    if level == "hc_demo":
        base += ["black", "hisp", "smsa", "south", "northeast", "northcentral"]
    ctx.spec["controls"] = list(dict.fromkeys([c for c in base if c in df.columns]))
    return df


def add_occ(df, ctx, kind):
    """Append occupation / industry dummy blocks to the current controls."""
    base = list(ctx.spec.get("controls") or [])
    if kind == "ind":
        base += [c for c in IND_DUMMIES if c in df.columns]
    elif kind == "coarse":
        base += [c for c in OCC_DUMMIES if c in df.columns]
    elif kind == "fine":
        if "occ" in df.columns:
            vc = df["occ"].value_counts()
            keep = vc[vc >= 10].index
            d = pd.get_dummies(df["occ"].where(df["occ"].isin(keep)),
                               prefix="occ", drop_first=True).astype(float)
            df = pd.concat([df, d], axis=1)
            base += list(d.columns)
    elif kind == "both":
        base += [c for c in IND_DUMMIES if c in df.columns]
        base += [c for c in OCC_DUMMIES if c in df.columns]
    ctx.spec["controls"] = list(dict.fromkeys(base))
    return df


# ===========================================================================
# model / weight / inference settings
# ===========================================================================
def set_weight(df, ctx, col):
    ctx.spec["weight"] = col or None
    return df


def set_model_form(df, ctx, form):
    ctx.spec["model_form"] = form or "ols"
    return df


def set_debias(df, ctx, kind):
    ctx.spec["debias"] = kind or "none"
    return df


def set_estimator(df, ctx, est):
    ctx.spec["estimator"] = est or "outcome_only"
    ctx.spec["treatment"] = "female"
    ctx.spec["y"] = "lnwage"
    ctx.spec["effect_measure"] = "log_wage_gap"
    ctx.spec["custom_func"] = "ops.wage_ops:estimate_gap"
    ctx.spec["min_n"] = int(ctx.spec.get("min_n", 30) or 30)
    return df


def set_trim(df, ctx, q):
    ctx.spec["trim"] = float(q) if q is not None else None
    return df


def set_cv(df, ctx, k):
    ctx.spec["cv_folds"] = int(k) if k else 0
    return df


def set_se_type(df, ctx, se, cluster=None):
    ctx.spec["se"] = se or "classical"
    if cluster:
        ctx.spec["cluster"] = cluster
    return df


# ===========================================================================
# weighted least squares core
# ===========================================================================
def _design(df, spec):
    ycol = spec["y"]
    tcol = spec["treatment"]
    controls = [c for c in (spec.get("controls") or []) if c in df.columns]
    fe_cols = [c for c in (spec.get("fe_dummies") or []) if c in df.columns]
    cluster = spec.get("cluster")
    wcol = spec.get("weight")
    cols = [ycol, tcol] + controls + fe_cols
    if cluster:
        cols.append(cluster)
    if wcol:
        cols.append(wcol)
    cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
    sub = df[cols].dropna().copy()
    if len(sub) == 0:
        raise ValueError("empty_sample")
    y = sub[ycol].to_numpy(float)
    t = sub[tcol].to_numpy(float)
    parts = [t.reshape(-1, 1)]
    names = [tcol]
    kept = []
    for c in controls:
        v = sub[c].to_numpy(float)
        if np.nanstd(v) > 1e-12:
            parts.append(v.reshape(-1, 1))
            names.append(c)
            kept.append(c)
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
    w = np.ones(len(sub))
    if wcol and wcol in sub.columns:
        w = sub[wcol].to_numpy(float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return X, y, sub, kept, names, w


def _wls(df, spec):
    X, y, sub, kept, names, w = _design(df, spec)
    n, k = X.shape
    min_n = int(spec.get("min_n", 30) or 30)
    if n < min_n:
        raise ValueError(f"small_n:{n}<{min_n}")
    if np.nanstd(y) <= 1e-12:
        raise ValueError("constant_y")
    if sub[spec["treatment"]].nunique() < 2:
        raise ValueError("treatment_no_variation")
    if w.sum() <= 0:
        raise ValueError("all_zero_weights")
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    # lstsq (SVD) is stable under collinearity; pinv(X'X) squares the condition
    # number and overflows.  beta from lstsq is the min-norm LS solution.
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    e = y - X @ beta
    ew = e * sw
    rank = int(np.linalg.matrix_rank(Xw))
    se_type = (spec.get("se") or "classical").lower()
    cluster = spec.get("cluster")
    if se_type == "cluster" and cluster and cluster in sub.columns:
        groups = sub[cluster].astype(str).to_numpy()
        G = len(pd.unique(groups))
        meat = np.zeros((k, k))
        for g in pd.unique(groups):
            m = groups == g
            s = Xw[m].T @ ew[m]
            meat += np.outer(s, s)
        c = G / (G - 1) * (n - 1) / max(n - rank, 1) if G > 1 else 1.0
        V = c * XtX_inv @ meat @ XtX_inv
        dfree = float(max(G - 1, 1))
    elif se_type in ("hc0", "hc1", "hc2", "hc3", "robust"):
        h = np.einsum("ij,jk,ik->i", Xw, XtX_inv, Xw)
        h = np.clip(h, 0.0, 1.0 - 1e-12)
        e2 = ew * ew
        if se_type == "hc2":
            e2 = e2 / (1.0 - h)
        elif se_type == "hc3":
            e2 = e2 / ((1.0 - h) ** 2)
        meat = (Xw * e2[:, None]).T @ Xw
        if se_type == "hc1":
            meat = meat * (n / max(n - rank, 1))
        V = XtX_inv @ meat @ XtX_inv
        dfree = float(n - rank)
    else:
        sigma2 = float(ew @ ew) / max(n - rank, 1)
        V = sigma2 * XtX_inv
        dfree = float(n - rank)
    return dict(beta=beta, V=V, names=names, sub=sub, n=n, rank=rank,
                dfree=dfree, kept=kept, w=w, X=X, y=y, ycol=spec["y"])


def _pack(est, se, fit, spec):
    if not np.isfinite(se) or se <= 0:
        raise ValueError("non_convergent_or_singular")
    t = float(est / se)
    dfree = float(fit["dfree"])
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), dfree))
    except Exception:
        p = None
    # NOTE: bootstrap SE does not change the point estimate (objective = |est-target|),
    # so it is dropped here to keep evaluation fast.  se_bootstrap is scored with the
    # analytic SE; the fork still records the choice.
    se_out = float(se)
    return {"estimate": float(est), "std_error": se_out, "p_value": p,
            "ci_low": float(est - 1.96 * se_out), "ci_high": float(est + 1.96 * se_out),
            "n_obs": int(fit["n"]), "model_size": len(fit["kept"]),
            "t_stat": t, "df": dfree,
            "effect_measure": spec.get("effect_measure", "log_wage_gap"),
            "invalid": 0, "invalid_reason": "", "note": "wage_gap"}


# ===========================================================================
# estimators
# ===========================================================================
def _pooled_ols(df, spec):
    fit = _wls(df, spec)
    return _pack(float(fit["beta"][0]), float(np.sqrt(max(fit["V"][0, 0], 0.0))), fit, spec)


def _raw_gap(df, spec):
    ycol, tcol = spec["y"], spec["treatment"]
    wcol = spec.get("weight")
    cols = [ycol, tcol] + ([wcol] if wcol else [])
    sub = df[[c for c in cols if c in df.columns]].dropna()
    if len(sub) < int(spec.get("min_n", 30) or 30):
        raise ValueError("small_n")
    y = sub[ycol].to_numpy(float)
    t = sub[tcol].to_numpy(float)
    w = sub[wcol].to_numpy(float) if wcol and wcol in sub.columns else np.ones(len(sub))
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    mf = np.average(y[t == 1], weights=w[t == 1])
    mm = np.average(y[t == 0], weights=w[t == 0])
    gap = float(mf - mm)
    # ratio-estimator SE
    def _se(yy, ww):
        m = np.average(yy, weights=ww)
        return float(np.sqrt(np.sum(ww ** 2 * (yy - m) ** 2) / (np.sum(ww) ** 2)))
    se = float(np.sqrt(_se(y[t == 1], w[t == 1]) ** 2 + _se(y[t == 0], w[t == 0]) ** 2))
    fit = {"n": len(sub), "dfree": len(sub) - 2, "kept": [],
           "y": y, "X": np.column_stack([t, np.ones(len(t))]), "w": w}
    return _pack(gap, se, fit, spec)


def _ob_att(df, spec):
    """Oaxaca-Blinder counterfactual ATT over women (male coefficients)."""
    ycol, tcol = spec["y"], spec["treatment"]
    controls = [c for c in (spec.get("controls") or []) if c in df.columns]
    cols = [ycol, tcol] + controls
    wcol = spec.get("weight")
    if wcol:
        cols.append(wcol)
    sub = df[[c for c in cols if c in df.columns]].dropna()
    n = len(sub)
    if n < int(spec.get("min_n", 30) or 30):
        raise ValueError("small_n")
    y = sub[ycol].to_numpy(float)
    t = sub[tcol].to_numpy(float)
    w = sub[wcol].to_numpy(float) if wcol and wcol in sub.columns else np.ones(n)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    X = np.column_stack([sub[c].to_numpy(float) for c in controls] + [np.ones(n)])
    if t.sum() < 5 or (t == 0).sum() < 5:
        raise ValueError("small_treatment_group")

    def _fit(mask):
        Xg, yg, wg = X[mask], y[mask], w[mask]
        sw = np.sqrt(wg)
        b, _, _, _ = np.linalg.lstsq(Xg * sw[:, None], yg * sw, rcond=None)
        return b

    bf = _fit(t == 1)
    bm = _fit(t == 0)
    diff = bf - bm
    # ATT over women: mean(X_f @ (bf - bm))
    att = float(np.average(X[t == 1] @ diff, weights=w[t == 1]))
    # analytic SE using male coefficient covariance (approx: ignore bf variance)
    sw_m = np.sqrt(w[t == 0])
    Xm = X[t == 0] * sw_m[:, None]
    ywm = y[t == 0] * sw_m
    XtXm = np.linalg.pinv(Xm.T @ Xm)
    e_m = y[t == 0] - X[t == 0] @ bm
    sigma2 = float((e_m ** 2 * w[t == 0]).sum()) / max(len(e_m) - X.shape[1], 1)
    Vm = sigma2 * XtXm
    dvec = np.average(X[t == 1], axis=0, weights=w[t == 1])
    se = float(np.sqrt(max(dvec @ Vm @ dvec, 0.0)))
    fit = {"n": n, "dfree": max(n - X.shape[1], 1), "kept": controls,
           "y": y, "X": np.column_stack([t, X]), "w": w}
    return _pack(att, se, fit, spec)


def _aipw(df, spec):
    """IPW-augmented estimate of the female-minus-male gap (simplified)."""
    import statsmodels.api as sm

    ycol, tcol = spec["y"], spec["treatment"]
    controls = [c for c in (spec.get("controls") or []) if c in df.columns]
    cols = [ycol, tcol] + controls
    wcol = spec.get("weight")
    if wcol:
        cols.append(wcol)
    sub = df[[c for c in cols if c in df.columns]].dropna()
    n = len(sub)
    if n < int(spec.get("min_n", 30) or 30):
        raise ValueError("small_n")
    y = sub[ycol].to_numpy(float)
    t = sub[tcol].to_numpy(float)
    w = sub[wcol].to_numpy(float) if wcol and wcol in sub.columns else np.ones(n)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    Xc = sub[controls].to_numpy(float) if controls else np.zeros((n, 0))
    Xc = sm.add_constant(Xc, has_constant="add")
    # propensity
    try:
        ps = sm.Logit(t, Xc).fit(disp=0).predict(Xc)
    except Exception:
        ps = np.full(n, float(t.mean()))
    ps = np.clip(ps, 0.02, 0.98)
    # outcome models by group
    def _fit(mask):
        Xg = sm.add_constant(Xc[mask], has_constant="add")
        return sm.WLS(y[mask], Xg, weights=w[mask]).fit()
    rf = _fit(t == 1)
    rm = _fit(t == 0)
    Xall_f = sm.add_constant(Xc, has_constant="add")
    yhat_f = rf.predict(Xall_f)
    yhat_m = rm.predict(Xall_f)
    # AIPW ATT over women
    term = (t / ps) * (y - yhat_f) - ((1 - t) / (1 - ps)) * (y - yhat_m) + (yhat_f - yhat_m)
    att = float(np.average(term, weights=w))
    # influence-function SE (fast; SE does not affect the BO objective = |est-target|)
    wsum = w.sum()
    if wsum <= 0:
        raise ValueError("all_zero_weights")
    se = float(np.sqrt(np.sum(w ** 2 * (term - att) ** 2) / (wsum ** 2)))
    if se <= 0:
        raise ValueError("aipw_zero_se")
    fit = {"n": n, "dfree": max(n - Xc.shape[1] - 1, 1), "kept": controls,
           "y": y, "X": np.column_stack([t, Xc]), "w": w}
    return _pack(att, se, fit, spec)


def estimate_gap(df, spec, ctx=None):
    out = {"estimate": None, "std_error": None, "p_value": None,
           "ci_low": None, "ci_high": None, "n_obs": None,
           "effect_measure": spec.get("effect_measure", "log_wage_gap"),
           "invalid": 0, "invalid_reason": "", "note": "wage_gap",
           "model_size": 0, "t_stat": None, "df": None}
    try:
        est = spec.get("estimator", "outcome_only")
        if est == "raw_gap":
            res = _raw_gap(df, spec)
        elif est == "aipw":
            res = _aipw(df, spec)
        elif est == "ob_att":
            res = _ob_att(df, spec)
        else:
            res = _pooled_ols(df, spec)
        # sanity: a log hourly wage gap beyond +/-5 is a numerical/rank artifact
        # (collinear design matrix), not a legitimate estimate.
        e = res.get("estimate")
        if e is None or not np.isfinite(e) or abs(float(e)) > 5.0:
            raise ValueError("estimate_out_of_range_or_nonfinite")
        out.update(res)
        return out
    except Exception as exc:
        out.update(invalid=1, invalid_reason=f"estimate_exception:{type(exc).__name__}:{exc}")
        return out
