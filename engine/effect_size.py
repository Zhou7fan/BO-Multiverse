# -*- coding: utf-8 -*-
"""Generic standardized-effect-size estimators (paper-agnostic).

Exp2 compares a reproduced analysis with the *paper's own* success rule.  The
multi100/Nature analytical-robustness benchmark standardizes every reported
test statistic to a Cohen's d (see multi100 ``R/cohens_d.R`` and
``R/standard_correlation_coefficient.R``):

    r = t / sqrt(t^2 + df)          (t test)
    d = 2r / sqrt(1 - r^2) = 2t / sqrt(df)

That last identity is why a paired t(14)=2.123 becomes d=1.135 in multi100,
even though a within-subject d_z would be 0.548.  We follow the benchmark's
convention exactly so targets and reproduced values are on the same scale.

The engine intentionally does not know any paper's variable names; the space
yaml supplies ``y``, ``group`` / ``y1`` / ``y0``, ``treatment`` etc.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def d_from_t(t: float, df: float) -> float:
    """Benchmark-compatible Cohen's d from a t statistic."""
    if df is None or df <= 0:
        return float("nan")
    return float(2.0 * t / math.sqrt(df))


def r_from_t(t: float, df: float) -> float:
    if df is None or df <= 0:
        return float("nan")
    return float(t / math.sqrt(t * t + df))


def d_from_r(r: float) -> float:
    r = float(r)
    if not np.isfinite(r) or abs(r) >= 1:
        return float("nan")
    return float(2.0 * r / math.sqrt(1.0 - r * r))


def se_d_two_group(n1: int, n0: int, d: float) -> float:
    if n1 <= 1 or n0 <= 1:
        return float("nan")
    return float(math.sqrt((n1 + n0) / (n1 * n0) + (d * d) / (2.0 * (n1 + n0))))


def _t_p(t: float, df: float) -> float:
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        from scipy.stats import norm
        return float(2.0 * norm.sf(abs(t)))


def _sign_value(x: float, spec: Dict[str, Any]) -> float:
    mode = (spec.get("sign") or "as_is").lower()
    if mode in ("absolute", "abs", "magnitude"):
        return abs(x)
    if mode in ("reverse", "flip", "-1"):
        return -x
    return x


def _pack(estimate: float, se: float, t: Optional[float], df: Optional[float],
          n: int, note: str, effect_measure: str = "cohens_d",
          spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    z = 1.959963984540054
    if spec:
        estimate = _sign_value(estimate, spec)
        if t is not None:
            t = _sign_value(t, spec)
    out = {
        "estimate": float(estimate) if np.isfinite(estimate) else None,
        "std_error": float(se) if se is not None and np.isfinite(se) else None,
        "p_value": None, "ci_low": None, "ci_high": None,
        "n_obs": int(n), "effect_measure": effect_measure,
        "invalid": 0, "invalid_reason": "", "note": note,
        "model_size": 0,
    }
    if out["estimate"] is None or out["std_error"] is None or out["std_error"] <= 0:
        out["invalid"] = 1
        out["invalid_reason"] = "non_convergent_d"
        return out
    if t is not None and df is not None and df > 0:
        out["p_value"] = _t_p(t, df)
    out["ci_low"] = out["estimate"] - z * out["std_error"]
    out["ci_high"] = out["estimate"] + z * out["std_error"]
    return out


def estimate_group_d(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Two-group standardized mean difference via the paper's t -> d rule."""
    ycol = spec["y"]
    gcol = spec.get("group") or spec.get("treatment")
    if not gcol:
        raise ValueError("standardized_effect/two_group requires spec['group']")
    sub = df[[ycol, gcol]].dropna()
    vals = spec.get("group_values")
    if vals:
        ctrl, trt = vals[0], vals[1]
        sub = sub[sub[gcol].isin([ctrl, trt])]
        y1 = sub.loc[sub[gcol] == trt, ycol].to_numpy(float)
        y0 = sub.loc[sub[gcol] == ctrl, ycol].to_numpy(float)
    else:
        levels = list(pd.unique(sub[gcol]))
        if len(levels) != 2:
            raise ValueError("two_group needs exactly 2 levels, got %r" % (levels,))
        y0 = sub.loc[sub[gcol] == levels[0], ycol].to_numpy(float)
        y1 = sub.loc[sub[gcol] == levels[1], ycol].to_numpy(float)
    n1, n0 = len(y1), len(y0)
    if n1 < int(spec.get("min_group_n", 3)) or n0 < int(spec.get("min_group_n", 3)):
        return {"invalid": 1, "invalid_reason": "small_group:%d/%d" % (n1, n0),
                "estimate": None, "std_error": None, "n_obs": n1 + n0}
    m1, m0 = float(np.mean(y1)), float(np.mean(y0))
    v1, v0 = float(np.var(y1, ddof=1)), float(np.var(y0, ddof=1))
    se_diff = math.sqrt(v1 / n1 + v0 / n0)
    t = (m1 - m0) / se_diff
    dfree = n1 + n0 - 2
    d = d_from_t(t, dfree)
    se = se_d_two_group(n1, n0, d)
    return _pack(d, se, t, dfree, n1 + n0, "cpu_d_two_group", spec=spec)


def estimate_paired_d(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Paired t -> benchmark d (2t/sqrt(df), df=n-1)."""
    if spec.get("y1") and spec.get("y0"):
        sub = df[[spec["y1"], spec["y0"]]].dropna()
        diff = (sub[spec["y1"]] - sub[spec["y0"]]).to_numpy(float)
    else:
        ycol = spec["y"]
        if spec.get("diff_col"):
            diff = df[spec["diff_col"]].dropna().to_numpy(float)
        else:
            raise ValueError("paired requires spec['y1']/['y0'] or ['diff_col']")
    n = len(diff)
    if n < int(spec.get("min_n", 3)):
        return {"invalid": 1, "invalid_reason": "small_n:%d" % n,
                "estimate": None, "std_error": None, "n_obs": n}
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    t = md / (sd / math.sqrt(n))
    dfree = n - 1
    d = d_from_t(t, dfree)
    se = 2.0 / math.sqrt(dfree)
    return _pack(d, se, t, dfree, n, "cpu_d_paired_benchmark", spec=spec)


def estimate_correlation_d(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    xcol, ycol = spec.get("x") or spec.get("treatment"), spec["y"]
    sub = df[[xcol, ycol]].dropna()
    n = len(sub)
    if n < int(spec.get("min_n", 3)):
        return {"invalid": 1, "invalid_reason": "small_n:%d" % n,
                "estimate": None, "std_error": None, "n_obs": n}
    r = float(np.corrcoef(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float))[0, 1])
    d = d_from_r(r)
    se = abs(2.0 / (1.0 - r * r) ** 1.5) * math.sqrt((1.0 - r * r) ** 2 / max(n - 1, 1))
    t = r * math.sqrt((n - 2) / (1.0 - r * r)) if abs(r) < 1 else None
    return _pack(d, se, t, n - 2, n, "cpu_d_from_correlation", spec=spec)


def estimate_reg_coeff_d(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """multi100's convention for a reported regression coefficient: beta/sqrt(n)."""
    beta = spec.get("coef")
    se_beta = spec.get("coef_se")
    n = spec.get("n") or len(df)
    if beta is None:
        raise ValueError("reg_coeff requires spec['coef']")
    d = float(beta) / math.sqrt(n)
    se = float(se_beta) / math.sqrt(n) if se_beta is not None else None
    return _pack(d, se, None, None, int(n), "cpu_d_reg_coeff", spec=spec)
