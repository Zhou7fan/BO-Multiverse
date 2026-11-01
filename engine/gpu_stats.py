# -*- coding: utf-8 -*-
"""gpu_stats: batched closed-form estimation library on torch+CUDA (paper-agnostic).

Covers contract §4:
- raw_gap     : weighted mean difference (= the treatment coefficient of y~[1,t] WLS).
- pooled_ols  : weighted OLS (controls / FE dummies / constant), treatment coefficient.
- ob_att      : Oaxaca-Blinder counterfactual ATT from two-group weighted OLS (on the treated).
SE supports classical / HC0 / HC1 / HC2 / HC3 / one-dimensional cluster.
Design matrices of different universes are padded+masked and solved in batches via torch.linalg; the batch size adapts to 8GB,
OOM automatically halves the batch and peak VRAM is recorded. logit(IRLS)/quantile(median)/custom go through the CPU fallback.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import effect_size
import scipy.stats

from . import torch_utils

log = logging.getLogger("multiverse.gpu_stats")

Z95 = scipy.stats.norm.ppf(0.975)  # 1.9599639845


# --------------------------------------------------------------------------- #
@dataclass
class Design:
    """One design to estimate (matrices already extracted from the df)."""
    X: np.ndarray            # (n,k), includes the constant column (if intercept)
    y: np.ndarray            # (n,)
    w: np.ndarray            # (n,)
    treat_idx: int           # column index of treatment in X
    se: str = "classical"
    cluster: Optional[np.ndarray] = None
    n: int = 0
    k: int = 0

    def __post_init__(self):
        self.n, self.k = self.X.shape


@dataclass
class WLSOut:
    estimate: float
    std_error: float
    cov_treat: float
    n: int
    k: int
    note: str = ""


# --------------------------------------------------------------------------- #
# batched WLS solving (GPU)
# --------------------------------------------------------------------------- #
def _wls_batch_gpu(designs: List[Design], device, budget_bytes: int = 8 * 1024 ** 3
                   ) -> List[WLSOut]:
    import torch
    B0 = len(designs)
    nmax = max(d.n for d in designs)
    kmax = max(d.k for d in designs)
    # greedily pack the batch
    batch = max(1, min(B0, torch_utils.adaptive_batch_size(nmax, kmax, budget_bytes)))
    results: List[Optional[WLSOut]] = [None] * B0

    def run_chunk(chunk_idx: List[int], bs: int) -> bool:
        m = len(chunk_idx)
        Xb = np.zeros((m, nmax, kmax), dtype=np.float64)
        yb = np.zeros((m, nmax), dtype=np.float64)
        wb = np.zeros((m, nmax), dtype=np.float64)
        mask = np.zeros((m, nmax), dtype=np.float64)
        treat = np.zeros(m, dtype=np.int64)
        clb = np.zeros((m, nmax), dtype=np.int64)
        for j, di in enumerate(chunk_idx):
            d = designs[di]
            Xb[j, :d.n, :d.k] = d.X
            yb[j, :d.n] = d.y
            wb[j, :d.n] = d.w
            mask[j, :d.n] = 1.0
            treat[j] = d.treat_idx
            if d.cluster is not None:
                # local cluster encoding
                _, cl = np.unique(d.cluster, return_inverse=True)
                clb[j, :d.n] = cl
        try:
            Xt = torch.from_numpy(Xb).to(device)
            yt = torch.from_numpy(yb).to(device)
            msk = torch.from_numpy(mask).to(device)
            wt = torch.from_numpy(wb).to(device) * msk
            # A = X' W X ; B = X' W y
            Xw = Xt * wt.unsqueeze(-1)               # (m,n,k)
            A = torch.einsum("bnk,bnl->bkl", Xw, Xt)  # (m,k,k)
            B = torch.einsum("bnk,bn->bk", Xw, yt)
            # b = A^-1 B
            solve = torch.linalg.solve(A, B)           # (m,k)
            invA = torch.linalg.inv(A)
            resid = yt - torch.einsum("bnk,bk->bn", Xt, solve)  # (m,n)
            wr = wt * resid * msk                     # weighted score residual
            se_type = designs[chunk_idx[0]].se
            if se_type == "classical":
                scale = (wr * resid * msk).sum(dim=1) / (
                    torch.tensor([d.n - d.k for d in designs], device=device, dtype=torch.float64))
                cov = torch.einsum("bkl,b->bkl", invA, scale)
            else:
                x = Xt
                if se_type == "hc0":
                    M = torch.einsum("bnk,bnl,bn->bkl", x, x, wr * wr * msk)
                elif se_type == "hc1":
                    M = torch.einsum("bnk,bnl,bn->bkl", x, x, wr * wr * msk)
                    dfcorr = torch.tensor(
                        [d.n / (d.n - d.k) for d in designs], device=device, dtype=torch.float64)
                    M = torch.einsum("bkl,b->bkl", M, dfcorr)
                elif se_type in ("hc2", "hc3"):
                    hat = torch.einsum("bnk,bkl,bnl->bn", x, invA, x) * wt  # hat diag
                    one_minus_h = torch.clamp(1.0 - hat, min=1e-8)
                    weight = (wr * wr * msk) / (one_minus_h ** (2.0 if se_type == "hc3" else 1.0))
                    M = torch.einsum("bnk,bnl,bn->bkl", x, x, weight)
                elif se_type == "cluster":
                    clt = torch.from_numpy(clb).to(device)
                    score = (wr * msk).unsqueeze(-1) * x               # (m,n,k)
                    Gmax = int(clt.max().item()) + 1
                    onehot = torch.nn.functional.one_hot(clt, num_classes=Gmax).to(torch.float64)
                    onehot = onehot * msk.unsqueeze(-1)                 # (m,n,G)
                    S = torch.einsum("bng,bnk->bgk", onehot, score)     # (m,G,k)
                    M = torch.einsum("bgk,bgl->bkl", S, S)
                    # small-sample correction (n-1)/(n-k) * G/(G-1)
                    Gcount = onehot.sum(dim=1).clamp(min=1)             # (m,G)
                    Geff = (Gcount > 0).sum(dim=1).to(torch.float64)
                    npar = torch.tensor([d.k for d in designs], device=device, dtype=torch.float64)
                    nn = torch.tensor([d.n for d in designs], device=device, dtype=torch.float64)
                    corr = ((nn - 1) / (nn - npar)) * (Geff / (Geff - 1).clamp(min=1))
                    M = torch.einsum("bkl,b->bkl", M, corr)
                else:
                    raise ValueError("unknown se " + se_type)
                cov = invA @ M @ invA
            # treatment coefficient and SE
            treat_idx = torch.tensor([d.treat_idx for d in designs], device=device, dtype=torch.long)
            row = torch.arange(m, device=device)
            est = solve[row, treat_idx]
            var = cov[row, treat_idx, treat_idx]
            out_est = est.cpu().numpy()
            out_se = np.sqrt(np.clip(var.cpu().numpy(), 0, None))
            for j, di in enumerate(chunk_idx):
                results[di] = WLSOut(float(out_est[j]), float(out_se[j]),
                                     float(var[j]), designs[di].n, designs[di].k,
                                     note="gpu_batch")
            return True
        except torch.linalg.LinAlgError:
            return False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    idx = list(range(B0))
    i = 0
    while i < B0:
        chunk = idx[i:i + batch]
        ok = run_chunk(chunk, len(chunk))
        if not ok and len(chunk) > 1:
            # OOM/singular: halve and retry (each item must eventually pass)
            half = max(1, len(chunk) // 2)
            for s in range(0, len(chunk), half):
                sub = chunk[s:s + half]
                ok2 = run_chunk(sub, len(sub))
                if not ok2:
                    for di in sub:
                        results[di] = WLSOut(float("nan"), float("nan"), float("nan"),
                                             designs[di].n, designs[di].k, note="singular_or_oom")
        elif not ok:
            results[chunk[0]] = WLSOut(float("nan"), float("nan"), float("nan"),
                                       designs[chunk[0]].n, designs[chunk[0]].k,
                                       note="singular_or_oom")
        i += len(chunk)
    return results


def wls_batch(designs: List[Design]) -> List[WLSOut]:
    """Group by (SE type, design column count k) then solve in batches; different k cannot share a batch (a zero column makes A singular)."""
    groups: Dict[Any, List[int]] = {}
    for i, d in enumerate(designs):
        groups.setdefault((d.se, d.k), []).append(i)
    out: List[Optional[WLSOut]] = [None] * len(designs)
    device = torch_utils.get_device()
    for key, idxs in groups.items():
        sub = [designs[i] for i in idxs]
        if device.type == "cuda":
            torch_utils.reset_peak_memory()
        res = _wls_batch_gpu(sub, device)
        for j, di in enumerate(idxs):
            out[di] = res[j]
    try:
        import torch
        torch.cuda.empty_cache()  # release the GPU caching allocator and host pinned staging, avoiding accumulation across chunk commits
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# CPU fallback: logit(IRLS) / quantile(median) / custom
# --------------------------------------------------------------------------- #
def _logit_irls(X: np.ndarray, y: np.ndarray, w: np.ndarray, treat_idx: int,
                max_iter: int = 25, tol: float = 1e-8) -> WLSOut:
    p = X.shape[1]
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        grad = X.T @ (w * (y - mu))
        Wd = w * mu * (1.0 - mu)
        H = (X * Wd[:, None]).T @ X
        step = np.linalg.solve(H + 1e-8 * np.eye(p), grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    Wd = w * mu * (1 - mu)
    cov = np.linalg.inv((X * Wd[:, None]).T @ X + 1e-8 * np.eye(p))
    se = np.sqrt(max(cov[treat_idx, treat_idx], 0))
    return WLSOut(float(beta[treat_idx]), float(se), float(cov[treat_idx, treat_idx]),
                  X.shape[0], p, note="cpu_logit_irls")


def _quantile_median(X: np.ndarray, y: np.ndarray, w: np.ndarray, treat_idx: int,
                     q: float = 0.5) -> WLSOut:
    """True median regression (statsmodels QuantReg, q=0.5), CPU fallback with timing.

    statsmodels QuantReg does not support analytic weights; when weights exist, rows are replicated by integer frequency weights
    (freq-weight QuantReg), matching the weighted-sample convention.
    """
    import statsmodels.api as sm
    # statsmodels QuantReg has no analytic weights. Frequency-weight replication would
    # blow 30k rows up to 600k rows for survey weights (e.g. famwgt summing to 600k), 80s+ per fit,
    # and BO evaluates every query, which is unacceptable. Here:
    #   - small sample / unit weights: keep the original convention (unweighted / freq_weighted);
    #   - large sample or heavy weights: sample with replacement by weight down to 20,000 rows, then run QuantReg,
    #     as a computational approximation of "weighted quantile regression" (fixed seed, reproducible).
    rep = np.round(w).astype(int)
    n = X.shape[0]
    if rep.sum() <= n + 1 or np.all(rep <= 1):
        Xd, yd = X, y
        weight_note = "unweighted"
    elif n <= 20000 and rep.sum() <= 200000:
        Xd = np.repeat(X, rep, axis=0)
        yd = np.repeat(y, rep, axis=0)
        weight_note = "freq_weighted"
    else:
        rng = np.random.default_rng(20261020)
        p = np.asarray(w, dtype=float)
        p = p / p.sum()
        idx = rng.choice(n, size=min(20000, n), replace=True, p=p)
        Xd, yd = X[idx], y[idx]
        weight_note = "weighted_subsample_20000"
    m = sm.QuantReg(yd, Xd).fit(q=q, maxiter=2000)
    est = float(m.params[treat_idx])
    se = float(m.bse[treat_idx])
    return WLSOut(est, se, float(m.cov_params()[treat_idx, treat_idx]),
                  X.shape[0], X.shape[1],
                  note="cpu_quantreg_median_%s" % weight_note)


# --------------------------------------------------------------------------- #
# build Designs from df + spec and estimate
# --------------------------------------------------------------------------- #
def _build_matrix(df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, pd.DataFrame]:
    ycol = spec["y"]; tcol = spec["treatment"]
    controls = list(spec.get("controls", []) or [])
    fe_cols = list(spec.get("fe_dummies", []) or [])
    wcol = spec.get("weight")
    cols = [ycol, tcol] + controls
    if wcol:
        cols.append(wcol)
    cols += [c for c in fe_cols if c not in cols]
    sub = df[cols].dropna()
    if wcol:
        sub = sub[sub[wcol] != 0]
    y = sub[ycol].to_numpy(dtype=np.float64)
    t = sub[tcol].to_numpy(dtype=np.float64)
    w = sub[wcol].to_numpy(dtype=np.float64) if wcol else np.ones(len(sub))
    parts = [t.reshape(-1, 1)]
    kept_controls = []
    for c in controls:
        v = sub[c].to_numpy(dtype=np.float64)
        if np.nanstd(v) <= 1e-12:
            continue  # constant columns (e.g. farmer, all 0 after filtering) are collinear with the intercept, drop them
        parts.append(v.reshape(-1, 1)); kept_controls.append(c)
    for fc in fe_cols:
        _dum = pd.get_dummies(sub[fc], prefix=fc, drop_first=True)
        if _dum.shape[1] > 100:
            import sys as _sys
            print("DUMMY_EXPLODE fc=%r sub.shape=%s nunique=%d"
                  % (fc, sub.shape, sub[fc].nunique()), file=_sys.stderr, flush=True)
        d = _dum.astype(float)
        d = d.loc[:, d.std(axis=0) > 1e-12]
        parts.append(d.to_numpy())
    if spec.get("intercept", True):
        parts.append(np.ones((len(sub), 1)))
    X = np.hstack(parts) if parts else np.ones((len(sub), 1))
    treat_idx = 0  # treatment is always placed in the first column
    spec["_kept_controls"] = kept_controls
    return X, y, w, treat_idx, sub


def is_standardized_estimator(estimator: str) -> bool:
    return estimator in ("standardized_effect", "cohen_d")


def _estimate_standardized(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Cohen's d from a group contrast, paired contrast, correlation or OLS t.

    All conversions follow multi100's benchmark formulas so exp2 targets
    (original_cohens_d) and reproduced estimates are directly comparable.
    """
    src = (spec.get("d_source") or "two_group").lower()
    if src == "two_group":
        return effect_size.estimate_group_d(df, spec)
    if src == "paired":
        return effect_size.estimate_paired_d(df, spec)
    if src == "correlation":
        return effect_size.estimate_correlation_d(df, spec)
    if src == "reg_coeff":
        return effect_size.estimate_reg_coeff_d(df, spec)
    if src == "from_coef":
        min_n = int(spec.get("min_n", 30))
        X, y, w, treat_idx, sub = _build_matrix(df, spec)
        n = len(y)
        if n < min_n:
            return {"invalid": 1, "invalid_reason": "small_n:%d<%d" % (n, min_n),
                    "estimate": None, "std_error": None, "n_obs": n}
        se_type = spec.get("se", "classical")
        des = [Design(X, y, w, treat_idx=treat_idx, se=se_type,
                      cluster=sub[spec["cluster"]].to_numpy() if se_type == "cluster" else None)]
        r = wls_batch(des)[0]
        if not np.isfinite(r.estimate) or not np.isfinite(r.std_error) or r.std_error <= 0:
            return {"invalid": 1, "invalid_reason": "non_convergent_or_singular",
                    "estimate": None, "std_error": None, "n_obs": n}
        t = float(r.estimate) / float(r.std_error)
        dfree = float(spec.get("df") or max(n - X.shape[1], 1))
        d = effect_size.d_from_t(t, dfree)
        se_d = 2.0 / np.sqrt(dfree) if dfree > 0 else float("nan")
        out = effect_size._pack(d, se_d, t, dfree, n, "cpu_d_from_coef", spec=spec)
        out["model_size"] = len(spec.get("controls", []) or [])
        return out
    return {"invalid": 1, "invalid_reason": "unknown_d_source:%s" % src,
            "estimate": None, "std_error": None, "n_obs": None}


def estimate_one(df: pd.DataFrame, spec: Dict[str, Any], exp_dir: str = ""
                ) -> Dict[str, Any]:
    """Run the terminal estimation on one universe's final df + ctx.spec, returning the result fields."""
    min_n = int(spec.get("min_n", 30))
    estimator = spec.get("estimator", "pooled_ols")
    se = spec.get("se", "classical")
    out = {"estimate": None, "std_error": None, "p_value": None, "ci_low": None,
           "ci_high": None, "n_obs": None, "effect_measure": spec.get("effect_measure", ""),
           "invalid": 0, "invalid_reason": "", "note": "", "model_size": len(spec.get("controls", []) or [])}
    try:
        if is_standardized_estimator(estimator):
            res = _estimate_standardized(df, spec)
            out.update(res)
            out["effect_measure"] = spec.get("effect_measure",
                                             res.get("effect_measure", "cohens_d"))
            return out
        # the custom estimator needs no standard y/treatment design matrix; call the paper's ops directly.
        if estimator == "custom":
            fq = spec.get("custom_func", "")
            mod_name, func_name = fq.split(":", 1)
            if exp_dir and exp_dir not in sys.path:
                sys.path.insert(0, exp_dir)
            import importlib
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
            f = getattr(mod, func_name)
            res = f(df, spec, {}) or {}
            out.update({k: res.get(k) for k in
                        ["estimate", "std_error", "p_value", "ci_low", "ci_high",
                         "n_obs", "invalid", "invalid_reason", "note", "model_size"]})
            out["effect_measure"] = spec.get("effect_measure", out.get("effect_measure", ""))
            return out

        X, y, w, treat_idx, sub = _build_matrix(df, spec)
        n = len(y)
        out["n_obs"] = int(n)
        if n < min_n:
            out["invalid"] = 1; out["invalid_reason"] = "small_n:%d<%d" % (n, min_n)
            return out
        if np.sum(w) <= 0:
            out["invalid"] = 1; out["invalid_reason"] = "all_zero_weights"
            return out

        if estimator == "raw_gap":
            # X = [const, treatment]
            Xd = np.hstack([np.ones((n, 1)), X[:, :1]])
            des = [Design(Xd, y, w, treat_idx=1, se="classical")]
            r = wls_batch(des)[0]
            est, se_v = r.estimate, r.std_error

        elif estimator == "pooled_ols":
            des = [Design(X, y, w, treat_idx=treat_idx, se=se,
                          cluster=sub[spec["cluster"]].to_numpy() if se == "cluster" else None)]
            r = wls_batch(des)[0]
            est, se_v = r.estimate, r.std_error

        elif estimator == "ob_att":
            tcol = spec["treatment"]
            t = sub[tcol].to_numpy()
            # X = [treatment, controls..., fe..., intercept]; dropping treatment gives the counterfactual design
            Xc = X[:, 1:]
            m1 = t == 1; m0 = t == 0
            if m1.sum() < min_n or m0.sum() < min_n:
                out["invalid"] = 1; out["invalid_reason"] = "ob_att_small_group"
                return out
            # two-group weighted OLS (CPU/numpy closed form, only two fits)
            Xw0 = Xc[m0] * w[m0][:, None]
            beta0 = np.linalg.solve(Xc[m0].T @ Xw0, Xw0.T @ y[m0])
            att = float(np.average(y[m1] - Xc[m1] @ beta0, weights=w[m1]))
            est = att
            # analytic SE: Var(mean y1) + g' V0 g
            mean_y1 = np.average(y[m1], weights=w[m1])
            var_mean1 = np.average((y[m1] - mean_y1) ** 2, weights=w[m1]) / max(m1.sum(), 1)
            Ainv = np.linalg.inv(Xc[m0].T @ Xw0 + 1e-10 * np.eye(Xc.shape[1]))
            g = np.average(Xc[m1], weights=w[m1], axis=0)
            se_v = float(np.sqrt(max(var_mean1 + g @ Ainv @ g, 0.0)))

        elif estimator == "logit":
            r = _logit_irls(X, y, w, treat_idx)
            est, se_v = r.estimate, r.std_error

        elif estimator == "quantile_median":
            r = _quantile_median(X, y, w, treat_idx)
            est, se_v = r.estimate, r.std_error

        elif estimator == "custom":
            fq = spec.get("custom_func", "")
            mod_name, func_name = fq.split(":", 1)
            if exp_dir and exp_dir not in sys.path:
                sys.path.insert(0, exp_dir)
            import importlib
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
            f = getattr(mod, func_name)
            res = f(df, spec, {})
            out.update({k: res.get(k) for k in
                        ["estimate", "std_error", "p_value", "ci_low", "ci_high", "n_obs"]})
            out["note"] = "cpu_custom"
            return out
        else:
            out["invalid"] = 1; out["invalid_reason"] = "unknown_estimator:%s" % estimator
            return out

        if not np.isfinite(est) or not np.isfinite(se_v) or se_v <= 0:
            out["invalid"] = 1; out["invalid_reason"] = "non_convergent_or_singular"
            return out
        out["estimate"] = float(est)
        out["std_error"] = float(se_v)
        z = est / se_v
        out["p_value"] = float(2 * (1 - scipy.stats.norm.cdf(abs(z))))
        out["ci_low"] = float(est - Z95 * se_v)
        out["ci_high"] = float(est + Z95 * se_v)
        if estimator == "ob_att":
            out["note"] = "ob_att_analytic_se"
    except Exception as e:
        out["invalid"] = 1
        out["invalid_reason"] = "estimate_exception:%s:%s" % (type(e).__name__, e)
    return out
