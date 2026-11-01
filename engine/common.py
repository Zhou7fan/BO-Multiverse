# -*- coding: utf-8 -*-
"""Paper-agnostic common conventions: result_contract fields, support judgment, objective formula.

All numeric decisions come from the space yaml's hypothesis / support_decision_rule (or a targets.json in the same directory);
this module hardcodes no paper-specific column names, thresholds, or target values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# result_contract fixed output fields (contract §5)
RESULT_FIELDS = [
    "universe_id", "estimate", "std_error", "p_value", "ci_low", "ci_high",
    "effect_measure", "n_obs", "support", "invalid", "invalid_reason",
    "runtime_seconds", "estimator", "target", "objective",
]


@dataclass
class SupportConfig:
    """Parsed from the yaml hypothesis + support_decision_rule (or targets.json)."""
    direction: str = "negative"          # negative | positive | nonnull
    null_value: float = 0.0
    test_type: str = "two_sided"         # two_sided | one_sided
    alpha: float = 0.05
    close_tol: float = 0.03
    targets_by_estimator: Dict[str, float] = field(default_factory=dict)
    effect_measure: str = ""
    require_significance: bool = True
    # v0.2: exp2 strict-reproduction objective.  "normalized_distance" makes
    # |estimate-target|/close_tol the whole objective (1.0 == at tolerance),
    # so a single target_obj=1.0 is comparable across papers/tolerances.
    objective_mode: str = "legacy"          # legacy | normalized_distance
    success_metric: str = "support"         # support | tolerance
    direction_check: bool = True

    @classmethod
    def from_space(cls, space: Dict[str, Any], targets_json: Optional[Dict[str, Any]] = None) -> "SupportConfig":
        hyp = space.get("hypothesis", {}) or {}
        rule = space.get("support_decision_rule", {}) or {}
        cfg = cls(
            direction=hyp.get("direction", rule.get("direction", "negative")),
            null_value=float(hyp.get("null_value", rule.get("null_value", 0.0))),
            test_type=hyp.get("test_type", rule.get("test_type", "two_sided")),
            alpha=float(rule.get("alpha", hyp.get("alpha", 0.05))),
            close_tol=float(rule.get("close_tol", 0.03)),
            targets_by_estimator=dict(rule.get("targets_by_estimator", {}) or {}),
            effect_measure=hyp.get("effect_measure", rule.get("effect_measure", "")),
            require_significance=bool(rule.get("require_significance", True)),
            objective_mode=str(rule.get("objective_mode", "legacy")),
            success_metric=str(rule.get("success_metric", "support")),
            direction_check=bool(rule.get("direction_check", True)),
        )
        # targets.json (paper-level shared targets) can supplement/override
        if targets_json:
            cfg.direction = targets_json.get("direction", cfg.direction)
            cfg.null_value = float(targets_json.get("null_value", cfg.null_value))
            cfg.test_type = targets_json.get("test_type", cfg.test_type)
            cfg.alpha = float(targets_json.get("alpha", cfg.alpha))
            cfg.close_tol = float(targets_json.get("close_tol", cfg.close_tol))
            cfg.effect_measure = targets_json.get("effect_measure", cfg.effect_measure) or cfg.effect_measure
            cfg.require_significance = bool(targets_json.get("require_significance", cfg.require_significance))
            cfg.objective_mode = str(targets_json.get("objective_mode", cfg.objective_mode))
            cfg.success_metric = str(targets_json.get("success_metric", cfg.success_metric))
            cfg.direction_check = bool(targets_json.get("direction_check", cfg.direction_check))
            for k, v in (targets_json.get("targets_by_estimator", {}) or {}).items():
                cfg.targets_by_estimator[k] = float(v)
        return cfg

    def target_for(self, estimator: str) -> Optional[float]:
        if estimator in self.targets_by_estimator:
            return self.targets_by_estimator[estimator]
        # fallback: take the first target value
        return next(iter(self.targets_by_estimator.values()), None)


def _is_direction_ok(estimate: float, direction: str) -> bool:
    if direction == "negative":
        return estimate < 0
    if direction == "positive":
        return estimate > 0
    if direction == "nonnull":
        return estimate != 0
    if direction in ("none", "any", ""):
        return True
    return True


def _is_significant(ci_low: float, ci_high: float, null_value: float, test_type: str) -> bool:
    """Whether the 95% interval excludes null_value."""
    if test_type == "one_sided":
        # one-sided: the whole interval on one side of null suffices (approximated here by two-sided exclusion, conservative)
        return not (ci_low <= null_value <= ci_high)
    return not (ci_low <= null_value <= ci_high)


def judge_support(estimate: Optional[float], ci_low: Optional[float], ci_high: Optional[float],
                  estimator: str, cfg: SupportConfig) -> Dict[str, Any]:
    """Return the support judgment detail. Callers set invalid to inconclusive first."""
    out = {"support": "inconclusive", "direction_ok": False, "significant": False,
           "close": False, "wrong_sign": False, "target": cfg.target_for(estimator)}
    if estimate is None or ci_low is None or ci_high is None:
        return out
    out["direction_ok"] = (True if not cfg.direction_check
                           else _is_direction_ok(estimate, cfg.direction))
    out["wrong_sign"] = (not out["direction_ok"]) if cfg.direction_check else False
    out["significant"] = _is_significant(ci_low, ci_high, cfg.null_value, cfg.test_type)
    tgt = out["target"]
    out["close"] = (tgt is not None) and (abs(estimate - tgt) <= cfg.close_tol)
    if cfg.success_metric == "tolerance":
        # exp2: the paper's own success rule is |d_ours - d_original| <= tol.
        # Direction/significance are reported, not required.
        out["support"] = "supports" if out["close"] else "inconclusive"
        return out
    if out["wrong_sign"]:
        out["support"] = "contradicts"
    elif out["direction_ok"] and (out["significant"] or not cfg.require_significance) and out["close"]:
        out["support"] = "supports"
    else:
        out["support"] = "inconclusive"
    return out


def objective(estimate: Optional[float], estimator: str, invalid: bool,
              support_detail: Dict[str, Any], model_size: int, cfg: SupportConfig) -> float:
    """Contract §6: paper-agreement loss, lower is better.

    obj = 10*invalid + 2*wrong_sign + 1*not_significant
          + min(|est-target|/close_tol, 3) + 1e-4*model_size
    """
    if invalid or estimate is None:
        return 10.0 + 1e-4 * max(model_size, 0)
    tgt = support_detail.get("target")
    dist_term = min(abs(estimate - tgt) / max(cfg.close_tol, 1e-12), 3.0) if tgt is not None else 3.0
    if cfg.objective_mode == "normalized_distance":
        # exp2: no sign/significance terms; 1.0 == exactly at the paper tolerance.
        return dist_term + 1e-4 * max(model_size, 0)
    wrong_sign = 1.0 if support_detail.get("wrong_sign") else 0.0
    not_significant = 0.0 if (support_detail.get("significant") or not cfg.require_significance) else 1.0
    return 2.0 * wrong_sign + 1.0 * not_significant + dist_term + 1e-4 * max(model_size, 0)


def invalid_result(universe_id: str, assignment: Dict[str, str], reason: str,
                   estimator: str = "", runtime: float = 0.0) -> Dict[str, Any]:
    """Build the standard record for one invalid universe (all fields of contract §5)."""
    rec = {f: None for f in RESULT_FIELDS}
    rec.update({
        "universe_id": universe_id,
        "assignment": dict(assignment),
        "estimate": None, "std_error": None, "p_value": None,
        "ci_low": None, "ci_high": None,
        "effect_measure": "", "n_obs": None,
        "support": "inconclusive",
        "invalid": 1, "invalid_reason": reason,
        "runtime_seconds": runtime, "estimator": estimator,
        "target": None, "objective": 10.0,
    })
    return rec
