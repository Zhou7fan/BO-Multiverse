# -*- coding: utf-8 -*-
"""space_loader: loads multi_space.yaml, menu-backtracking validation, DFS static-pruning enumeration and counting, encoding.

Paper-agnostic: it only understands the yaml structure and the global menu, not any paper-specific column names/thresholds.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

log = logging.getLogger("multiverse.space_loader")

SKIP = "__skip__"

# Strict-reproduction roles (exp2).  A fork's role decides whether it may be
# searched at all; `ai_extension` is never part of a reproduction space.
ROLE_PINNED = "pinned_explicit"
ROLE_UNCERTAIN = "paper_uncertainty"
ROLE_ALTERNATIVE = "paper_explicit_alternative"
ROLE_AI_EXTENSION = "ai_extension"
ROLES = {ROLE_PINNED, ROLE_UNCERTAIN, ROLE_ALTERNATIVE, ROLE_AI_EXTENSION}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MENU = os.environ.get(
    "MULTIVERSE_MENU", os.path.join(_REPO_ROOT, "menu", "pipeline_menu.yaml"))


class SpaceValidationError(Exception):
    """Structural errors such as a yaml fork that cannot be traced back to a menu option."""


# --------------------------------------------------------------------------- #
# Menu
# --------------------------------------------------------------------------- #
def load_menu(menu_path: str = DEFAULT_MENU) -> Dict[str, set]:
    """Return the {(step_id, option_id)} set plus option -> optional/repeatable meta info."""
    with open(menu_path, "r", encoding="utf-8") as f:
        menu = yaml.safe_load(f)
    pairs = set()
    opt_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for step in menu.get("pipeline", []):
        sid = step["id"]
        for opt in step.get("options", []):
            pairs.add((sid, opt["id"]))
            opt_meta[(sid, opt["id"])] = {
                "optional": bool(opt.get("optional", False)),
                "repeatable": bool(opt.get("repeatable", False)),
                "label": opt.get("label", ""),
            }
    log.info("Menu loaded: %d steps / %d options", len({p[0] for p in pairs}), len(pairs))
    return {"pairs": pairs, "opt_meta": opt_meta}


# --------------------------------------------------------------------------- #
# requires parsing
# --------------------------------------------------------------------------- #
def parse_requires(requires: List[str]) -> List[Tuple[str, str, Optional[str]]]:
    """Parse 'column:age' / 'fork:f_estimator=ob_att' into (kind, target, value)."""
    out = []
    for r in requires or []:
        r = str(r).strip()
        if r.startswith("column:"):
            out.append(("column", r.split(":", 1)[1], None))
        elif r.startswith("fork:"):
            body = r.split(":", 1)[1]
            if "=" in body:
                fk, fv = body.split("=", 1)
                out.append(("fork", fk.strip(), fv.strip()))
            else:
                out.append(("fork", body.strip(), None))
        else:
            out.append(("raw", r, None))
    return out


class Choice:
    __slots__ = ("choice_id", "label", "code", "ops_module", "func", "params",
                 "produces", "requires", "validity", "is_skip")

    def __init__(self, d: Dict[str, Any], is_skip: bool = False):
        self.choice_id = d.get("choice_id", SKIP)
        self.label = d.get("label", self.choice_id)
        self.code = d.get("code")
        self.ops_module = d.get("ops_module")
        self.func = d.get("func")
        self.params = dict(d.get("params", {}) or {})
        self.produces = list(d.get("produces", []) or [])
        self.requires = parse_requires(d.get("requires", []) or [])
        self.validity = d.get("validity", "True")
        self.is_skip = is_skip

    def fork_requires(self) -> List[Tuple[str, str]]:
        return [(t, v) for (k, t, v) in self.requires if k == "fork"]

    def column_requires(self) -> List[str]:
        return [t for (k, t, v) in self.requires if k == "column"]


class Fork:
    __slots__ = ("fork_id", "step", "option", "label", "optional", "ordered",
                 "choices", "reference", "index", "role", "source", "paper_choices")

    def __init__(self, d: Dict[str, Any], index: int):
        self.fork_id = d["fork_id"]
        self.step = d["step"]
        self.option = d["option"]
        self.label = d.get("label", self.fork_id)
        self.optional = bool(d.get("optional", False))
        self.ordered = bool(d.get("ordered", True))
        self.choices = [Choice(c) for c in (d.get("choices", []) or [])]
        if self.optional:
            self.choices.append(Choice({"choice_id": SKIP, "label": "Skip this fork"}, is_skip=True))
        self.reference = d.get("reference")
        self.index = index
        self.role = d.get("role")
        self.source = d.get("source", "")
        self.paper_choices = list(d.get("paper_choices", []) or [])

    @property
    def n_choices(self) -> int:
        return len(self.choices)


# --------------------------------------------------------------------------- #
# SpaceLoader
# --------------------------------------------------------------------------- #
class SpaceLoader:
    def __init__(self, space_yaml: str, menu_path: str = DEFAULT_MENU,
                 targets_json: Optional[str] = None):
        self.space_yaml = os.path.abspath(space_yaml)
        self.exp_dir = os.path.dirname(self.space_yaml)
        with open(self.space_yaml, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.menu = load_menu(menu_path)
        self.forks: List[Fork] = [Fork(d, i) for i, d in enumerate(self.raw.get("forks", []))]
        self.fork_index = {fk.fork_id: fk for fk in self.forks}
        self.validation_errors: List[str] = []
        self._validate_menu()
        self._validate_roles()
        # Precompute: starting from fork fi, which earlier forks the downstream choices' "static constraints" reference.
        # All paths through which subtree counting depends on the prefix:
        #   (a) forks referenced as quoted names in choice.validity (backward validity pruning);
        #   (b) the target forks of choice.fork:requires (checked against the prefix by _is_consistent when backward;
        #       when forward they become pins, already part of the memo key's pinned section, which does not affect correctness).
        # The memo key must cover both kinds, otherwise backward constraints would cause incorrect reuse.
        nf = len(self.forks)
        self._downstream_mentioned: List[set] = [set() for _ in range(nf + 1)]
        for fi in range(nf - 1, -1, -1):
            s = set(self._downstream_mentioned[fi + 1])
            for c in self.forks[fi].choices:
                s |= self._mentioned_forks(c.validity or "")
                for (tf, _tv) in c.fork_requires():
                    s.add(tf)
            self._downstream_mentioned[fi] = s
        self._memo: Dict[Tuple, int] = {}
        self.universe_count = self._count_all()
        self.raw.setdefault("meta", {})
        self.raw["meta"]["universe_count"] = self.universe_count
        self.raw["meta"]["fork_count"] = len(self.forks)
        self.raw["meta"]["engine_version"] = "0.1.0"

    # ---------------- menu-backtracking validation ---------------- #
    def _validate_menu(self) -> None:
        pairs = self.menu["pairs"]
        for fk in self.forks:
            if (fk.step, fk.option) not in pairs:
                msg = (f"[menu backtracking failed] fork '{fk.fork_id}' references "
                       f"step='{fk.step}' / option='{fk.option}' outside the menu")
                self.validation_errors.append(msg)
                log.error(msg)
            for c in fk.choices:
                for (k, t, v) in c.requires:
                    if k == "fork" and t not in self.fork_index:
                        msg = (f"[unknown dependency fork] fork '{fk.fork_id}' choice '{c.choice_id}' "
                               f"requires nonexistent fork '{t}'")
                        self.validation_errors.append(msg)
                        log.error(msg)
        if self.validation_errors:
            raise SpaceValidationError(
                "menu/dependency validation failed at %d place(s):\n  - " % len(self.validation_errors)
                + "\n  - ".join(self.validation_errors))

    # ---------------- role / penalty validation for strict reproduction ---------------- #
    def _validate_roles(self) -> None:
        """exp2/exp3: every fork must declare a role; AI extensions are forbidden.

        Strict enforcement is opt-in: `meta.strict_roles: true`.  Legacy spaces
        without that flag load unchanged (reference_penalty is ignored by the
        evaluator, with a warning).
        """
        raw_meta = self.raw.get("meta", {}) or {}
        exp = self.raw.get("experiment", {}) or {}
        mode = exp.get("mode") or raw_meta.get("mode")
        # Strict enforcement is opt-in via meta.strict_roles.  New exp2/exp3
        # spaces set it explicitly; legacy spaces (pre-v0.2) without roles still
        # load, with reference_penalty ignored by the evaluator.
        strict = bool(raw_meta.get("strict_roles", False))
        if not strict:
            if mode in ("reproduction", "perturbation"):
                log.warning("space mode=%s but meta.strict_roles is not true; "
                            "role/penalty validation skipped", mode)
            return
        errors: List[str] = []
        for fk in self.forks:
            real = [c for c in fk.choices if not c.is_skip]
            if fk.role not in ROLES:
                errors.append(
                    "[missing role] fork '%s' declares no role; strict reproduction requires one of %s"
                    % (fk.fork_id, sorted(ROLES)))
                continue
            if fk.role == ROLE_AI_EXTENSION:
                errors.append(
                    "[ai_extension forbidden] fork '%s' is an AI extension; a reproduction space must not contain it"
                    % fk.fork_id)
            elif fk.role == ROLE_PINNED:
                if len(real) != 1:
                    errors.append(
                        "[pinned_explicit must be unique] fork '%s' has %d real choices"
                        % (fk.fork_id, len(real)))
                if fk.reference and fk.reference not in {c.choice_id for c in fk.choices}:
                    errors.append(
                        "[pinned reference missing] fork '%s' reference='%s'"
                        % (fk.fork_id, fk.reference))
            elif fk.role == ROLE_UNCERTAIN:
                if len(real) < 2:
                    errors.append(
                        "[paper_uncertainty needs at least 2 choices] fork '%s' has only %d"
                        % (fk.fork_id, len(real)))
            elif fk.role == ROLE_ALTERNATIVE:
                if len(real) < 2:
                    errors.append(
                        "[paper_explicit_alternative needs at least 2 choices] fork '%s' has only %d"
                        % (fk.fork_id, len(real)))
                if fk.paper_choices:
                    unknown = {c.choice_id for c in real} - set(fk.paper_choices)
                    if unknown:
                        errors.append(
                            "[paper_choices mismatch] fork '%s' has extra %s"
                            % (fk.fork_id, sorted(unknown)))
        penalty = float(raw_meta.get("reference_penalty", 0.0) or 0.0)
        if penalty != 0.0:
            errors.append(
                "[reference_penalty removed] meta.reference_penalty=%s; strict reproduction "
                "pins the paper's explicit choices via pinned_explicit, no deviation penalty allowed" % penalty)
        if errors:
            raise SpaceValidationError(
                "role/penalty validation failed at %d place(s):\n  - " % len(errors) + "\n  - ".join(errors))

    # ---------------- static backward validity pruning ---------------- #
    _FORK_NAME_RE = None

    def _mentioned_forks(self, validity: str) -> set:
        import re
        if not validity:
            return set()
        toks = re.findall(r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]""", validity)
        return {t for t in toks if t in self.fork_index}

    def _static_validity(self, choice: Choice, prefix: Dict[str, str]) -> Optional[bool]:
        """Statically evaluate choice.validity.

        - All forks referenced in the expression are already assigned in the prefix (backward reference) -> evaluate for real, prune when False.
        - References a not-yet-assigned forward fork, or the expression contains runtime objects like df -> return None (kept as runtime invalid).
        """
        v = (choice.validity or "True").strip()
        if v in ("True", ""):
            return True
        mentioned = self._mentioned_forks(v)
        if any(fk not in prefix for fk in mentioned):
            return None  # forward reference, not statically decidable
        # the expression references runtime objects like df -> leave to runtime, do not prune statically (evaluating with df=None would misjudge).
        import re
        if re.search(r"\bdf\b", v):
            return None
        from types import SimpleNamespace
        ctx = SimpleNamespace(space=dict(prefix), params=dict(choice.params),
                              tables={}, spec={}, rng=None, exp_dir=None)
        ns = {"ctx": ctx, "np": __import__("numpy"), "pd": __import__("pandas"),
              "df": None}
        try:
            return bool(eval(v, {"__builtins__": {}}, ns))
        except Exception:
            return None  # other runtime dependencies -> judged at runtime

    # ---------------- static-pruning DFS counting ---------------- #
    def _admissible_choices(self, fi: int, pinned: Dict[str, set],
                            prefix: Optional[Dict[str, str]] = None) -> List[Choice]:
        fk = self.forks[fi]
        prefix = prefix or {}
        allowed = []
        for c in fk.choices:
            if c.is_skip and fk.fork_id in pinned and pinned[fk.fork_id]:
                continue  # skip not allowed when pinned
            if fk.fork_id in pinned and pinned[fk.fork_id]:
                if c.choice_id not in pinned[fk.fork_id]:
                    continue
            sv = self._static_validity(c, prefix)
            if sv is False:
                continue  # backward validity statically False -> prune
            allowed.append(c)
        return allowed

    def _is_consistent(self, fi: int, c: Choice, prefix: Dict[str, str]) -> bool:
        for (tf, tv) in c.fork_requires():
            if tf in prefix:
                if tv is not None and prefix[tf] != tv:
                    return False
        return True

    def _apply_pins(self, fi: int, c: Choice, pinned: Dict[str, set]) -> Dict[str, set]:
        new = {k: set(v) for k, v in pinned.items()}
        assigned_ids = {self.forks[j].fork_id for j in range(fi)}
        for (tf, tv) in c.fork_requires():
            if tf in assigned_ids:
                continue  # already in prefix, checked by _is_consistent
            new.setdefault(tf, set()).add(tv)  # future fork: statically pin the value
        return new

    def _count(self, fi: int, prefix: Dict[str, str], pinned: Dict[str, set]) -> int:
        if fi == len(self.forks):
            return 1
        # the memo key only contains earlier forks actually referenced by downstream validity; the rest of the prefix does not affect subtree counting
        dm = sorted(self._downstream_mentioned[fi])
        pref_tup = tuple(prefix.get(fk) for fk in dm)
        key = (fi, pref_tup, tuple(sorted((k, tuple(sorted(v))) for k, v in pinned.items())))
        if key in self._memo:
            return self._memo[key]
        total = 0
        for c in self._admissible_choices(fi, pinned, prefix):
            if not self._is_consistent(fi, c, prefix):
                continue
            sub = self._count(fi + 1, {**prefix, self.forks[fi].fork_id: c.choice_id},
                              self._apply_pins(fi, c, pinned))
            total += sub
        self._memo[key] = total
        return total

    def _count_all(self) -> int:
        return self._count(0, {}, {})

    # ---------------- enumeration (small spaces / tests) ---------------- #
    def iter_assignments(self):
        def dfs(fi, prefix, pinned):
            if fi == len(self.forks):
                yield dict(prefix)
                return
            for c in self._admissible_choices(fi, pinned, prefix):
                if not self._is_consistent(fi, c, prefix):
                    continue
                yield from dfs(fi + 1, {**prefix, self.forks[fi].fork_id: c.choice_id},
                               self._apply_pins(fi, c, pinned))
        yield from dfs(0, {}, {})

    # ---------------- uniform random sampling (without materializing the whole space) ---------------- #
    def sample_assignments(self, n: int, seed: int = 20261020) -> List[Dict[str, str]]:
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            prefix: Dict[str, str] = {}
            pinned: Dict[str, set] = {}
            for fi in range(len(self.forks)):
                cand = [c for c in self._admissible_choices(fi, pinned, prefix)
                        if self._is_consistent(fi, c, prefix)]
                weights = []
                for c in cand:
                    w = self._count(fi + 1, {**prefix, self.forks[fi].fork_id: c.choice_id},
                                    self._apply_pins(fi, c, pinned))
                    weights.append(max(w, 0))
                weights = np.asarray(weights, dtype=float)
                if weights.sum() <= 0:
                    break
                pick = rng.choice(len(cand), p=weights / weights.sum())
                c = cand[pick]
                prefix[self.forks[fi].fork_id] = c.choice_id
                pinned = self._apply_pins(fi, c, pinned)
            else:
                out.append(prefix)
        return out

    # ---------------- universe_id stable hash ---------------- #
    def universe_id(self, assignment: Dict[str, str]) -> str:
        h = hashlib.sha1()
        for fk in self.forks:
            h.update(("%s=%s;" % (fk.fork_id, assignment.get(fk.fork_id, SKIP))).encode("utf-8"))
        return h.hexdigest()[:16]

    # ---------------- BOCS one-hot encoding ---------------- #
    def block_sizes(self) -> List[int]:
        return [fk.n_choices for fk in self.forks]

    def encode_onehot(self, assignment: Dict[str, str]) -> np.ndarray:
        vec = np.zeros(sum(self.block_sizes()), dtype=np.int8)
        off = 0
        for fk in self.forks:
            cid = assignment.get(fk.fork_id, SKIP)
            idx = next((i for i, c in enumerate(fk.choices) if c.choice_id == cid), None)
            if idx is not None:
                vec[off + idx] = 1
            off += fk.n_choices
        return vec

    def decode_onehot(self, vec: np.ndarray) -> Dict[str, str]:
        off = 0
        out = {}
        for fk in self.forks:
            block = vec[off:off + fk.n_choices]
            ones = np.where(np.asarray(block) == 1)[0]
            if len(ones) == 1:
                out[fk.fork_id] = fk.choices[int(ones[0])].choice_id
            else:
                out[fk.fork_id] = None  # invalid block pattern (0 or multiple 1s)
            off += fk.n_choices
        return out

    # ---------------- COMBO categorical path graphs ---------------- #
    def combo_graphs(self) -> List[Dict[str, Any]]:
        graphs = []
        for fk in self.forks:
            v = fk.n_choices
            A = np.zeros((v, v), dtype=float)
            if v > 1:
                if fk.ordered:
                    for i in range(v - 1):
                        A[i, i + 1] = A[i + 1, i] = 1.0
                else:
                    A[:] = 1.0
                    np.fill_diagonal(A, 0.0)
            # normalized Laplacian eigendecomposition (Fourier basis)
            deg = A.sum(axis=1)
            deg[deg == 0] = 1.0
            D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
            L_norm = np.eye(v) - D_inv_sqrt @ A @ D_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(L_norm)
            graphs.append({"fork_id": fk.fork_id, "n_vertices": v, "adjacency": A,
                           "laplacian_eigvals": eigvals, "fourier_basis": eigvecs,
                           "ordered": fk.ordered})
        return graphs

    # ---------------- convenience access ---------------- #
    def get_fork(self, fork_id: str) -> Fork:
        return self.fork_index[fork_id]

    def choice_of(self, fork_id: str, assignment: Dict[str, str]) -> Choice:
        fk = self.fork_index[fork_id]
        cid = assignment.get(fork_id, SKIP)
        for c in fk.choices:
            if c.choice_id == cid:
                return c
        raise KeyError(f"fork {fork_id} has no choice {cid}")

    def reference_assignment(self) -> Optional[Dict[str, str]]:
        ref = {}
        for fk in self.forks:
            if fk.reference:
                ref[fk.fork_id] = fk.reference
            elif fk.role == ROLE_PINNED:
                real = [c for c in fk.choices if not c.is_skip]
                if len(real) != 1:
                    return None
                ref[fk.fork_id] = real[0].choice_id
            elif fk.optional:
                ref[fk.fork_id] = SKIP
            else:
                return None
        return ref

    def role_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for fk in self.forks:
            out[fk.role or "unassigned"] = out.get(fk.role or "unassigned", 0) + 1
        return out

    def searchable_forks(self) -> List[Fork]:
        """Forks the optimizer may vary (more than one admissible real choice)."""
        return [fk for fk in self.forks
                if len([c for c in fk.choices if not c.is_skip]) > 1]

