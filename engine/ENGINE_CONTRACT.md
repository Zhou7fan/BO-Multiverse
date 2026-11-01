# Engine Execution Contract (ENGINE CONTRACT, v0.2.0)

This file is the **paper-agnostic** interface contract. The generic engine (`engine/`) and each paper's
instantiation artifacts (`papers/<id>/multiverse/<exp>/multi_space.yaml` + `ops/`) must strictly comply with this file.
Neither side may change field semantics on its own; extensions may only add optional fields, never alter the meaning of existing fields.
The menu `menu/pipeline_menu.yaml` is the sole source of forks; this contract does not change "a fork must be traceable back to a menu option".

---

## 1. Directories and imports

- One experiment space = one directory `<exp_dir>` (e.g. `papers/<id>/multiverse/exp2_reproduction/`), containing
  `multi_space.yaml`, `ops/` (a Python package, may have `__init__.py`), `results/`.
- In `multi_space.yaml`, `ops_module: "ops.xxx"` resolves to `<exp_dir>/ops/xxx.py`;
  `func` is the function name in that module. The engine prepends `<exp_dir>` to `sys.path` before execution and restores it afterwards.
- Complex implementations may only live in that experiment's `ops/`; paper-specific column names/thresholds/constants **may only** appear in that paper's
  yaml / ops, and **must never** be written into `engine/` or the menu.

## 2. Choice code contract

- Each choice provides inline `code` (defining `def apply(df, ctx)`) or `ops_module`+`func`
  (optionally `params`, a dict merged into `ctx.params`).
- `apply(df, ctx) -> pandas.DataFrame`: the input is the current main table "after all upstream forks have run in order",
  and it returns the processed main table (in-place or a new table are both fine, but a DataFrame must be returned).
- The execution environment provides `pandas as pd`, `numpy as np`, torch (`engine.torch_utils`).
- A choice **only does data processing and settings registration; fitting models inside apply is forbidden, as is hardcoding results/conclusions**.
- The context `ctx` contains at least:
  - `ctx.params`  : dict, the current choice's params (with globals merged in).
  - `ctx.space`   : dict, `{fork_id: choice_id}`, the values decided so far (usable for requires logic).
  - `ctx.tables`  : dict, multi-table/auxiliary data (lazy-loaded, added on demand).
  - `ctx.spec`    : dict, the **model settings registry**. Choices of model-settings forks write keys via
    `ctx.spec.update({...})` (see §4), but **do not estimate here**.
  - `ctx.rng`     : numpy random generator, fixed seed `20261020` (determinism.seed).
  - `ctx.exp_dir` : absolute path of the experiment directory (for locating resources bundled with ops).

## 3. requires / produces / validity (pruning and runtime legality)

- `produces`: the list of column names/objects this choice produces.
- `requires`: dependencies, of the form `column:age` (the column must exist), `fork:f_estimator=ob_att`
  (another fork must take a given value). The engine checks them before executing the choice; if unmet, that universe is recorded invalid.
- `validity`: a **Python boolean expression as a string**, whose namespace contains `df`, `np`, `pd`, `ctx`.
  It is evaluated before execution (against the input df); if False, record invalid and skip (`invalid_reason` records which fork/condition).
- Runtime exceptions, too-small samples (n<`min_n`, default 30), missing key columns, singular/non-convergent matrices, all-zero weights
  are all caught via try/except and recorded as `invalid=1, invalid_reason=<type:detail>`, **never aborting the enumeration**.
- Pruning based on static declarations (requires/fork) happens during DFS; validity is judged at execution time.

## 4. Model settings registry ctx.spec (terminal estimation)

So that the final estimation can use GPU batched closed-form solving, **no model/inference fork fits anything itself**; instead it registers into `ctx.spec`.
After all forks have run, the engine reads `ctx.spec` uniformly, calls the batched estimation library, and produces the result_contract.

`ctx.spec` keys (defaults filled by the engine when missing):

| key | meaning | values |
|---|---|---|
| `y` | dependent variable column name | str (e.g. `lnwage`) |
| `treatment` | focal treatment column name (the variable whose coefficient/gap is taken) | str (e.g. `female`) |
| `controls` | list of control variable column names (already includes the dummies to be built) | list[str] |
| `weight` | weight column name or None | str / None |
| `estimator` | estimator family | see below |
| `se` | standard error type | `classical`/`hc0`/`hc1`/`hc2`/`hc3`/`cluster` |
| `cluster` | clustering column (when se=cluster) | str / None |
| `fe_dummies` | columns to expand into fixed-effect dummies | list[str] (drop_first) |
| `intercept` | whether to add a constant term | bool (default true) |
| `effect_measure` | effect size label | str (e.g. `log_wage_gap`) |
| `min_n` | minimum sample size | int (default 30) |

Built-in `estimator` (the engine must give a **torch+CUDA batched closed-form solution**, with treatment as the focus):
- `raw_gap`: weighted mean difference `E_w[y|t=1]-E_w[y|t=0]` (analytic SE: the classic SE of a weighted mean difference).
- `pooled_ols`: weighted OLS (including controls/fe/constant), returns the `treatment` coefficient; SE per `se`:
  classical / HC0–HC3 / one-dimensional cluster.
- `ob_att`: two-group weighted OLS split by treatment, returns the Oaxaca–Blinder counterfactual ATT
  (the weighted mean of ŷ_1−ŷ_0 over the treated); SE uses the classic decomposition variance (analytic), noted in `note`.
- All of the above return a single treatment-related scalar `estimate`.
- Extensions (optional, CPU fallback with separate timing): `logit` (IRLS, returns the treatment average marginal effect/coefficient),
  `quantile_median` (quantile regression, iterative/LP). CPU is allowed for these families; the engine must note the CPU path and its timing in the log.
- Estimators truly inexpressible with the built-ins: the choice sets `ctx.spec['estimator']='custom'` and
  `ctx.spec['custom_func']='ops.mod:func'`, with function signature `f(df, spec, ctx)->dict`,
  returning estimate/std_error/p_value/ci_low/ci_high/n_obs/effect_measure; runs on CPU with timing.

## 5. result_contract (one record per universe, fixed fields)

`universe_id` (stable hash of the fork-prefix combination), `assignment` ({fork:choice}), `estimate`,
`std_error`, `p_value`, `ci_low`, `ci_high`, `effect_measure`, `n_obs`,
`support` (supports/contradicts/inconclusive, per §6), `invalid`, `invalid_reason`,
`runtime_seconds`; plus `estimator`, `target`, `objective` (for auditing).
Confidence level defaults to 95%. p: defaults per `hypothesis.test_type` (two_sided or one_sided).
Sharded to disk under `results/` as parquet (preferred) or csv, fixed rows per shard (e.g. 50k), file names include the experiment and shard number.

## 6. Support judgment and optimization objective (support_decision_rule / objective)

- The space yaml writes `direction` (negative/positive/nonnull), `null_value`,
  `test_type` under `hypothesis`, and a `support_decision_rule`:
  - `targets_by_estimator`: `{estimator: paper-reported target value}` (each family compared against its own value when the paper reports different ones);
  - `close_tol`: point-estimate closeness threshold (default 0.03 log point for the log wage gap, overridable by yaml);
  - `alpha`: default 0.05.
- `support` judgment (engine implementation):
  - invalid → `inconclusive`;
  - direction matches `direction` (estimate<0 for negative);
  - significant: the one-sided/two-sided 95% interval excludes `null_value`;
  - close: `|estimate - target(estimator)| <= close_tol`.
  - **supports = direction correct AND significant AND close**; direction correct but not significant/not close = inconclusive;
    wrong direction = contradicts.
- The scalar objective the optimizer **minimizes** (paper-agreement loss, lower is better; uniformly implemented in the engine's `optimizer_adapter`):
  `obj = 10*invalid + 2*wrong_sign + 1*not_significant + min(|est-target|/close_tol, 3) + 1e-4*model_size`
  - `model_size` is the number of control variables (a slight parsimony penalty, tiny weight, only used to break ties).
  - This definition is shared by all three experiments; the reference path is **only used to take the target, never as a warm-up point**.

## 7. Enumeration, memoization and batched execution

- Load yaml → DFS in fork order; the static phase prunes via requires(`fork:`) and optional skip;
  count `universe_count` (product + static pruning, exact integer, written back to `meta.universe_count`).
- **Memoize** by fork prefix: the df produced by the same prefix (same string of choices) is computed only once and cached
  (key = prefix choice tuple; DataFrames use copy-on-write / cached references).
- Execution:
  - transformation forks (pandas) are reused per prefix on CPU;
  - terminal estimation batches by "same df signature + estimator family", pads/masks the different design matrices on the GPU and
    solves them in batches with `torch.linalg` (batch size adapts to fill 8GB, halves automatically when VRAM runs low, peak VRAM recorded);
  - stacking CPU multi-processing for pandas transforms is allowed, but must be noted in the log.
- Large-scale spec curves use **random sampling** (default 50,000 valid universes per space, fixed seed);
  fully running the million-scale space is neither required nor expected; BO executes within its query budget.

## 8. Optimizer encoding (optimizer_adapter)

- Unified interface: `suggest(observed) -> assignment`; `evaluate(assignment) -> objective` (internally runs one universe).
- BOCS: each fork is a block of **one-hot binary bits**; exactly one bit per block must be set, invalid block patterns incur a large penalty;
  binary vector length = Σ fork choice counts. SA moves are restricted to "flips within a block".
- COMBO: each fork is one **categorical variable**, `n_vertices=#choices`, adjacency matrix is a **path graph**
  (choices adjacent in yaml order; a complete graph is allowed for unordered families, marked by yaml `ordered:false`);
  the graph Fourier frequencies/basis are precomputed from an eigendecomposition of the adjacency matrix.
- warm-up: AI-generated diverse specs (covering different values of each fork), the **same warm-up** for all methods/seeds;
  the reference path never enters the warm-up.

## 9. Instantiation scale and self-checks

- Each space targets `universe_count >= 1,000,000` (exact count, written back); when short, add
  forks/choices under a reasonable, defensible premise, and state the rationale in `experiment.instantiation_notes`; aim for 1e6–1e7,
  avoiding meaningless bloat.
- Each choice **must first pass a smoke test on a small sample (e.g. a 300-row draw)**:
  inline code runs, produced columns actually appear, requires/validity behave correctly, terminal estimation returns the full fields.
  Smoke results go to `results/smoke_test.json` (per choice: ok/error/elapsed).
- The yaml's `experiment.instantiation_notes` must spell out: which forks were activated, which menu option each corresponds to,
  and why; the `reference` field marks the paper's real choices (E2/E3).

---

## 10. New in v0.2: strict reproduction (exp2/exp3)

### 10.1 fork role

When `meta.strict_roles: true`, every fork **must** declare a `role`, otherwise `SpaceLoader` raises an error.
(v0.2 makes this explicitly opt-in; legacy exp1/exp2/exp3 spaces without roles still load, but
`reference_penalty` is ignored. New exp2 spaces must write `strict_roles: true` in meta.)

| role | meaning | searched |
|---|---|---|
| `pinned_explicit` | explicitly stated in the paper; only one real choice | no (fixed) |
| `paper_uncertainty` | unspecified by the paper, but present in the menu; at least 2 choices | yes |
| `paper_explicit_alternative` | the paper itself reports several versions; at least 2 choices | yes (restricted by `paper_choices`) |
| `ai_extension` | the AI's own idea, not done by the paper | **forbidden** in a reproduction space |

- `meta.reference_penalty` must be 0; nonzero raises an error. v0.2 removed that code path from `Evaluator`.
- A `pinned_explicit` fork may have only one real choice; `reference` must point to it (or be omitted).
- Every universe satisfies the paper's explicit settings by construction, so no "deviation from reference" penalty is needed.

### 10.2 Effect size (Cohen's d) and success judgment

Three new optional keys in `support_decision_rule`:

```yaml
support_decision_rule:
  direction: none                  # none = no direction judgment
  close_tol: 0.05                  # the paper's own tolerance (d scale)
  require_significance: false
  success_metric: tolerance        # supports iff |est-target| <= tol
  objective_mode: normalized_distance
```

- With `objective_mode: normalized_distance`,
  `obj = 10*invalid + min(|est-target|/close_tol, 3) + 1e-4*model_size`;
  `obj <= 1.0` exactly equals "within the paper's tolerance", and `target_obj` is uniformly set to 1.0.
- With `success_metric: tolerance`, `support` is decided by `close` alone; direction/significance are only recorded, not part of the judgment.
- `estimator: standardized_effect` (alias `cohen_d`) is implemented by `engine/effect_size.py`,
  with `d_source` one of `two_group` / `paired` / `from_coef` / `correlation` / `reg_coeff`;
  the conversion follows multi100's convention `d = 2t/sqrt(df)` (consistent with `R/cohens_d.R`).

### 10.3 Full enumeration and the human baseline

- `engine/full_space.py`: streaming DFS that enumerates **all** universes, estimates one by one, shards to disk,
  and aggregates `success_rate` / `best_objective` / `reference`. Million-scale spaces no longer must be sampled;
  exp2's small spaces (1e4–1e5) should be run to completion.
- `engine/bo/human_baseline.py`: reads the original d and the human reanalysis d from the multi100 processed CSVs,
  defines human success as `|d_reanalysis - d_original| <= 0.05`; supports
  `success_at_k` (probability of at least one success in k draws without replacement), aligned with BO's
  `budget_success_total/adaptive`.
- Main convention of the human baseline: the number of human reanalyses per study (4–7) is the number of attempts;
  compare "humans succeed at least once" with "BO succeeds at least once in its first k evaluations" at the same k.

