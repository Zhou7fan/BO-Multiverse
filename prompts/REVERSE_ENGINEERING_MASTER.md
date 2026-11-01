# Reverse-engineering Multiverse + BO master experiment prompt

> Usage: copy this whole file into a new conversation. The new conversation only "runs experiments, produces results, visualizes" —
> do not redesign the menu, and do not write paper-specific things back into the menu.

## 0. The task in one sentence

This is not multiverse analysis for multiverse analysis's sake.  
It is fundamentally **reverse engineering**: having seen the paper's reported results, infer which analysis steps it took, which
options it activated, and what choice each option took.  
The multiverse is only the search space of "all possible analysis paths"; BO (BOCS / COMBO) is the searcher.  
So the only metric that matters is: **how many queries it takes to find the path consistent with the paper's reported results**.  
A bigger solution space is better — the bigger the space, the more it proves BO's sample efficiency over random search.

Working directory: `C:\Users\Administrator\Desktop\Multiverse Ducksuck`  
Machine: Windows 11, NVIDIA RTX 4060 Ti (8GB), Python 3.13 (Anaconda).

## 1. The three-layer structure (get this straight first)

### 1.1 The global menu: fully paper-agnostic

`menu/pipeline_menu.yaml`: pipeline → 8 steps → 81 options.

- An option is "a kind of operation you may or may not do" — a hint for the AI, not a fixed pipeline.
- The menu contains **no concrete choice**: no concrete column names, thresholds, year windows, adding a constant, subsets.
- The menu has **no H1–H6, no paper-specific options like income concept**.
  Concept operationalization is uniformly `map_hypothesis_concept`: the AI first judges whether the paper's "hypothesis concept → data variable"
  option exists, then generates several choices per concept. The income theme is just one instance of it.
- The menu is as complete as possible at the option level; choices are generated on the spot by the AI after reading the paper + dataset,
  **and may be entirely absent from the menu**.
- Each paper instantiates its own subspace in its own folder; different papers' subspaces differ.

### 1.2 The paper subspace: paper-specific

`papers/<paper_id>/multiverse/exp{1,2,3}_*/multi_space.yaml`

- `fork` = one activated instance of an option; repeatable options can become several forks.
- `choice` = that fork's concrete value/implementation (columns, thresholds, constants, subsets, code).
- `universe` = every fork takes one choice, yielding one complete analysis path.
- `reference` = the path the paper determines verbatim in Experiments 2/3; used only as the target baseline,
  **never as a warm-up point**.

### 1.3 The engine: paper-agnostic

`engine/`: loads multi_space, executes choices, GPU batched estimation, objective, BO driving.  
Key files:

- `engine/ENGINE_CONTRACT.md`: the space/choice/result/objective contract.
- `engine/bo/INTERFACE.md`: the unified `suggest` interface for BOCS / COMBO / random search.
- `engine/bo/reverse_benchmark.py`: the reverse benchmark driver this prompt uses (writes CSV per query,
  computes queries-to-target, draws convergence charts).
- `engine/bo/bocs/`, `engine/bo/combo/`: the real BOCS / COMBO implementations.

## 2. The three experiments (information isolation, strictly enforced)

### Experiment 1: blind (largest space)

The sub-agent may only see:

- The dataset (`data/` or the files the space's `data_paths` point to);
- The hypothesis statement extracted from the abstract;
- The menu `menu/pipeline_menu.yaml`.

It physically cannot read the paper's PDF text or `code/`.  
It walks the menu step by step, option by option, judging "does this paper/this data need this option",
then generates the choices itself; it does not aim to reproduce the paper, so its space is the largest of the three.  
The target is still the paper's reported results (provided by the evaluator's targets; the agent cannot see the actual values).

### Experiment 2: reproduction (anchored on the paper)

The sub-agent can see the dataset, the hypothesis, the menu, and the paper's text and code. It must:

1. First reproduce, item by item, the analyses the paper really did, setting every explicit step / option / choice as `reference`;
2. Then check against the menu for gaps: the forks the paper **implicitly fixed but never explained, which exist in the menu**
   (e.g. missing values filled with the mean or 0, whether to winsorize, which group of variables was controlled for), and expand
   choices for each such fork;
3. The space is built around the paper's real uncertainty, not self-invented unrelated analyses.

### Experiment 3: perturbation (reproduction + local wobble)

Exactly the same as Experiment 2, with extra requirements:

- For every **explicitly determined** choice of the paper, the AI generates **3–5 neighboring choices** around it
  (e.g. paper adds 5 to a column → generate adding 1 / 3 / 7 / 9; paper winsorizes at 1% → 0.5% / 2% / 3%).
- These neighboring choices are generated sensibly by the AI from the data; they **need not be in the menu**.

## 3. Objective function and convergence metrics

The evaluator already turns one universe into a scalar objective (lower is better):

```
obj = 10*invalid + 2*wrong_sign + 1*not_significant
      + min(|estimate - paper_target| / close_tol, 3)
      + 1e-4*model_size
```

- `paper_target` comes from `targets_by_estimator` in `targets.json`;
- `invalid` = code won't run / sample too small / singular matrix etc., straight 10-point penalty;
- `close_tol` defaults to 0.03.

**success threshold**: by default `obj <= 0.05` (changeable with `--target-obj`).  
That means: direction correct, significant, point estimate near the paper target (about |estimate-target| ≤ 0.0015,
plus the 1e-4 model_size penalty). The reason not to use 0.1: exp3's random warm-up occasionally
already reaches about 0.06, so 0.1 would let every method hit during warm-up and lose any comparative meaning.

Metrics that must be reported:

1. The `best_so_far` vs BO query curve (the warm-up's initial best is recorded separately; the curve starts at BO query 0);
2. `queries_to_target`: the first query with `best_so_far <= 0.05`;
3. `first_support_query`: the first query with support=supports;
4. `success_rate`: how many seeds reach the target within budget;
5. wall-clock, average time per suggest, average time per eval;
6. Space size (universe_count) and the valid ratio (if sampling results already exist).

## 4. How to run (three sub-agents in parallel)

Start one sub-agent per experiment; each sub-agent only touches its own `exp*` directory.
Recommended per experiment:

- Methods: `BOCS-SA`, `COMBO`, `RandomSearch` (random baseline);
- seeds: `0,1,2`;
- per run: `--warmup 20 --queries 50`;
- `--target-obj 0.05`;
- one output directory per experiment: `papers/<paper_id>/multiverse/exp*/results/reverse_bo/`.

Single-run command template:

```powershell
python engine/bo/reverse_benchmark.py `
  --space "papers/<paper_id>/multiverse/exp2_reproduction/multi_space.yaml" `
  --targets "papers/<paper_id>/multiverse/targets.json" `
  --exp exp2 --out "papers/<paper_id>/multiverse/exp2_reproduction/results/reverse_bo" `
  --method BOCS-SA --seed 0 --warmup 20 --queries 50 --target-obj 0.05
```

All experiments (3 methods × 3 seeds run sequentially, written query by query, interruptible):

```powershell
python engine/bo/reverse_benchmark.py `
  --space "papers/<paper_id>/multiverse/exp2_reproduction/multi_space.yaml" `
  --targets "papers/<paper_id>/multiverse/targets.json" `
  --exp exp2 --out "papers/<paper_id>/multiverse/exp2_reproduction/results/reverse_bo" `
  --methods BOCS-SA,COMBO,RandomSearch --seeds 0,1,2 `
  --warmup 20 --queries 50 --target-obj 0.05 --max-seconds 1800
```

After the runs finish, aggregate/plot (reads existing CSV/JSON, does not re-run):

```powershell
python engine/bo/reverse_benchmark.py `
  --space "...multi_space.yaml" --targets "...targets.json" --exp exp2 `
  --out "...results/reverse_bo" `
  --methods BOCS-SA,COMBO,RandomSearch --seeds 0,1,2 --aggregate-only
```

## 5. Deliverables

Each experiment produces, under its own `results/reverse_bo/`:

- `BOCS-SA/seed{0,1,2}.csv`, `COMBO/seed{0,1,2}.csv`, `RandomSearch/seed{0,1,2}.csv`;
- a `seed*_summary.json` per run;
- `curves_mean.csv`, `summary_table.csv`, `aggregate.json`;
- `convergence_best_so_far.png` (best objective vs query, mean ± range);
- `queries_to_target.png` (bar chart of final best and queries-to-target);
- `REPORT.md`: space size, reference obj, method comparison table, convergence conclusions, failure/timeout records.

Also make one cross-experiment master report (via another sub-agent or direct aggregation):

- The three space sizes (exp1 largest) and their valid ratios;
- The three convergence curves side by side;
- A `queries_to_target` table for BOCS vs COMBO vs random;
- Answer explicitly: "how much faster is BO than random search? Is the advantage more obvious in bigger spaces?"

## 6. Constraints

- The menu `menu/pipeline_menu.yaml` is paper-agnostic: **no paper-specific option/choice may be added**.
- Paper-specific things may only appear in that paper's `multi_space.yaml` / `ops/`.
- Experiment 1's sub-agent must not read the PDF text or `code/`; Experiments 2/3 may.
- The real BOCS / COMBO must be called; substituting a plain GP / plain optimizer is not allowed.
- The `reference path` serves only as the target baseline, not as warm-up.
- All numbers must come from real runs; failures, timeouts, invalid must be recorded — do not report only successes.
- Keep a full trail throughout: commands, parameters, logs, CSVs, figures, reports.
