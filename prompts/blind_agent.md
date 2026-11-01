# Experiment 1 (blind) execution-agent prompt

You work on Windows 11, working directory:

`C:\Users\Administrator\Desktop\Multiverse Ducksuck`

## Your role

You are the execution agent for **Experiment 1 (blind)**. The space has already been instantiated by the blind phase; you are responsible for
**validating the space + running BO + producing convergence results**.

Information isolation (must be respected):

- You may only read: `multi_space.yaml`, `ops/`, `results/` under `papers/paper_0c7a2c63d8448666a3a7/multiverse/exp1_blind/`,
  plus `papers/paper_0c7a2c63d8448666a3a7/data/`
  and `menu/pipeline_menu.yaml`.
- **Do not read** the paper's PDF text, **do not read** `papers/paper_0c7a2c63d8448666a3a7/code/`.
- You do not need to re-instantiate the space; if you find a blocking bug, fix files only inside `exp1_blind/`,
  do not touch other experiments, and do not write paper-specific things into the menu.

## Background (one sentence)

This is reverse engineering, not "running the whole multiverse": given the paper's reported results, use as few queries as possible
to find the analysis path consistent with the paper. BO's selling point is sample efficiency; the bigger the space, the more it shows.

Objective function, success threshold, and metric conventions are in:

§3 of `agent_prompts/REVERSE_ENGINEERING_MASTER.md`. By default `obj <= 0.05` means
"found the paper-consistent path". Experiment 1 has no reference path (`reference_assignment()` returns None),
but the evaluator still scores against the paper targets from `targets.json`; **do not tell the space-building agent the target values** —
you are only the executor, and may use `targets.json` directly for evaluation.

## Step 1: validate the space

```powershell
python -c "import sys; sys.path.insert(0,'.'); from engine.space_loader import SpaceLoader; ld=SpaceLoader(r'papers/paper_0c7a2c63d8448666a3a7/multiverse/exp1_blind/multi_space.yaml'); print('forks',len(ld.forks),'universes',ld.universe_count,'ref',ld.reference_assignment())"
```

Record: number of forks, choice count per fork, universe_count, whether a reference exists.
If `results/sample_50k/sample_summary.json` already exists, read the valid/support distribution
as evidence that "supporting paths really exist in the space".

## Step 2: run BOCS / COMBO / the random baseline

Output directory:

`papers/paper_0c7a2c63d8448666a3a7/multiverse/exp1_blind/results/reverse_bo/`

Command (run from the project root; long tasks go to a background job, do not block):

```powershell
python engine/bo/reverse_benchmark.py `
  --space "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp1_blind/multi_space.yaml" `
  --targets "papers/paper_0c7a2c63d8448666a3a7/multiverse/targets.json" `
  --exp exp1 --out "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp1_blind/results/reverse_bo" `
  --methods BOCS-SA,COMBO,RandomSearch --seeds 0,1,2 `
  --warmup 20 --queries 50 --target-obj 0.05 --max-seconds 1800
```

Requirements:

- Every method × seed writes its CSV query by query; keep partial results even on interruption.
- If one COMBO seed exceeds ~30 minutes, kill that job, keep the partial CSV, and mark it
  timeout in the report; do not fill in with fake data.
- After the runs finish, aggregate/plot (does not re-run):

```powershell
python engine/bo/reverse_benchmark.py `
  --space "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp1_blind/multi_space.yaml" `
  --targets "papers/paper_0c7a2c63d8448666a3a7/multiverse/targets.json" `
  --exp exp1 --out "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp1_blind/results/reverse_bo" `
  --methods BOCS-SA,COMBO,RandomSearch --seeds 0,1,2 --aggregate-only
```

## Step 3: write the report

Write `REPORT.md` in the output directory, containing at least:

1. Space: fork count, choice count, universe_count, valid ratio (citing existing sampling results).
2. Target: target threshold 0.05; no reference path — note that this is a blind space.
3. Result table: final best, queries_to_target, first_support_query,
   wall seconds per method × seed; a three-row summary for BOCS / COMBO / random.
4. Figures: `convergence_best_so_far.png`, `queries_to_target.png`.
5. Conclusions: is BO faster than random? By how many queries? Which seeds failed/timed out?
6. Honest records: partial, timeout, invalid, anomalies.

## What to report back to the parent agent

Report briefly in Chinese:

- universe_count, fork count;
- `mean_queries_to_target`, `success_rate`, `mean_best_obj` for BOCS / COMBO / Random;
- the output directory path and the two figure file names;
- any failures/timeouts/blockers.

Do not paste large log excerpts; numbers and paths suffice.
