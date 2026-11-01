# Experiment 2 (reproduction) execution-agent prompt

You work on Windows 11, working directory:

`C:\Users\Administrator\Desktop\Multiverse Ducksuck`

## Your role

You are the execution agent for **Experiment 2 (reproduction)**. The space was built by the instantiation
agent "that can see the paper's text and code"; you are responsible for **validating the reference path + running BO + producing convergence results**.

You may read:

- The PDF, `code/`, `data/` under `papers/paper_0c7a2c63d8448666a3a7/`;
- `papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/`;
- `menu/pipeline_menu.yaml`, `engine/`.

Do not modify the menu; do not modify other experiments' directories; the shared engine code is considered frozen. On a blocking bug,
copy the needed files into this experiment's `results/reverse_bo/local_engine/` and modify them there, and explain in the report.

## Background (one sentence)

This is reverse engineering, not "running the whole multiverse": given the paper's reported results, use as few queries as possible
to find the analysis path consistent with the paper. Experiment 2's space is anchored on the paper: reference path = the
step / option / choice the paper determines verbatim; the remaining forks are the "concealed choices" that the paper
implicitly fixed but that exist in the menu, expanded by the instantiation agent.

Objective function and metric conventions are in §3 of `agent_prompts/REVERSE_ENGINEERING_MASTER.md`.
By default `obj <= 0.05` means "found the paper-consistent path"; the reference path's obj is usually around 0.005.

## Step 1: validate the reference path

```powershell
python -c "import sys,os; sys.path.insert(0,'.'); from engine.space_loader import SpaceLoader; ld=SpaceLoader(r'papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/multi_space.yaml'); print('forks',len(ld.forks),'universes',ld.universe_count); print(ld.reference_assignment())"
```

Then read:

- `papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/results/reference_check.json`
- `.../results/notes.md`

Confirm the reference path reproduces the paper's three estimates on the 2011 wave (ob_att −0.0880, raw_gap −0.2314,
pooled_ols −0.1505), n_obs=4824. If reference_check is not PASS, fix the space/ops first,
then run BO; do not run with a wrong reference.

Manually evaluate the reference's objective:

```powershell
python -c "import sys,os; sys.path.insert(0,'.'); from engine.space_loader import SpaceLoader; from engine.choice_runtime import ChoiceExecutor; from engine.optimizer_adapter import Evaluator; p=r'papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/multi_space.yaml'; ld=SpaceLoader(p); ev=Evaluator(ld,ChoiceExecutor(ld),targets_json=r'papers/paper_0c7a2c63d8448666a3a7/multiverse/targets.json'); r=ld.reference_assignment(); print('ref_obj',ev.evaluate(r)); print(ev.record(r))"
```

## Step 2: run BOCS / COMBO / the random baseline

Output directory:

`papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/results/reverse_bo/`

Command (long tasks go to a background job):

```powershell
python engine/bo/reverse_benchmark.py `
  --space "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/multi_space.yaml" `
  --targets "papers/paper_0c7a2c63d8448666a3a7/multiverse/targets.json" `
  --exp exp2 --out "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/results/reverse_bo" `
  --methods BOCS-SA,COMBO,RandomSearch --seeds 0,1,2 `
  --warmup 20 --queries 50 --target-obj 0.05 --max-seconds 1800
```

Requirements:

- Every method × seed writes its CSV query by query; keep partials on interruption.
- Kill any COMBO seed exceeding ~30 minutes, keep the partial, report timeout.
- After the runs finish, aggregate/plot:

```powershell
python engine/bo/reverse_benchmark.py `
  --space "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/multi_space.yaml" `
  --targets "papers/paper_0c7a2c63d8448666a3a7/multiverse/targets.json" `
  --exp exp2 --out "papers/paper_0c7a2c63d8448666a3a7/multiverse/exp2_reproduction/results/reverse_bo" `
  --methods BOCS-SA,COMBO,RandomSearch --seeds 0,1,2 --aggregate-only
```

## Step 3: write the report

`results/reverse_bo/REPORT.md` must contain at least:

1. Space: fork count, choice count, universe_count; the reference path's fork→choice mapping
   (may cite the table in `results/notes.md`).
2. Reference obj (about 0.005) and the paper's three target values; the reference is used only as the target, not as warm-up.
3. Result table: final best, queries_to_target,
   first_support_query, success_rate, wall seconds for BOCS / COMBO / Random.
4. Figures: `convergence_best_so_far.png`, `queries_to_target.png`.
5. Conclusions: how fast does BO find the reference-consistent path? BOCS vs COMBO vs random;
   sample efficiency in the 8.96M space.
6. Honest records of partial/timeout/invalid/anomalies.

## What to report back to the parent agent

Report briefly in Chinese:

- universe_count, reference obj;
- `mean_queries_to_target`, `success_rate`, `mean_best_obj` for the three methods;
- the output directory and figure file names;
- failures/timeouts/blockers.

Do not paste large log excerpts.
