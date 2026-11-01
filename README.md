# Open materials — "Reading Science Backwards"

Pre-experiment materials for the four-paper study: the operation menu in its
final form (no per-paper provenance listed), the code, prompts, and adapters
used to build the spaces and run the searches, the multiverse space
definitions for the four papers, and the scripts that reproduce the
appendix tables and figures. Search logs are not bundled; the materials here
are what a re-run starts from.

```
menu/
  operation_menu.json   final operation menu (families / methods / rows)
engine/                 specification runtime and space loader; BOCS, COMBO,
                        random search (space_loader.py, choice_runtime.py, bo/)
core/                   experiment runner, evaluator, codec, algorithms,
                        report/plot builders (the four-paper harness)
algos_ext/              adapters for the third-party optimizers (CASMOPOLITAN,
                        Bounce, BODi, MerCBO, HEBO, bo_pr, GP-*, BoTorch, Ax,
                        BayBE, Summit, bayes_optim); see algos_ext/THIRD_PARTY.md
scripts/                scripts that reproduce the appendix tables and figures
                        (fingerprints, dense counts, attacks, config check)
spaces/
  wage_gap/             gender wage gap (PSID; target -0.1505, tol 0.0005)
  bureaucracy/          Chinese bureaucracy (target 0.336, tol 0.0034)
  turnout/              globalization & turnout (target 0.5384, tol 0.0054)
  charity/              charitable giving (target 1.8477, tol 0.0554)
```

Each space contains `multi_space.yaml` (the fork/choice definitions and
universe size), `ops/` (executable operations), `targets.json` (target value
and tolerance), and `reference_verification.json` where available.

## Rerunning

Search protocol: 17 Bayesian / combinatorial optimizers plus a random-uniform
baseline, 3 seeds each, 20 warm-up draws + 100 further steps, objective
`|estimate - target|` with tolerance from `targets.json`.

The harness that evaluates a specification and runs the optimizers is
`core/` (runner, evaluator, algorithms) on top of `engine/`
(`engine/space_loader.py`, `engine/choice_runtime.py`, `engine/bo/`). Point it
at one of the spaces here, e.g.:

```
python core/runner.py --space spaces/wage_gap/multi_space.yaml \
  --algo COMBO --seed 0 --out results/wage_gap --warmup 20 --budget 100
```

See `scripts/README.md` for the scripts behind each appendix table and figure.

## Data

Data files are not bundled (large / access-restricted). Download before rerunning:

- wage_gap: PSID 1990-2019 (https://psidonline.isr.umich.edu), Blau-Kahn 2017
  extract as in the GSB DBI career-wage-gaps replication package; see
  `data/build_wage_gap_sample.py` (`data/psid_blau_kahn_2017.csv`).
- bureaucracy: Multi100 replication data, https://osf.io/8w4se
  (`data/replication.dta`).
- turnout: Multi100 replication data, https://osf.io/qfm2g (`data/Turnout.dta`).
- charity: Multi100 replication data, https://osf.io/hd8e6
  (`data/workingdata2009.csv`).

Candidate pools (`pool*.npz`, 200k uniform points for the acquisition / random
sampler) are regenerated from `meta.selection_seed` in each `multi_space.yaml`
via `sample_assignments` in the main repository; they are not bundled.

## Notes on completeness

- Search logs and the dense-sampling draws are not bundled. The scripts above
  regenerate them from the frozen spaces, so every appendix number can be
  recomputed from the materials here.
- DEIDENTIFICATION.md lists the renames applied to the menu.
