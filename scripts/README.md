# Reproduction scripts

Scripts that reproduce the computations behind the tables and figures of
the technical appendix. All scripts run from the repository root with the
project virtual environment (Python 3.12; numpy, pandas, scikit-learn,
statsmodels, scipy, torch, botorch, gpytorch).

| Appendix item | Script |
|---|---|
| Table 2 (search for the published value) | `core/report.py` over the per-seed logs |
| Table 4 (analyses that reproduce a value) | `core/report.py` (hitting-time estimates from `Table 2` random medians) |
| Table 5 (wage-gap map) | `spaces/wage_gap/multi_space.yaml` |
| Table 6 (optimizer comparison, first run) | `logs/wage_gap/runs_all.csv`, `logs/wage_gap/summary.md` |
| Table 7 (distinct estimates within tolerance) | `harvest_minmax.py` / `compare_ref.py` over `logs/wage_gap` |
| Table 8 (decision fingerprints) | `fingerprint.py` (traces + dense draws) |
| Direct counts (dense uniform sampling) | `dense_count.py` |
| Attack searches (Section 5.5, 7.4) | `attack_runner.py`, then `collect_attack.py` |
| Figure 5-8 (trajectory plates) | generated from the per-seed traces (four-paper report plates) |
| Wage-gap search traces (Figure 3-4) | `logs/wage_gap/results_p17_v2_data.json` (plates from the first-run report) |

Usage examples:

```
python scripts/dense_count.py --space spaces/charity/multi_space.yaml \
    --n 10000 --workers 24 --precision 0.00005 --out results/dense
python scripts/dense_count.py --space spaces/wage_gap/multi_space.yaml \
    --n 3000 --workers 24 --precision 0.00005 --out results/dense
python scripts/fingerprint.py --space spaces/bureaucracy/multi_space.yaml \
    --precision 0.0005 --traces results/bureaucracy --dense results/dense/bureaucracy_dense_draws.csv \
    --out results/fingerprints
python scripts/attack_runner.py --space spaces/wage_gap/multi_space.yaml \
    --algo GP-EI --seed 0 --direction 1 --out results/attack --warmup 20 --budget 100
```

Dense-sampling sizes used in the appendix: 10,000 draws for the turnout and
charitable-giving spaces, and 3,000 for the bureaucracy and wage-gap spaces.

The dense-sampling draws and attack runs behind the numbers in the appendix
are not bundled (they were run on the project server); the scripts regenerate
them deterministically from the space definitions above.
