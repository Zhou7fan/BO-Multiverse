# Agent C notes (experiment 4)

## Status

| algorithm | adapter | smoke (3 warmup + 3 budget) | status |
|---|---|---|---|
| `BoTorch` | `algos_ext/agent_c.py::BoTorchFast` | 3.1 s, best_d 0.5505, no error | ok |
| `bo_pr` | `algos_ext/agent_c.py::BoPR` | 9.6 s, best_d 0.5618, no error | ok |
| `BayBE` | `algos_ext/agent_c.py::BayBE` | 6.9 s, best_d 0.5618, no error | ok |
| `Ax` | `algos_ext/agent_c.py::AxClient` | 6.1 s, best_d 0.5618, no error | ok |
| `Gryffin` | import attempt in `agent_c.py` | `No module named 'resource'`; no wheel, C build required | not_run |
| `SMAC3` | import attempt in `agent_c.py` | partial import; `No module named 'ConfigSpace'`; pyrfr unavailable | not_run |

Logs and machine-readable records are in `experiment4/algorithms_status/<name>.{log,json}`.

## Environment note

The shared base Anaconda 3.12 environment was being modified concurrently while
this work ran (numpy was upgraded to 2.5.4, which broke its pandas build).  To
keep Agent C's results reproducible and not interfere with other agents, smoke
runs use an isolated venv:

```
experiment4/venvs/agent_c/Scripts/python.exe   # Python 3.11.9
numpy 1.26.4, pandas 2.2.2, scipy 1.14.1, sklearn 1.5.2, torch 2.4.1+cu124
```

The venv was created from conda env `gg` with `--system-site-packages`; BotTorch,
GPyTorch and BoTorch 0.18.1 are loaded lazily from
`D:\Anaconda\Lib\site-packages` via `_ensure_botorch()` in the adapter.  On the
remote runner botorch is already importable, so that fallback is a no-op.  No
packages were installed into the base environment by Agent C.

## Vendoring

* `algos_ext/vendor/bo_pr/discrete_mixed_bo/` - minimal files copied from
  `experiment3/cbo_survey/repos/bo_pr`: `input.py`, `ste.py`, `kernels.py`,
  `wrapper.py`, `probabilistic_reparameterization.py`, `gen.py`,
  `initializers.py`, `optimize.py`, `__init__.py`.
  Two BoTorch 0.18 compatibility shims were applied:
  * `gen.py`: local `NLC_TOL` fallback and a local `_filter_kwargs`
    (`botorch.optim.utils._filter_kwargs` was removed).
  * `initializers.py`: `is_nonnegative` fallback to
    `botorch.optim.initializers`.
* `algos_ext/vendor/py/` - extracted PyPI wheels for BayBE 0.15.0
  (+ attrs 26.1, cattrs 26.2, exceptiongroup) and Ax 1.3.1 (+ plotly,
  pyre-extensions, markdown, graphviz, pymoo).  Adapters prepend this directory
  to `sys.path` only when constructing BayBE/Ax.

## Algorithm mappings

* **BoTorch** - core's `BoTorchGP` is not edited.  The new adapter fits
  `SingleTaskGP` on `codec.ordinal(X)` and scores a random 20k subset of the
  500k pool per query (analytic EI); this reduces a full run from tens of
  minutes per query to ~0.3-1 s.
* **bo_pr** - real probabilistic reparameterization: categorical one-hot input,
  `get_kernel("mixed_categorical")` GP, `ExpectedImprovement` wrapped in
  `MCProbabilisticReparameterization` (64 MC samples), bo_pr's own
  `optimize_acqf`; final candidate is argmax-rounded to choice indices.
* **BayBE** - `SearchSpace.from_dataframe` over a 20k candidate subset (the full
  cartesian product is ~5e7), `Campaign` with a minimize target, default
  two-phase recommender.  Measurements are fed from the runner's observations.
  Recommendations are restricted to that 20k candidate set by design.
* **Ax** - `ax.api.client.Client`, one categorical `ChoiceParameterConfig` per
  non-constant fork, Ax's Center+Sobol+MBM strategy; warm-up and fallback
  observations are attached as completed trials; each suggestion is a pending
  trial closed on the next call.
* **Gryffin** - source-only PyPI sdist (`--only-binary` finds nothing); import
  fails at Unix-only `resource` before the Cython kernels can be compiled.  No
  Windows wheel and no C compiler on this host.
* **SMAC3** - source import raises `No module named 'ConfigSpace'`; SMAC's
  `__init__` partially swallows it.  `ConfigSpace`/`pyrfr` are required and
  `pyrfr` has no Windows wheel.

## Smoke commands

Smoke output was intentionally written to `algos_ext/_smoke` rather than
`results/` to respect the instruction not to touch `results/`:

```
cd experiment4 && ./venvs/agent_c/Scripts/python.exe core/runner.py \
  --space spaces/marshall/multi_space.yaml --algo <NAME> --seed 0 \
  --out algos_ext/_smoke --warmup 3 --budget 3 --max-seconds 600
```

All four working adapters were additionally checked on `spaces/jiang` (seed 1)
and with 5 warm-up + 5 budget proposals on `spaces/marshall`; no errors.

Remote commands for the full runs are in `algos_ext/run_agent_c.sh`
(5 papers x 3 seeds for BoTorch, bo_pr, BayBE, Ax; out `results/agent_c`,
warmup 20, budget 100, max-seconds 3600, pool-n 500000).
