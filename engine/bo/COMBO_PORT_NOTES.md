# COMBO Port Notes

## Environment verdict: original could NOT be run -> faithful minimal port

The task first tries to run the **original** COMBO under a `combo37` (Python 3.7)
conda env with `torch==1.6.0` CPU. This was attempted **4 times** and every
attempt died in conda's *transaction execution* step with
`[Errno 36] Resource deadlock avoided`, regardless of stdout/stderr handling or
`Start-Process -Wait`. Logs:

- `engine/bo_logs_conda_create.txt`
- `D:/Anaconda/envs/combo37_create.log`
- `D:/Anaconda/envs/c_out.log` / `c_err.log`

The packages (python-3.7.16 etc.) download fine; conda then hard-deadlocks while
linking them inside this shell's job object, so the env is never created. With no
Python 3.7 interpreter, torch 1.6.0 (which has no modern-Python builds) cannot be
installed. Per the task's fallback clause, COMBO was therefore **minimally and
faithfully ported** to the main interpreter (Python 3.13, torch 2.6.0 CPU).

## What was ported (file-by-file)

Original root: a folder whose Chinese name means "references - combinatorial
Bayesian optimization" (`COMBO-master/COMBO-master/COMBO/`).
Port root: `engine/bo/combo/`.

| Original | Ported file | Notes |
|---|---|---|
| `config.py` | `combo/config.py` | `experiment_directory()` patched to `engine/bo/logs/combo_exp` (original raised on unknown host). |
| `graphGP/modules/gp_modules.py` | `graphgp/modules/gp_modules.py` | verbatim. |
| `graphGP/kernels/graphkernel.py` | `graphgp/kernels/graphkernel.py` | verbatim, relative imports. |
| `graphGP/kernels/diffusionkernel.py` | `graphgp/kernels/diffusionkernel.py` | **verbatim algebra** (Cartesian-product product kernel, `exp(-beta*freq)`, mean-normalization HACK, 1e-6 stabilizer). |
| `graphGP/means/{mean,constant}.py` | `graphgp/means/*` | verbatim. |
| `graphGP/likelihoods/{likelihood,gaussian}.py` | `graphgp/likelihoods/*` | verbatim. |
| `graphGP/models/{gp,gp_regression}.py` | `graphgp/models/*` | verbatim. |
| `graphGP/inference/inference.py` | `graphgp/inference/inference.py` | **torch2 fixes only**: `torch.cholesky(upper=False)` -> `torch.linalg.cholesky`; `torch.triangular_solve(b,L)` -> `torch.linalg.solve_triangular(L,b,upper=False)`. Jitter ladder unchanged. |
| `graphGP/sampler/tool_partition.py` | `graphgp/sampler/tool_partition.py` | `torch.ger` -> `torch.outer`; Kronecker/`group_input`/`ungroup_input`/`direct_product` algebra unchanged. |
| `graphGP/sampler/tool_slice_sampling.py` | `graphgp/sampler/tool_slice_sampling.py` | verbatim. |
| `graphGP/sampler/priors.py` | `graphgp/sampler/priors.py` | verbatim (horseshoe edge-weight prior, partition entropy prior, constants). |
| `graphGP/sampler/sample_hyper.py` | `graphgp/sampler/sample_hyper.py` | verbatim slice sampling of const mean / kernel amp / noise var. |
| `graphGP/sampler/sample_edgeweight.py` | `graphgp/sampler/sample_edgeweight.py` | verbatim slice sampling of per-fork `log_beta` (edge-weight posterior). |
| `graphGP/sampler/sample_posterior.py` | `graphgp/sampler/sample_posterior.py` | verbatim sweep loop; removed the stdout progress bar. |
| `acquisition/acquisition_functions.py` | `acquisition/acquisition_functions.py` | verbatim EI for minimization. |
| `acquisition/acquisition_marginalization.py` | `acquisition/acquisition_marginalization.py` | verbatim EI marginalization over posterior samples + predictive statistics. |
| `acquisition/acquisition_optimizers/graph_utils.py` | `.../graph_utils.py` | verbatim `neighbors` / Cartesian-product neighbors. |
| `acquisition/acquisition_optimizers/greedy_ascent.py` | `.../greedy_ascent.py` | verbatim. |
| `acquisition/acquisition_optimizers/starting_points.py` | `.../starting_points.py` | verbatim; **fixed** the original's broken import (`acquisition_function` -> `acquisition_functions`). |
| `acquisition/acquisition_optimization.py` | `acquisition/acquisition_optimization.py` | kept `next_evaluation`; **removed** the multiprocessing/psutil auto-scaling block (sequential only) and the commented-out simulated-annealing path (it was already disabled upstream). |
| `experiments/test_functions/multiple_categorical.py` (PestControl/Centroid) | `combo_optimizer.py::build_path_graphs` | path-graph adjacency + Laplacian eigendecomp; `torch.symeig` -> `torch.linalg.eigh`. |

## What was NOT replaced

- The surrogate is still the **graph diffusion GP** (product of per-fork graph
  diffusion kernels over the graph Cartesian product) -- NOT a plain categorical
  GP / one-hot RBF kernel.
- Edge weights (`log_beta`) and kernel amplitude/noise are still **posterior
  sampled** (slice sampling) exactly as upstream.
- The acquisition is still **EI marginalized over posterior samples**, optimized
  by **greedy ascent on the graph Cartesian product** (1-Hamming neighbors under
  the sampled partition/edge matrices). No plain EI over a dense grid.

## Numerical parity check

`combo/parity_check.py` (result in `logs/combo_port_parity.txt`):
- path-graph Laplacian eigenvalues vs numpy: max diff `2.4e-7`;
- ported `DiffusionKernel` Gram matrix vs an independent numpy re-derivation of
  the same product diffusion kernel: max rel error `2.7e-7` -> **PASS**.

This confirms the port's core kernel math matches the original formulation to
numerical precision (the remaining difference is float32 vs float64 ordering).
The end-to-end optimization behavior on the synthetic categorical target is in
`logs/synthetic_summary.csv` / `synthetic_best_vs_query.png`.
