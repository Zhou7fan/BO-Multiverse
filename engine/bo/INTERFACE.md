# BO Optimizer Interface (engine/bo)

Four optimizers share one ask–tell protocol so `engine/optimizer_adapter` can
drive them interchangeably. All operate on the **choice-index assignment** space
defined in `ENGINE_CONTRACT.md` §8.

## Unified representation

- **assignment** : 1-D `int` array of length `n_forks`. Entry `assignment[f]` is
  the choice index picked at fork `f`, in `[0, n_vertices[f])`.
- **observed_X** : `(n_obs, n_forks)` array (or list of assignments) of all
  assignments evaluated so far, **including the warm-ups**.
- **observed_y** : `(n_obs,)` float array of the external objective at those
  assignments. LOWER is better; the external evaluator adds a large penalty for
  invalid (0-hot / multi-hot) configurations.
- A valid assignment always has exactly one hot bit per fork.

## Common constructor

```python
Optimizer(n_vertices, warmups=None, seed=0, **algo_kwargs)
```

- `n_vertices` : list of choice counts per fork, e.g. `[4,4,4,3,3,3]`.
- `warmups`    : list of assignments (optional; the adapter passes them in via
  `observed_X/y` on the first `suggest` call instead).
- `seed`       : RNG seed for determinism.

## Common method

```python
assignment = optimizer.suggest(observed_X, observed_y)   # -> shape (n_forks,) int
```

Each `suggest` call = one BO iteration: the optimizer **retrains its surrogate /
resamples its posterior** on the supplied observations and returns the single next
assignment to evaluate. The external evaluator then computes its `objective` and
calls `suggest` again with the appended observation.

---

## 1. BOCS-SA  (`bo/bocs/optimizers.py::BOCS_SA`)

- Encodes each assignment as a flattened **one-hot block** binary vector
  (`bo/bocs/encoding.py::BlockEncoder`): fork `f` occupies bits
  `[offsets[f], offsets[f+1])`, exactly one 1.
- Surrogate: order-2 Bayesian linear model with the **horseshoe** prior
  (`bo/bocs/LinReg.py`, `bo/bocs/bhs.py`) -- drawn posterior coefficient `alpha`.
- Acquisition: simulated annealing on the surrogate with **block-internal flips**
  (pick a fork, move its hot bit to another choice in the same fork).
- Constructor extras: `order=2`, `lam=1e-4`, `sa_iter=200`, `sa_reruns=5`.

## 2. BOCS-SMC  (`bo/bocs/optimizers.py::BOCS_SMC`)

- Same block-one-hot encoding and same order-2 horseshoe surrogate.
- Acquisition: faithful SMC port (`bo/bocs/smc.py`) -- geometric-bridge tempering
  on `rho`, weighted logistic-regression move kernel, importance weights, ESS
  bisection, systematic resampling, Metropolis moves -- then a block-internal
  coordinate polish on the surrogate to land on the one-hot manifold.
- Constructor extras: `order=2`, `lam=1e-4`, `smc_particles=20`, `smc_budget=200`.

## 3. COMBO  (`bo/combo/combo_optimizer.py::COMBOOptimizer`)

- Operates directly on choice indices (each fork = one categorical vertex).
- Each fork's graph is a **path graph**; Fourier frequencies/basis are precomputed
  from the Laplacian eigendecomposition (`build_path_graphs`).
- Surrogate: **graph diffusion GP** over the graph Cartesian product
  (`graphgp/kernels/diffusionkernel.py`) with slice-sampled edge weights and
  kernel hyperparameters.
- Acquisition: EI **marginalized over posterior samples**, optimized by greedy
  ascent on the graph Cartesian product.
- Constructor extras: `n_post=10` (posterior samples for EI), `n_burn=50`.

## 4. RandomSearch  (`bo/random_search.py::RandomSearch`)

- Uniform draw over the valid one-hot space each `suggest`. No surrogate. Baseline.

---

## Wiring to `engine/optimizer_adapter`

```
n_vertices   = [len(choices_f) for each activated fork]
warmups      = adapter's fixed diverse spec points (never the reference path)
opt          = <chosen class>(n_vertices, warmups, seed)
obs_X, obs_y = (warmups, [evaluate(w) for w in warmups])
for query in range(budget):
    a        = opt.suggest(obs_X, obs_y)     # choice-index assignment
    uni      = run_universe(a)               # adapter runs one multiverse
    y        = objective(uni)                 # lower = better (invalid -> big penalty)
    obs_X.append(a); obs_y.append(y)
```

Encoding/decoding notes:
- BOCS classes do the one-hot encode/decode internally (`BlockEncoder`); the
  adapter only ever speaks choice-index assignments.
- COMBO consumes choice indices natively.
- Invalid assignments are never emitted by the BO classes (block-internal moves);
  the external evaluator's large-penalty clause exists for safety, not because the
  optimizers propose invalid points.
