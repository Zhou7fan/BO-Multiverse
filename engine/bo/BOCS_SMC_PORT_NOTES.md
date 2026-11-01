# BOCS-SMC Python Port Notes

This file documents the faithful port of the **BOCS SMC acquisition** from MATLAB
(`algorithms/binary_smc.m` + `algorithms/SMC_Code/*.m`) to Python
(`engine/bo/bocs/smc.py`). The statistical surrogate is unchanged: the order-2
Bayesian linear model with the horseshoe prior (`LinReg.py` + `bhs.py`, copied
verbatim from `BOCSpy/`). SMC is *only* an acquisition / proposal mechanism; it
never replaces the surrogate.

## Why SMC had to be ported

The reference `BOCSpy/` ships only `BOCS.py` with **SA** and **SDP-l1**
acquisitions. The SMC acquisition exists only in the MATLAB tree
(`algorithms/binary_smc.m`, `algorithms/SMC_Code/*.m`). To get a real
BOCS-SMC optimizer we ported it line-by-line.

## MATLAB function -> Python function map

| MATLAB (`algorithms/...`)                         | Python (`bocs/smc.py`)        | Notes |
|---|---|---|
| `binary_smc.m`                                    | `binary_smc()`                | Main loop: tempering `rho`, loop while particle diversity > `delta`; returns best particle + its objective. |
| `SMC_Code/samp_stats.m`                            | `samp_stats()`                | Weighted mean / cov / lower-triangular correlation. Returns `(s_mean, s_cov, s_corr)` to match MATLAB's 3-output signature. |
| `SMC_Code/fit_logistic.m`                          | `fit_logistic()`              | Weighted IRLS filling the triangular proposal matrix `logistic_A`. Same constants (`delta=0.01`, `epsilon=0.1`, `eps_mean=0.02`, `eps_corr=0.075`, `min_marg_prob=1e-10`, `iter_max=50`). |
| `SMC_Code/evaluate_logit.m`                        | `evaluate_logit()`             | Product-of-Bernoulli density `q(model)` under triangular `logistic_A`. |
| `SMC_Code/logistic_sample.m`                       | `logistic_sample()`            | Sequential Bernoulli proposal draw; returns `(sample, q)`. |
| `SMC_Code/importance_weights.m`                   | `importance_weights()`         | Geometric-bridge weight update `w_new = w_old * exp((new_rho-old_rho)*f)`, renormalized, NaNs -> 0. |
| `SMC_Code/find_tolerance.m`                        | `find_tolerance()`             | Bisection on `rho` targeting `ESS = alpha * old_ESS`; same bounds (`lower=rho`, `upper=5rho`, `eps=rho/1e3`, `<=500` iters). |
| `SMC_Code/eff_sample_size.m`                       | `eff_sample_size()`           | `1/sum(w^2)`, 0 if all NaN. |
| `SMC_Code/smc_resample.m`                         | `smc_resample()`               | Resample + reset uniform weights. |
| `SMC_Code/binary_resampling.m`                    | `systematic_resample()`        | Systematic resampling (1-indexed MATLAB -> 0-indexed Python). |
| `SMC_Code/move_particles.m`                       | `move_particles()`             | Metropolis move with logistic proposal; accept `exp(rho(df))*q_old/q_new`; loop until diversity change `<0.02` or `>0.95`, max 100 iters. |
| `SMC_Code/unique_particles.m`                     | `unique_particles()`           | `#unique rows / n_particles`. |
| `discrete_sample.m`, `pascal_row.m`               | (not ported)                   | Helper utilities not on the SMC call path (used by other MATLAB variants); nothing in `binary_smc.m` calls them. |

## Numerical differences and rationale

1. **RNG**: MATLAB uses the global `rand`/`binornd` stream. The Python port takes
   an injected `numpy.random.Generator` (`rng=`); `logistic_sample` implements
   `binornd(1,p)` as `rng.random() < p`. This makes the 5-seed experiment
   deterministic per seed. Exact sample-for-sample match to MATLAB is not claimed
   (different RNG), but the *distribution* and algorithm are identical.

2. **`fit_logistic` correlated-regressor selection**. MATLAB builds
   `L_i = abs(s_corr(i,1:i-1)) > eps_corr` and then writes
   `Z = [models(:,L_i), ones]`. In MATLAB `L_i` is a length-`i-1` logical mask
   over the prefix of columns `< i`. In Python we reproduce this unambiguously
   with integer indices `corr_cols = np.where(abs(s_corr[i,:i]) > eps_corr)` and
   select `models[:, corr_cols]`. Algebra (the IRLS normal equations
   `(Z'WZ + eps I) a = Z'W(y-p) - eps a_old`) is identical.

3. **Stable sigmoid**: the IRLS update and `evaluate_logit` clip the logit to
   `[-30, 30]` to avoid `exp` overflow warnings on large `z_a`; values are
   unchanged to machine precision in the saturated regime.

4. **Posterior sign convention**. MATLAB sets `ps.objective = @(x) -f(x)` and
   tempers `pi(x) ~ exp(rho * ps.objective) = exp(-rho*f(x))`. We minimize `f`
   (the surrogate); the port keeps `-objective = f`, identical convention.

## Deliberate adaptation to the block-one-hot problem

The raw `binary_smc` operates on an arbitrary flattened binary vector. Our
feasible set is the **one-hot manifold** (one hot bit per fork block). In
`bocs/optimizers.py::BOCS_SMC.suggest`:

- particles start from the trained surrogate;
- `binary_smc` runs the faithful tempering/logistic-proposal dynamics on the
  flattened vector (this is the SMC acquisition);
- the returned particle is **projected** onto the one-hot manifold (per-block
  argmax) and then polished with **block-internal coordinate moves** on the same
  surrogate -- exactly the "neighborhood moves restricted to within a block"
  rule, symmetric to how BOCS-SA only flips within a block.

The external evaluator is responsible for penalizing any invalid (0 or >1 hot)
configuration; our optimizers never emit one.
