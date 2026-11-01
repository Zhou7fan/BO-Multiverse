# BOCS-SMC : faithful Python port of the MATLAB SMC acquisition in
#   algorithms/binary_smc.m + algorithms/SMC_Code/*.m
#
# Original authors: Ricardo Baptista and Matthias Poloczek (2018)
# Port author: Multiverse engine (see BOCS_SMC_PORT_NOTES.md for the
#   MATLAB -> Python function mapping and numerical differences).
#
# The SMC particle system targets a posterior pi(x) ~ exp(rho * f(x)) over the
# binary space, where f is the (negative) surrogate objective.  As rho grows
# via the geometric-bridge tempering schedule, mass concentrates on the low
# f region.  A weighted logistic regression (logistic_A) is fitted as an MCMC
# proposal, particles are resampled when ESS drops, and moved with a
# Metropolis kernel.  This module operates on flattened binary vectors and
# mirrors the MATLAB algebra 1:1.

import numpy as np


# ---------------------------------------------------------------------------
# eff_sample_size.m  ->  eff_sample_size
# ---------------------------------------------------------------------------
def eff_sample_size(weights):
    if np.all(np.isnan(weights)):
        return 0.0
    return 1.0 / np.sum(weights ** 2)


# ---------------------------------------------------------------------------
# unique_particles.m -> unique_particles
# ---------------------------------------------------------------------------
def unique_particles(binary_particles):
    n_models = binary_particles.shape[0]
    uniq = np.unique(binary_particles, axis=0)
    return uniq.shape[0] / float(n_models)


# ---------------------------------------------------------------------------
# samp_stats.m       ->  samp_stats
# ---------------------------------------------------------------------------
def samp_stats(models, weights):
    n_samples, n_vars = models.shape
    weights = weights.reshape(-1)
    # weighted mean per coordinate
    s_mean = weights @ models                      # (n_vars,)
    s_cov = np.zeros((n_vars, n_vars))
    s_corr = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1):
            cov_ij = np.sum(weights * models[:, i] * models[:, j])
            s_cov[i, j] = cov_ij
            denom = np.sqrt(s_mean[i] * (1 - s_mean[i]) *
                            s_mean[j] * (1 - s_mean[j]))
            s_corr[i, j] = (cov_ij - s_mean[i] * s_mean[j]) / denom if denom > 0 else 0.0
    return s_mean, s_cov, s_corr


# ---------------------------------------------------------------------------
# evaluate_logit.m   ->  evaluate_logit
# ---------------------------------------------------------------------------
def evaluate_logit(logistic_A, model):
    # product-of-Bernoulli proposal density q(model) under triangular logistic_A
    q = 1.0
    n_vars = model.shape[0]
    for i in range(n_vars):
        logit_p = logistic_A[i, i] + logistic_A[i, :i] @ model[:i]
        p = 1.0 / (1.0 + np.exp(-logit_p))
        q *= (p ** model[i]) * ((1.0 - p) ** (1.0 - model[i]))
    return q


# ---------------------------------------------------------------------------
# logistic_sample.m  ->  logistic_sample
# ---------------------------------------------------------------------------
def logistic_sample(n_vars, logistic_A, rng):
    binary_sample = np.zeros(n_vars)
    p = 1.0
    for j in range(n_vars):
        if j == 0:
            regressor = logistic_A[j, j]
        else:
            regressor = logistic_A[j, j] + logistic_A[j, :j] @ binary_sample[:j]
        pr = 1.0 / (1.0 + np.exp(-regressor))
        bit = 1.0 if rng.random() < pr else 0.0
        binary_sample[j] = bit
        p = p * pr if bit == 1.0 else p * (1.0 - pr)
    return binary_sample, p


# ---------------------------------------------------------------------------
# fit_logistic.m     ->  fit_logistic
# ---------------------------------------------------------------------------
def fit_logistic(models, weights, logistic_A):
    # Weighted IRLS logistic regression that fills the (lower-triangular)
    # proposal matrix A row by row.  Faithful to fit_logistic.m.
    n_samples, n_vars = models.shape
    delta = 0.01
    epsilon = 0.1
    eps_mean = 0.02
    eps_corr = 0.075
    min_marg_prob = 1e-10
    iter_max = 50

    s_mean, _, s_corr = samp_stats(models, weights)

    for i in range(n_vars):
        if s_mean[i] <= eps_mean or s_mean[i] >= (1.0 - eps_mean):
            s_mean_adj = s_mean[i] * (1.0 - min_marg_prob) + min_marg_prob * 0.5
            logistic_A[i, i] = np.log(s_mean_adj) - np.log(1.0 - s_mean_adj)
            logistic_A[i, :i] = 0.0
            continue

        # indices (among columns 0..i-1) whose correlation with i exceeds threshold
        corr_cols = np.where(np.abs(s_corr[i, :i]) > eps_corr)[0]
        Z = np.column_stack([models[:, corr_cols], np.ones(n_samples)])
        y = models[:, i]
        a_old = np.append(logistic_A[i, corr_cols], logistic_A[i, i])

        error = np.inf
        counter = 0
        while error > delta and counter < iter_max:
            # p_log = sigmoid(Z a_old); q_log = p(1-p), vectorized (stable)
            z_a = np.clip(Z @ a_old, -30, 30)
            p_log = 1.0 / (1.0 + np.exp(-z_a))
            q_log = p_log * (1.0 - p_log)
            n_nugget = Z.shape[1]
            W = weights.reshape(-1) * q_log
            H = (Z.T * W) @ Z + epsilon * np.eye(n_nugget)
            g = (Z.T * weights.reshape(-1)) @ (y - p_log) - epsilon * a_old
            a_new = np.linalg.solve(H, g)
            error = np.max(np.abs(a_new - a_old))
            a_old = a_new
            counter += 1

        logistic_A[i, corr_cols] = a_new[:-1]
        logistic_A[i, i] = a_new[-1]

    return logistic_A


# ---------------------------------------------------------------------------
# importance_weights.m -> importance_weights
# ---------------------------------------------------------------------------
def importance_weights(models, model_val, old_wts, old_rho, new_rho):
    n_models = models.shape[0]
    w_new = np.zeros(n_models)
    for i in range(n_models):
        post_new = new_rho * model_val[i]
        post_old = old_rho * model_val[i]
        w_new[i] = old_wts[i] * np.exp(post_new - post_old)
    nan_mask = np.isnan(w_new)
    w_new[nan_mask] = 0.0
    s = np.sum(w_new)
    if s > 0:
        w_new = w_new / s
    return w_new


# ---------------------------------------------------------------------------
# binary_resampling.m -> systematic_resample  (systematic resampling)
# ---------------------------------------------------------------------------
def systematic_resample(binary_models, weights, rng):
    n_samples, n_var = binary_models.shape
    x_new = np.zeros((n_samples, n_var))
    idx = np.zeros(n_samples, dtype=int)
    u = rng.random()
    v = weights.reshape(-1) * n_samples
    c = v[0]
    counter = 0
    for i in range(n_samples):
        while c < u:
            counter += 1
            c += v[counter]
        x_new[i] = binary_models[counter]
        idx[i] = counter
        u = u + 1.0
    return x_new, idx


# ---------------------------------------------------------------------------
# smc_resample.m     ->  smc_resample
# ---------------------------------------------------------------------------
def smc_resample(ps, rng):
    new_models, new_idx = systematic_resample(ps['models'], ps['weights'], rng)
    n_models = new_models.shape[0]
    ps['models'] = new_models
    ps['model_val'] = ps['model_val'][new_idx]
    ps['weights'] = np.ones(n_models) / n_models
    return ps


# ---------------------------------------------------------------------------
# find_tolerance.m   ->  find_tolerance
# ---------------------------------------------------------------------------
def find_tolerance(ps):
    alpha = ps['alpha_star']
    rho = ps['rho']
    old_wts = ps['weights']
    lower_bd = rho
    upper_bd = 5.0 * rho
    epsilon = rho / 1e3
    old_ess = eff_sample_size(old_wts)
    ess_star = alpha * old_ess
    rho_t = (lower_bd + upper_bd) / 2.0
    ps_weights = importance_weights(ps['models'], ps['model_val'], ps['weights'], ps['rho'], rho_t)
    count_temp = 0
    while abs(upper_bd - lower_bd) >= epsilon:
        ps_weights = importance_weights(ps['models'], ps['model_val'], ps['weights'], ps['rho'], rho_t)
        if eff_sample_size(ps_weights) > ess_star:
            lower_bd = rho_t
            rho_t = (rho_t + upper_bd) / 2.0
        else:
            upper_bd = rho_t
            rho_t = (rho_t + lower_bd) / 2.0
        count_temp += 1
        if count_temp > 500:
            break
    while np.all(np.isnan(ps_weights)):
        rho_t = rho_t - epsilon
        ps_weights = importance_weights(ps['models'], ps['model_val'], ps['weights'], ps['rho'], rho_t)
    ps['weights'] = ps_weights
    ps['rho'] = rho_t
    return ps


# ---------------------------------------------------------------------------
# move_particles.m   ->  move_particles
# ---------------------------------------------------------------------------
def move_particles(ps, rng):
    n_vars = ps['n_vars']
    model_orig = ps['models'].copy()
    model_val = ps['model_val'].copy()
    rho = ps['rho']
    logistic_A = ps['logistic_A']

    n_models = model_orig.shape[0]
    diversity_old = unique_particles(model_orig)
    mh_accept_vect = []
    total_fn_count = ps['total_fn_count'][-1]

    iter_temp = 0
    while True:
        new_model = np.zeros((n_models, n_vars))
        new_mval = np.zeros(n_models)
        accept_ratio = 0.0

        for i in range(n_models):
            prop_model, _ = logistic_sample(n_vars, logistic_A, rng)
            q_new = evaluate_logit(logistic_A, prop_model)
            q_old = evaluate_logit(logistic_A, model_orig[i])
            total_fn_count += 1
            model_obj_new = ps['objective'](prop_model)
            log_post_new = rho * model_obj_new
            log_post_old = rho * model_val[i]
            accept_prob = np.exp(log_post_new - log_post_old) * (q_old / q_new)
            if rng.random() < min(1.0, accept_prob):
                new_model[i] = prop_model
                new_mval[i] = model_obj_new
                accept_ratio += 1.0
            else:
                new_model[i] = model_orig[i]
                new_mval[i] = model_val[i]

        model_orig = new_model
        model_val = new_mval
        mh_accept_vect.append(accept_ratio / n_models)

        diversity_new = unique_particles(new_model)
        if abs(diversity_new - diversity_old) < 0.02 or diversity_new > 0.95:
            break
        diversity_old = diversity_new
        iter_temp += 1
        if iter_temp > 100:
            break

    ps['models'] = model_orig
    ps['model_val'] = model_val
    ps['mh_accept'] = min(mh_accept_vect)
    ps['total_fn_count'] = np.append(ps['total_fn_count'], total_fn_count)
    return ps


# ---------------------------------------------------------------------------
# binary_smc.m       ->  binary_smc
# ---------------------------------------------------------------------------
def binary_smc(objective, n_vars, n_models=10, evalBudget=100,
               init_rho=1e-5, alpha=0.90, delta=0.2, rng=None):
    """Run SMC acquisition over the flattened binary space.

    objective(x) : callable on a 1-D binary row vector, returns the surrogate
                   value f(x) we want to MINIMIZE.  (Internally we target
                   pi(x) ~ exp(-rho * f(x)).)
    Returns the best binary row vector found (1-D, length n_vars).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    ps = {}
    ps['n_vars'] = n_vars
    ps['rho'] = init_rho
    ps['alpha_star'] = alpha
    ps['mh_accept'] = 1.0

    # initial uniform particles
    ps['models'] = (rng.random((n_models, n_vars)) < 0.5).astype(float)
    # target posterior is exp(rho * (-objective))
    ps['objective'] = lambda x: -1.0 * objective(x)
    ps['model_val'] = np.array([ps['objective'](ps['models'][i]) for i in range(n_models)])
    ps['total_fn_count'] = np.array([n_models], dtype=float)
    ps['weights'] = np.ones(n_models) / n_models
    ps['logistic_A'] = np.eye(n_vars)

    ps = find_tolerance(ps)
    ps['ess'] = eff_sample_size(ps['weights'])
    ps['pdiv'] = unique_particles(ps['models'])

    best_idx = np.argmax(ps['weights'])
    model_best = ps['models'][best_idx].copy()
    obj_best = -1.0 * ps['objective'](model_best)

    counter = 1
    while ps['pdiv'] > delta:
        ps['logistic_A'] = fit_logistic(ps['models'], ps['weights'], ps['logistic_A'])
        ps = smc_resample(ps, rng)
        ps = move_particles(ps, rng)
        ps = find_tolerance(ps)
        counter += 1
        ps['ess'] = eff_sample_size(ps['weights'])
        ps['pdiv'] = unique_particles(ps['models'])

        cur_idx = np.argmax(ps['weights'])
        model_temp = ps['models'][cur_idx]
        obj_temp = -1.0 * ps['objective'](model_temp)
        if obj_temp <= obj_best:
            model_best = model_temp.copy()
            obj_best = obj_temp

        if ps['total_fn_count'][-1] > evalBudget:
            break

    return model_best, obj_best
