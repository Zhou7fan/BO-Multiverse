# Reproduces the BOCSpy example_bqp.py (quadratic binary problem) using the
# copied & import-fixed modules in this package, SA acquisition only.
# Verifies that LinReg (order-2 Bayesian linear model + horseshoe bhs) trains
# and that BOCS-SA drives the best-f-vs-query down. Saves a plot + log.

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bo.bocs.LinReg import LinReg
from bo.bocs.quad_mat import quad_mat
from bo.bocs.sample_models import sample_models


def sa_optimize(surrogate, n_vars, n_iter, rng):
    old_x = sample_models(1, n_vars)
    old_obj = surrogate(old_x)
    best_x, best_obj = old_x.copy(), old_obj
    T = 1.0
    for t in range(n_iter):
        T = 0.8 * T
        flip = rng.integers(0, n_vars)
        new_x = old_x.copy()
        new_x[0, flip] = 1.0 - new_x[0, flip]
        new_obj = surrogate(new_x)
        if (new_obj < old_obj) or (rng.random() < np.exp((old_obj - new_obj) / T)):
            old_x, old_obj = new_x, new_obj
        if new_obj < best_obj:
            best_x, best_obj = new_x.copy(), new_obj
    return best_x, best_obj


def main():
    rng = np.random.default_rng(10)
    np.random.seed(10)
    n_vars = 10
    evalBudget = 50
    n_init = 10
    lam = 1e-4
    Q = quad_mat(n_vars, 10)

    def model(x):
        return (x.dot(Q) * x).sum(axis=1)

    def penalty(x):
        return lam * np.sum(x, axis=1)

    x_vals = sample_models(n_init, n_vars)
    y_vals = model(x_vals)
    inputs = {'n_vars': n_vars, 'x_vals': x_vals, 'y_vals': y_vals}

    LR = LinReg(n_vars, 2)
    objs = []
    for t in range(evalBudget - n_init):
        LR.train(inputs)
        alpha_t = LR.alpha
        stat = lambda x: LR.surrogate_model(x, alpha_t) + penalty(x)
        best_x, _ = sa_optimize(stat, n_vars, n_iter=200, rng=rng)
        y_new = model(best_x)
        inputs['x_vals'] = np.vstack([inputs['x_vals'], best_x])
        inputs['y_vals'] = np.hstack([inputs['y_vals'], y_new])
        objs.append(float(y_new.reshape(-1)[0] + penalty(best_x).reshape(-1)[0]))

    # brute-force optimum
    grid = np.array([[int(b) for b in f'{i:0{n_vars}b}'] for i in range(2 ** n_vars)], dtype=float)
    opt_f = np.min(model(grid) + penalty(grid))
    cum_best = np.minimum.accumulate(objs)

    log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'bocs_bqp_sa_smoke.txt')
    with open(log_path, 'w') as fh:
        fh.write(f"BQP n_vars={n_vars}, budget={evalBudget}, n_init={n_init}\n")
        fh.write(f"brute-force optimum f*={opt_f:.6f}\n")
        fh.write(f"BOCS-SA final best f={cum_best[-1]:.6f}, gap={cum_best[-1]-opt_f:.6f}\n")
        fh.write("iter, obj, cum_best\n")
        for i, o in enumerate(objs):
            fh.write(f"{i}, {o:.6f}, {cum_best[i]:.6f}\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(objs)), np.abs(cum_best - opt_f), 'r-o', ms=3, label='BOCS-SA')
    ax.set_yscale('log')
    ax.set_xlabel('query t (after warmup)')
    ax.set_ylabel('|best f - f*|')
    ax.set_title('BOCS-SA on quadratic binary problem (BQP)')
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), '..', 'logs', 'bocs_bqp_sa_best_vs_query.png')
    fig.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")
    print(f"opt_f={opt_f:.4f} final_best={cum_best[-1]:.4f} gap={cum_best[-1]-opt_f:.4f}")


if __name__ == '__main__':
    main()
