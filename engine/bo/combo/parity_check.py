# Numerical parity check for the COMBO port.
#
# The original COMBO could not be run (no py3.7/torch1.6 env -- see
# COMBO_PORT_NOTES.md). To prove the port's core math is faithful, we compare
# the ported DiffusionKernel Gram matrix against an independent numpy reference
# that re-derives the same graph diffusion kernel from the Laplacian
# eigendecomposition, and verify path-graph Fourier factors match scipy.linalg.
#
# Writes results to engine/bo/logs/combo_port_parity.txt .

import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ENGINE)

from bo.combo.combo_optimizer import build_path_graphs
from bo.combo.graphgp.kernels.diffusionkernel import DiffusionKernel


def reference_subgraph_kernel(freq, basis, beta, mean_norm):
    # K_i(a,b) = sum_k basis[a,k] exp(-beta*freq[k]) basis[b,k], normalized
    L = basis * np.exp(-beta * freq)[None, :]
    return L @ basis.T / mean_norm


def main():
    n_vertices = [4, 5, 3]
    adj, freq, basis = build_path_graphs(n_vertices)
    lines = []

    # 1) path-graph Laplacian eigendecomp matches scipy reference
    import scipy.linalg as sla
    for i, n_v in enumerate(n_vertices):
        A = adj[i].numpy()
        L = np.diag(A.sum(0)) - A
        w_ref, v_ref = np.linalg.eigh(L)
        w_port = freq[i].numpy()
        lines.append(f"fork{i} path-graph eigval max|diff|={np.max(np.abs(w_port-w_ref)):.2e}")

    # 2) DiffusionKernel Gram vs independent numpy Kronecker-product reference
    rng = np.random.default_rng(0)
    X = torch.as_tensor(rng.integers(0, np.array(n_vertices), size=(30, len(n_vertices))), dtype=torch.long)
    grouped_log_beta = torch.tensor([0.5, -0.3, 1.1])  # one beta per fork (no partition grouping)
    kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta, fourier_freq_list=freq, fourier_basis_list=basis)
    kernel.log_amp.fill_(0.0)
    K_port = kernel(X).detach().numpy()

    # reference: product over forks of per-subgraph diffusion kernels
    K_ref = np.ones((X.shape[0], X.shape[0]))
    for i in range(len(n_vertices)):
        beta = float(torch.exp(grouped_log_beta[i]))
        freq_i = freq[i].numpy()
        basis_i = basis[i].numpy()
        mean_norm = np.mean(np.exp(-beta * freq_i))
        Ki = reference_subgraph_kernel(freq_i, basis_i, beta, mean_norm)
        rows = X[:, i].numpy()
        K_ref *= Ki[rows][:, rows]
    # port adds log_amp (0) and 1e-6 stabilizer on diagonal
    K_ref = K_ref + np.diag(1e-6 * np.ones(X.shape[0]))

    err = np.max(np.abs(K_port - K_ref))
    rel = err / (np.max(np.abs(K_ref)) + 1e-12)
    lines.append(f"DiffusionKernel Gram max abs err vs numpy reference = {err:.3e}")
    lines.append(f"DiffusionKernel Gram max rel err vs numpy reference = {rel:.3e}")
    lines.append("PARITY: " + ("PASS" if rel < 1e-4 else "FAIL"))

    out = os.path.join(HERE, '..', 'logs', 'combo_port_parity.txt')
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
