import math

import torch

from .graphkernel import GraphKernel


class DiffusionKernel(GraphKernel):
	"""
	Graph diffusion kernel over the vertices of the graph Cartesian product.
	Ported verbatim from COMBO/graphGP/kernels/diffusionkernel.py: the kernel is
	the product (Cartesian product of sub-graphs) of per-subgraph diffusion
	kernels K_i(x_i, x_i') = sum_k phi_k(x_i) exp(-beta_i lambda_k) phi_k(x_i').
	Edge scale (beta) is NOT baked into the module; it is sampled downstream.
	"""
	def __init__(self, grouped_log_beta, fourier_freq_list, fourier_basis_list):
		super(DiffusionKernel, self).__init__(grouped_log_beta, fourier_freq_list, fourier_basis_list)

	def forward(self, x1, x2=None, diagonal=False):
		if diagonal:
			assert x2 is None

		stabilizer = 0
		if x2 is None:
			x2 = x1
			if diagonal:
				stabilizer = 1e-6 * x1.new_ones(x1.size(0), 1, dtype=torch.float32)
			else:
				stabilizer = torch.diag(1e-6 * x1.new_ones(x1.size(0), dtype=torch.float32))

		full_gram = 1
		for i in range(len(self.fourier_freq_list)):
			beta = torch.exp(self.grouped_log_beta[i])
			fourier_freq = self.fourier_freq_list[i]
			fourier_basis = self.fourier_basis_list[i]

			subvec1 = fourier_basis[x1[:, i]]
			subvec2 = fourier_basis[x2[:, i]]
			freq_transform = torch.exp(-beta * fourier_freq)

			if diagonal:
				factor_gram = torch.sum(subvec1 * freq_transform.unsqueeze(0) * subvec2, dim=1, keepdim=True)
			else:
				factor_gram = torch.matmul(subvec1 * freq_transform.unsqueeze(0), subvec2.t())
			# HACK for numerical stability for scalability
			full_gram *= factor_gram / torch.mean(freq_transform)

		return torch.exp(self.log_amp) * (full_gram + stabilizer)
