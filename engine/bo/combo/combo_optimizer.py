# COMBO objective + unified optimizer over the multi-fork categorical space.
#
# Faithful to COMBO/main.py:
#   - each fork is a categorical variable with n_vertices[f] choices;
#   - each fork's graph is a PATH graph (choice indices adjacent in yaml order);
#   - Fourier frequencies/basis are precomputed from the Laplacian eigendecomp;
#   - the surrogate is the graph-diffusion GP (DiffusionKernel) with posterior
#     sampling of edge weights (log_beta) and partition;
#   - the acquisition is EI marginalized over posterior samples, optimized by
#     greedy ascent on the graph Cartesian product.
#
# Unified interface (contract section 8):
#   suggest(observed_X, observed_y) -> assignment  (1-D int array, length n_fork)
# observed_X : (n_obs, n_fork) int choice indices; observed_y : (n_obs,) float.

import numpy as np
import torch

from .graphgp.kernels.diffusionkernel import DiffusionKernel
from .graphgp.models.gp_regression import GPRegression
from .graphgp.sampler.sample_posterior import posterior_sampling
from .acquisition.acquisition_optimization import next_evaluation
from .acquisition.acquisition_functions import expected_improvement
from .acquisition.acquisition_marginalization import inference_sampling


def build_path_graphs(n_vertices):
	"""Return (adj_mat_list, fourier_freq_list, fourier_basis_list) for path graphs."""
	adj_mat_list = []
	fourier_freq = []
	fourier_basis = []
	for n_v in n_vertices:
		adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
		adj_mat_list.append(adjmat.float())
		laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
		# torch.symeig (removed in torch2) -> torch.linalg.eigh
		eigval, eigvec = torch.linalg.eigh(laplacian)
		fourier_freq.append(eigval.float())
		fourier_basis.append(eigvec.float())
	return adj_mat_list, fourier_freq, fourier_basis


class COMBOOptimizer(object):
	acquisition_name = 'COMBO'

	def __init__(self, n_vertices, warmups=None, seed=0, n_post=10, n_burn=50, log=None):
		self.n_vertices = np.asarray(n_vertices, dtype=int)
		self.n_forks = len(self.n_vertices)
		self.warmups = warmups
		self.seed = seed
		self.n_post = n_post
		self.n_burn = n_burn
		self.log = log if log is not None else []
		self.adj_mat_list, self.fourier_freq, self.fourier_basis = build_path_graphs(self.n_vertices)
		self.suggest_count = 0

	def _log(self, msg):
		self.log.append(f"[COMBO suggest#{self.suggest_count}] {msg}")

	def suggest(self, observed_X, observed_y):
		self.suggest_count += 1
		torch.manual_seed(self.seed + self.suggest_count)
		np.random.seed(self.seed + self.suggest_count)

		eval_inputs = torch.as_tensor(np.asarray(observed_X), dtype=torch.long)
		if eval_inputs.dim() == 1:
			eval_inputs = eval_inputs.unsqueeze(0)
		eval_outputs = torch.as_tensor(np.asarray(observed_y), dtype=torch.float32).view(-1, 1)
		n_vertices = self.n_vertices

		grouped_log_beta = torch.ones(len(self.fourier_freq))
		kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta,
		                         fourier_freq_list=self.fourier_freq,
		                         fourier_basis_list=self.fourier_basis)
		surrogate_model = GPRegression(kernel=kernel)
		surrogate_model.init_param(eval_outputs)

		log_beta = eval_outputs.new_zeros(eval_inputs.size(1))
		sorted_partition = [[m] for m in range(eval_inputs.size(1))]

		# burn-in posterior sampling of hyperparams + edge weights
		ps_burn = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices,
		                             self.adj_mat_list, log_beta, sorted_partition,
		                             n_sample=1, n_burn=self.n_burn, n_thin=1)
		log_beta = ps_burn[1][0]
		sorted_partition = ps_burn[2][0]
		self._log(f"burned-in edge-weight/partition posterior (n_burn={self.n_burn})")

		# draw posterior samples for EI marginalization
		sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices,
		                                      self.adj_mat_list, log_beta, sorted_partition,
		                                      n_sample=self.n_post, n_burn=0, n_thin=1)
		hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples = sample_posterior
		self._log(f"drew {len(hyper_samples)} posterior GP samples (graph diffusion kernel, "
		          f"edge-weight posterior sampled via slice)")

		reference = torch.min(eval_outputs, dim=0)[0].item()
		x_opt = eval_inputs[torch.argmin(eval_outputs)]
		inference_samples = inference_sampling(eval_inputs, eval_outputs, n_vertices,
		                                       hyper_samples, log_beta_samples, partition_samples,
		                                       freq_samples, basis_samples)
		suggestion, mean, std, var = next_evaluation(x_opt, eval_inputs, inference_samples, partition_samples,
		                                             edge_mat_samples, n_vertices, expected_improvement,
		                                             reference, log=self.log)
		assignment = suggestion.reshape(-1).numpy().astype(int)
		self._log(f"marginalized-EI greedy-ascent on graph Cartesian product -> assignment={assignment.tolist()}")
		return assignment
