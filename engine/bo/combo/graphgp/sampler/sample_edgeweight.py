import numpy as np
import torch

from ..inference.inference import Inference
from .tool_partition import group_input
from .tool_slice_sampling import univariate_slice_sampling
from .priors import log_prior_edgeweight


def slice_edgeweight(model, input_data, output_data, n_vertices, log_beta,
                     sorted_partition, fourier_freq_list, fourier_basis_list, ind):
	updated_subset_ind = [(ind in subset) for subset in sorted_partition].index(True)
	updated_subset = sorted_partition[updated_subset_ind]
	log_beta_rest = torch.sum(log_beta[updated_subset]) - log_beta[ind]
	grouped_log_beta = torch.stack([torch.sum(log_beta[subset]) for subset in sorted_partition])
	model.kernel.grouped_log_beta = grouped_log_beta
	model.kernel.fourier_freq_list = fourier_freq_list
	model.kernel.fourier_basis_list = fourier_basis_list
	grouped_input_data = group_input(input_data=input_data, sorted_partition=sorted_partition, n_vertices=n_vertices)
	inference = Inference(train_data=(grouped_input_data, output_data), model=model)

	def logp(log_beta_i):
		log_prior = log_prior_edgeweight(log_beta_i)
		if np.isinf(log_prior):
			return log_prior
		model.kernel.grouped_log_beta[updated_subset_ind] = log_beta_rest + log_beta_i
		log_likelihood = float(-inference.negative_log_likelihood(hyper=model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(log_beta[ind])
	x1 = univariate_slice_sampling(logp, x0)
	log_beta[ind] = x1
	model.kernel.grouped_log_beta[updated_subset_ind] = log_beta_rest + x1
	return log_beta
