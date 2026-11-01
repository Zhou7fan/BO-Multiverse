import sys
import time
import copy
import numpy as np

from .sample_hyper import slice_hyper
from .sample_edgeweight import slice_edgeweight
from .tool_partition import direct_product
from ...config import PROGRESS_BAR_LEN


def posterior_sampling(model, input_data, output_data, n_vertices, adj_mat_list,
                       log_beta, sorted_partition, n_sample, n_burn=0, n_thin=1, log=None):
	hyper_samples = []
	log_beta_samples = []
	partition_samples = []
	freq_samples = []
	basis_samples = []
	edge_mat_samples = []

	partition_sample = sorted_partition
	log_beta_sample = log_beta
	fourier_freq_list = model.kernel.fourier_freq_list
	fourier_basis_list = model.kernel.fourier_basis_list
	edge_mat_list = [direct_product(adj_mat_list, subset) for subset in sorted_partition]

	n_sample_total = n_sample * n_thin + n_burn
	for s in range(0, n_sample_total):
		slice_hyper(model, input_data, output_data, n_vertices, sorted_partition=partition_sample)
		shuffled_beta_ind = list(range(len(adj_mat_list)))
		np.random.shuffle(shuffled_beta_ind)
		for beta_ind in shuffled_beta_ind:
			log_beta_sample = slice_edgeweight(model, input_data, output_data, n_vertices,
			                                   log_beta=log_beta_sample, sorted_partition=partition_sample,
			                                   fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list,
			                                   ind=beta_ind)
		if s >= n_burn and (s - n_burn + 1) % n_thin == 0:
			hyper_samples.append(model.param_to_vec())
			log_beta_samples.append(log_beta_sample.clone())
			partition_samples.append(copy.deepcopy(partition_sample))
			freq_samples.append([elm.clone() for elm in fourier_freq_list])
			basis_samples.append([elm.clone() for elm in fourier_basis_list])
			edge_mat_samples.append([elm.clone() for elm in edge_mat_list])
	return hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples
