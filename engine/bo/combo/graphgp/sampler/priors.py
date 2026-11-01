import numpy as np

from .tool_partition import compute_group_size

LOG_LOWER_BND = -12.0
LOG_UPPER_BND = 20.0
STABLE_MEAN_RNG = 1.0
GRAPH_SIZE_LIMIT = 1024 + 2


def log_prior_constmean(constmean, output_min, output_max):
	output_mid = (output_min + output_max) / 2.0
	output_rad = (output_max - output_min) * STABLE_MEAN_RNG / 2.0
	if constmean < output_mid - output_rad or output_mid + output_rad < constmean:
		return -float('inf')
	stable_dev = output_rad / 2.0
	return -np.log(stable_dev) - 0.5 * (constmean - output_mid) ** 2 / stable_dev ** 2


def log_prior_noisevar(log_noise_var):
	if log_noise_var < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(10000.0)) < log_noise_var:
		return -float('inf')
	return np.log(np.log(1.0 + (0.1 / np.exp(log_noise_var)) ** 2))


def log_prior_kernelamp(log_amp, output_var, kernel_min, kernel_max):
	if log_amp < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(10000.0)) < log_amp:
		return -float('inf')
	log_amp_lower = np.log(output_var) - np.log(kernel_max)
	log_amp_upper = np.log(output_var) - np.log(max(kernel_min, 1e-100))
	log_amp_mid = 0.5 * (log_amp_upper + log_amp_lower)
	log_amp_rad = 0.5 * (log_amp_upper - log_amp_lower)
	log_amp_std = log_amp_rad / 2.0
	return -np.log(log_amp_std) - 0.5 * (log_amp - log_amp_mid) ** 2 / log_amp_std ** 2


def log_prior_edgeweight(log_beta_i):
	if log_beta_i < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(100.0)) < log_beta_i:
		return -float('inf')
	tau = 5.0
	return np.log(np.log(1.0 + 2.0 / (np.exp(log_beta_i) / tau) ** 2))


def log_prior_partition(sorted_partition, n_vertices):
	if len(sorted_partition) == 1 or compute_group_size(sorted_partition=sorted_partition, n_vertices=n_vertices) > GRAPH_SIZE_LIMIT:
		return -float('inf')
	else:
		prob_mass = np.array([np.sum(np.log(n_vertices[subset])) for subset in sorted_partition])
		prob_mass /= np.sum(prob_mass)
		entropy_mass = -np.sum(prob_mass * np.log(prob_mass))
		max_log = np.sum(np.log(n_vertices))
		thr_log = np.log(GRAPH_SIZE_LIMIT)
		n_chunk = int(np.floor(max_log / thr_log))
		prob_base = np.array([np.log(GRAPH_SIZE_LIMIT) for _ in range(n_chunk)] + [max_log - n_chunk * thr_log])
		prob_base /= np.sum(prob_base)
		entropy_base = -np.sum(prob_base * np.log(prob_base))
		return np.log(entropy_mass - entropy_base) * 5
