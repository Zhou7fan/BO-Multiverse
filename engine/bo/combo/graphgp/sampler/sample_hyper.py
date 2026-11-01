import numpy as np
import torch

from ..inference.inference import Inference
from .tool_partition import group_input
from .tool_slice_sampling import univariate_slice_sampling
from .priors import log_prior_constmean, log_prior_noisevar, log_prior_kernelamp


def slice_hyper(model, input_data, output_data, n_vertices, sorted_partition):
	grouped_input_data = group_input(input_data=input_data, sorted_partition=sorted_partition, n_vertices=n_vertices)
	inference = Inference(train_data=(grouped_input_data, output_data), model=model)
	slice_constmean(inference)
	slice_kernelamp(inference)
	slice_noisevar(inference)


def slice_constmean(inference):
	output_min = torch.min(inference.train_y).item()
	output_max = torch.max(inference.train_y).item()
	def logp(constmean):
		log_prior = log_prior_constmean(constmean, output_min=output_min, output_max=output_max)
		if np.isinf(log_prior):
			return log_prior
		inference.model.mean.const_mean.fill_(constmean)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=inference.model.param_to_vec()))
		return log_prior + log_likelihood
	x0 = float(inference.model.mean.const_mean)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.mean.const_mean.fill_(x1)


def slice_noisevar(inference):
	def logp(log_noise_var):
		log_prior = log_prior_noisevar(log_noise_var)
		if np.isinf(log_prior):
			return log_prior
		inference.model.likelihood.log_noise_var.fill_(log_noise_var)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=inference.model.param_to_vec()))
		return log_prior + log_likelihood
	x0 = float(inference.model.likelihood.log_noise_var)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.likelihood.log_noise_var.fill_(x1)


def slice_kernelamp(inference):
	output_var = torch.var(inference.train_y).item()
	kernel_min = np.prod([torch.mean(torch.exp(-fourier_freq[-1])).item() / torch.mean(torch.exp(-fourier_freq)).item() for fourier_freq in inference.model.kernel.fourier_freq_list])
	kernel_max = np.prod([torch.mean(torch.exp(-fourier_freq[0])).item() / torch.mean(torch.exp(-fourier_freq)).item() for fourier_freq in inference.model.kernel.fourier_freq_list])
	def logp(log_amp):
		log_prior = log_prior_kernelamp(log_amp, output_var, kernel_min, kernel_max)
		if np.isinf(log_prior):
			return log_prior
		inference.model.kernel.log_amp.fill_(log_amp)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=inference.model.param_to_vec()))
		return log_prior + log_likelihood
	x0 = float(inference.model.kernel.log_amp)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.kernel.log_amp.fill_(x1)
