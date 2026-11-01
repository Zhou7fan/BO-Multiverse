import torch

from ..graphgp.kernels.diffusionkernel import DiffusionKernel
from ..graphgp.models.gp_regression import GPRegression
from ..graphgp.inference.inference import Inference
from ..graphgp.sampler.tool_partition import group_input

from .acquisition_functions import expected_improvement


def acquisition_expectation(x, inference_samples, partition_samples, n_vertices, acquisition_func=expected_improvement,
                            reference=None):
	# EI marginalized over posterior samples. Ported verbatim.
	if x.dim() == 1:
		x = x.unsqueeze(0)

	acquisition_sample_list = []
	for s in range(len(inference_samples)):
		hyper = inference_samples[s].model.param_to_vec()
		grouped_x = group_input(x, sorted_partition=partition_samples[s], n_vertices=n_vertices)
		pred_dist = inference_samples[s].predict(grouped_x, hyper=hyper, verbose=False)
		pred_mean_sample = pred_dist[0].detach()
		pred_var_sample = pred_dist[1].detach()
		acquisition_sample_list.append(acquisition_func(pred_mean_sample[:, 0], pred_var_sample[:, 0],
		                                                reference=reference))
	return torch.stack(acquisition_sample_list, 1).sum(1, keepdim=True)


def inference_sampling(input_data, output_data, n_vertices, hyper_samples, log_beta_samples, partition_samples,
                       freq_samples, basis_samples):
	inference_samples = []
	for s in range(len(hyper_samples)):
		grouped_log_beta = torch.stack([torch.sum(log_beta_samples[s][subset]) for subset in partition_samples[s]])
		kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta,
		                         fourier_freq_list=freq_samples[s], fourier_basis_list=basis_samples[s])
		model = GPRegression(kernel=kernel)
		model.vec_to_param(hyper_samples[s])
		grouped_input_data = group_input(input_data=input_data, sorted_partition=partition_samples[s],
		                                 n_vertices=n_vertices)
		inference = Inference((grouped_input_data, output_data), model=model)
		inference_samples.append(inference)
	return inference_samples


def prediction_statistic(x, inference_samples, partition_samples, n_vertices):
	if x.dim() == 1:
		x = x.unsqueeze(0)
	mean_sample_list = []
	std_sample_list = []
	var_sample_list = []
	for s in range(len(inference_samples)):
		grouped_x = group_input(input_data=x, sorted_partition=partition_samples[s], n_vertices=n_vertices)
		pred_dist = inference_samples[s].predict(grouped_x)
		mean_sample_list.append(pred_dist[0].data)
		std_sample_list.append((pred_dist[1] ** 0.5).data)
		var_sample_list.append(pred_dist[1].data)
	return torch.cat(mean_sample_list, 1).mean(1, keepdim=True), \
	       torch.cat(std_sample_list, 1).mean(1, keepdim=True), \
	       torch.cat(var_sample_list, 1).mean(1, keepdim=True)
