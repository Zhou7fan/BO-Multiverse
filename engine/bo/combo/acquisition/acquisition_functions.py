import torch
from torch.distributions.normal import Normal


def expected_improvement(mean, var, reference):
	# EI for minimization, on graphs (no gradients needed). Ported verbatim.
	predictive_normal = Normal(mean.new_zeros(mean.size()), mean.new_ones(mean.size()))
	std = torch.sqrt(var)
	standardized = (-mean + reference) / std
	return (std * torch.exp(predictive_normal.log_prob(standardized)) + (-mean + reference) * predictive_normal.cdf(standardized)).clamp(min=0)
