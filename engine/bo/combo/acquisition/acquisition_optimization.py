import time
import numpy as np
import torch

from .acquisition_optimizers.starting_points import optim_inits
from .acquisition_optimizers.greedy_ascent import greedy_ascent
from .acquisition_functions import expected_improvement
from .acquisition_marginalization import prediction_statistic
from .acquisition_optimizers.graph_utils import neighbors


MAX_N_ASCENT = float('inf')


def next_evaluation(x_opt, input_data, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                    acquisition_func=expected_improvement, reference=None, log=None):
	# Sequential-only port of COMBO/acquisition/acquisition_optimization.next_evaluation.
	# (parallel multiprocessing / psutil auto-scaling removed; greedy ascent runs serially.)
	def _log(msg):
		if log is not None:
			log.append(msg)

	_log('acq: selecting initial points')
	x_inits, acq_inits = optim_inits(x_opt, inference_samples, partition_samples, edge_mat_samples, n_vertices,
	                                 acquisition_func, reference)
	n_inits = x_inits.size(0)

	ga_args_list = [(x_inits[i], inference_samples, partition_samples, edge_mat_samples,
	                 n_vertices, acquisition_func, MAX_N_ASCENT, reference) for i in range(n_inits)]
	ga_return_values = [greedy_ascent(*(ga_args_list[i])) for i in range(n_inits)]
	ga_opt_vrt, ga_opt_acq = zip(*ga_return_values)

	opt_vrt = list(ga_opt_vrt[:])
	opt_acq = list(ga_opt_acq[:])

	acq_sort_inds = np.argsort(-np.array(opt_acq))
	suggestion = None
	for i in range(len(opt_vrt)):
		ind = acq_sort_inds[i]
		if not torch.all(opt_vrt[ind] == input_data, dim=1).any():
			suggestion = opt_vrt[ind]
			break
	if suggestion is None:
		for i in range(len(opt_vrt)):
			ind = acq_sort_inds[i]
			nbds = neighbors(opt_vrt[ind], partition_samples, edge_mat_samples, n_vertices, uniquely=True)
			for j in range(nbds.size(0)):
				if not torch.all(nbds[j] == input_data, dim=1).any():
					suggestion = nbds[j]
					break
			if suggestion is not None:
				break
	if suggestion is None:
		suggestion = torch.cat(tuple([torch.randint(low=0, high=int(n_v), size=(1, 1)) for n_v in n_vertices]), dim=1).long()

	mean, std, var = prediction_statistic(suggestion, inference_samples, partition_samples, n_vertices)
	return suggestion, mean, std, var
