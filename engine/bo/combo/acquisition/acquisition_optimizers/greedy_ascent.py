import torch

from ..acquisition_functions import expected_improvement
from ..acquisition_marginalization import acquisition_expectation
from .graph_utils import neighbors


def greedy_ascent(x_init, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                  acquisition_func=expected_improvement, max_n_ascent=float('inf'), reference=None):
	n_ascent = 0
	x = x_init
	max_acquisition = acquisition_expectation(x, inference_samples, partition_samples, n_vertices,
	                                          acquisition_func, reference)
	while n_ascent < max_n_ascent:
		x_nbds = neighbors(x, partition_samples, edge_mat_samples, n_vertices, uniquely=True)
		nbds_acquisition = acquisition_expectation(x_nbds, inference_samples, partition_samples, n_vertices,
		                                            acquisition_func, reference)
		max_nbd_acquisition, max_nbd_ind = torch.max(nbds_acquisition, 0)
		if max_nbd_acquisition > max_acquisition:
			max_acquisition = max_nbd_acquisition
			x = x_nbds[max_nbd_ind.item()]
			n_ascent += 1
		else:
			break
	return x, max_acquisition.item()
