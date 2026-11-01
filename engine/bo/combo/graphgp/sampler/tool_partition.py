import numpy as np

import torch


def np_kron(mat1, mat2):
	np_mat1 = mat1.numpy()
	np_mat2 = mat2.numpy()
	return torch.from_numpy(np.kron(np_mat1, np_mat2))


def kronecker(mat1, mat2):
	# kronecker product between 2 2D tensors (torch.ger -> torch.outer in torch2)
	s1 = mat1.size()
	s2 = mat2.size()
	return torch.outer(mat1.reshape(-1), mat2.reshape(-1)).reshape(s1[0], s1[1], s2[0], s2[1]).permute([0, 2, 1, 3]).reshape(s1[0] * s2[0], s1[1] * s2[1])


def sort_partition(partition):
	lowest_ind_key_dict = {min(subset): sorted(subset) for subset in partition}
	sorted_partition = []
	for k in sorted(lowest_ind_key_dict.keys()):
		sorted_partition.append(lowest_ind_key_dict[k])
	return sorted_partition


def compute_unit_in_group(sorted_partition, n_vertices):
	unit_in_group = []
	for subset in sorted_partition:
		ind_units = list(np.flip(np.cumprod((n_vertices[subset][1:][::-1])))) + [1]
		unit_in_group.append(ind_units)
	return unit_in_group


def compute_group_size(sorted_partition, n_vertices):
	complexity = sum([np.prod(n_vertices[subset]) ** 3 for subset in sorted_partition])
	max_size = max([np.prod(n_vertices[subset]) for subset in sorted_partition])
	return max_size


def group_input(input_data, sorted_partition, n_vertices):
	unit_in_group = compute_unit_in_group(sorted_partition, n_vertices)
	grouped_input = input_data.new_zeros(input_data.size(0), len(sorted_partition))
	for g in range(len(sorted_partition)):
		for ind, unit in zip(sorted_partition[g], unit_in_group[g]):
			grouped_input[:, g] += input_data[:, ind] * unit
	return grouped_input


def ungroup_input(grouped_input, sorted_partition, n_vertices):
	unit_in_group = compute_unit_in_group(sorted_partition, n_vertices)
	input_data = grouped_input.new_zeros(grouped_input.size(0), sum([len(subset) for subset in sorted_partition]))
	for g in range(len(sorted_partition)):
		subset = sorted_partition[g]
		unit = unit_in_group[g]
		elm_ind = 0
		remainder = grouped_input[:, g]
		while elm_ind < len(subset) - 1:
			input_data[:, subset[elm_ind]] = remainder // unit[elm_ind]
			remainder = remainder % unit[elm_ind]
			elm_ind += 1
		input_data[:, subset[-1]] = remainder
	return input_data


def direct_product(adj_mat_list, subset):
	elm = subset[0]
	grouped_adjacency = adj_mat_list[elm]
	for ind in range(1, len(subset)):
		elm = subset[ind]
		grouped_adjacency = kronecker(grouped_adjacency, adj_mat_list[elm])
	return grouped_adjacency
