import numpy as np

import torch
import torch.nn as nn


class Inference(nn.Module):
	# Port of COMBO/graphGP/inference/inference.py.
	# torch-2.6 adaptations:
	#   torch.cholesky(A, upper=False) -> torch.linalg.cholesky(A) (lower L)
	#   torch.triangular_solve(b, L)   -> torch.linalg.solve_triangular(L, b, upper=False)

	def __init__(self, train_data, model):
		super(Inference, self).__init__()
		self.model = model
		self.train_x = train_data[0]
		self.train_y = train_data[1]
		self.output_min = torch.min(self.train_y)
		self.output_max = torch.max(self.train_y)
		self.mean_vec = None
		self.gram_mat = None
		self.cholesky = None
		self.jitter = 0

	def gram_mat_update(self, hyper=None):
		if hyper is not None:
			self.model.vec_to_param(hyper)
		self.mean_vec = self.train_y - self.model.mean(self.train_x.float())
		self.gram_mat = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x.float()))

	def cholesky_update(self, hyper):
		self.gram_mat_update(hyper)
		eye_mat = torch.diag(self.gram_mat.new_ones(self.gram_mat.size(0)))
		for jitter_const in [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
			chol_jitter = torch.trace(self.gram_mat).item() * jitter_const
			try:
				self.cholesky = torch.linalg.cholesky(self.gram_mat + eye_mat * chol_jitter)
				self.jitter = chol_jitter
				return
			except RuntimeError:
				pass
		raise RuntimeError('Absolute entry values of Gram matrix are between %.4E~%.4E with trace %.4E' %
		                   (torch.min(torch.abs(self.gram_mat)).item(), torch.max(torch.abs(self.gram_mat)).item(),
		                    torch.trace(self.gram_mat).item()))

	def predict(self, pred_x, hyper=None, verbose=False):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.cholesky_update(hyper)

		k_pred_train = self.model.kernel(pred_x, self.train_x)
		k_pred = self.model.kernel(pred_x, diagonal=True)

		chol_solver = torch.linalg.solve_triangular(self.cholesky, torch.cat([k_pred_train.t(), self.mean_vec], 1), upper=False)
		chol_solve_k = chol_solver[:, :-1]
		chol_solve_y = chol_solver[:, -1:]

		pred_mean = torch.mm(chol_solve_k.t(), chol_solve_y) + self.model.mean(pred_x)
		pred_quad = (chol_solve_k ** 2).sum(0).view(-1, 1)
		pred_var = k_pred - pred_quad

		if hyper is not None:
			self.cholesky_update(param_original)

		if verbose:
			numerically_stable = (pred_var >= 0).all()
			zero_pred_var = (pred_var <= 0).all()
			return pred_mean, pred_var.clamp(min=1e-8), numerically_stable, zero_pred_var
		else:
			return pred_mean, pred_var.clamp(min=1e-8)

	def negative_log_likelihood(self, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.cholesky_update(hyper)
		mean_vec_sol = torch.linalg.solve_triangular(self.cholesky, self.mean_vec, upper=False)
		nll = 0.5 * torch.sum(mean_vec_sol ** 2) + torch.sum(torch.log(torch.diag(self.cholesky))) + 0.5 * self.train_y.size(0) * np.log(2 * np.pi)
		if hyper is not None:
			self.cholesky_update(param_original)
		return nll
