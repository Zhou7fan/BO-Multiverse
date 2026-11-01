# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import numpy as np
from itertools import combinations
from .bhs import bhs

class LinReg:

	# fallback-ladder trigger log (class-level; workers clear it at the start of a run and tally it at the end)
	fallback_log = []

	def __init__(self, nVars, order):
		self.nVars = nVars
		self.order = order

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def setupData(self):

		# limit data to unique points
		X, x_idx = np.unique(self.xTrain, axis=0, return_index=True)
		y = self.yTrain[x_idx]

		# set upper threshold
		infT = 1e6

		# separate samples based on Inf output
		y_Infidx  = np.where(np.abs(y) > infT)[0]
		y_nInfidx = np.setdiff1d(np.arange(len(y)), y_Infidx)

		# save samples in two sets of variables
		self.xInf = X[y_Infidx,:]
		self.yInf = y[y_Infidx]

		self.xTrain = X[y_nInfidx,:]
		self.yTrain = y[y_nInfidx]

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	@staticmethod
	def _fit_bhs(D, y):
		# limited attempts; bhs itself does not retry. Returns (beta, b0) on success, None on failure.
		for _ in range(2):
			try:
				ag, a0, _, _, _ = bhs(D, y, 400, 0, 1)
				if ag is not None and np.isfinite(ag).all():
					return ag, float(a0)
			except Exception:
				pass
		return None

	def train(self, inputs):

		self.xTrain = inputs['x_vals']
		self.yTrain = inputs['y_vals']
		self.setupData()

		Xraw = self.xTrain
		yraw = self.yTrain
		nSamps = Xraw.shape[0]
		tag = 'native'

		# ---- Step 1: original order=2 design, straight to bhs ---- #
		D2 = self.order_effects(Xraw, self.order)
		keep2 = ~np.all(D2 == 0, axis=0)
		D2z = D2[:, keep2] if keep2.sum() else D2
		res = self._fit_bhs(D2z, yraw)
		if res is not None:
			pad = np.zeros(D2.shape[1]); pad[keep2] = res[0][:, -1]
			self.alpha = np.append(res[1], pad)
			return

		# ---- Step 2: y standardization + column normalization (design still order=2) ---- #
		tag = 'y-standardize'
		ys = (yraw - yraw.mean()) / (yraw.std() + 1e-12)
		D2n = (D2z - D2z.mean(0)) / (D2z.std(0) + 1e-12)
		res = self._fit_bhs(D2n, ys)
		if res is not None:
			pad = np.zeros(D2.shape[1]); pad[keep2] = res[0][:, -1]
			self.alpha = np.append(yraw.mean(), pad)
		else:
			# ---- Step 3: order=1 design (p~n, horseshoe fastmvg can work) ---- #
			tag = 'order=1'
			self.order = 1
			D1 = self.order_effects(Xraw, 1)
			keep1 = ~np.all(D1 == 0, axis=0)
			D1z = D1[:, keep1] if keep1.sum() else D1
			D1n = (D1z - D1z.mean(0)) / (D1z.std(0) + 1e-12)
			ys = (yraw - yraw.mean()) / (yraw.std() + 1e-12)
			res = self._fit_bhs(D1n, ys)
			if res is not None:
				pad = np.zeros(D1.shape[1]); pad[keep1] = res[0][:, -1]
				self.alpha = np.append(yraw.mean(), pad)
			else:
				# ---- Step 4: order=1 ridge-regression fallback (only replaces the surrogate, keeps the outer SA/SMC) ---- #
				tag = 'ridge'
				self.order = 1
				D1 = self.order_effects(Xraw, 1)
				Xd = np.column_stack([np.ones(nSamps), D1])
				lam = 1.0
				try:
					beta = np.linalg.solve(Xd.T @ Xd + lam * np.eye(Xd.shape[1]), Xd.T @ yraw)
				except Exception:
					beta = np.zeros(Xd.shape[1])
				self.alpha = beta

		LinReg.fallback_log.append(tag)
		try:
			print('[FALLBACK] %s order=%d' % (tag, self.order))
		except Exception:
			pass

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def surrogate_model(self, x, alpha):
		# SURROGATE_MODEL: Function evaluates the linear model
		# Assumption: input x only contains one row

		# generate x_all (all basis vectors) based on model order
		x_all = np.append(1, self.order_effects(x, self.order))

		# check if x maps to an Inf output (if so, barrier=Inf)
		barrier = 0.
		if self.xInf.shape[0] != 0:
			if np.equal(x, self.xInf).all(axis=1).any():
				barrier = np.inf

		# compute and return objective with barrier
		out = np.dot(x_all,alpha) + barrier
		
		return out

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def order_effects(self, x_vals, ord_t):
		# order_effects: Function computes data matrix for all coupling
		# orders to be added into linear regression model.

		# Find number of variables
		n_samp, n_vars = x_vals.shape

		# Generate matrix to store results
		x_allpairs = x_vals

		for ord_i in range(2,ord_t+1):

			# generate all combinations of indices (without diagonals)
			offdProd = np.array(list(combinations(np.arange(n_vars),ord_i)))

			# generate products of input variables
			x_comb = np.zeros((n_samp, offdProd.shape[0], ord_i))
			for j in range(ord_i):
				x_comb[:,:,j] = x_vals[:,offdProd[:,j]]
			x_allpairs = np.append(x_allpairs, np.prod(x_comb,axis=2),axis=1)

		return x_allpairs


# -- END OF FILE --