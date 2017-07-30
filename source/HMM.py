# -*- coding: utf-8 -*-
u"""隠れマルコフモデルの実装."""
import numpy as np


class HMM(object):
	u"""隠れマルコフモデル."""

	def __init__(self, model, hidden=3):
		u"""
		初期化.

		Argument
			hidden : 隠れ変数の数

		Attribute
			hidden : 隠れ変数の数
			trans  : 遷移行列
			init   : 初期状態確率
			model  : 隠れ変数に対応したモデル
		"""
		self.hidden = hidden
		self.trans = np.tile(np.ones(hidden, dtype=float) / hidden, (hidden, 1))
		self.init = np.ones(hidden, dtype=float) / hidden
		self.model = model

	def fit(self, x, maxiter=1000, tol=1e-3):
		u"""
		モデルの当てはめ.

		EMアルゴリズム(バウム・ウェルチアルゴリズム)によってモデルパラメータを推定する.

		Argument
			x : 観測された系列データ (shape = (時刻, 次元))
		"""
		trans = self.trans
		init = self.init
		hidden = self.hidden
		model = self.model
		prob = lambda x: model.get_prob(x, range(self.hidden))

		lh_old = -1e+24
		diff = 1

		ite = 0
		while diff > tol and ite < maxiter:
			lh, alpha, c = self.forward(x)
			# print lh

			_, beta = self.backward(x, c)
			gamma = alpha * beta
			for i in range(hidden):
				tmp = trans[i, :] * np.sum(
					alpha[:-1, i][:, np.newaxis] * beta[1:, :]
					* prob(x[1:]).T / c[1:, np.newaxis], axis=0)
				trans[i, :] = tmp / np.sum(gamma[:-1, i])
			init = gamma[0, :]

			model.fit(x, gamma)       # P(x | z)のパラメータ更新
			self.trans = trans
			self.init = init

			diff = np.abs(lh - lh_old)
			lh_old = lh
			ite += 1

		if ite == maxiter:
			print "Maximum iteration was achieved!!"

	def predict(self, x):
		u"""
		ビタビアルゴリズム(対数スケール).

		隠れ変数の最尤推定を行う.
		
		Return
			seq  : 最尤推定した状態系列
			maxP : seqの完全対数尤度
		"""
		trans = self.trans
		init = self.init
		prob = lambda x: self.model.get_prob(x, range(self.hidden)).ravel()

		psis = [np.log(init * prob(x[0]))]
		phi = []
		for t in range(1, x.shape[0]):
			psi = psis[-1]
			tmp = psi[:, np.newaxis] + np.log(trans)
			psis.append(np.amax(tmp, axis=0) + np.log(prob(x[t])))
			phi.append(np.argmax(tmp, axis=0))
		maxi = np.argmax(psis[-1])
		maxP = psis[-1][maxi]
		seq = [maxi]
		for p in phi[::-1]:
			seq.append(p[maxi])
			maxi = seq[-1]
		return (seq, maxP)

	def forward(self, x):
		u"""
		前向きアルゴリズム(スケーリングあり).

		Return
			lh    : xの尤度 P(x)
			alpha : alpha
			c     : スケーリング係数
		"""
		trans = self.trans
		init = self.init
		prob = lambda x: self.model.get_prob(x, range(self.hidden)).ravel()

		T = x.shape[0]
		alpha = np.zeros((T, self.hidden))
		c = np.zeros(T)

		tmp = init * prob(x[0])
		c[0] = np.sum(tmp)
		alpha[0, :] = tmp / c[0]
		for t in range(1, T):
			tmp = np.sum(alpha[t - 1, :][:, np.newaxis] * trans, axis=0) * prob(x[t])
			c[t] = np.sum(tmp)
			alpha[t, :] = tmp / c[t]
		lh = np.sum(np.log(c))
		return (lh, alpha, c)

	def backward(self, x, c):
		u"""後ろ向きアルゴリズム."""
		trans = self.trans
		prob = lambda x: self.model.get_prob(x, range(self.hidden)).ravel()

		T = x.shape[0]
		beta = np.ones((T, self.hidden))
		for t in range(1, T):
			i = -t
			beta[i - 1, :] = np.sum(
				trans * (prob(x[i]) * beta[i, :])[np.newaxis, :], axis=1
			) / c[i]
		return (np.sum(np.log(c)), beta)












