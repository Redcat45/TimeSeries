# -*- coding: utf-8 -*-
u"""HMMに使うモデル."""
import numpy as np
from scipy.misc import factorial


class Categorical(object):
	u"""カテゴリカル分布."""

	def __init__(self, hidden, c):
		u"""
		初期化.

		Argument
			hidden  : 隠れ状態数
			c       : カテゴリの数

		Attribute
			hidden  : 隠れ状態数
			c       : カテゴリ数
			p       : 隠れ状態毎のカテゴリの生起確率
		"""
		self.hidden = hidden
		self.c = c
		np.random.seed(0)
		self.p = np.random.random((hidden, c))
		self.p = self.p / np.sum(self.p, axis=1)[:, np.newaxis]

	def get_prob(self, xseq, states):
		u"""xの生起確率を返す."""
		probs = []
		try:
			iter(states)
		except:
			states = [states]
		try:
			iter(xseq)
		except:
			xseq = [xseq]

		for s in states:
			sprob = []
			for x in xseq:
				sprob.append(self.p[s, x])
			probs.append(sprob)
		return np.array(probs)

	def fit(self, xseq, gamma):
		u"""モデルの当てはめ."""
		p = self.p
		for i in range(self.hidden):
			g = np.sum(gamma[:, i])
			for c in range(self.c):
				p[i, c] = np.sum(gamma[:, i][xseq == c]) / g
		self.p = p


class Poisson(object):
	u"""ポアソン分布."""

	def __init__(self, hidden):
		u"""
		初期化.

		Argument
			hidden  : 隠れ状態数

		Attribute
			hidden  : 隠れ状態数
			param   : ポアソン分布の平均パラメータ
		"""
		self.hidden = hidden
		self.param = np.array([1., 2.])

	def get_prob(self, xseq, states):
		u"""xの生起確率を返す."""
		param = self.param
		try:
			iter(states)
		except:
			states = [states]

		probs = []
		for s in states:
			probs.append(
				param[s]**xseq * np.exp(-param[s]) / factorial(xseq)
			)
		return np.array(probs)

	def fit(self, xseq, gamma):
		u"""モデルの当てはめ."""
		param = self.param
		for i in range(self.hidden):
			g = np.sum(gamma[:, i])
			param[i] = np.sum(gamma[:, i] * xseq) / g
		self.param = param















