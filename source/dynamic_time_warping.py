# -*- coding: utf-8 -*-
u"""
DynamicTimeWarping(動的時間伸縮法)の実装.

参考: wikipedia
"""
import numpy as np

w = 6      # windowサイズ


def dtw(vec1, vec2, wsize=1):
	u"""実装."""
	w = wsize
	d = np.zeros([len(vec1) + 1, len(vec2) + 1])
	d[:] = np.inf
	d[0, 0] = 0

	for i in xrange(1, d.shape[0]):
		for j in xrange(np.maximum(i - w, 1), np.minimum(i + w, d.shape[1])):
			cost = abs(vec1[i - 1] - vec2[j - 1])
			d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
	# print d
	return d[-1][-1]




































