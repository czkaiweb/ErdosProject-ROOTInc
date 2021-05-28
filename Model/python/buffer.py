#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:13:28 2018
"""

import numpy as np
import random
from collections import deque


class MemoryBuffer(object):

	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr

	def __len__(self):
		return self.len

	def add(self, state, action, reward, next_state):
		transition = (state, action, reward, next_state)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)
