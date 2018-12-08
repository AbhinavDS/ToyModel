import numpy as np
import random
from collections import deque


class MemoryBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.buffer_p = deque(maxlen=size)
		self.buffer_wt = deque(maxlen=size)
		self.maxSize = size
		self.len = 0
		self.len_p = 0
		self.uniques = set()

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		batch = []
		count = min(count, self.len)
		# batch = random.sample(self.buffer, count)
		batch = random.choices(self.buffer,weights=self.buffer_wt,k=count)
		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr
	def agg_sample(self,count):
		return self.sample(count)

		batch = []

		count_p = min(int(count/2), self.len_p)
		batch = random.sample(self.buffer_p, count_p)
		s_arr_p = np.float32([arr[0] for arr in batch])
		a_arr_p = np.float32([arr[1] for arr in batch])
		r_arr_p = np.float32([arr[2] for arr in batch])
		s1_arr_p = np.float32([arr[3] for arr in batch])

		count_n = count - count_p
		batch = random.sample(self.buffer, count_n)
		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		if count_p != 0:
			s_arr = np.concatenate((s_arr,s_arr_p),axis=0)
			a_arr = np.concatenate((a_arr,a_arr_p),axis=0)
			r_arr = np.concatenate((r_arr,r_arr_p),axis=0)
			s1_arr = np.concatenate((s1_arr,s1_arr_p),axis=0)
		return s_arr, a_arr, r_arr, s1_arr

	
	def len(self):
		return self.len

	def agg_add(self, s, a, r, s1):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""
		return self.add(s,a,r,s1)
		transition = (s,a,r,s1)
		if r[0]:
			self.len_p += 1
			if self.len_p > self.maxSize:
				self.len_p = self.maxSize
			self.buffer_p.append(transition)
		else:
			self.len += 1
			if self.len > self.maxSize:
				self.len = self.maxSize
			self.buffer.append(transition)
	def add(self, s, a, r, s1):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""
		transition = (s,a,r,s1)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)
		self.buffer_wt.append(5+r)
		# self.uniques.add(r)