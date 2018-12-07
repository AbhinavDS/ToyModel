from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
# import psutil
import gc

from . import rl_train
from . import buffer
from . import utils

class RLModule:
	def __init__(self, params):
		self.MAX_STEPS = 1#000
		self.MAX_BUFFER = 1000000
		self.params = params

		S_DIM = 2*params.img_width + params.feature_size
		A_DIM = 4
		A_MAX = 1
		P_DIM = 1
		self.reward_dim = 2

		self.ram = buffer.MemoryBuffer(self.MAX_BUFFER)
		self.trainer = rl_train.Trainer(S_DIM, A_DIM, A_MAX, P_DIM, self.ram, params.batch_size, reward_dim=self.reward_dim)
	
	def get_new_state(self,state):
		#return numpy array
		return state

	def step(self,c, s, gt, A, mask, proj_pred, proj_gt):
		s_avg = torch.mean(s, dim=1)
		state = np.float32(np.concatenate((proj_gt,proj_pred, s_avg.cpu().numpy()),axis=1))
		for r in range(self.MAX_STEPS):
			action, prob = self.trainer.get_exploration_action(state)
			# if _ep%5 == 0:
			# 	# validate every 5th episode
			# 	action = trainer.get_exploitation_action(state)
			# else:
			# 	# get action based on observation, use exploration policy here
			# 	action = trainer.get_exploration_action(state)

			#new_observation, reward, done, info = env.step(action)
			reward = utils.calculate_reward(action, prob, c, A, gt, mask, self.params, self.reward_dim)
			new_state = self.get_new_state(state) # expects numpy array
			
			# # dont update if this is validation
			# if _ep%50 == 0 or _ep>450:
			# 	continue

			# if done:
			# 	new_state = None
			# else:
			# 	new_state = np.float32(new_observation)
			# 	# push this exp in ram
			# 	ram.add(state, action, reward, new_state)

			for s,a,r,n in zip(state, action, reward, new_state):
				self.ram.add(s,a,r,n)

			state = new_state

			# perform optimization
			self.trainer.optimize()
			# if done:
			# 	break

		# check memory consumption and clear memory
		gc.collect()
		# process = psutil.Process(os.getpid())
		# print(process.memory_info().rss)
		action, prob = self.trainer.get_exploitation_action(state)
		reward = utils.calculate_reward(action, prob, c, A, gt, mask, self.params, self.reward_dim)[0]
		# if _ep%100 == 0:
		# 	trainer.save_models(_ep)
		
		return (action,reward)

