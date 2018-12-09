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

		self.ram = buffer.MemoryBuffer(self.MAX_BUFFER)
		self.trainer = rl_train.Trainer(S_DIM, A_DIM, A_MAX, self.ram, params.batch_size, critic_step=3)
	
	def get_new_state(self,state):
		#return numpy array
		return state

	def step(self,c, s, gt, A, mask, proj_pred, proj_gt,_ep):
		s_avg = torch.mean(s, dim=1)
		state = np.float32(np.concatenate((proj_gt,proj_pred, s_avg.cpu().numpy()),axis=1))
		for r in range(self.MAX_STEPS):
			action = self.trainer.get_exploration_action(state)
			# if _ep%5 == 0:
			# 	# validate every 5th episode
			# 	action = trainer.get_exploitation_action(state)
			# else:
			# 	# get action based on observation, use exploration policy here
			# 	action = trainer.get_exploration_action(state)

			#new_observation, reward, done, info = env.step(action)
			reward,_ = utils.calculate_reward(action,c,A,gt,mask,self.params)
			new_state = self.get_new_state(state)#expects numpy array
			
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
				self.ram.agg_add(s,a,r,n)

			state = new_state

			# perform optimization
			self.trainer.optimize()
			# if done:
			# 	break

		# check memory consumption and clear memory
		gc.collect()
		# process = psutil.Process(os.getpid())
		# print(process.memory_info().rss)
		action = self.trainer.get_exploitation_action(state)
		reward,intersections = utils.calculate_reward(action,c,A,gt,mask,self.params)
		# if _ep%200 == 0:
		# 	self.trainer.save_models((_ep)%10000)
		
		return (action,reward,intersections)

	def step_test(self,c, s, gt, A, mask, proj_pred, proj_gt):
		self.trainer.actor.eval()
		self.trainer.target_actor.eval()
		self.trainer.critic.eval()
		self.trainer.target_critic.eval()
		s_avg = torch.mean(s, dim=1)
		state = np.float32(np.concatenate((proj_gt,proj_pred, s_avg.cpu().numpy()),axis=1))
		gc.collect()
		action = self.trainer.get_exploitation_action(state)
		reward = utils.calculate_reward(action,c,A,gt,mask,self.params)
		return (action,reward)

	def load(self, count):
		self.trainer.load_models(count)


