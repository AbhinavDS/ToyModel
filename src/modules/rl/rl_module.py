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
	def __init__(self, params, prefix):
		self.MAX_STEPS = 1#000
		self.MAX_BUFFER = 1000000
		self.params = params
		self.prefix = "" if prefix == "" else prefix+"_"
		self.path = os.path.join(self.params.save_model_dirpath, "rl")
		self.path = os.path.join(self.path, self.prefix)

		S_DIM = 2*params.img_width + params.feature_size
		A_DIM = 4
		A_MAX = 1

		self.ram = buffer.MemoryBuffer(self.MAX_BUFFER)
		self.trainer = rl_train.Trainer(S_DIM, A_DIM, A_MAX, self.ram, params.batch_size, critic_step=3)
		self._ep=0
		
	
	def get_new_state(self,state):
		#return numpy array
		return state

	def step(self,c, s, gt, A, mask, proj_pred, proj_gt, to_split = True):
		s_avg = torch.mean(s, dim=1)
		state = np.float32(np.concatenate((proj_gt,proj_pred, s_avg.cpu().detach().numpy()),axis=1))
		self._ep += 1
		for step in range(self.MAX_STEPS):
			action = self.trainer.get_exploration_action(state)
			# if _ep%5 == 0:
			# 	# validate every 5th episode
			# 	action = trainer.get_exploitation_action(state)
			# else:
			# 	# get action based on observation, use exploration policy here
			# 	action = trainer.get_exploration_action(state)

			#new_observation, reward, done, info = env.step(action)
			reward, _ = utils.calculate_reward(action,c,A,gt,mask,self.params)
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
		# action = self.trainer.get_exploitation_action(state)
		action, pred_genus = self.trainer.get_final_action(state)
		reward, intersections = utils.calculate_reward(action,c,A,gt,mask,self.params)
		gt_genus = utils.calculate_genus(c, to_split)
		self.trainer.genus_step(state, gt_genus)

		if self._ep%200 == 0:
			self.ep = self._ep%1000
			self.trainer.save_models(self._ep, path=self.path)
		
		return (action,reward,intersections, pred_genus, gt_genus)

	def step_test(self,c, s, gt, A, mask, proj_pred, proj_gt):
		self.trainer.actor.eval()
		self.trainer.target_actor.eval()
		self.trainer.critic.eval()
		self.trainer.target_critic.eval()
		s_avg = torch.mean(s, dim=1).detach()
		state = np.float32(np.concatenate((proj_gt,proj_pred, s_avg.cpu().numpy()),axis=1))
		gc.collect()
		action, genus = self.trainer.get_final_action(state)
		intersections  = utils.get_intersections(action, c, A, self.params)
		return (action, genus, intersections, genus, genus)

	def load(self,count):
		self.trainer.load_models(count, path=self.path)


	def genus_step(self, state, gt_genus):
		state = Variable(torch.from_numpy(state))
		gt_genus = Variable(torch.from_numpy(gt_genus))