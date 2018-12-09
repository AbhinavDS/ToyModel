from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

from . import utils
from . import model
from . import genus

LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 0.01


class Trainer:

	def __init__(self, state_dim, action_dim, action_lim, ram, batch_size, critic_step=1):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.batch_size = batch_size
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = model.Critic(self.state_dim, self.action_dim)
		self.target_critic = model.Critic(self.state_dim, self.action_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)
		self.critic_step = critic_step

		self.genus = genus.Genus(self.state_dim)
		self.genus_optimizer = torch.optim.Adam(self.genus.parameters(),LEARNING_RATE)
		self.criterionG = nn.NLLLoss()
		
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state))
		action = self.target_actor.forward(state).detach()
		return action.data.numpy()

	def get_final_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state))
		action = self.target_actor.forward(state).detach()
		genus = torch.argmax(self.genus.forward(state),dim=-1).detach()
		return action.data.numpy(), genus.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state))
		action = self.actor.forward(state).detach()
		new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1,a1,r1,s2 = self.ram.agg_sample(self.batch_size)

		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))

		for i in range(self.critic_step):
			# ---------------------- optimize critic ----------------------
			# Use target actor exploitation policy here for loss evaluation
			
			# a2 = self.target_actor.forward(s2).detach()
			# next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
			
			# y_exp = r + gamma*Q'( s2, pi'(s2))
			y_expected = r1 #+ GAMMA*next_val
			# y_pred = Q( s1, a1)
			y_predicted = torch.squeeze(self.critic.forward(s1, a1))
			# compute critic loss, and update the critic
			#print(y_predicted,y_expected,"hi")
			loss_critic = F.smooth_l1_loss(y_predicted, y_expected.squeeze())
			self.critic_optimizer.zero_grad()
			loss_critic.backward()
			self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.soft_update(self.target_critic, self.critic, TAU)

		# if self.iter % 100 == 0:
		if self.batch_size > 1:
			y_1 = y_predicted.data.numpy()[0]
			r_1 = r1.data.numpy()[0]
		else:
			y_1 = y_predicted.data.numpy()
			r_1 = r1.data.numpy()
		print ('Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
			' Loss_critic :- ', loss_critic.data.numpy(), ' Critic Pred Reward :- ', y_1, ' Actual Reward :- ', r_1)
		self.iter += 1

	
	def genus_step(self, state, gt_genus):
		state = Variable(torch.from_numpy(state))
		gt_genus = torch.from_numpy(gt_genus)
		gt_genus.requires_grad  = False
		pred_genus = self.genus.forward(state)
		loss_genus = self.criterionG(pred_genus, gt_genus)
		self.genus_optimizer.zero_grad()
		loss_genus.backward()
		self.genus_optimizer.step()

	def save_models(self, episode_count):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		torch.save(self.target_actor.state_dict(), '/home/abhinavds/Documents/Projects/ToyModel/ckpt/rl/Models_genus/' + str(episode_count) + '_actor.pt')
		torch.save(self.target_critic.state_dict(), '/home/abhinavds/Documents/Projects/ToyModel/ckpt/rl/Models_genus/' + str(episode_count) + '_critic.pt')
		torch.save(self.genus.state_dict(), '/home/abhinavds/Documents/Projects/ToyModel/ckpt/rl/Models_genus/' + str(episode_count) + '_genus.pt')
		print ('Models saved successfully')

	def load_models(self, episode):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.actor.load_state_dict(torch.load('/home/abhinavds/Documents/Projects/ToyModel/ckpt/rl/Models_genus/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('/home/abhinavds/Documents/Projects/ToyModel/ckpt/rl/Models_genus/' + str(episode) + '_critic.pt'))
		self.genus.load_state_dict(torch.load('/home/abhinavds/Documents/Projects/ToyModel/ckpt/rl/Models_genus/' + str(episode) + '_genus.pt'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		print ('Models loaded succesfully')