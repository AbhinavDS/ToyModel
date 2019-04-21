import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim, params):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.image_C = params.num_image_feats_layers_rl
		self.image_H = params.rl_image_resolution[0]
		self.image_W = params.rl_image_resolution[1]

		self.hidden_dim = 1000

		self.image_dim = (self.image_C * self.image_H * self.image_W)
		self.feature_dim = self.state_dim - self.image_dim

		new_H = self.image_H
		new_W = self.image_W

		self.conv_block1 = nn.Sequential(
			nn.Conv2d(self.image_C, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)

		new_H = new_H//2 + 1
		new_W = new_W//2 + 1
		
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)
		
		new_H = new_H//2 + 1
		new_W = new_W//2 + 1
		
		self.conv_dim = (128) * new_H * new_W
		
		self.fci1 = nn.Linear(self.conv_dim, self.hidden_dim)
		
		self.fcf1 = nn.Linear(self.feature_dim, self.hidden_dim)
		
		self.fca1 = nn.Linear(self.action_dim,self.hidden_dim)
		
		self.fc2 = nn.Linear(3*self.hidden_dim,self.hidden_dim)
		
		self.fc3 = nn.Linear(self.hidden_dim,1)
		
		self.a = nn.ReLU()
		
	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		img_flat = state.narrow(1, 0, self.image_dim)
		feat_flat = state.narrow(1, self.image_dim, self.feature_dim)

		img = img_flat.view(-1, self.image_C, self.image_H, self.image_W)

		conv_feats = self.conv_block2(self.conv_block1(img)).view(img.size(0), -1)

		i1 = self.a(self.fci1(conv_feats))
		f1 = self.a(self.fcf1(feat_flat))
		a1 = self.a(self.fca1(action))
		x = torch.cat((i1,f1,a1),dim=1)

		x = self.a(self.fc2(x))
		x = self.fc3(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim, params):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.image_C = params.num_image_feats_layers_rl
		self.image_H = params.rl_image_resolution[0]
		self.image_W = params.rl_image_resolution[1]

		self.image_dim = (self.image_C * self.image_H * self.image_W)
		self.feature_dim = self.state_dim - self.image_dim

		new_H = self.image_H
		new_W = self.image_W

		self.conv_block1 = nn.Sequential(
			nn.Conv2d(self.image_C, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)

		new_H = new_H//2 + 1
		new_W = new_W//2 + 1
		
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)
		
		new_H = new_H//2 + 1
		new_W = new_W//2 + 1
		
		self.conv_dim = (128) * new_H * new_W

		self.fci1 = nn.Linear(self.conv_dim, 256)
		
		self.fcf1 = nn.Linear(self.feature_dim, 256)
		
		self.fc1 = nn.Linear(512,256)
		
		self.fc2 = nn.Linear(256,128)
		
		self.fc3 = nn.Linear(128,64)
		
		self.fc4 = nn.Linear(64,action_dim)

		
	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		img_flat = state.narrow(1, 0, self.image_dim)
		feat_flat = state.narrow(1, self.image_dim, self.feature_dim)
		img = img_flat.view(-1, self.image_C, self.image_H, self.image_W)
		
		conv_featsa = self.conv_block2(self.conv_block1(img))
		conv_feats = conv_featsa.view(img.size(0), -1)
		
		i1 = F.relu(self.fci1(conv_feats))
		f1 = F.relu(self.fcf1(feat_flat))
		s1 = torch.cat((i1,f1), dim=1)
		x = F.relu(self.fc1(s1))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = F.tanh(self.fc4(x))

		action = action * self.action_lim

		return action


