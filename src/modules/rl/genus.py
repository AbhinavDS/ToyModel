import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from  src import dtype, dtypeL, dtypeB

class Genus(nn.Module):
	def __init__(self, state_dim):
		super(Genus, self).__init__()
		
		self.state_dim = state_dim
		
		self.fc1 = nn.Linear(state_dim,256)
		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,64)
		self.fc4 = nn.Linear(64,2)
		self.softmax = nn.LogSoftmax(dim=-1)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.xavier_uniform_(self.fc3.weight)
		nn.init.xavier_uniform_(self.fc4.weight)

	def forward(self, state):
		"""
		returns split probability
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = self.softmax(x)
		return x
