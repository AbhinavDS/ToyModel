import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from  src import dtype, dtypeL, dtypeB

class Splitter(nn.Module):
	def __init__(self, image_width, dim_size):
		super(Splitter, self).__init__()
		self.image_width = image_width
		self.dim_size = dim_size #Coordinate dimension
		self.layers1 = nn.ModuleList()
		self.layers2 = nn.ModuleList()
		self.layers3 = nn.ModuleList()
		# self.add_layer(self.layers1, torch.nn.Conv1d(i))
		# self.add_layer(self.layers2, nn.Linear(2*feature_size,feature_size))
		# self.add_layer(nn.Linear(2*feature_size,feature_size))
		# for i in range(depth):
		# 	self.add_layer(nn.Linear(self.feature_size,self.feature_size))
		# 	self.add_layer(nn.Linear(self.feature_size,self.feature_size))
		# self.W_p_c = nn.Linear(self.dim_size,self.feature_size)
		# self.W_p_s = nn.Linear(self.feature_size,self.feature_size)
		# self.W_p = nn.Linear(2*self.feature_size,self.feature_size)
		# self.W_c = nn.Linear(self.feature_size,self.dim_size)
		# self.W_ic = nn.Linear(self.dim_size,self.feature_size)
		# self.W_lol = nn.Linear(self.feature_size,self.dim_size)
		# self.a = nn.Tanh()
		# # Initialize weights according to the Xavier Glorot formula
		# nn.init.xavier_uniform_(self.W_p_c.weight)
		# nn.init.xavier_uniform_(self.W_p_s.weight)
		# nn.init.xavier_uniform_(self.W_p.weight)
		# nn.init.xavier_uniform_(self.W_c.weight)
		# nn.init.xavier_uniform_(self.W_ic.weight)
		# nn.init.xavier_uniform_(self.W_lol.weight)

	def add_layer(self, layers, layer,init=True):
		layers.append(layer)
		if init:
			nn.init.xavier_uniform_(layers[-1].weight)

	def forward(self, proj_c, proj_gt):
		#proj_c: batch_size x image_width_1d
		#proj_gt: batch_size x image_width_1d
		
		# temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		# # #s = self.a(self.W_s(s_prev) + self.W_x(x_prev))
		
		# c_f = self.a(self.W_p_c(c_prev))
		# s_f = self.a(self.W_p_s(s_prev))
		# feature_from_state = self.a(self.W_p(torch.cat((c_f,s_f),dim=2)))
		# # #concat_feature = torch.cat((c_prev,s_prev),dim=1)
		# # #feature_from_state = self.a(self.W_p_2(self.a(self.W_p_1(concat_feature))))
		# x = torch.cat((feature_from_state,x_prev),dim=2)
		# x = self.a(self.layers[0](x)+torch.bmm(temp_A,self.layers[1](x)))
		# for i in range(2,len(self.layers),2):
		# 	x = self.a(self.layers[i](x)+torch.bmm(temp_A,self.layers[i+1](x)) + x)
		# #c = self.a(self.W_c(x)+c_prev)
		# # c = self.a(self.W_p_s(s_prev[0,:]))#+c_prev)
		# c = self.a(self.W_lol(x))
		# s = s_prev
		# #print(x)
		# #c = c.view((-1,2))
		return x

	def calculateReward(points, c, A, gt):
		x1, y1, x2, y2 = points
		return 1
