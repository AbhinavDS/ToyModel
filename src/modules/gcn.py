import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from  src import dtype, dtypeL, dtypeB

class GCN(nn.Module):
	def __init__(self, feature_size, dim_size, depth, weights_init='zero', residual_change=False):
		super(GCN, self).__init__()
		self.residual_change = residual_change
		self.feature_size = feature_size
		self.dim_size = dim_size #Coordinate dimension
		self.layers = nn.ModuleList()
		self.depth = depth
		self.add_layer(nn.Linear(2*feature_size,feature_size))
		self.add_layer(nn.Linear(2*feature_size,feature_size))
		for i in range(depth):
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
		self.W_p_c = nn.Linear(self.dim_size,self.feature_size)
		self.W_p_s = nn.Linear(self.feature_size,self.feature_size)
		self.W_p = nn.Linear(2*self.feature_size,self.feature_size)
		self.W_c = nn.Linear(self.feature_size,self.dim_size)
		self.W_ic = nn.Linear(self.dim_size,self.feature_size)
		self.W_final = nn.Linear(self.feature_size,self.dim_size)
		self.a = nn.Tanh()
		if weights_init == 'zero':
			self.zero_init()
		else:
			self.xavier_init()
	
	# Initialize weights according to the Xavier Glorot formula
	def xavier_init(self):
		nn.init.xavier_uniform_(self.W_p_c.weight)
		nn.init.xavier_uniform_(self.W_p_s.weight)
		nn.init.xavier_uniform_(self.W_p.weight)
		nn.init.xavier_uniform_(self.W_c.weight)
		nn.init.xavier_uniform_(self.W_ic.weight)
		nn.init.xavier_uniform_(self.W_final.weight)

	def zero_init(self):
		nn.init.constant_(self.W_p_c.weight,0)
		nn.init.constant_(self.W_p_s.weight,0)
		nn.init.constant_(self.W_p.weight,0)
		nn.init.constant_(self.W_c.weight,0)
		nn.init.constant_(self.W_ic.weight,0)
		nn.init.constant_(self.W_final.weight,0)

	def add_layer(self,layer,init=True):
		self.layers.append(layer)
		if init:
			nn.init.xavier_uniform_(self.layers[-1].weight)

	def forward(self, x_prev, s_prev, c_prev, A):
		#x: batch_size x V x feature_size
		#s: batch_size x V x feature_size
		#c: batch_size x V x dim_size
		#W: feature_size x feature_size
		#A: batch_size x V x V
		
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		# #s = self.a(self.W_s(s_prev) + self.W_x(x_prev))
		print ("X1")
		assert (not torch.isnan(x_prev).any())
		print ("W1")
		assert (not torch.isnan(c_prev).any())
		
		# print (self.W_p_c.weight)
		
		c_f = self.a(self.W_p_c(c_prev))
		s_f = self.a(self.W_p_s(s_prev))
		print ("c1")
		assert (not torch.isnan(c_f).any())
		print ("s1")
		assert (not torch.isnan(s_f).any())
		feature_from_state = self.a(self.W_p(torch.cat((c_f,s_f),dim=2)))
		print ("ff1")
		assert (not torch.isnan(feature_from_state).any())
		# #concat_feature = torch.cat((c_prev,s_prev),dim=1)
		# #feature_from_state = self.a(self.W_p_2(self.a(self.W_p_1(concat_feature))))
		x = torch.cat((feature_from_state,x_prev),dim=2)
		print ("X2")
		assert (not torch.isnan(x).any())
		x = self.a(self.layers[0](x)+torch.bmm(temp_A,self.layers[1](x)))
		print ("X3")
		assert (not torch.isnan(x).any())
		for i in range(2,len(self.layers),2):
			x = self.a(self.layers[i](x)+torch.bmm(temp_A,self.layers[i+1](x)) + x)
			print ("X4",i)
			assert (not torch.isnan(x).any())
		# c = self.a(self.W_c(x)+c_prev)
		# c = self.a(self.W_p_s(s_prev[0,:]))#+c_prev)
		c = self.a(self.W_final(x))
		print ("X5")
		assert (not torch.isnan(x).any())
		print ("W2")
		assert (not torch.isnan(c).any())
		if self.residual_change:
			c = c + c_prev
			print ("W3")
			assert (not torch.isnan(c).any())
		s = s_prev
		#print(x)
		#c = c.view((-1,2))
		return x, s, c

	def embed(self,c):
		return self.a(self.W_ic(c))