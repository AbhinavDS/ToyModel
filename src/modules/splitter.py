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
		self.activation = nn.ReLU()

		self.feature_layers = nn.ModuleList()
		self.add_layer(self.feature_layers, nn.Conv1d(1, 16, 3)) # n_channels, out_channels, kernel_size
		self.add_layer(self.feature_layers, self.activation, init=False)
		self.add_layer(self.feature_layers, nn.MaxPool1d(2), init=False)
		self.add_layer(self.feature_layers, nn.Conv1d(16, 32, 3)) # n_channels, out_channels, kernel_size
		self.add_layer(self.feature_layers, self.activation, init=False)
		self.add_layer(self.feature_layers, nn.MaxPool1d(2), init=False)
		self.add_layer(self.feature_layers, nn.Conv1d(32, 64, 3)) # n_channels, out_channels, kernel_size
		self.add_layer(self.feature_layers, self.activation, init=False)
		self.add_layer(self.feature_layers, nn.MaxPool1d(2), init=False)

		self.merged_fc = nn.Linear(int(self.image_width/8),int(self.image_width/8))
		self.output_line = nn.Linear(int(self.image_width/8), dim_size*2)
		self.output_value = nn.Linear(int(self.image_width/8) + dim_size*2, 1)
		nn.init.xavier_uniform_(self.merged_fc.weight)
		nn.init.xavier_uniform_(self.output_line.weight)
		nn.init.xavier_uniform_(self.output_value.weight)#I wrote code .
												

	def add_layer(self, layers, layer,init=True):
		layers.append(layer)
		if init:
			nn.init.xavier_uniform_(layers[-1].weight)

	def forward(self, proj_c, proj_gt):
		"""
		proj_c: batch_size x image_width_1d
		proj_gt: batch_size x image_width_1d
		x: batch_size x 4
		"""

		x1 = proj_c
		x2 = proj_gt
		for layer in self.feature_layers:
			x1 = layer(x1)
			x2 = layer(x2)

		x = torch.cat((x1,x2), dim=1)
		x = self.merged_fc(x)
		x = self.activation(x)
		line = self.output_line(x)
		line = self.activation(line)

		x = torch.cat((x,line), dim=1)
		v = self.output_value(x)
		v = self.output_value(v)

		return line, v

	def calculate_reward(self, points, c, Pid, gt, mask):
		p1, q1, p2, q2 = points
		batch_size = c.size(0)
		reward = torch.zeros((batch_size)).type(dtype)
		num_intersections = 0
		edges = []
		for b in range(batch_size):
			num_verts = Pid[b].shape[0]
			for i in range(num_verts):
				if num_intersections > 2:
					break
				for j in range(i,num_verts):
					if num_intersections > 2:
						break
					if Pid[b][i,j]:
						x1, y1, x2, y2 = c[b,i,0].item(),c[b,i,1].item(),c[b,j,0].item(),c[b,j,1].item()
						if(self.intersect(p1,q1,p2,q2,x1,y1,x2,y2)):
							num_intersections += 1
							edges.append([i,j])
			if num_intersections != 2:
				reward[b] = 0
				continue
			elif Pid[b][edges[0][0],edges[0][1]] != Pid[b][edges[1][0],edges[1][1]]:
				reward[b] = 0
				continue
			else:
				masked_gt = gt[b].masked_select(mask[b].unsqueeze(1).repeat(1,self.dim_size)).reshape(-1, self.dim_size)
				pos = 0
				neg = 0
				start = 0
				for i in range(masked_gt.size(0)):
					if masked_gt[i,0].item() == -2:
						start = (i+1)	
						continue
					if i+1 == masked_gt.size(0) or masked_gt[i+1,0].item() == -2:
						x1, y1, x2, y2 = masked_gt[i,0].item(), masked_gt[i,1].item(), masked_gt[start,0].item(), masked_gt[start,1].item()
					else:
						x1, y1, x2, y2 = masked_gt[i,0].item(), masked_gt[i,1].item(), masked_gt[i+1,0].item(), masked_gt[i+1,1].item()
					if self.line(p1,q1,p2,q2,x1,y1) > 0:
						pos += 1
					else:
						neg += 1
					if(self.intersect(p1,q1,p2,q2,x1,y1,x2,y2)):
						reward[b] = 0
						break
					else:
						reward[b] = 1
				if(pos == 0 or neg == 0):
					reward[b] = 0
		return reward

	def line(self,p1,q1,p2,q2,x1,y1):
		return (p2-p1)*y1 - (q2-q1)*x1 -(q1*p2-q2*p1)

	def intersect(self,p1,q1,p2,q2,x1,y1,x2,y2):
		#x1,y1 substituted in line joining p1,q1, p2,q2 is line(p1,q1,p2,q2,x1,y1)
		return self.line(p1,q1,p2,q2,x1,y1)*self.line(p1,q1,p2,q2,x2,y2) < 0 and self.line(x1,y1,x2,y2,p1,q1)*self.line(x1,y1,x2,y2,p2,q2) < 0