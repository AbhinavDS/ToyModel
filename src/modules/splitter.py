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
		self.add_layer(self.merge_layers, self.activation, init=False)
		self.add_layer(self.feature_layers, nn.MaxPool1d(2))
		self.add_layer(self.feature_layers, nn.Conv1d(16, 32, 3)) # n_channels, out_channels, kernel_size
		self.add_layer(self.merge_layers, self.activation, init=False)
		self.add_layer(self.feature_layers, nn.MaxPool1d(2))
		self.add_layer(self.feature_layers, nn.Conv1d(32, 64, 3)) # n_channels, out_channels, kernel_size
		self.add_layer(self.merge_layers, self.activation, init=False)
		self.add_layer(self.feature_layers, nn.MaxPool1d(2))

		self.merged_fc = nn.Linear(int(self.image_width/8),int(self.image_width/8))
		self.output_line = nn.Linear(int(self.image_width/8), dim_size*2)
		self.output_value = nn.Linear(int(self.image_width/8) + dim_size*2, 1)
		nn.init.xavier_uniform_(self.merged_fc.weight)
		nn.init.xavier_uniform_(self.output_line.weight)
		nn.init.xavier_uniform_(self.output_value.weight)

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

	def calculateReward(points, c, A, gt):
		x1, y1, x2, y2 = points
		return 1
