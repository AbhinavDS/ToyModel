import time
import torch
from torch import optim
import numpy as np
import random
import math
from data import dataLoader
from src.util import utils
from src.loss.chamfer_loss import ChamferLoss
from src.loss.normal_loss import NormalLoss
from src.loss.laplacian_loss import LaplacianLoss
from src.loss.edge_loss import EdgeLoss
from src.modules.deformer import Deformer
from src.modules.vertex_adder import VertexAdder


class Model(nn.Module):
	def __init__(self, num_blocks, params):
		super(Model, self).__init__()
		self.params = params
		self.deformer = None
		if torch.cuda.is_available():
			self.deformer = [Deformer(feature_size,dim_size,params.depth).cuda() for i in range(num_blocks + 1)]
		else:
			self.deformer = [Deformer(feature_size,dim_size,params.depth).cuda() for i in range(num_blocks + 1)]

		if params.load_model_path:
			self.load(params.load_model_path)	
		
		self.adder = VertexAdder(params.add_prob).cuda()
	
		self.criterionC = ChamferLoss()
		self.criterionN = NormalLoss()
		self.criterionL = LaplacianLoss()
		self.criterionE = EdgeLoss()
		self.optimizer_params = []
	
		for i in range(num_blocks + 1):
			self.optimizer_params += deformer[i].parameters()

	
	def load(self, path, dict=None):
		# if none load all
		deformer.load_state_dict(torch.load(params.load_model_path))
		pass

	def save(self, path, dict=None):
		# if none save all
		pass

	def create_start_data(self):
		c,x,A = utils.inputMesh(self.params.feature_size)# x is c with zeros appended, x=f ..pixel2mesh
		c = np.expand_dims(c, 0)
		c = np.tile(c,(self.params.batch_size, 1, 1))
		x = np.expand_dims(x, 0)
		x = np.tile(x,(self.params.batch_size, 1, 1))
		A = np.expand_dims(A, 0)
		A = np.tile(A,(self.params.batch_size, 1, 1))
		x = torch.Tensor(x).type(dtype)
		c = torch.Tensor(c).type(dtype)
		return x, c, A

	def forward1(self, x, c, s, A):
		loss = 0.0
		closs = 0.0
		laploss = 0
		nloss = 0
		eloss = 0

		# Vertex addition
		for block in range(num_blocks-self.params.start_block):
			x, c, A, s = adder.forward(x, c, A, s)
		
		x = deformer[0].embed(c)
		c_prev = c
		x, s, c = deformer[0].forward(x,s,c_prev,A)
		norm = c.size(1) * (num_blocks + 1)
		laploss = criterionL(c_prev, c, A) / norm
		closs = criterionC(c, gt, mask) / norm
		eloss = criterionE(c, A) / norm
		nloss = criterionN(c, gt, gtnormals, A, mask) / norm
		

		for block in range(num_blocks):
		# for block in range(1):
			if block < self.params.start_block:
				x, c, A, s = adder.forward(x, c, A, s)
			c_prev = c
			x, s, c = deformer[block + 1].forward(x,s,c_prev,A)
		
			norm = c.size(1)# * (num_blocks + 1)
			laploss += (criterionL(c_prev, c, A)/norm)
			closs += (criterionC(c, gt, mask)/norm)
			eloss += (criterionE(c, A)/norm)
			nloss += (criterionN(c, gt, gtnormals, A, mask)/norm)
		
		loss = closs + params.lambda_n*nloss + params.lambda_lap*laploss + params.lambda_e*eloss
		total_closs +=closs/len(train_data)
		total_laploss +=laploss/len(train_data)
		total_nloss +=nloss/len(train_data)
		total_eloss +=eloss/len(train_data)
		total_loss += loss/len(train_data)
			
		proj_pred = utils.flatten_pred_batch(utils.getPixels(c), A, params)

		return x, c, s, A, total_loss, proj_pred

	def forward2(self):
		return x_new, c_new, A_new, s_new

	def split(self):
		return x_new, c_new, A_new, s_new

