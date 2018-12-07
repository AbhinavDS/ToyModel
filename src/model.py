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
from  src import dtype, dtypeL, dtypeB

class Model(nn.Module):
	def __init__(self, params):
		super(Model, self).__init__()
		self.params = params
		self.deformer = None
		if torch.cuda.is_available():
			self.deformer = [Deformer(self.params.feature_size, self.params.dim_size, self.params.depth).cuda() for i in range(self.params.num_blocks + 1)]
		else:
			self.deformer = [Deformer(self.params.feature_size, self.params.dim_size, self.params.depth).cuda() for i in range(self.params.num_blocks + 1)]

		if params.load_model_path:
			self.load(params.load_model_path)	
		
		self.adder = VertexAdder(params.add_prob).cuda()
	
		self.criterionC = ChamferLoss()
		self.criterionN = NormalLoss()
		self.criterionL = LaplacianLoss()
		self.criterionE = EdgeLoss()
		self.optimizer_params = []
	
		for i in range(len(deformer)):
			self.optimizer_params += deformer[i].parameters()

		self.loss = 0.0
		self.closs = 0.0
		self.nloss = 0.0
		self.laploss = 0.0
		self.eloss = 0.0

		self.deformer1_dict = {}
		for i in range(len(self.deformer)):
			self.deformer1_dict["Deformer_"+str(i)] = self.deformer[i]
		
	def load(self, path, load_dict=None):
		checkpoint = torch.load(path)

		model_dict = {}
		for i in range(len(self.deformer)):
			model_dict["Deformer_"+str(i)] = self.deformer[i]
		
		if load_dict:
			for key in load_dict:
				load_dict[key] = model_dict[key]
		else:
			# if none save all
			load_dict = model_dict

		for key in load_dict:
			load_dict[key].load_state_dict(checkpoint[key])

	def save(self, path, save_dict=None):
		model_dict = {}
		for i in range(len(self.deformer)):
			model_dict["Deformer_"+str(i)] = self.deformer[i].state_dict()
		
		if save_dict:
			for key in save_dict:
				save_dict[key] = model_dict[key]
		else:
			# if none save all
			save_dict = model_dict
		torch.save(save_dict, path)

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

	def clear_losses(self):
		self.loss = 0.0
		self.closs = 0.0
		self.laploss = 0.0
		self.nloss = 0.0
		self.eloss = 0.0

	def forward1(self, x, c, s, A):
		self.clear_losses()
		# Vertex addition
		for block in range(self.params.num_blocks-self.params.start_block):
			x, c, A, s = self.adder.forward(x, c, A, s)
		
		x = self.deformer[0].embed(c)
		c_prev = c
		x, s, c = self.deformer[0].forward(x,s,c_prev,A)
		norm = c.size(1) * (num_blocks + 1)
		self.laploss = criterionL(c_prev, c, A) / norm
		self.closs = criterionC(c, gt, mask) / norm
		self.eloss = criterionE(c, A) / norm
		self.nloss = criterionN(c, gt, gtnormals, A, mask) / norm
		

		for block in range(self.params.num_blocks):
		# for block in range(1):
			if block < self.params.start_block:
				x, c, A, s = self.adder.forward(x, c, A, s)
			c_prev = c
			x, s, c = self.deformer[block + 1].forward(x,s,c_prev,A)
		
			norm = c.size(1)# * (num_blocks + 1)
			self.laploss += (criterionL(c_prev, c, A)/norm)
			self.closs += (criterionC(c, gt, mask)/norm)
			self.eloss += (criterionE(c, A)/norm)
			self.nloss += (criterionN(c, gt, gtnormals, A, mask)/norm)
		
		self.loss = self.closs + self.params.lambda_n*self.nloss + self.params.lambda_lap*self.laploss + self.params.lambda_e*self.eloss
			
		proj_pred = utils.flatten_pred_batch(utils.getPixels(c), A, params)

		return x, c, s, A, proj_pred

	def forward2(self):
		return None

	def split(self):
		return None

