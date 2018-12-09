import time
import torch
import torch.nn as nn
import numpy as np
import random
import math
from src.data import dataLoader
from src.util import utils
from src.loss.chamfer_loss import ChamferLoss
from src.loss.normal_loss import NormalLoss
from src.loss.laplacian_loss import LaplacianLoss
from src.loss.edge_loss import EdgeLoss
from src.modules.deformer import Deformer
from src.modules.vertex_adder import VertexAdder
from src.modules.vertex_splitter import VertexSplitter
from src.modules.rl.rl_module import RLModule

from  src import dtype, dtypeL, dtypeB

class Model(nn.Module):
	def __init__(self, params):
		super(Model, self).__init__()
		self.params = params
		self.deformer = None
		if torch.cuda.is_available():
			self.deformer = [Deformer(self.params.feature_size, self.params.dim_size, self.params.depth).cuda() for i in range(1 + self.params.num_blocks1 + self.params.num_blocks2)]
		else:
			self.deformer = [Deformer(self.params.feature_size, self.params.dim_size, self.params.depth).cuda() for i in range(1 + self.params.num_blocks1 + self.params.num_blocks2)]
		
		self.adder = VertexAdder(params.add_prob).cuda()
		
		self.splitter = VertexSplitter().cuda()

		self.criterionC = ChamferLoss()
		self.criterionN = NormalLoss()
		self.criterionL = LaplacianLoss()
		self.criterionE = EdgeLoss()
		self.optimizer_params1 = []
		self.optimizer_params2 = []
	
		for i in range(0, self.params.num_blocks1 + 1):
			self.optimizer_params1 += self.deformer[i].parameters()

		for i in range(self.params.num_blocks1 + 1, len(self.deformer)):
			self.optimizer_params2 += self.deformer[i].parameters()

		self.loss = 0.0
		self.closs = 0.0
		self.nloss = 0.0
		self.laploss = 0.0
		self.eloss = 0.0

		self.deformer1_dict = {}
		for i in range(len(self.deformer)):
			self.deformer1_dict["Deformer_"+str(i)] = self.deformer[i]

		self.rl_module = RLModule(params)
		self.unfreezed = True

		if params.load_model_path:
			self.load(params.load_model_path, count=9000)	

	
	def freeze1(self, unfreezed):
		self.unfreezed = unfreezed
		for param in self.optimizer_params1:
			param.requires_grad = unfreezed

	def freeze2(self, unfreezed):
		self.unfreezed = unfreezed
		for param in self.optimizer_params2:
			param.requires_grad = unfreezed

	def load_rl(self, count):
		self.rl_module.load(count)


	def load(self, path, load_dict=None, count=None):
		checkpoint = torch.load(path)
		print ("RESTORING...")
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
			if key in checkpoint:
				load_dict[key].load_state_dict(checkpoint[key])

		if count:
			self.load_rl(count)

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

	def forward1(self, x, c, s, A, gt, gtnormals, mask):
		self.clear_losses()
		# Vertex addition
		for block in range(self.params.num_blocks1-self.params.start_block):
			x, c, A, s = self.adder.forward(x, c, A, s)
		
		x = self.deformer[0].embed(c)
		c_prev = c
		x, s, c = self.deformer[0].forward(x,s,c_prev,A)
		norm = c.size(1) * (self.params.num_blocks1 + 1)
		self.laploss = self.criterionL(c_prev, c, A) / norm
		self.closs = self.criterionC(c, gt, mask) / norm
		self.eloss = self.criterionE(c, A) / norm
		self.nloss = self.criterionN(c, gt, gtnormals, A, mask) / norm
		

		for block in range(1, self.params.num_blocks1+1):
			if block < self.params.start_block:
				x, c, A, s = self.adder.forward(x, c, A, s)
			c_prev = c
			x, s, c = self.deformer[block].forward(x,s,c_prev,A)
		
			norm = c.size(1) * (self.params.num_blocks1 + 1)
			self.laploss += (self.criterionL(c_prev, c, A)/norm)
			self.closs += (self.criterionC(c, gt, mask)/norm)
			self.eloss += (self.criterionE(c, A)/norm)
			self.nloss += (self.criterionN(c, gt, gtnormals, A, mask)/norm)
		
		self.loss = self.closs + self.params.lambda_n*self.nloss + self.params.lambda_lap*self.laploss + self.params.lambda_e*self.eloss
			
		proj_pred = utils.flatten_pred_batch(utils.getPixels(c), A, self.params)

		return x, c, s, A, proj_pred

	def forward2(self, x, c, s, A, gt, gtnormals, mask):
		self.clear_losses()		
		for block in range(self.params.num_blocks1 + 1, len(self.deformer)):
			x, c, A, s = self.adder.forward(x, c, A, s)
			c_prev = c
			x, s, c = self.deformer[block].forward(x,s,c_prev,A)
		
			norm = c.size(1) * (self.params.num_blocks2)
			self.laploss += (self.criterionL(c_prev, c, A)/norm)
			self.closs += (self.criterionC(c, gt, mask)/norm)
			self.eloss += (self.criterionE(c, A)/norm)
			self.nloss += (self.criterionN(c, gt, gtnormals, A, mask)/norm)
		
		self.loss = 0
		self.loss = self.closs + self.params.lambda_n*self.nloss + self.params.lambda_lap*self.laploss + self.params.lambda_e*self.eloss
			
		proj_pred = utils.flatten_pred_batch(utils.getPixels(c), A, self.params)

		return x, c, s, A, proj_pred

	def split(self, c, x, gt, A, mask, proj_pred, proj_gt, ep, test):
		if test:
			return self.rl_module.step_test(c, x, gt, A, mask, proj_pred, proj_gt)
		else:
			return self.rl_module.step(c, x, gt, A, mask, proj_pred, proj_gt, ep)
