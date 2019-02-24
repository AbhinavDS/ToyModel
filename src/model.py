import os
import time
import torch
import torch.nn as nn
import numpy as np
import random
import math
from src.data import dataLoader
from src.util import utils
from src.modules.deformer_block import DeformerBlock
from src.modules.vertex_splitter import VertexSplitter
from src.modules.rl.rl_module import RLModule

from src import dtype, dtypeL, dtypeB

class Model():
	def __init__(self, params, load_models=False):
		super(Model, self).__init__()
		self.params = params
		self.deformer_block1 = DeformerBlock(self.params, self.params.num_gcns1, self.params.initial_adders, True, weights_init='xavier')
		self.deformer_block2 = DeformerBlock(self.params, self.params.num_gcns2, 0, False, residual_change=True)

		self.splitter_block = VertexSplitter().cuda()

		self.optimizer_params1 = []
		self.optimizer_params2 = []
	
		for i in range(0, self.params.num_gcns1):
			self.optimizer_params1 += self.deformer_block1[i].parameters()
		for i in range(0, self.params.num_gcns2):
			self.optimizer_params2 += self.deformer_block2[i].parameters()

		# Create dictionary for saving
		self.deformer_block_dict = {}
		for i in range(len(self.deformer_block1)):
			self.deformer_block_dict["Deformer1_"+str(i)] = self.deformer_block1[i]
		for i in range(len(self.deformer_block2)):
			self.deformer_block_dict["Deformer2_"+str(i)] = self.deformer_block2[i]

		self.rl_module = [RLModule(params, str(i)) for i in range(self.params.num_rl)]
		self.last_epoch = 0
		if load_models:
			self.load(os.path.join(params.load_model_dirpath,'model.toy'))
			self.load_rl(self.params.load_rl_count)
		

	def load_rl(self, count):
		for i in range(len(self.rl_module)):
			self.rl_module[i].load(count)
			self.rl_module[i]._ep = count

	def load(self, path):
		checkpoint = torch.load(path)
		print ("RESTORING...")
		for key in self.deformer_block_dict.keys():
			self.deformer_block_dict[key].load_state_dict(checkpoint[key])
		self.last_epoch = checkpoint["last_epoch"]
		
	def save(self, path, epoch):
		checkpoint = {}
		print ("SAVING...")
		for key in self.deformer_block_dict.keys():
			checkpoint[key] = self.deformer_block_dict[key].state_dict()
		checkpoint["last_epoch"] = epoch
		torch.save(checkpoint, path)

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

	def split(self, c, x, gt, A, mask, proj_pred, proj_gt, test, block, to_split = True):
		if test:
			return self.rl_module[block].step_test(c, x, gt, A, mask, proj_pred, proj_gt)
		else:
			self.rl_module[block].step(c, x, gt, A, mask, proj_pred, proj_gt, to_split)
			return self.rl_module[block].step(c, x, gt, A, mask, proj_pred, proj_gt, to_split)
