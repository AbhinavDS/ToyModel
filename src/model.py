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

	def forward1(self):
		return x_new, c_new, A_new, s_new

	def forward2(self):
		return x_new, c_new, A_new, s_new

	def split(self):
		return x_new, c_new, A_new, s_new

