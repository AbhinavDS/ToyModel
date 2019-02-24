import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

from src.loss.chamfer_loss import ChamferLoss
from src.loss.edge_loss import EdgeLoss
from src.loss.laplacian_loss import LaplacianLoss
from src.loss.normal_loss import NormalLoss
from src.modules.gcn import GCN
from src.modules.vertex_adder import VertexAdder
from src.util import utils
from src import dtype, dtypeL, dtypeB

class DeformerBlock(nn.Module):
	def __init__(self, params, num_gcns, initial_adders, embed, weights_init='zero', residual_change=False):
		super(DeformerBlock, self).__init__()
		self.params = params
		self.num_gcns = num_gcns
		self.initial_adders = initial_adders
		self.embed = embed
		assert (self.num_gcns > 0, "Number of gcns is 0")
		
		self.deformer_block = [GCN(self.params.feature_size, self.params.dim_size, self.params.depth, weights_init=weights_init, residual_change=residual_change).cuda() for _ in range(self.num_gcns)]
		self.adder = VertexAdder(params.add_prob).cuda()

		self.criterionC = ChamferLoss()
		self.criterionN = NormalLoss()
		self.criterionL = LaplacianLoss()
		self.criterionE = EdgeLoss()
		self.set_loss_to_zero()
		
	def __getitem__(self, key):
		return self.deformer_block[key]

	def __len__(self):
		return len(self.deformer_block)
		
	def set_loss_to_zero(self):
		self.loss = 0.0
		self.closs = 0.0
		self.laploss = 0.0
		self.nloss = 0.0
		self.eloss = 0.0

	def forward(self, x, c, s, A, Pid, gt, gtnormals, mask):
		self.set_loss_to_zero()
		total_blocks = self.initial_adders + self.num_gcns

		for _ in range(self.initial_adders):
			x, c, A, Pid, s = self.adder.forward(x, c, A, Pid, s)
		
		if self.embed:
			x = self.deformer_block[0].embed(c)

		for gcn in range(self.num_gcns):
			if gcn + self.initial_adders < self.num_gcns:
				x, c, A, Pid, s = self.adder.forward(x, c, A, Pid, s)

			c_prev = c
			x, s, c = self.deformer_block[gcn].forward(x,s,c_prev,A)
			norm = c.size(1) * (self.num_gcns)
			self.laploss += (self.criterionL(c_prev, c, A)/norm)
			self.closs += (self.criterionC(c, gt, mask)/norm)
			self.eloss += (self.criterionE(c, A)/norm)
			self.nloss += (self.criterionN(c, gt, gtnormals, A, mask)/norm)
		self.loss = self.closs + self.params.lambda_n*self.nloss + self.params.lambda_lap*self.laploss + self.params.lambda_e*self.eloss
		proj_pred = utils.flatten_pred_batch(utils.getPixels(c), A, self.params)
		return x, c, s, A, Pid, proj_pred