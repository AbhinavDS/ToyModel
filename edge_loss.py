import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class EdgeLoss(nn.Module):

	def __init__(self):
		super(EdgeLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, pred, A):
		temp_A = Variable(torch.Tensor(A).type(torch.FloatTensor),requires_grad=False)
		
		# Might not make sense in 2D
		edges = self.normal_pred(pred, temp_A)
		loss = torch.mul(edges, edges)
		loss = torch.sum(loss)
		return loss

	def normal_pred(self, x, A):
		num_points_x, points_dim = x.size()
		x = x.permute(1,0)
		x = x.repeat(1,num_points_x).view(points_dim, num_points_x, num_points_x)
		x_t = x.transpose(1,2)
		x_diff = x_t - x
		
		# Filter out non-neighbours		
		A = A.unsqueeze(0)
		A = A.repeat(points_dim,1,1)
		x_diff = torch.mul(x_diff, A)
		return x_diff