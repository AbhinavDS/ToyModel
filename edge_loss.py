import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
class EdgeLoss(nn.Module):

	def __init__(self):
		super(EdgeLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, pred, A):
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		
		# Might not make sense in 2D
		edges = self.regularizer(pred, temp_A)
		loss = torch.mul(edges, edges)
		loss = torch.sum(loss)
		return loss

	def regularizer(self, x, A):
		batch_size, num_points_x, points_dim = x.size()
		x = x.permute(0,2,1)
		x = x.repeat(1,1,num_points_x).view(batch_size, points_dim, num_points_x, num_points_x)
		x_t = x.transpose(3,2)
		x_diff = x_t - x
		
		# Filter out non-neighbours		
		A = A.unsqueeze(1)
		A = A.repeat(1,points_dim,1,1)
		x_diff = torch.mul(x_diff, A)
		return x_diff