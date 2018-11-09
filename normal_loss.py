import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
class NormalLoss(nn.Module):

	def __init__(self):
		super(NormalLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, preds, gts, gts_normals, A):
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		
		# Get normals of nearest gt vertex
		P = self.batch_pairwise_dist(gts, preds)
		nearest_gt = torch.argmin(P, 1)
		q = gts_normals[nearest_gt][0]

		# Calculate difference for each pred vertex, use adj mat to filter out non-neighbours
		diff_neighbours = self.normal_pred(preds, temp_A)
		
		# Calculate final loss
		return self.calculate_loss(diff_neighbours, q)

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

	def calculate_loss(self, p_k, nq):
		points_dim, num_points_x, _ = p_k.size()
		nq = nq.transpose(1,0)
		nq = nq.repeat(1,num_points_x).view(points_dim, num_points_x, num_points_x)
		inner_product = torch.mul(nq,p_k)
		inner_product = torch.sum(inner_product, dim=0)
		inner_product_sq = torch.mul(inner_product,inner_product)
		return torch.sum(inner_product_sq)


	def batch_pairwise_dist(self,x,y):
		x = x.unsqueeze(0)
		y = y.unsqueeze(0)
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)
		#brk()
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P