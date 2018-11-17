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

	def forward(self, preds, gts, gts_normals, A, mask):
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		
		# Get normals of nearest gt vertex
		P = self.batch_pairwise_dist(gts, preds)
		repeated_mask = mask.unsqueeze(2).repeat(1,1,preds.size(1))

		nearest_gt = torch.argmin(P, 1)
		for i in range(gts.size(0)):
			newP = P[i].masked_select(repeated_mask[i])
			newP = newP.reshape(1,-1, preds.size(1))
			nearest_gt[i] = torch.argmin(newP, 1)
		
		nearest_gt = nearest_gt.unsqueeze(2).repeat(1,1,2)
		q = torch.gather(gts_normals, 1, nearest_gt)
		
		# Calculate difference for each pred vertex, use adj mat to filter out non-neighbours
		diff_neighbours = self.normal_pred(preds, temp_A)
		
		# Calculate final loss
		return self.calculate_loss(diff_neighbours, q)

	def normal_pred(self, x, A):
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

	def calculate_loss(self, p_k, nq):
		batch_size, points_dim, num_points_x, _ = p_k.size()
		nq = nq.transpose(2,1)
		nq = nq.repeat(1, 1, num_points_x).view(batch_size, points_dim, num_points_x, num_points_x)
		inner_product = torch.mul(nq, p_k)
		
		inner_product = torch.sum(inner_product, dim=1)
		inner_product_sq = torch.mul(inner_product,inner_product)
		return torch.sum(inner_product_sq)


	def batch_pairwise_dist(self,x,y):
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