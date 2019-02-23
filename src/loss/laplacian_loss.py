import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
class LaplacianLoss(nn.Module):

	def __init__(self):
		super(LaplacianLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, pred1, pred2, A):
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		
		lap_pred1 = pred1 - self.centroid(pred1, temp_A)
		lap_pred2 = pred2 - self.centroid(pred2, temp_A)
		loss = lap_pred2 - lap_pred1
		loss = torch.mul(loss,loss)
		loss = torch.sum(loss)
		return loss

	def centroid(self, x, A):
		batch_size, num_points_x, points_dim = x.size()
		num_neighbours = torch.sum(A, dim=1)
		neighbours = x.permute(0,2,1)
		neighbours = neighbours.repeat(1, 1, num_points_x).view(batch_size, points_dim, num_points_x, num_points_x)

		# Filter out non-neighbours		
		A = A.unsqueeze(1)
		A = A.repeat(1, points_dim,1,1)
		neighbours = torch.mul(neighbours, A)

		neighbours = torch.sum(neighbours, dim = 3)
		neighbours = neighbours.permute(0,2,1)
		num_neighbours = num_neighbours.unsqueeze(2).repeat(1,1,2)
		# print ("NN ",num_neighbours)
		neighbours = torch.div(neighbours, num_neighbours)
		assert (not torch.isnan(neighbours).any())
		return neighbours