import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class LaplacianLoss(nn.Module):

	def __init__(self):
		super(LaplacianLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self, pred1, pred2, A):
		temp_A = Variable(torch.Tensor(A).type(torch.FloatTensor),requires_grad=False)
		
		lap_pred1 = pred1 - self.centroid(pred1, temp_A)
		lap_pred2 = pred2 - self.centroid(pred2, temp_A)

		loss = lap_pred2 - lap_pred1
		loss = torch.mul(loss,loss)
		loss = torch.sum(loss)
		return loss

	def centroid(self, x, A):
		# print ("A", x,)
		num_points_x, points_dim = x.size()
		num_neighbours = torch.sum(A, dim=0)
		
		neighbours = x.permute(1,0)
		neighbours = neighbours.repeat(1,num_points_x).view(points_dim, num_points_x, num_points_x)

		# Filter out non-neighbours		
		A = A.unsqueeze(0)
		A = A.repeat(points_dim,1,1)
		neighbours = torch.mul(neighbours, A)
		neighbours = torch.sum(neighbours, dim = 2)
		neighbours = neighbours / num_neighbours
		neighbours = neighbours.permute(1,0)
		return neighbours