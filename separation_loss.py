import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SeparationLoss(nn.Module):

	def __init__(self):
		super(SeparationLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self,verts,gts,A):
		num_verts = verts.size(0)
		peri = 0
		for i in range(num_verts):
			for j in range(i+1,num_verts):
				if(A[i,j]):
					peri+=self.edge_length(verts[i,:],verts[j,:])
		gtperi = self.edge_length(gts[0,:],gts[-1,:])					
		num_gt = gts.size(0)
		for i in range(num_gt-1):
			gtperi+=self.edge_length(gts[i,:],gts[i+1,:])
		return (peri - gtperi)**2
	def edge_length(self,x,y):
		return torch.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)