import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SeparationLoss(nn.Module):

	def __init__(self):
		super(SeparationLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def separationloss(self,verts,gts,A):
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

	def forward(self,verts,gts,A):
		num_verts = verts.size(0)
		r = 0
		centroid = torch.mean(verts,dim=0)
		for i in range(num_verts):
			r+=self.edge_length(verts[i,:],centroid)/num_verts
		gt_centroid = torch.mean(gts,dim=0)
		num_gt = gts.size(0)
		gtr = 0
		for i in range(num_gt):
			gtr+=self.edge_length(gts[i,:],gt_centroid)/num_gt
		return (r - gtr)**2 + self.separationloss(verts,gts,A)
	def edge_length(self,x,y):
		return torch.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)