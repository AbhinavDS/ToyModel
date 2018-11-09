import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class NormalLoss(nn.Module):

	def __init__(self):
		super(NormalLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()        

	def forward(self,preds,gts):
		P = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(P, 1)
		loss_1 = torch.sum(mins)
		mins, _ = torch.min(P, 2)
		loss_2 = torch.sum(mins)

		return loss_1 + loss_2


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

	def normalise(self, vector):
		normal2 = v
		return [normal2[0]/math.sqrt(normal2[1]**2+normal2[0]**2), normal2[1]/math.sqrt(normal2[1]**2+normal2[0]**2)]

	def get_normals(self, points):
		# for each polygon in the batch
		# get normal for each vertex
		# reshape with dimension 2
		# subtract with vertices of previous row
		# multiply with -1 and swap columns 
		# normalize them
		# 
		return


def writeNormals(file,polygons):

	allnormals = []
	for p in range(len(polygons)):
		polygon  = polygons[p]
		polygonnormals = []
		for v in range(len(polygon)):
			vertp = polygon[v-1]
			vert = polygon[v]
			vertn = polygon[(v+1)%len(polygon)]
			# vertp = [317,322]
			# vert = [327,310]
			# vertn = [329,302]
			normal1 = [vert[1]-vertp[1],vertp[0]-vert[0]]
			normal1 = normalise(normal1)

			normal2 = [vertn[1]-vert[1],vert[0]-vertn[0]]
			normal2 = normalise(normal2)
			normal = [0,0]
			normal[0] = normal1[0] + normal2[0]
			normal[1] = normal1[1] + normal2[1]
			normal = normalise(normal)
			# print(vertp,vert,vertn,normal1,normal2,normal)
			# w = input("eui")
			if(v == 0):
				file.write(str(normal[0])+','+str(normal[1]))
			else:
				file.write(','+str(normal[0])+','+str(normal[1]))
			polygonnormals.append(normal)
		allnormals.append(polygonnormals)
		if(p!=len(polygons)-1):
			file.write(PAD_TOKEN)			
	file.write('\n')
	return allnormals
