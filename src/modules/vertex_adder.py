import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from  src import dtype, dtypeL, dtypeB

class VertexAdder(nn.Module):
	def __init__(self,toss):
		super(VertexAdder, self).__init__()
		self.toss = toss

	def forward(self, x_prev, c_prev, A):
		# dim A: batch_size x verices x verices
		batch_size = A.shape[0]
		feature_size = x_prev.size(2)
		dim_size = c_prev.size(2)
		num_vertices = A.shape[1] * np.ones(batch_size)
		Ar = np.reshape(A, (batch_size, -1))
		final_num_vertices = num_vertices +  np.count_nonzero(Ar, axis=1)/2

		num_vertices = int(num_vertices[0])
		final_num_vertices = int(final_num_vertices[0])
		A_new = np.zeros((batch_size, final_num_vertices, final_num_vertices))

		#v_index: batch_size
		v_index = np.ones(batch_size,dtype=np.int)*num_vertices#first new vertex added here
		v_index = np.expand_dims(v_index, 1)
		#x_prev: batch x num_vert x feat 
		x_new =  torch.cat((x_prev,torch.zeros(batch_size,final_num_vertices-num_vertices,feature_size).type(dtype)),dim=1)
		c_new =  torch.cat((c_prev,torch.zeros(batch_size,final_num_vertices-num_vertices,dim_size).type(dtype)),dim=1)
		for i in range(num_vertices):
			for j in range(i+1, num_vertices):				
				mask = np.expand_dims(A[:,i,j],1)
				#mask: batch_size
				toss = np.random.uniform()
				if toss < self.toss:
					continue:
				temp_A_new = np.zeros((batch_size, final_num_vertices, final_num_vertices))

				#add vertex between them
				np.put_along_axis(temp_A_new[:,i,:], v_index, mask, axis=1)
				np.put_along_axis(temp_A_new[:,:,i], v_index, mask, axis=1)
				np.put_along_axis(temp_A_new[:,:,j], v_index, mask, axis=1)
				np.put_along_axis(temp_A_new[:,j,:], v_index, mask, axis=1)

				A_new += temp_A_new

				#x_new : batch x final_num_vert x feat
				tmask = torch.LongTensor(mask).type(dtype)
				tv_index = torch.LongTensor(v_index).type(dtypeL)			
				x_v = ((x_prev[:,i,:] + x_prev[:,j,:])/2)*tmask#batch x feat
				c_v = ((c_prev[:,i,:] + c_prev[:,j,:])/2)*tmask
				x_v = x_v.unsqueeze(1)
				c_v = c_v.unsqueeze(1)
				tv_index = tv_index.unsqueeze(1)
				x_new.scatter_add_(1,tv_index.repeat(1, 1, feature_size), x_v)
				c_new.scatter_add_(1,tv_index.repeat(1, 1, dim_size), c_v)
				v_index += mask.astype(int)
				v_index = v_index % final_num_vertices

		return x_new, c_new, A_new

