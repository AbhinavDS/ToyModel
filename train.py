import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import dataLoader
import chamfer_loss, normal_loss, laplacian_loss, edge_loss, separation_loss
from torch.autograd import Variable
if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
	dtypeL = torch.cuda.LongTensor
	dtypeB = torch.cuda.ByteTensor
else:
	dtype = torch.FloatTensor
	dtypeL = torch.LongTensor
	dtypeB = torch.ByteTensor

torch.set_printoptions(threshold=23679250035)
np.set_printoptions(threshold=13756967)
class Deformer(nn.Module):
	def __init__(self, feature_size, dim_size,depth):
		super(Deformer, self).__init__()
		self.feature_size = feature_size
		self.dim_size = dim_size#Coordinate dimension
		self.layers = nn.ModuleList()
		self.depth = depth
		self.add_layer(nn.Linear(2*feature_size,feature_size))
		self.add_layer(nn.Linear(2*feature_size,feature_size))
		for i in range(depth):
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
		self.W_p_c = nn.Linear(self.dim_size,self.feature_size)
		self.W_p_s = nn.Linear(self.feature_size,self.feature_size)
		self.W_p = nn.Linear(2*self.feature_size,self.feature_size)
		self.W_c = nn.Linear(self.feature_size,self.dim_size)
		self.W_ic = nn.Linear(self.dim_size,self.feature_size)
		self.W_lol = nn.Linear(self.feature_size,self.dim_size)
		self.a = nn.Tanh()
		# Initialize weights according to the Xavier Glorot formula
		nn.init.xavier_uniform_(self.W_p_c.weight)
		nn.init.xavier_uniform_(self.W_p_s.weight)
		nn.init.xavier_uniform_(self.W_p.weight)
		nn.init.xavier_uniform_(self.W_c.weight)
		nn.init.xavier_uniform_(self.W_ic.weight)
		nn.init.xavier_uniform_(self.W_lol.weight)

	def add_layer(self,layer,init=True):
		self.layers.append(layer)
		if init:
			nn.init.xavier_uniform_(self.layers[-1].weight)

	def forward(self, x_prev, s_prev,c_prev, A):
		#x: batch_size x V x feature_size
		#s: batch_size x V x feature_size
		#c: batch_size x V x dim_size
		#W: feature_size x feature_size
		#A: batch_szie x V x V
		
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		# #s = self.a(self.W_s(s_prev) + self.W_x(x_prev))
		
		c_f = self.a(self.W_p_c(c_prev))
		s_f = self.a(self.W_p_s(s_prev))
		feature_from_state = self.a(self.W_p(torch.cat((c_f,s_f),dim=2)))
		# #concat_feature = torch.cat((c_prev,s_prev),dim=1)
		# #feature_from_state = self.a(self.W_p_2(self.a(self.W_p_1(concat_feature))))
		x = torch.cat((feature_from_state,x_prev),dim=2)
		x = self.a(self.layers[0](x)+torch.bmm(temp_A,self.layers[1](x)))
		for i in range(2,len(self.layers),2):
			x = self.a(self.layers[i](x)+torch.bmm(temp_A,self.layers[i+1](x)) + x)
		#c = self.a(self.W_c(x)+c_prev)
		# c = self.a(self.W_p_s(s_prev[0,:]))#+c_prev)
		c = self.a(self.W_lol(x))
		s = s_prev
		#print(x)
		#c = c.view((-1,2))
		return x, s, c

	def forwardCX(self,c):
		return self.a(self.W_ic(c))

class vertexAdd(nn.Module):
	def __init__(self):
		super(vertexAdd, self).__init__()
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

def create_mask(gt, seq_len):
	# seq_len: batch_size x 1
	mask = np.arange(gt.size(1))
	mask = np.expand_dims(mask, 0)
	mask = np.tile(mask,(batch_size, 1))
	seq_len_check = np.reshape(seq_len, (-1, 1))
	seq_len_check = np.tile(seq_len_check,(1, gt.size(1)))
	condition = (mask < seq_len_check)
	return condition.astype(np.uint8)

if __name__=="__main__":
	# MAKE THE DATA
	train_data, train_data_normal, seq_len, feature_size, dim_size = dataLoader.getData()
	print ("Loaded")
	batch_size = 50
	num_epochs = 2000
	lr = 1e-5
	num_blocks = 0
	depth = 10#increasing depth needs reduction in lr

	# RUN TRAINING AND TEST
	if torch.cuda.is_available():
		deformer = Deformer(feature_size,dim_size,depth).cuda()
	else:
		deformer = Deformer(feature_size,dim_size,depth)

	# deformer.load_state_dict(torch.load('model_10000.toy'))
	
	adder = vertexAdd().cuda()
	criterionC = chamfer_loss.ChamferLoss()
	criterionN = normal_loss.NormalLoss()
	criterionL = laplacian_loss.LaplacianLoss()
	criterionE = edge_loss.EdgeLoss()
	criterionS = separation_loss.SeparationLoss()
	optimizer = optim.Adam(deformer.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
	#optimizer = optim.Adagrad(deformer.parameters(), lr=lr,lr_decay=5e-3)
	c,_,_ = dataLoader.inputMesh(feature_size)


	k = 0
	show_stat = 500
	for epoch in range(0, num_epochs):
		scheduler.step()
		ex_indices = [i for i in range(0, len(train_data))]
		
		#random.shuffle(ex_indices)
		total_loss = 0.0
		total_closs = 0
		total_laploss = 0
		total_nloss = 0
		total_eloss = 0
		total_sloss = 0
		for i in range(0, len(ex_indices), batch_size):
			idx = ex_indices[i: i+batch_size]
			optimizer.zero_grad()

			# input 
			s = torch.Tensor(train_data[idx]).type(dtype).unsqueeze(1).repeat(1,3,1)
			c,x,A = dataLoader.inputMesh(feature_size)
			c = np.expand_dims(c, 0)
			c = np.tile(c,(batch_size, 1, 1))
			x = np.expand_dims(x, 0)
			x = np.tile(x,(batch_size, 1, 1))
			A = np.expand_dims(A, 0)
			A = np.tile(A,(batch_size, 1, 1))
			x = torch.Tensor(x).type(dtype)
			c = torch.Tensor(c).type(dtype)
			gt = torch.Tensor(dataLoader.generateGT(train_data[idx])).type(dtype)#vertices x dim_size
			gtnormals = torch.Tensor(dataLoader.generateGT(train_data_normal[idx])).type(dtype)#vertices x dim_size

			mask = create_mask(gt, seq_len[idx])
			mask = torch.Tensor(mask).type(dtypeB)
			
			gt.requires_grad = False
			loss = 0.0
			closs = 0.0
			sloss = 0.0
			
			# Vertex addition
			num_ias = int(np.log2(feature_size/30)) 
			for ias in range(num_ias):
				x, c, A = adder.forward(x,c,A)
				s = torch.cat((s,s),dim=1)

			x = deformer.forwardCX(c)
			x, s, c1 = deformer.forward(x,s,c,A)

			laploss = criterionL(c, c1, A)
			c = c1
			closs = criterionC(c, gt, mask)
			eloss = criterionE(c, A)
			nloss = criterionN(c, gt, gtnormals, A, mask)
			

			for block in range(num_blocks):
				#x, c, A = adder.forward(x,c,A)
				#s = torch.cat((s,s),dim=0)
				x, s, c = deformer.forward(x,s,c,A)
				
				closs += criterionC(c,gt)
				if(epoch > 10000):
					sloss += criterionS(c,gt,A)
			
			loss = closs + 0.0001*nloss + 0.6*(laploss + 0.33*eloss) #+ sloss
			total_closs +=closs/len(train_data)
			total_laploss +=laploss/len(train_data)
			total_nloss +=nloss/len(train_data)
			total_eloss +=eloss/len(train_data)
			total_sloss +=sloss/len(train_data)
			total_loss += loss/len(train_data)
				
			if (k % show_stat == 0):
				masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
				dataLoader.drawPolygons(dataLoader.getPixels(c[0]),dataLoader.getPixels(masked_gt),color='red',out='pred.png',A=A[0])
				print("Loss on epoch %i, iteration %i: LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f,S:%f" % (epoch, k, optimizer.param_groups[0]['lr'], total_loss,total_closs,total_laploss,total_nloss,total_eloss,total_sloss))
				torch.save(deformer.state_dict(),'model_10000.toy')
			else:
				loss.backward()#retain_graph=True)
				optimizer.step()
				
			k += batch_size

	#Normal loss
	#Blocks
	#Vertex adder in block

# Add batch sizes
# Maybe check for examples from same number of vertices. 