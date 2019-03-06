import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from  src import dtype, dtypeL, dtypeB

class GCN(nn.Module):
	def __init__(self, feature_size, image_feature_size, dim_size, depth, weights_init='zero', residual_change=False):
		super(GCN, self).__init__()
		self.residual_change = residual_change
		self.feature_size = feature_size
		self.image_feature_size = image_feature_size
		self.dim_size = dim_size #Coordinate dimension
		self.layers = nn.ModuleList()
		self.depth = depth
		self.add_layer(nn.Linear(2*feature_size,feature_size))
		self.add_layer(nn.Linear(2*feature_size,feature_size))
		for i in range(depth):
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
		self.W_p_c = nn.Linear(self.dim_size,self.feature_size)
		self.W_p_s = nn.Linear(self.image_feature_size,self.feature_size)
		self.W_p = nn.Linear(2*self.feature_size,self.feature_size)
		self.W_c = nn.Linear(self.feature_size,self.dim_size)
		self.W_ic = nn.Linear(self.dim_size,self.feature_size)
		self.W_final = nn.Linear(self.feature_size,self.dim_size)
		self.a = nn.Tanh()
		if weights_init == 'zero':
			self.zero_init()
		else:
			self.xavier_init()
	
	# Initialize weights according to the Xavier Glorot formula
	def xavier_init(self):
		nn.init.xavier_uniform_(self.W_p_c.weight)
		nn.init.xavier_uniform_(self.W_p_s.weight)
		nn.init.xavier_uniform_(self.W_p.weight)
		nn.init.xavier_uniform_(self.W_c.weight)
		nn.init.xavier_uniform_(self.W_ic.weight)
		nn.init.xavier_uniform_(self.W_final.weight)

	def zero_init(self):
		nn.init.constant_(self.W_p_c.weight,0)
		nn.init.constant_(self.W_p_s.weight,0)
		nn.init.constant_(self.W_p.weight,0)
		nn.init.constant_(self.W_c.weight,0)
		nn.init.constant_(self.W_ic.weight,0)
		nn.init.constant_(self.W_final.weight,0)

	def add_layer(self,layer,init=True):
		self.layers.append(layer)
		if init:
			nn.init.xavier_uniform_(self.layers[-1].weight)

	def forward(self, x_prev, c_prev, A, image_feats):
		#x: batch_size x V x feature_size
		#c: batch_size x V x dim_size
		#W: feature_size x feature_size
		#A: batch_size x V x V
		#image_feats: tuple of 3 feature maps each of (batch_size x C x H x W)
		
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)

		c_f = self.a(self.W_p_c(c_prev))
		s_prev = self.extract_features(image_feats, c_prev).detach()
		s_f = self.a(self.W_p_s(s_prev))

		feature_from_state = self.a(self.W_p(torch.cat((c_f,s_f),dim=2)))

		x = torch.cat((feature_from_state,x_prev),dim=2)
		x = self.a(self.layers[0](x)+torch.bmm(temp_A,self.layers[1](x)))
		for i in range(2,len(self.layers),2):
			x = self.a(self.layers[i](x)+torch.bmm(temp_A,self.layers[i+1](x)) + x)
		
		c = self.a(self.W_final(x))
		if self.residual_change:
			c = c + c_prev
		return x, c

	def embed(self,c):
		return self.a(self.W_ic(c))


	def bilinearFeatures(self, c, featureMap):
		#c: batch_size x V x dim_size
		#featureMap: batch_size x channels x height x width
		x1y1 = torch.floor(c)
		x2y2 = torch.ceil(c)
		num_verts = c.size(1)
		batch_size, nchannels, maxHeight, maxWidth = featureMap.size()
		
		x1 = x1y1[:,:,0].type(dtypeL).unsqueeze(1).unsqueeze(2).expand(-1, nchannels, maxHeight, -1) #batch_size x channels x height x num_verts
		x2 = x2y2[:,:,0].type(dtypeL).unsqueeze(1).unsqueeze(2).expand(-1, nchannels, maxHeight, -1) #batch_size x channels x height x num_verts
		y1 = x1y1[:,:,1].type(dtypeL).unsqueeze(1).unsqueeze(3).expand(-1, nchannels, -1, num_verts) #batch_size x channels x num_verts x num_verts
		y2 = x2y2[:,:,1].type(dtypeL).unsqueeze(1).unsqueeze(3).expand(-1, nchannels, -1, num_verts) #batch_size x channels x num_verts x num_verts

		featureMapX1 = featureMap.gather(3,x1)#batch_size x channels x height x num_verts
		featureMapX2 = featureMap.gather(3,x2)

		featureMapX1Y1 = featureMapX1.gather(2,y1).diagonal(dim1=2,dim2=3).permute(0,2,1)
		featureMapX1Y2 = featureMapX1.gather(2,y2).diagonal(dim1=2,dim2=3).permute(0,2,1)

		featureMapX2Y1 = featureMapX2.gather(2,y1).diagonal(dim1=2,dim2=3).permute(0,2,1)
		featureMapX2Y2 = featureMapX2.gather(2,y2).diagonal(dim1=2,dim2=3).permute(0,2,1) #batch_size x num_verts x channels 

		
		#norm = torch.mul((x2y2[:,:,0] - x1y1[:,:,0]),(x2y2[:,:,1] - x1y1[:,:,1])).unsqueeze(-1).expand(-1,-1,nchannels) #not needed its already 1; also gives zero when both values are zero
		weightsX1Y1 = torch.mul((x2y2[:,:,0] - c[:,:,0]),(x2y2[:,:,1] - c[:,:,1])).unsqueeze(-1).expand(-1,-1,nchannels)
		weightsX1Y2 = torch.mul((x2y2[:,:,0] - c[:,:,0]),(c[:,:,1] - x1y1[:,:,1])).unsqueeze(-1).expand(-1,-1,nchannels)
		weightsX2Y1 = torch.mul((c[:,:,0] - x1y1[:,:,0]),(x2y2[:,:,1] - c[:,:,1])).unsqueeze(-1).expand(-1,-1,nchannels)
		weightsX2Y2 = torch.mul((c[:,:,0] - x1y1[:,:,0]),(c[:,:,1] - x1y1[:,:,1])).unsqueeze(-1).expand(-1,-1,nchannels)

		feat = torch.mul(featureMapX1Y1, weightsX1Y1)
		feat += torch.mul(featureMapX1Y2, weightsX1Y2)
		feat += torch.mul(featureMapX2Y1, weightsX2Y1)
		feat += torch.mul(featureMapX2Y2, weightsX2Y2)

		return feat.type(dtype)

	def extract_features(self, features, c):
		#c: batch_size x V x dim_size
		#output: batch_size x V x feature_size

		conv_3_3, conv_4_3, conv_5_3 = features
		scaled_c = (c.clone().detach() - (-1)) / 2.0 # scaling things back to 0 to 1 from -1 to 1	
		
		_, _, height, width  = conv_3_3.size()
		c_3_3 = scaled_c.clone()
		c_3_3[:,:,0] *= (width-1)
		c_3_3[:,:,1] *= (height-1)
		feats_3_3 = self.bilinearFeatures(c_3_3, conv_3_3)
		
		_, _, height, width  = conv_4_3.size()
		c_4_3 = scaled_c.clone()
		c_3_3[:,:,0] *= (width-1)
		c_3_3[:,:,1] *= (height-1)
		feats_4_3 = self.bilinearFeatures(c_4_3, conv_4_3)

		_, _, height, width  = conv_5_3.size()
		c_5_3 = scaled_c.clone()
		c_3_3[:,:,0] *= (width-1)
		c_3_3[:,:,1] *= (height-1)
		feats_5_3 = self.bilinearFeatures(c_5_3, conv_5_3)

		concat_feats = torch.cat([feats_3_3, feats_4_3, feats_5_3], 2)

		return concat_feats.detach()