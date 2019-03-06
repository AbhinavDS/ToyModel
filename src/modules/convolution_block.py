import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
from torch.autograd import Variable

class ConvolutionBlock(nn.Module):
	def __init__(self, params, normalize=False):
		super(ConvolutionBlock, self).__init__()
		self.params = params
		self.normalize = normalize
		self.mean = torch.tensor([0.485, 0.456, 0.406]).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		self.std = torch.tensor([0.229, 0.224, 0.225]).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		self.model = torch_models.vgg16_bn(pretrained=True)
		self.model.eval()
		out_list = [22, 32, 42]
		self.layer_3_3 =  nn.Sequential(*list(self.model.features.children())[:out_list[0]])
		self.layer_4_3 =  nn.Sequential(*list(self.model.features.children())[:out_list[1]])
		self.layer_5_3 =  nn.Sequential(*list(self.model.features.children())[:out_list[2]])
		self.conv_3_3 = None
		self.conv_4_3 = None
		self.conv_5_3 = None
	
	def concatFeatureMaps(self, image):
		if self.normalize:
			image = (image - self.mean).div(self.std)
		self.conv_3_3 = self.layer_3_3(image)
		self.conv_4_3 = self.layer_4_3(image)
		self.conv_5_3 = self.layer_5_3(image)

	def bilinearFeatures(self, c, featureMap):
		#c: batch_size x V x dim_size
		#featureMap: batch_size x channels x height x width
		
		x1y1 = torch.floor(c)
		x2y2 = torch.ceil(c)
		num_verts = c.size(1)
		batch_size, nchannels, maxHeight, maxWidth = featureMap.size()

		x1 = x1y1[:,:,0].long().squeeze().unsqueeze(1).unsqueeze(2).expand(-1, nchannels, maxHeight, -1) #batch_size x channels x height x num_verts
		x2 = x2y2[:,:,0].long().squeeze().unsqueeze(1).unsqueeze(2).expand(-1, nchannels, maxHeight, -1) #batch_size x channels x height x num_verts
		y1 = x1y1[:,:,1].long().squeeze().unsqueeze(1).unsqueeze(3).expand(-1, nchannels, -1, num_verts) #batch_size x channels x num_verts x num_verts
		y2 = x2y2[:,:,1].long().squeeze().unsqueeze(1).unsqueeze(3).expand(-1, nchannels, -1, num_verts) #batch_size x channels x num_verts x num_verts

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

		return feat

	def forward(self, c):
		#c: batch_size x V x dim_size
		#output: batch_size x V x feature_size
		assert (self.conv_3_3 is not None)
		assert (self.conv_4_3 is not None)
		assert (self.conv_5_3 is not None)
		out_3_3 = None
		out_4_3 = None
		out_5_3 = None
		c_3_3 = c.clone().detach()/8
		c_4_3 = c_3_3 / 2
		c_5_3 = c_4_3 / 2
		feats_3_3 = self.bilinearFeatures(c_3_3, self.conv_3_3)
		feats_4_3 = self.bilinearFeatures(c_4_3, self.conv_4_3)
		feats_5_3 = self.bilinearFeatures(c_5_3, self.conv_5_3)

		concat_feats = torch.cat([feats_3_3, feats_4_3, feats_5_3], 2)
		return concat_feats


params = '1'
a = ConvolutionBlock(params,normalize=True)
image= torch.randn(4,3, 224,224)
c = [[[1,2],[1,10],[0,0]],[[2,4],[5,123],[223,223]],[[12,54],[65,23],[23,54]],[[10,10],[100,100],[34,53]]]
c = torch.tensor(c).float()
print (image.size(), c.size())
a.concatFeatureMaps(image)
print (a(c))

