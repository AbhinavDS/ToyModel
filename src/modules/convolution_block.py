import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
from torch.autograd import Variable

from src import dtype, dtypeL, dtypeB

class ConvolutionBlock(nn.Module):
	def __init__(self, params, normalize=False):
		super(ConvolutionBlock, self).__init__()
		self.params = params
		self.normalize = normalize
		self.mean = torch.tensor([0.485, 0.456, 0.406]).type(dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		self.std = torch.tensor([0.229, 0.224, 0.225]).type(dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		self.model = torch_models.vgg16_bn(pretrained=True).type(dtype)
		self.model.eval()
		out_list = [22, 32, 42]
		self.layer_3_3 =  nn.Sequential(*list(self.model.features.children())[:out_list[0]])
		self.layer_4_3 =  nn.Sequential(*list(self.model.features.children())[:out_list[1]])
		self.layer_5_3 =  nn.Sequential(*list(self.model.features.children())[:out_list[2]])
		# self.conv_3_3 = None
		# self.conv_4_3 = None
		# self.conv_5_3 = None
	
	def forward(self, image):
		if self.normalize:
			image = (image - self.mean).div(self.std)
		conv_3_3 = self.channel_normalize(self.layer_3_3(image))
		conv_4_3 = self.channel_normalize(self.layer_4_3(image))
		conv_5_3 = self.channel_normalize(self.layer_5_3(image))
		return (conv_3_3.detach(), conv_4_3.detach(), conv_5_3.detach())

	def channel_normalize(self,x):
		x = x.permute(1,0,2,3)
		orig_shape = x.size()
		x = x.reshape((x.size(0), -1))
		max_x = x.max(1, keepdim=True)[0]
		max_x[max_x == 0] = 1
		x_normed = x / max_x
		x_normed = x_normed.reshape(orig_shape)
		x_normed = x_normed.permute(1,0,2,3)
		return x_normed

	

if __name__=="__main__":
	params = '1'
	a = ConvolutionBlock(params,normalize=True)
	image= torch.randn(4,3,224,224)
	c = [[[1,2],[1,10],[0,0]],[[2,4],[5,123],[223,223]],[[12,54],[65,23],[23,54]],[[10,10],[100,100],[34,53]]] # check if our c is scaled
	c = torch.tensor(c).type(dtype)
	print (image.size(), c.size())
	feats = a.forward(image)
	c1,c2,c3 = feats
	print (c1.size(), c2.size(), c3.size())
