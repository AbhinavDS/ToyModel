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
		self.conv_3_3 = None
		self.conv_4_3 = None
		self.conv_5_3 = None
	
	def forward(self, image):
		if self.normalize:
			image = (image - self.mean).div(self.std)
		self.conv_3_3 = self.layer_3_3(image)
		self.conv_4_3 = self.layer_4_3(image)
		self.conv_5_3 = self.layer_5_3(image)
		return (self.conv_3_3.detach(), self.conv_4_3.detach(), self.conv_5_3.detach())

	

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
