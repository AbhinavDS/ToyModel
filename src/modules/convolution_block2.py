import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.modules.tgg import TGG
from src import dtype, dtypeL, dtypeB

class ConvolutionBlock(nn.Module):
	def __init__(self, params, normalize=False):
		super(ConvolutionBlock, self).__init__()
		self.params = params
		self.normalize = normalize
		self.mean = torch.tensor([0.485, 0.456, 0.406]).type(dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		self.std = torch.tensor([0.229, 0.224, 0.225]).type(dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		self.model = TGG().cuda()
		self.model.load_state_dict(torch.load('/home/abhinavds/Documents/Projects/Toy/ImageStory/ckpt_tgg/state_recon_model1_2.tgg'))
		self.model.eval()
	def forward(self, image):
		if self.normalize:
			image = (image - self.mean).div(self.std)
		# output = self.model(image)[1]
		# return (output.detach(), output.detach(), output.detach())
		conv3, conv4, conv5 = self.model(image)
		return (conv3.detach(), conv4.detach(), conv5.detach())

	

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
