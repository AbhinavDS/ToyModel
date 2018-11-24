import torch
if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
	dtypeL = torch.cuda.LongTensor
	dtypeB = torch.cuda.ByteTensor
else:
	dtype = torch.FloatTensor
	dtypeL = torch.LongTensor
	dtypeB = torch.ByteTensor
