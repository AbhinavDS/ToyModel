import time
import torch
from torch import optim
import numpy as np
import random
import math
from data import dataLoader
from src.model import Model
from  src import dtype, dtypeL, dtypeB
#from torchviz import make_dot, make_dot_from_trace

def train_model(params):
	# MAKE THE DATA loader
	max_vertices, feature_size, data_size = dataLoader.getMetaData(params)
	dim_size = params.dim_size
	train_data_loader = dataLoader.getDataLoader(params)
	num_blocks = int(math.ceil(np.log2(max_vertices))) - 1 - 2 #(since we start with 3 vertices already)
	params.num_blocks = num_blocks
	params.max_vertices = max_vertices
	params.data_size = data_size
	params.feature_size = feature_size
	params.start_block = 2
	print("Num Blocks: " + str(num_blocks))
	
	iter_count = 0
	model = Model(params)
	optimizer = optim.Adam(model.optimizer_params, lr=params.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = params.step_size, gamma=params.gamma)

	for epoch in range(params.num_epochs):
		scheduler.step()

		total_closs = 0
		total_laploss = 0
		total_nloss = 0
		total_eloss = 0
		total_loss = 0
		
		start_time = time.time()
		for i in range(int(math.ceil(data_size/params.batch_size))):
			optimizer.zero_grad()

			train_data, train_data_normal, seq_len, proj_gt = next(train_data_loader)

			# input 
			gt = torch.Tensor(utils.reshapeGT(params,train_data)).type(dtype)#vertices x dim_size
			gtnormals = torch.Tensor(utils.reshapeGT(params,train_data_normal)).type(dtype)#vertices x dim_size

			
			mask = utils.create_mask(gt, seq_len)
			mask = torch.Tensor(mask).type(dtypeB)

			s = torch.Tensor(train_data).type(dtype).unsqueeze(1).repeat(1,3,1)

			gt.requires_grad = False
			gtnormals.requires_grad = False

			x, c, A = model.create_start_data()
			x, c, s, A, proj_pred = model.forward1(x, c, s, A)

			total_closs += model.closs/len(train_data)
			total_laploss += model.laploss/len(train_data)
			total_nloss += model.nloss/len(train_data)
			total_eloss += model.eloss/len(train_data)
			total_loss += model.loss/len(train_data)
			
			if (iter_count % params.show_stat == 0):
				masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color='red',out='results/pred.png',A=A[0])
				print("Loss on epoch %i, iteration %i: LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, iter_count, optimizer.param_groups[0]['lr'], loss, closs, laploss, nloss, eloss))
				model.save(params.save_model_path)	
			model.loss.backward()
			optimizer.step()				
			iter_count += params.batch_size
		end_time = time.time()
		print ("Epoch Completed, Time taken: %f"%(end_time-start_time))
		print("Loss on epoch %i,  LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, optimizer.param_groups[0]['lr'], total_loss,total_closs,total_laploss,total_nloss,total_eloss))