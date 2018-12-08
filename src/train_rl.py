import time
import torch
from torch import optim
import numpy as np
import random
import math
from src.util import utils
from src.data import dataLoader
from src.model import Model
from  src import dtype, dtypeL, dtypeB
#from torchviz import make_dot, make_dot_from_trace

import src.modules.rl.utils as utils_rl

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
		condition = False #epoch < 1000
		
		if not condition and not model.freeze_state:
			model.freeze(False)

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
			x, c, s, A, proj_pred = model.forward1(x, c, s, A, gt, gtnormals, mask)

			total_closs += model.closs/len(train_data)
			total_laploss += model.laploss/len(train_data)
			total_nloss += model.nloss/len(train_data)
			total_eloss += model.eloss/len(train_data)
			total_loss += model.loss/len(train_data)
				
			
			masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
			if (iter_count % params.show_stat == 0) and condition:
				masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color='red',out='results/pred_rl%s.png'%params.sf,A=A[0])
				print("Loss on epoch %i, iteration %i: LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, iter_count, optimizer.param_groups[0]['lr'], model.loss, model.closs, model.laploss, model.nloss, model.eloss))
				# model.save(params.save_model_path)

			if condition:
				model.loss.backward()
				optimizer.step()				
				iter_count += params.batch_size
			else:
				action, reward = model.split2(c, x, gt, A, mask, proj_pred, proj_gt, epoch * params.data_size + i)
				print (action[0],reward[0],"Image")
				color = 'red' if (reward[0]==20) else ('yellow' if reward[0] else 'blue')
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color=color,out='results/pred_rl%s.png'%params.sf,A=A[0], line=(action[0][0],action[0][1],action[0][2],action[0][3]))
				iter_count += params.batch_size
		end_time = time.time()
		if condition:
			print ("Epoch Completed, Time taken: %f"%(end_time-start_time))
			print("Loss on epoch %i; Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, total_loss,total_closs,total_laploss,total_nloss,total_eloss))

