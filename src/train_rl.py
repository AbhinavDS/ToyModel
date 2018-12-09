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
	params.num_blocks1 = num_blocks
	params.num_blocks2 = 2
	params.max_vertices = max_vertices
	params.data_size = data_size
	params.feature_size = feature_size
	params.start_block = 2
	print("Num Blocks: " + str(num_blocks))
	
	iter_count = 0
	model = Model(params)
	optimizer1 = optim.Adam(model.optimizer_params1, lr=params.lr)
	scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size = params.step_size, gamma=params.gamma)

	optimizer2 = optim.Adam(model.optimizer_params2, lr=params.lr)
	scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size = params.step_size, gamma=params.gamma)

	re_init_train2 = False
	for epoch in range(params.num_epochs):
		# Train this part
		condition_train1 = False#epoch < 1000 #False
		condition_train2 = True#True#epoch > 3000 #True#epoch > 210
		test_rl = False#True

		if condition_train1:
			optimizer = optimizer1
			scheduler = scheduler1
		else:
			model.freeze1(False)

		if condition_train2:
			optimizer = optimizer2
			scheduler = scheduler2
			if not model.unfreezed:
				model.freeze2(True)
			if re_init_train2:
				re_init_train2 = False
				model.reinit2()
		else:
			model.freeze2(False)
		
		total_closs = 0
		total_laploss = 0
		total_nloss = 0
		total_eloss = 0
		total_loss = 0
		
		start_time = time.time()
		for i in range(int(math.ceil(data_size/params.batch_size))):
			if condition_train1 or condition_train2:
				optimizer.zero_grad()

			train_data, train_data_normal, seq_len, proj_gt = next(train_data_loader)

			# input 
			gt = torch.Tensor(utils.reshapeGT(params,train_data)).type(dtype)#vertices x dim_size
			gtnormals = torch.Tensor(utils.reshapeGT(params,train_data_normal)).type(dtype)#vertices x dim_size

			
			mask = utils.create_mask(gt, seq_len)
			mask = torch.Tensor(mask).type(dtypeB)
			loss_mask = utils.create_loss_mask(gt) # also masks extra padded -2 between polygons for loss calculation
			loss_mask = torch.Tensor(loss_mask).type(dtypeB)

			s = torch.Tensor(train_data).type(dtype).unsqueeze(1).repeat(1,3,1)

			gt.requires_grad = False
			gtnormals.requires_grad = False
			mask.requires_grad = False
			loss_mask.requires_grad = False

			x, c, A = model.create_start_data()
			x, c, s, A, proj_pred = model.forward1(x, c, s, A, gt, gtnormals, loss_mask)

			total_closs += model.closs/len(train_data)
			total_laploss += model.laploss/len(train_data)
			total_nloss += model.nloss/len(train_data)
			total_eloss += model.eloss/len(train_data)
			total_loss += model.loss/len(train_data)

			masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
			Pid = np.copy(A)
			epoch_set = int(num_epochs/(2*params.num_polygons+1))# 1000
			epoch_id = epoch/epoch_set
			if not condition_train1:#[1,2,3,4,5]
				train_till_block = int((epoch_id+1)/2) # [1,1,2,2,3] when train_till_block trains all split_i <= that also train and train_till_block trains only in even epoch sets
				for block_i in range(train_till_block):
					is_last_block = (block_i == int((int(num_epochs/epoch_set) + 1)/2))#3
					action, reward, intersections, pred_genus, gt_genus = model.split(c, x, gt, Pid, mask, proj_pred, proj_gt, epoch * params.data_size + i, test_rl or  epoch_id%2 == 0, to_split = not is_last_block)
				
					print (action[0],reward[0],pred_genus[0], gt_genus[0], intersections[0],"Image")
					A, Pid = model.splitter.forward(Pid,intersections)

					if is_last_block:
						break
				
					# Split and send forward
					x, c, s, A, proj_pred = model.forward2(x.detach(), c.detach(), s.detach(), A, gt, gtnormals, loss_mask)
					total_closs += model.closs/len(train_data)
					total_laploss += model.laploss/len(train_data)
					total_nloss += model.nloss/len(train_data)
					total_eloss += model.eloss/len(train_data)
					total_loss += model.loss/len(train_data)
					if (epoch_id%2 == 0 and block_i <= train_till_block):
						model.loss.backward(retain_graph = True)
						model.clear_losses()
						
				if epoch_id%2 == 0 and train_till_block < int((int(num_epochs/epoch_set) + 1)/2):#3
					optimizer.step()
			
				
				color = 'red' if (reward[0]==1) else ('yellow' if reward[0] else 'blue')
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color=color,out='results/pred_rl%s.png'%params.sf,A=A[0], line=(action[0][0],action[0][1],action[0][2],action[0][3]))
				
			if condition_train1:# or condition_train2:
				model.loss.backward()
				optimizer.step()				
			

			if (iter_count % params.show_stat == 0) and (condition_train1):
				masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color='red',out='results/pred_rl%s.png'%params.sf,A=A[0])
				print("Loss on epoch %i, iteration %i: LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, iter_count, optimizer.param_groups[0]['lr'], model.loss, model.closs, model.laploss, model.nloss, model.eloss))
			model.save(params.save_model_path)
			iter_count += params.batch_size
		end_time = time.time()
		if condition_train1 or condition_train2:
			print ("Epoch Completed, Time taken: %f"%(end_time-start_time))
			print("Loss on epoch %i; Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, total_loss,total_closs,total_laploss,total_nloss,total_eloss))
			scheduler.step()
		