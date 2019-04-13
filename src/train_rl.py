import os
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

import src.modules.rl.utils as utils_rl

def train_model(params):
	max_vertices, feature_size, data_size = dataLoader.getMetaData(params)
	dim_size = params.dim_size
	train_data_loader = dataLoader.getDataLoader(params)
	num_gcns = int(math.ceil(np.log2(max_vertices))) - 2 # - 2 #(since we start with 3 vertices already)
	
	params.save_model_dirpath = os.path.join(params.save_model_dirpath, params.sf)
	load_models = (os.path.realpath(params.load_model_dirpath) == os.path.realpath(params.save_model_dirpath))
	if not os.path.isdir(params.save_model_dirpath):
		os.makedirs(params.save_model_dirpath)
		os.makedirs(os.path.join(params.save_model_dirpath,'rl'))
		load_models = False
	elif not load_models:
		print (params.save_model_dirpath+" already exists. Overwrite not possible.")
		return
	params.num_gcns1 = num_gcns
	params.num_gcns2 = 1
	params.num_rl = 3
	params.max_vertices = max_vertices
	params.data_size = data_size
	params.feature_size = 128
	params.depth = 0
	params.image_feature_size = 768*25 #1280 #filters of conv_3_3 + conv_4_3 + conv_5_3
	params.initial_adders = 2
	print("Num GCNs: " + str(num_gcns))
	
	iter_count = 0
	model = Model(params, load_models=load_models)
	params.start_epoch = model.last_epoch
	optimizer1 = optim.Adam(model.optimizer_params1, lr=params.lr)
	scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size = params.step_size, gamma=params.gamma)

	optimizer2 = optim.Adam(model.optimizer_params2, lr=params.lr*10)
	scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size = params.step_size*10, gamma=params.gamma*1.1)

	num_iters = int(math.ceil(data_size/params.batch_size))
	num_blocks = (2*params.num_polygons)
	# iters_per_block = int(num_iters/num_blocks)
	iters_per_block = params.iters_per_block
	total_iters = 0
	for epoch in range(params.start_epoch, params.num_epochs):
		start_time = time.time()
		total_closs = 0
		total_laploss = 0
		total_nloss = 0
		total_eloss = 0
		total_loss = 0
		for iters in range(num_iters):
			# Train this part
			test_rl = False#True
			total_iters += 1
			block_id = int(total_iters/iters_per_block)%num_blocks
			# block_id = int(iters/iters_per_block)
			### For now TODO: REMOVE LATER
			if block_id == num_blocks-1:
				continue
			print ("##############")
			print ("BLOCK_ID:: ", block_id, " EPOCH_NO:: ", epoch)
			print ("##############")
			################
			if block_id == 0:
				optimizer = optimizer1
			else:
				optimizer = optimizer2
			optimizer.zero_grad()
			#################

			# Iterator
			train_data, train_data_normal, seq_len, proj_gt, train_data_images = next(train_data_loader)

			# Input 
			gt = torch.Tensor(utils.reshapeGT(params,train_data)).type(dtype) #vertices x dim_size
			gtnormals = torch.Tensor(utils.reshapeGT(params,train_data_normal)).type(dtype) #vertices x dim_size
			gt_images = torch.Tensor(train_data_images).type(dtype)
			image_feats = model.convolution_block.forward(gt_images)

			# Masks for batch size
			mask = utils.create_mask(gt, seq_len)
			mask = torch.Tensor(mask).type(dtypeB)
			loss_mask = utils.create_loss_mask(gt) # also masks extra padded -2 between polygons for loss calculation
			loss_mask = torch.Tensor(loss_mask).type(dtypeB)

			gt.requires_grad = False
			gtnormals.requires_grad = False
			mask.requires_grad = False
			loss_mask.requires_grad = False

			x, c, A = model.create_start_data()
			Pid = np.copy(A)

			# Start Processing
			x, c, A, Pid, proj_pred = model.deformer_block1.forward(x, c, A, Pid, gt, gtnormals, loss_mask, image_feats)

			norm = len(train_data) * (int((block_id)/2)+1)
			total_closs += model.deformer_block1.closs.item()/norm
			total_laploss += model.deformer_block1.laploss.item()/norm
			total_nloss += model.deformer_block1.nloss.item()/norm
			total_eloss += model.deformer_block1.eloss.item()/norm
			total_loss += model.deformer_block1.loss.item()/norm

			masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
			
			if block_id == 0:
				color = 'red'# if reward[0] else 'blue'
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color=color,out='results/pred_rl_%s.png'%params.sf,A=A[0])#, line=(action[0][0],action[0][1],action[0][2],action[0][3]))
				model.deformer_block1.loss.backward()
				# model.deformer_block1.set_loss_to_zero()
				optimizer.step()
			else: #[1,2,3,4,5] no 0 [S,SD,SDS,SDSD,SDSDS]
				action, reward = None, None
				# block iter iterates over the sequence SDSDS or whatever given by block id.
				# trains last splitter block (when block_id is S,SDS,SDSDS)denoted by the odd block_id or all the deformer blocks if even block id (when SD,SDSD)
				for block_iter in range(block_id):
					is_last_block = (block_iter==num_blocks-1)
					if block_iter%2 == 0:
						action, reward, intersections, pred_genus, gt_genus = model.split(c, x, gt, Pid, mask, proj_pred, proj_gt, test_rl or  block_id%2 == 0 or not(block_iter==block_id-1), int(block_iter/2), to_split = not is_last_block)		# ep = epoch * params.data_size + iters
						print (action[0],reward[0],pred_genus[0], gt_genus[0], intersections[0],"Image")
						A, Pid = model.splitter_block.forward(Pid,intersections)
					else:
						x, c, A, Pid,  proj_pred = model.deformer_block2.forward(x.detach(), c.detach(), A, Pid, gt, gtnormals, loss_mask, image_feats)
						total_closs += model.deformer_block2.closs.item()/norm
						total_laploss += model.deformer_block2.laploss.item()/norm
						total_nloss += model.deformer_block2.nloss.item()/norm
						total_eloss += model.deformer_block2.eloss.item()/norm
						total_loss += model.deformer_block2.loss.item()/norm
						if (block_id%2 == 0):
							model.deformer_block2.loss.backward(retain_graph = True)
							# model.deformer_block2.set_loss_to_zero()
						else:
							# model.deformer_block2.set_loss_to_zero()
							pass

					if block_iter == block_id-1:
						color = 'red' if (block_id%2==0 or is_last_block) else 'blue'
						utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color=color,out='results/pred_rl_%s.png'%params.sf,A=A[0], line=(action[0][0],action[0][1],action[0][2],action[0][3]))

				if block_id%2 == 0:
					optimizer.step()
			
			print (block_id, (iter_count % params.show_stat == 0), (block_id==0), block_id%2 == 0)
			if (iter_count % params.show_stat == 0) and (block_id==0) and block_id%2 == 0 :
				masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color='red',out='results/pred_rl_%s.png'%params.sf,A=A[0])
				if block_id == 0:
					b1 = model.deformer_block1
					print("Loss on epoch %i, iteration %i: LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, iter_count, optimizer.param_groups[0]['lr'], b1.loss, b1.closs, b1.laploss, b1.nloss, b1.eloss))
				else:
					b2 = model.deformer_block2
					print("Loss on epoch %i, iteration %i: LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, iter_count, optimizer.param_groups[0]['lr'], b2.loss, b2.closs, b2.laploss, b2.nloss, b2.eloss))
			
			iter_count += params.batch_size
		end_time = time.time()

		
		model.save(os.path.join(params.save_model_dirpath, "model.toy"), epoch)
		print ("Epoch Completed, Time taken: %f"%(end_time-start_time))
		print("Loss on epoch %i; LR %f; Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, optimizer.param_groups[0]['lr'], total_loss,total_closs,total_laploss,total_nloss,total_eloss))
		scheduler1.step()
		scheduler2.step()
		