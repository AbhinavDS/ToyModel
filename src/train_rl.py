import time
import torch
from torch import optim
import numpy as np
import random
import math
from data import dataLoader
from src.util import utils
from src.loss.chamfer_loss import ChamferLoss
from src.loss.normal_loss import NormalLoss
from src.loss.laplacian_loss import LaplacianLoss
from src.loss.edge_loss import EdgeLoss
from src.modules.deformer import Deformer
from src.modules.vertex_adder import VertexAdder

from  src import dtype, dtypeL, dtypeB
def train_model(params):
	# MAKE THE DATA loader
	max_vertices, feature_size, data_size = dataLoader.getMetaData(params)
	dim_size = params.dim_size
	train_data_loader = dataLoader.getDataLoader(params)
	num_blocks = int(math.ceil(np.log2(max_vertices))) - 1 #(since we start with 3 vertices already)
	print("Num Blocks: " + str(num_blocks))
	
	iter_count = 0
	# RUN TRAINING AND TEST
	if torch.cuda.is_available():
		deformer = Deformer(feature_size,dim_size,params.depth).cuda()
	else:
		deformer = Deformer(feature_size,dim_size,params.depth)

	if params.load_model_path:
		deformer.load_state_dict(torch.load(params.load_model_path))
	
	adder = VertexAdder().cuda()
	criterionC = ChamferLoss()
	criterionN = NormalLoss()
	criterionL = LaplacianLoss()
	criterionE = EdgeLoss()
	optimizer = optim.Adam(deformer.parameters(), lr=params.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = params.step_size, gamma=params.gamma)
	c,_,_ = utils.inputMesh(feature_size)


	for epoch in range(params.num_epochs):
		scheduler.step()
		total_loss = 0.0
		total_closs = 0
		total_laploss = 0
		total_nloss = 0
		total_eloss = 0
		total_sloss = 0
		start_time = time.time()
		for i in range(int(math.ceil(data_size/params.batch_size))):
			optimizer.zero_grad()
			train_data, train_data_normal, seq_len, proj_gt = next(train_data_loader)
			# input 
			s = torch.Tensor(train_data).type(dtype).unsqueeze(1).repeat(1,3,1)
			c,x,A = utils.inputMesh(feature_size)# x is c with zeros appended, x=f ..pixel2mesh
			c = np.expand_dims(c, 0)
			c = np.tile(c,(params.batch_size, 1, 1))
			x = np.expand_dims(x, 0)
			x = np.tile(x,(params.batch_size, 1, 1))
			A = np.expand_dims(A, 0)
			A = np.tile(A,(params.batch_size, 1, 1))
			x = torch.Tensor(x).type(dtype)
			c = torch.Tensor(c).type(dtype)
			gt = torch.Tensor(utils.reshapeGT(params,train_data)).type(dtype)#vertices x dim_size
			gtnormals = torch.Tensor(utils.reshapeGT(params,train_data_normal)).type(dtype)#vertices x dim_size

			mask = utils.create_mask(gt, seq_len)
			mask = torch.Tensor(mask).type(dtypeB)
			
			gt.requires_grad = False
			loss = 0.0
			closs = 0.0
			laploss = 0
			nloss = 0
			eloss = 0

			
			# Vertex addition

			x = deformer.embed(c)
			c_prev = c
			x, s, c = deformer.forward(x,s,c_prev,A)
			laploss = criterionL(c_prev, c, A)
			closs = criterionC(c, gt, mask)
			eloss = criterionE(c, A)
			nloss = criterionN(c, gt, gtnormals, A, mask)
			

			for block in range(num_blocks):
				x, c, A = adder.forward(x,c,A)
				s = torch.cat((s,s),dim=1)
				c_prev = c
				x, s, c = deformer.forward(x,s,c_prev,A)
				
				laploss += criterionL(c_prev, c, A)
				closs += criterionC(c, gt, mask)
				eloss += criterionE(c, A)
				nloss += criterionN(c, gt, gtnormals, A, mask)
				
			loss = closs + params.lambda_n*nloss + params.lambda_lap*laploss + params.lambda_e*eloss
			total_closs +=closs/len(train_data)
			total_laploss +=laploss/len(train_data)
			total_nloss +=nloss/len(train_data)
			total_eloss +=eloss/len(train_data)
			total_loss += loss/len(train_data)
				
			proj_pred = utils.flatten_pred_batch(utils.getPixels(c), A, params)

			if (iter_count % params.show_stat == 0):
				masked_gt = gt[0].masked_select(mask[0].unsqueeze(1).repeat(1,dim_size)).reshape(-1, dim_size)
				utils.drawPolygons(utils.getPixels(c[0]),utils.getPixels(masked_gt),proj_pred=proj_pred[0], proj_gt=proj_gt[0], color='red',out='results/pred.png',A=A[0])
				print("Loss on epoch %i, iteration %i: LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, iter_count, optimizer.param_groups[0]['lr'], loss, closs, laploss, nloss, eloss))
				torch.save(deformer.state_dict(), params.save_model_path)
			# else:
			loss = loss#/params.batch_size
			loss.backward()#retain_graph=True)
			optimizer.step()
				
			iter_count += params.batch_size
		end_time = time.time()
		print ("Epoch Completed, Time taken: %f"%(end_time-start_time))
		print("Loss on epoch %i,  LR = %f;Losses = T:%f,C:%f,L:%f,N:%f,E:%f" % (epoch, optimizer.param_groups[0]['lr'], total_loss,total_closs,total_laploss,total_nloss,total_eloss))

