import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from data import dataLoader
from utils import utils
from loss.chamfer_loss import ChamferLoss
from loss.normal_loss import NormalLoss
from loss.laplacian_loss import LaplacianLoss
from loss.edge_loss import EdgeLoss
from modules.deformer import Deformer
from modules.vertex_adder import VertexAdder

from torch.autograd import Variable
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtypeL = torch.cuda.LongTensor
    dtypeB = torch.cuda.ByteTensor
else:
    dtype = torch.FloatTensor
    dtypeL = torch.LongTensor
    dtypeB = torch.ByteTensor

torch.set_printoptions(threshold=23679250035)
np.set_printoptions(threshold=13756967)

if __name__=="__main__":
    # MAKE THE DATA
    test_data, feature_size, dim_size = dataLoader.getData(test=True)
    batch_size = 1
    num_epochs = 1
    lr = 5e-5
    num_blocks = 0
    depth = 10#increasing depth needs reduction in lr

    # RUN TRAINING AND TEST
    if torch.cuda.is_available():
        deformer = Deformer(feature_size,dim_size,depth).cuda()
    else:
        deformer = Deformer(feature_size,dim_size,depth)
    deformer.load_state_dict(torch.load('model_10000.toy'))
    adder = vertexAdd().cuda()
    criterionC = chamfer_loss.ChamferLoss()
    criterionN = normal_loss.NormalLoss()
    criterionL = laplacian_loss.LaplacianLoss()
    criterionE = edge_loss.EdgeLoss()
    criterionS = separation_loss.SeparationLoss()
    c,_,_ = dataLoader.inputMesh(feature_size)
    ex_indices = [i for i in range(0, len(test_data))]
    total_loss = 0.0
    total_closs = 0
    total_laploss = 0
    total_nloss = 0
    total_eloss = 0
    total_sloss = 0
    for idx in ex_indices:
        s = torch.Tensor(test_data[idx]).type(dtype).repeat(3,1)
        c,x,A = dataLoader.inputMesh(feature_size)
        x = torch.Tensor(x).type(dtype)
        c = torch.Tensor(c).type(dtype)
        gt = torch.Tensor(dataLoader.generateGT(test_data[idx])).type(dtype)#vertices x dim_size
        gtnormals = torch.Tensor(dataLoader.generateNormals(test=True)).type(dtype)

        gt.requires_grad = False
        loss = 0.0
        closs = 0.0
        sloss = 0.0
        
        num_ias = int(np.log2(feature_size/30)) 
        for ias in range(num_ias):
            x, c, A = adder.forward(x,c,A)
            s = torch.cat((s,s),dim=0)

        x = deformer.forwardCX(c)

        x, s, c1 = deformer.forward(x,s,c,A)

        for block in range(num_blocks):
            #x, c, A = adder.forward(x,c,A)
            #s = torch.cat((s,s),dim=0)
            c = c1
            x, s, c1 = deformer.forward(x,s,c,A)
            
            #closs += criterionC(c,gt)
        laploss = criterionL(c, c1, A)
        c = c1
        closs = criterionC(c, gt)
        nloss = criterionN(c, gt, gtnormals, A)
        eloss = criterionE(c, A)

        loss = closs + 0.0001*nloss + 0.6*(laploss + 0.33*eloss) #+ sloss
        total_closs +=closs/len(test_data)
        total_laploss +=laploss/len(test_data)
        total_nloss +=nloss/len(test_data)
        total_eloss +=eloss/len(test_data)
        total_sloss +=sloss/len(test_data)
        total_loss += loss/len(test_data)
        break
        #print(dataLoader.getPixels(c))
    dataLoader.drawPolygons(dataLoader.getPixels(c),dataLoader.getPixels(gt),color='red',out='pred_test.png',A=A)
        #w = input("Epoch over")
    print("Losses = T:%f,C:%f,L:%f,N:%f,E:%f,S:%f" % (total_loss,total_closs,total_laploss,total_nloss,total_eloss,total_sloss))
    #Blocks
    #Vertex adder in block