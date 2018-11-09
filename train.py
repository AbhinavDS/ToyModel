import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import dataLoader
import chamfer_loss,separation_loss
from torch.autograd import Variable
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
torch.set_printoptions(threshold=23679250035)
np.set_printoptions(threshold=13756967)
class Deformer(nn.Module):
    def __init__(self, feature_size, dim_size,depth):
        super(Deformer, self).__init__()
        self.feature_size = feature_size
        self.dim_size = dim_size#Coordinate dimension
        self.layers = nn.ModuleList()
        self.depth = depth
        self.add_layer(nn.Linear(2*feature_size,feature_size))
        self.add_layer(nn.Linear(2*feature_size,feature_size))
        for i in range(depth):
            self.add_layer(nn.Linear(self.feature_size,self.feature_size))
            self.add_layer(nn.Linear(self.feature_size,self.feature_size))
        self.W_p_c = nn.Linear(self.dim_size,self.feature_size)
        self.W_p_s = nn.Linear(self.feature_size,self.feature_size)
        self.W_p = nn.Linear(2*self.feature_size,self.feature_size)
        self.W_c = nn.Linear(self.feature_size,self.dim_size)
        self.W_ic = nn.Linear(self.dim_size,self.feature_size)
        self.W_lol = nn.Linear(self.feature_size,self.dim_size)
        self.a = nn.Tanh()
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform_(self.W_p_c.weight)
        nn.init.xavier_uniform_(self.W_p_s.weight)
        nn.init.xavier_uniform_(self.W_p.weight)
        nn.init.xavier_uniform_(self.W_c.weight)
        nn.init.xavier_uniform_(self.W_ic.weight)
        nn.init.xavier_uniform_(self.W_lol.weight)

    def add_layer(self,layer,init=True):
        self.layers.append(layer)
        if init:
            nn.init.xavier_uniform_(self.layers[-1].weight)

    def forward(self, x_prev, s_prev,c_prev, A):
        #x: batch_size x V x feature_size
        #s: batch_size x V x feature_size
        #c: batch_size x V x dim_size
        #W: feature_size x feature_size
        #A: batch_szie x V x V
        
        temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
        # #s = self.a(self.W_s(s_prev) + self.W_x(x_prev))
        
        c_f = self.a(self.W_p_c(c_prev))
        s_f = self.a(self.W_p_s(s_prev))
        feature_from_state = self.a(self.W_p(torch.cat((c_f,s_f),dim=1)))
        # #print("feature_from_state")
        # #print(feature_from_state)
        # #concat_feature = torch.cat((c_prev,s_prev),dim=1)
        # #feature_from_state = self.a(self.W_p_2(self.a(self.W_p_1(concat_feature))))
        x = torch.cat((feature_from_state,x_prev),dim=1)
        x = self.a(self.layers[0](x)+torch.mm(temp_A,self.layers[1](x)))
        for i in range(2,len(self.layers),2):
            x = self.a(self.layers[i](x)+torch.mm(temp_A,self.layers[i+1](x)) + x)
        #c = self.a(self.W_c(x)+c_prev)
        # c = self.a(self.W_p_s(s_prev[0,:]))#+c_prev)
        c = self.a(self.W_lol(x))
        s = s_prev
        #print(x)
        #c = c.view((-1,2))
        return x, s, c
    def forwardCX(self,c):
        return self.a(self.W_ic(c))

class vertexAdd(nn.Module):
    def __init__(self):
        super(vertexAdd, self).__init__()
    def forward(self,x_prev,c_prev,A):
        num_vertices = len(A)
        final_num_vertices = num_vertices + int(np.count_nonzero(A == 1)/2)
        A_new = np.zeros((final_num_vertices,final_num_vertices))
        v_index = num_vertices#first new vertex added here
        x_new =  x_prev
        c_new =  c_prev
        for i in range(num_vertices):
            for j in range(i+1,num_vertices):
                if(A[i,j] == 1):
                    #add vertex between them
                    A_new[i,v_index] = 1
                    A_new[v_index,i] = 1
                    A_new[v_index,j] = 1
                    A_new[j,v_index] = 1
                    x_v = ((x_prev[i,:] + x_prev[j,:])/2).unsqueeze(0)
                    c_v = ((c_prev[i,:] + c_prev[j,:])/2).unsqueeze(0)
                    x_new = torch.cat((x_new,x_v),dim=0)
                    c_new = torch.cat((c_new,c_v),dim=0)
                    v_index+=1
        #print(x_new.size())
        return x_new, c_new, A_new


if __name__=="__main__":
    # MAKE THE DATA
    train_data, feature_size, dim_size = dataLoader.getData()
    batch_size = 1
    num_epochs = 10000
    lr = 5e-5
    num_blocks = 0
    depth = 10#increasing depth needs reduction in lr

    # RUN TRAINING AND TEST
    deformer = Deformer(feature_size,dim_size,depth).cuda()
    adder = vertexAdd().cuda()
    criterionC = chamfer_loss.ChamferLoss()
    criterionS = separation_loss.SeparationLoss()
    optimizer = optim.Adam(deformer.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)
    #optimizer = optim.Adagrad(deformer.parameters(), lr=lr,lr_decay=5e-3)
    c,_,_ = dataLoader.inputMesh(feature_size)
    for epoch in range(0, num_epochs):
        scheduler.step()
        ex_indices = [i for i in range(0, len(train_data))]
        #random.shuffle(ex_indices)
        total_loss = 0.0
        total_closs = 0
        total_sloss = 0
        for idx in ex_indices:
            optimizer.zero_grad()
            s = torch.Tensor(train_data[idx]).type(dtype).repeat(3,1)
            c,x,A = dataLoader.inputMesh(feature_size)
            x = torch.Tensor(x).type(dtype)
            c = torch.Tensor(c).type(dtype)
            gt = torch.Tensor(dataLoader.generateGT(train_data[idx])).type(dtype)#vertices x dim_size
            #gtnormals = dataLoader.generateNormals()
            gt.requires_grad = False
            loss = 0.0
            closs = 0.0
            sloss = 0.0
            
            num_ias = int(np.log2(feature_size/3)) 
            for ias in range(num_ias):
                x, c, A = adder.forward(x,c,A)
                s = torch.cat((s,s),dim=0)

            x = deformer.forwardCX(c)
            x, s, c = deformer.forward(x,s,c,A)
            loss = criterionC(c,gt)
            for block in range(num_blocks):
                #x, c, A = adder.forward(x,c,A)
                #s = torch.cat((s,s),dim=0)
                x, s, c = deformer.forward(x,s,c,A)
                
                closs += criterionC(c,gt)
                if(epoch > 10000):
                    sloss += criterionS(c,gt,A)
                    loss = closs + 1*sloss
                else:
                    loss = closs
            
            total_closs +=closs/len(train_data)
            total_sloss +=sloss/len(train_data)
            total_loss += loss/len(train_data)
            loss.backward()#retain_graph=True)
            optimizer.step()
            #print(dataLoader.getPixels(c))
        dataLoader.drawPolygons(dataLoader.getPixels(c),dataLoader.getPixels(gt),color='red',out='pred.png',A=A)
            #w = input("Epoch over")
        print("Loss on epoch %i: LR = %f;%f,%f,%f" % (epoch,optimizer.param_groups[0]['lr'], total_loss,total_closs,total_sloss))

    #Normal loss
    #Blocks
    #Vertex adder in block