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
class Deformer(nn.Module):
    def __init__(self, feature_size, dim_size):
        super(Deformer, self).__init__()
        self.feature_size = feature_size
        self.dim_size = dim_size#Coordinate dimension
        self.W_1 = nn.Linear(2*self.feature_size,self.feature_size)
        self.W_n_1 = nn.Linear(2*self.feature_size,self.feature_size)
        self.W_2 = nn.Linear(self.feature_size,self.feature_size)
        self.W_n_2 = nn.Linear(self.feature_size,self.feature_size)
        self.W_3 = nn.Linear(self.feature_size,self.feature_size)
        self.W_n_3 = nn.Linear(self.feature_size,self.feature_size)
        
        #self.W_s = nn.Linear(self.feature_size,self.feature_size)
        #self.W_x = nn.Linear(self.feature_size,self.feature_size)
        self.W_p_1 = nn.Linear(self.feature_size+self.dim_size,self.feature_size)
        self.W_p_2 = nn.Linear(self.feature_size,self.feature_size)
        self.W_c = nn.Linear(self.feature_size,self.dim_size)
        self.a = nn.Tanh()
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.W_n_1.weight)
        nn.init.xavier_uniform_(self.W_2.weight)
        nn.init.xavier_uniform_(self.W_n_2.weight)
        nn.init.xavier_uniform_(self.W_3.weight)
        nn.init.xavier_uniform_(self.W_n_3.weight)
        
        #nn.init.xavier_uniform_(self.W_s.weight)
        #nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_p_1.weight)
        nn.init.xavier_uniform_(self.W_p_2.weight)
        nn.init.xavier_uniform_(self.W_c.weight)

    def forward(self, x_prev, s_prev,c_prev, A):
        #x: batch_size x V x feature_size
        #s: batch_size x V x feature_size
        #c: batch_size x V x dim_size
        #W: feature_size x feature_size
        #A: batch_szie x V x V
        temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
        #s = self.a(self.W_s(s_prev) + self.W_x(x_prev))
        feature_from_state = self.a(self.W_p_2(self.a(self.W_p_1(torch.cat((c_prev,s_prev),dim=1)))))
        x = torch.cat((feature_from_state,x_prev),dim=1)
        x = self.a(self.W_1(x) + torch.mm(temp_A,self.W_n_1(x)))#Graph convolution
        x = self.a(self.W_2(x) + torch.mm(temp_A,self.W_n_2(x)))
        x = self.a(self.W_3(x) + torch.mm(temp_A,self.W_n_3(x)))

        c = self.a(self.W_c(x))
        return x, s, c

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
    lr = 1e-4
    num_blocks = 3

    # RUN TRAINING AND TEST
    deformer = Deformer(feature_size,dim_size)
    adder = vertexAdd()
    criterionC = chamfer_loss.ChamferLoss()
    criterionS = separation_loss.SeparationLoss()
    #optimizer = optim.Adam(deformer.parameters(), lr=lr)
    optimizer = optim.Adagrad(deformer.parameters(), lr=lr,lr_decay=5e-4)
    c,_,_ = dataLoader.inputMesh(feature_size)
    dataLoader.drawPolygons(dataLoader.getPixels(c),color='red',out='pred.png')
    erw = input("dk")
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_data))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        total_closs = 0
        total_sloss = 0
        for idx in ex_indices:
            s = torch.Tensor(train_data[idx]).type(dtype).repeat(3,1)
            c,x,A = dataLoader.inputMesh(feature_size)
            x = torch.Tensor(x).type(dtype)
            c = torch.Tensor(c).type(dtype)
            gt = torch.Tensor(dataLoader.generateGT(train_data[idx])).type(dtype)#vertices x dim_size
            #print(s)
            gt.requires_grad = False
            optimizer.zero_grad()
            loss = 0.0
            closs = 0.0
            sloss = 0.0
            x, s, c = deformer.forward(x,s,c,A)
            for block in range(num_blocks):
                x, c, A = adder.forward(x,c,A)
                s = torch.cat((s,s),dim=0)
                x, s, c = deformer.forward(x,s,c,A)
            closs += criterionC(c,gt)
            sw = 2
            if(epoch < 0):
                sloss += criterionS(c,gt,A)
                loss = closs + sw*sloss
            else:
                loss = closs
            #w = input("Block over")
            total_closs +=closs/len(train_data)
            total_sloss +=sloss/len(train_data)
            total_loss += loss/len(train_data)
            loss.backward()#retain_graph=True)
            optimizer.step()
            #print(dataLoader.getPixels(c))
            dataLoader.drawPolygons(dataLoader.getPixels(c),color='red',out='pred.png',A=A)
            dataLoader.drawPolygons(dataLoader.getPixels(gt),color='green',out='gt.png')
            #w = input("Epoch over")
        print("Loss on epoch %i: %f,%f,%f" % (epoch, total_loss,total_closs,total_sloss))
