import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from  src import dtype, dtypeL, dtypeB

class VertexSplitter(nn.Module):
	def __init__(self):
		super(VertexSplitter, self).__init__()

	def forward(self, Pid, intersections):
		# action: np batch x 4
		# A: np batch x num_verts x num_verts
		# returns A and Pid given a Pid
		batch_size = Pid.shape[0]
		num_verts = Pid.shape[1]
		
		A = Pid
		for b in range(batch_size):
			A[b] = (Pid[b] > 0)
			edges = intersections[b]
			if edges is None:
				continue

			new_Pid = np.max(Pid[b]) + 1
			[edge1, edge2] = edges
			if len(set(edge1).union(set(edge2))) != 4:
				continue
			elif (Pid[b][edge1[0],edge2[0]] == 1) or (Pid[b][edge1[1],edge2[1]] == 1):
				continue

			old_Pid = Pid[b][edge1[0],edge1[1]]
		
			#set old edges to 0
			Pid[b][edge1[0],edge1[1]] = 0
			Pid[b][edge1[1],edge1[0]] = 0
			Pid[b][edge2[0],edge2[1]] = 0
			Pid[b][edge2[1],edge2[0]] = 0
			
			#get new edges
			new_edge1 = [edge1[0],edge2[0]]
			new_edge2 = [edge1[1],edge2[1]]
			edge1, edge2 = new_edge1, new_edge2

			#set new edges Pids
			Pid[b][edge1[0],edge1[1]] = old_Pid
			Pid[b][edge1[1],edge1[0]] = old_Pid
			Pid[b][edge2[0],edge2[1]] = new_Pid
			Pid[b][edge2[1],edge2[0]] = new_Pid

			curr_id = edge2[0]
			visited =set()
			visited.add(curr_id)
			found = True
			while(found):
				for j in range(0,num_verts):
					if Pid[b][curr_id,j] and j not in visited:
						Pid[b][curr_id,j] = new_Pid
						curr_id = j
						visited.add(curr_id)
						found = True
						break
					else:
						found = False
			A[b] = (Pid[b] > 0)
		return A, Pid
