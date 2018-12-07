import numpy as np
import torch
import shutil
import torch.autograd as Variable
from  src import dtype, dtypeL, dtypeB

def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

def calculate_split_reward(prob, c, Pid, gt, mask, params):
	total_gt = 0
	masked_gt = gt.masked_select(mask.unsqueeze(1).repeat(1,params.dim_size)).reshape(-1, params.dim_size)
	count = 1
	for i in range(masked_gt.size(0)):
		if count > 1:
			break
		if masked_gt[i,0].item() == -2:
			count += 1
		else:
			if count:
				count = 0
				total_gt += 1
	total_pred = np.max(Pid)
	split = (total_pred < total_gt)
	
	if split:
		return int(prob >= 0)
	else:
		return int(prob < 0)

def calculate_reward(points, prob, c, Pid, gt, mask, params, reward_dim):
	#points: batch_size x 4
	#Pid: adjacency matrix. Polygon id
	batch_size = c.size(0)
	reward = np.zeros((batch_size, reward_dim), dtype=np.float32)
	
	for b in range(batch_size):
		num_intersections = 0
		edges = []

		[p1, q1, p2, q2] = points[b].tolist()
		reward[b][1] = calculate_split_reward(prob[b], c[b], Pid[b], gt[b], mask[b], params)
		# [p1, p2] = points[b].tolist()
		# q1 = -1; q2 = 1;
		num_verts = Pid[b].shape[0]
		for i in range(num_verts):
			if num_intersections > 2:
				break
			for j in range(i,num_verts):
				if num_intersections > 2:
					break
				if Pid[b][i,j]:
					x1, y1, x2, y2 = c[b,i,0].item(),c[b,i,1].item(),c[b,j,0].item(),c[b,j,1].item()
					if(intersect(p1,q1,p2,q2,x1,y1,x2,y2)):
						num_intersections += 1
						edges.append([i,j])
		if num_intersections != 2:
			reward[b][0] = 0
			continue
		elif Pid[b][edges[0][0],edges[0][1]] != Pid[b][edges[1][0],edges[1][1]]:
			reward[b][0] = 0
			continue
		else:

			reward[b][0] += 1; 
			#continue

			masked_gt = gt[b].masked_select(mask[b].unsqueeze(1).repeat(1,params.dim_size)).reshape(-1, params.dim_size)
			pos = 0
			neg = 0
			start = 0
			for i in range(masked_gt.size(0)):
				if masked_gt[i,0].item() == -2:
					start = (i+1)	
					continue
				if i+1 == masked_gt.size(0) or masked_gt[i+1,0].item() == -2:
					x1, y1, x2, y2 = masked_gt[i,0].item(), masked_gt[i,1].item(), masked_gt[start,0].item(), masked_gt[start,1].item()
				else:
					x1, y1, x2, y2 = masked_gt[i,0].item(), masked_gt[i,1].item(), masked_gt[i+1,0].item(), masked_gt[i+1,1].item()
				if line(p1,q1,p2,q2,x1,y1) > 0:
					pos += 1
				else:
					neg += 1
				if(intersect(p1,q1,p2,q2,x1,y1,x2,y2)):
					reward[b][0] = 0
					break
			if(pos == 0 or neg == 0):
				reward[b][0] = 0
			else:
				reward[b][0] += 1
	return reward

def line(p1,q1,p2,q2,x1,y1):
	return (p2-p1)*y1 - (q2-q1)*x1 -(q1*p2-q2*p1)

def intersect(p1,q1,p2,q2,x1,y1,x2,y2):
	#x1,y1 substituted in line joining p1,q1, p2,q2 is line(p1,q1,p2,q2,x1,y1)
	return line(p1,q1,p2,q2,x1,y1)*line(p1,q1,p2,q2,x2,y2) < 0 and line(x1,y1,x2,y2,p1,q1)*line(x1,y1,x2,y2,p2,q2) < 0


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
	ou = OrnsteinUhlenbeckActionNoise(1)
	states = []
	for i in range(1000):
		states.append(ou.sample())
	import matplotlib.pyplot as plt

	plt.plot(states)
	plt.show()