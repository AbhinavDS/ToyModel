import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


img = np.load('image_feats.npy')
bs, channel, _, _ = img.shape
print (np.nonzero(img))

# for i in range(bs):
# 	for j in range(channel):
# 		cur_img = img[i, j, : , :]
# 		cur_img = cur_img / np.max(cur_img)
# 		print (np.nonzero(cur_img), cur_img[np.nonzero(cur_img)])
# 		pil_img = Image.fromarray(np.uint8(cur_img*255.0))
# 		pil_img.thumbnail((600,600), Image.ANTIALIAS)
# 		cur_img = np.asarray(pil_img)
# 		print (cur_img)
# 		plt.imshow(cur_img, cmap='gray')
# 		plt.show()
# 		input("sd")

# import torch

# def channel_normalize(x):
# 	x = x.permute(1,0,2,3)
# 	orig_shape = x.size()
# 	x = x.reshape((x.size(0), -1))
# 	max_x = x.max(1, keepdim=True)[0]
# 	max_x[max_x == 0] = 1
# 	x = x / max_x
# 	x = x.reshape(orig_shape)
# 	x = x.permute(1,0,2,3)
# 	return x_normed

# a = [
# 	[[[12,0,1,1],[100,0,3,4],[1,0,3,1]],
# 	[[121,1,11,11],[1100,10,13,14],[11,10,13,11]]
# 	],
# 	[
# 	[[12,0,31,1],[1030,0,3,4],[13,0,3,1]],
# 	[[1321,1,11,11],[11300,10,133,14],[113,130,13,113]]
# 	],
# 	[[[123,0,13,13],[1030,30,33,34],[13,3,33,13]],
# 	[[1221,12,121,121],[11200,120,123,124],[112,102,132,112]]]
# 	]
# a = np.array(a, dtype=float)
# print (a.shape)
# img = torch.from_numpy(a)
# print (img)
# img2 = normalize(img)
# print (img2)