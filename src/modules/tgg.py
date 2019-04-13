import torch.nn as nn
import torch.nn.functional as F


# FCN32s
class TGG(nn.Module):
	def __init__(self, learned_billinear=False):
		super(TGG, self).__init__()
		self.learned_billinear = learned_billinear

		self.conv_block1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)

		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)

		self.conv_block3 = nn.Sequential(
			nn.Conv2d(128, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)

		self.conv_block4 = nn.Sequential(
			# nn.Conv2d(256, 512, 3, padding=1),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(512, 512, 3, padding=1),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(512, 512, 3, padding=1),
			# nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)

		self.conv_block5 = nn.Sequential(
			# nn.Conv2d(512, 512, 3, padding=1),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(512, 512, 3, padding=1),
			# nn.ReLU(inplace=True),
			# nn.Conv2d(512, 512, 3, padding=1),
			# nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2, ceil_mode=True),
		)

		if self.learned_billinear:
			raise NotImplementedError

	def forward(self, x):
		conv1 = self.conv_block1(x)
		conv2 = self.conv_block2(conv1)
		conv3 = self.conv_block3(conv2)
		conv3 = self.channel_normalize(conv3)
		conv4 = self.conv_block4(conv3)
		conv5 = self.conv_block5(conv4)
		return conv3, conv4, conv5

	def channel_normalize(self,x):
		x = x.permute(1,0,2,3)
		orig_shape = x.size()
		x = x.reshape((x.size(0), -1))
		max_x = x.max(1, keepdim=True)[0]
		max_x[max_x == 0] = 1
		x_normed = x / max_x
		x_normed = x_normed.reshape(orig_shape)
		x_normed = x_normed.permute(1,0,2,3)
		return x_normed

	def init_vgg16_params(self, vgg16):
		blocks = [
			self.conv_block1,
			self.conv_block2,
			self.conv_block3,
			# self.conv_block4,
			# self.conv_block5,
		]

		ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
		features = list(vgg16.features.children())

		for idx, conv_block in enumerate(blocks):
			for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
				if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
					assert l1.weight.size() == l2.weight.size()
					assert l1.bias.size() == l2.bias.size()
					l2.weight.data = l1.weight.data
					l2.bias.data = l1.bias.data
