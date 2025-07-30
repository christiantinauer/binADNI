import torch.nn as nn
from deep_taylor_decomposition import LinearDeepTaylorDecomposition, Conv3DDeepTaylorDecomposition

class Model(nn.Module):
	def __init__(self, first_linear_in_features, with_softmax=False, with_dtd=False, dtd_unflattened_dim=None):
		super().__init__()
		self.with_softmax = with_softmax
		self.with_dtd = with_dtd

		self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
		self.conv1_relu = nn.ReLU()
		self.down_conv1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
		self.down_conv1_relu = nn.ReLU()

		self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
		self.conv2_relu = nn.ReLU()
		self.down_conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
		self.down_conv2_relu = nn.ReLU()

		self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
		self.conv3_relu = nn.ReLU()
		self.down_conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
		self.down_conv3_relu = nn.ReLU()

		self.conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
		self.conv4_relu = nn.ReLU()
		self.down_conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
		self.down_conv4_relu = nn.ReLU()

		self.flatten = nn.Flatten()

		self.dens1 = nn.Linear(in_features=first_linear_in_features, out_features=16)
		self.dens1_relu = nn.ReLU()

		self.dens2 = nn.Linear(in_features=16, out_features=2)
		
		if self.with_softmax or self.with_dtd:
			self.softmax = nn.Softmax(dim=1)

		if self.with_dtd:
			self.dens2_dtd = LinearDeepTaylorDecomposition(self.dens2)
			self.dens1_dtd = LinearDeepTaylorDecomposition(self.dens1)

			self.unflatten = nn.Unflatten(1, dtd_unflattened_dim)

			self.down_conv4_dtd = Conv3DDeepTaylorDecomposition(self.down_conv4)
			self.conv4_dtd = Conv3DDeepTaylorDecomposition(self.conv4)

			self.down_conv3_dtd = Conv3DDeepTaylorDecomposition(self.down_conv3)
			self.conv3_dtd = Conv3DDeepTaylorDecomposition(self.conv3)

			self.down_conv2_dtd = Conv3DDeepTaylorDecomposition(self.down_conv2)
			self.conv2_dtd = Conv3DDeepTaylorDecomposition(self.conv2)

			self.down_conv1_dtd = Conv3DDeepTaylorDecomposition(self.down_conv1)
			self.conv1_dtd = Conv3DDeepTaylorDecomposition(self.conv1)
			
	def forward(self, x):
		# Block 1
		c1 = self.conv1_relu(self.conv1(x))
		dc1 = self.down_conv1_relu(self.down_conv1(c1))

		# Block 2
		c2 = self.conv2_relu(self.conv2(dc1))
		dc2 = self.down_conv2_relu(self.down_conv2(c2))

		# Block 3
		c3 = self.conv3_relu(self.conv3(dc2))
		dc3 = self.down_conv3_relu(self.down_conv3(c3))
		
		# Block 4
		c4 = self.conv4_relu(self.conv4(dc3))
		dc4 = self.down_conv4_relu(self.down_conv4(c4))

		# Flatten
		f = self.flatten(dc4)

		# Dense 1
		d1 = self.dens1_relu(self.dens1(f))

		# Dense 2
		logits = self.dens2(d1)
		if not self.with_dtd:
			if self.with_softmax:
				return self.softmax(logits)
			else:
				return logits
		else:
			softmax = self.softmax(logits)

			R = self.dens2_dtd(softmax, d1)
			R = self.dens1_dtd(R, f)
			
			R = self.unflatten(R)

			R = self.down_conv4_dtd(R, c4)
			R = self.conv4_dtd(R, dc3)

			R = self.down_conv3_dtd(R, c3)
			R = self.conv3_dtd(R, dc2)

			R = self.down_conv2_dtd(R, c2)
			R = self.conv2_dtd(R, dc1)

			R = self.down_conv1_dtd(R, c1)
			heatmaps = self.conv1_dtd(R, x)

			if self.with_softmax:
				return softmax, heatmaps
			else:
				return logits, heatmaps
