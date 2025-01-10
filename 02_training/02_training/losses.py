import torch.nn as nn

class RelevanceGuidedLoss(nn.Module):
	def forward(self, output, target):
		return -(output * target).sum(dim=[1, 2, 3, 4]).mean()