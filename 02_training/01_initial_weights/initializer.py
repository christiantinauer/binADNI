import math
import torch

# he_uniform
# as in tensorflow
def init_weights(module):
	if type(module) == torch.nn.Linear or type(module) == torch.nn.Conv3d:
		torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
		torch.nn.init.zeros_(module.bias)
