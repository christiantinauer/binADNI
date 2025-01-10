import json

import torch

from model import Model
from initializer import init_weights

with open('random_seeds.json', 'r') as infile:
	random_seeds = json.load(infile)

for i in range(len(random_seeds)):
	for (input_space, input_shape, first_linear_in_features, dtd_unflattened_dim) in [
		('native',	(1, 1, 160, 240, 256),	8736,	 (8, 7, 12, 13)),
		('MNI',			(1, 1, 182, 218, 182),	5120,	 (8, 8, 10, 8)),
	]:
		torch.manual_seed(random_seeds[i])

		model = Model(first_linear_in_features=first_linear_in_features, with_softmax=True, with_dtd=True, dtd_unflattened_dim=dtd_unflattened_dim)
		model.apply(init_weights)

		# check if all dimensions/parameters work
		model(torch.ones(input_shape))

		model_state_dict = model.state_dict()
		
		# exclude double references to weights and biases
		# _dtd layers wrap the corresponding forward layers
		keys_to_remove = []
		for key in model_state_dict.keys():
			if '_dtd.for_layer.' in key:
				keys_to_remove.append(key)
		for key in keys_to_remove:
			model_state_dict.pop(key)
			
		torch.save(model_state_dict, str(i) + '_initial_weights_' + input_space + '.pth')
