import os
import json
import torch

from training import training

seed = 90
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../00_data')
weights_base_path = os.path.join(this_file_path, 'binADNI_weights')

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/CN_bootstrap.json'), 'r') as infile:
  CN_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/AD_bootstrap.json'), 'r') as infile:
  AD_bootstrap = json.load(infile)

device = 'cuda:0'
epochs = 20
batch_size = 6

for training_config in [
  # lr			rf		image																							mask                        									bet     	binarize		shape								in feat		dtd unflatten			params destination							initial weights
	# MNI152 nlin
	# (1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	True,				(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet_bin',					'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	# (1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	False,			(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet',							'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,		False,			(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin',									'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,		True,				(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bin',							'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	# native space
	# (1e-3,		.3,		'T1__bias_corrected.nii.gz',										'T1__brain_mask.nii.gz',											True,   	False,			(160, 240, 256),		8736,			(8, 7, 12, 13),		'T1_bet',							'../01_initial_weights/{0:1d}_initial_weights_native.pth'),
	# (1e-3,		.3,		'T1__bias_corrected.nii.gz',										'T1__brain_mask.nii.gz',											True,   	True,				(160, 240, 256),		8736,			(8, 7, 12, 13),		'T1_bet_bin',					'../01_initial_weights/{0:1d}_initial_weights_native.pth'),
	# (1e-3,		.3,		'T1__bias_corrected.nii.gz',										'T1__brain_mask.nii.gz',											False,		False,			(160, 240, 256),		8736,			(8, 7, 12, 13),		'T1',									'../01_initial_weights/{0:1d}_initial_weights_native.pth'),
	# (1e-3,		.3,		'T1__bias_corrected.nii.gz',										'T1__brain_mask.nii.gz',											False,		True,				(160, 240, 256),		8736,			(8, 7, 12, 13),		'T1_bin',							'../01_initial_weights/{0:1d}_initial_weights_native.pth'),
]:
	(lr, rf, image_name, mask_name, use_bet, use_binarize, input_shape, first_linear_in_features, dtd_unflattened_dim, params_destination, initial_weights) = training_config

	final_params_destination = os.path.join(weights_base_path, params_destination)
	if not os.path.exists(final_params_destination):
		os.makedirs(final_params_destination)
  
	# build up info
	saved_models = os.listdir(final_params_destination)
	saved_models.sort()

	training_runtime = {}
	for saved_model in saved_models:
  	# T1__bootstrap_index-01__initial_weights_index-01__001-tcl-0.856-vcl-0.846-tca-0.590-vca-0.824-tsrim-0.889-vsrim-0.998.pth
		bootstrap_index = int(saved_model[len('T1__bootstrap_index-'):len('T1__bootstrap_index-') + 2])
		initial_weights_index = int(saved_model[len('T1__bootstrap_index-01__initial_weights_index-'):len('T1__bootstrap_index-01__initial_weights_index-') + 2])
		epoch = int(saved_model[len('T1__bootstrap_index-01__initial_weights_index-01__'):len('T1__bootstrap_index-01__initial_weights_index-01__') + 3])

		bootstrap_runtime = training_runtime[bootstrap_index] if bootstrap_index in training_runtime else {}
		bootstrap_runtime[initial_weights_index] = (epoch, os.path.join(final_params_destination, saved_model))
		training_runtime[bootstrap_index] = bootstrap_runtime

	for bootstrap_index in range(1, 11): # len(CN_bootstrap)):
		training_data_paths = []
		for subject_folder in CN_bootstrap[bootstrap_index]['training']:
			training_data_paths.append((os.path.join(data_base_path, 'CN', subject_folder), 0))
		for subject_folder in AD_bootstrap[bootstrap_index]['training']:
			training_data_paths.append((os.path.join(data_base_path, 'AD', subject_folder), 1))
      
		validation_data_paths = []
		for subject_folder in CN_bootstrap[bootstrap_index]['validation']:
			validation_data_paths.append((os.path.join(data_base_path, 'CN', subject_folder), 0))
		for subject_folder in AD_bootstrap[bootstrap_index]['validation']:
			validation_data_paths.append((os.path.join(data_base_path, 'AD', subject_folder), 1))

		bootstrap_runtime = training_runtime[bootstrap_index] if bootstrap_index in training_runtime else {}

		for initial_weights_index in range(1, 4):
			(start_epoch, last_saved_model) = bootstrap_runtime[initial_weights_index] if initial_weights_index in bootstrap_runtime else (0, None)
			if start_epoch >= epochs:
				continue

			print(bootstrap_index)
			print(initial_weights_index)
			print(start_epoch)
			print(last_saved_model)

			training(last_saved_model, bootstrap_index, initial_weights_index, start_epoch, training_config, training_data_paths, validation_data_paths, device, epochs, batch_size, final_params_destination)
