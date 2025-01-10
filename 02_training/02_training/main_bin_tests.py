import os
import json
import torch

from training import training

seed = 90
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../00_data')
weights_base_path = os.path.join(this_file_path, 'binADNI_weights_30_epochs__new_normalizer')

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/CN_bootstrap.json'), 'r') as infile:
  CN_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/AD_bootstrap.json'), 'r') as infile:
  AD_bootstrap = json.load(infile)

device = 'cuda:1'
epochs = 30
batch_size = 20

for training_config in [
  # lr			rf		image																																		mask                        									bet     	normalize		binarizer		rg				shape								in feat		dtd unflatten			params destination							initial weights
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bin01375',				'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet_bin01375',		'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz',			'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bin0275',					'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz',			'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet_bin0275',			'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bin04125',				'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet_bin04125',		'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',													'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	True,				None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet',							'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
	(1e-3,		.3,		'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',													'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	True,				None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin',									'../01_initial_weights/{0:1d}_initial_weights_MNI.pth'),
]:
	(lr, rf, image_name, mask_name, use_bet, use_normalize, binarizer, use_rg, input_shape, first_linear_in_features, dtd_unflattened_dim, params_destination, initial_weights) = training_config

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
