import os
import json
import torch
import numpy as np
import nibabel as nib
from scipy.signal import find_peaks

from model import Model

seed = 90
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../00_data')
target_base_path = '/mnt/neuro/nas2/Work/binADNI/04_results/04_heatmaps_ISMRM2025/'

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/CN_bootstrap.json'), 'r') as infile:
  CN_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/AD_bootstrap.json'), 'r') as infile:
  AD_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../01_weights_selection/trainings_filtered.json'), 'r') as infile:
  selected_weights = json.load(infile)

bootstrap_index = 4
initial_weights_index = 1

device = 'cuda:1'

for training_config in [
  # image																																	mask                        									bet     	normalize		binarizer		rg				shape								in feat		dtd unflatten			params destination
	# MNI152 nlin
	('T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bin01375',				),
	('T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet_bin01375',		),
	('T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bin0275',					),
	('T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet_bin0275',			),
	('T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bin04125',				),
	('T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	False,			None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet_bin04125',		),
	('T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',													'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		True,   	True,				None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin_bet',							),
	('T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',													'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',		False,   	True,				None,				False,		(182, 218, 182),		5120,			(8, 8, 10, 8),		'T1_@MNI_nlin',									),
]:
	(image_name, mask_name, use_bet, use_normalize, binarizer, use_rg, input_shape, first_linear_in_features, dtd_unflattened_dim, model_name) = training_config
	
	weight_file = None
	for selected_weight in selected_weights[model_name]:
		if selected_weight['bootstrap_index'] == bootstrap_index and selected_weight['weights_index'] == initial_weights_index:
			weight_file = selected_weight['selected'][0]['file']
			break
	
	if weight_file is None:
		print('No weight file found.')
		exit()

	model = Model(first_linear_in_features=first_linear_in_features, with_softmax=True, with_dtd=True, dtd_unflattened_dim=dtd_unflattened_dim)
	model = model.to(device)

	loaded_state = torch.load(os.path.join(this_file_path, '../../02_training/02_training/binADNI_weights_30_epochs__new_normalizer', model_name, weight_file))
	model_state_dict = loaded_state['model_state_dict']
	model.load_state_dict(model_state_dict, strict=False)

	heatmaps_path = os.path.join(target_base_path, 'heatmaps', model_name)
	if not os.path.exists(heatmaps_path):
		os.makedirs(heatmaps_path)

	final_heatmaps_path = heatmaps_path + '/bootstrap_index-{0:02d}__initial_weights_index-{1:02d}'.format(bootstrap_index, initial_weights_index)
	if not os.path.exists(final_heatmaps_path):
		os.makedirs(final_heatmaps_path)

	subject_paths = []
	for subject in CN_bootstrap[bootstrap_index]['test']:
		subject_paths.append((os.path.join(data_base_path, 'CN', subject), subject))
	for subject in AD_bootstrap[bootstrap_index]['test']:
		subject_paths.append((os.path.join(data_base_path, 'AD', subject), subject))
	
	for subject_path, subject in subject_paths:
		# image
		image = nib.load(os.path.join(subject_path, image_name))
		image_data = np.asarray(image.dataobj)

		# mask
		mask = nib.load(os.path.join(subject_path, mask_name))				
		mask_data = np.asarray(mask.dataobj)

		# normalize voxel intensity values to highest peak in histogram of brain voxels (= white matter peak)
		if use_normalize:
			masked_image_data = image_data * mask_data
			hist, bin_edges = np.histogram(masked_image_data[masked_image_data > 0], bins=196)
			peaks, _ = find_peaks(hist, height=hist.max() / 2, prominence=1, width=4)
			# print(peaks)
			if len(peaks) <= 2:
				normalizer_bin_index = peaks[-1]
			else:
				# normalizer_bin_index = (int)(peaks[-2] + (peaks[-1] - peaks[-2]) * (bin_edges[peaks[-2]] / bin_edges[peaks[-1]]) / 2.)
				normalizer_bin_index = (int)(peaks[-2] + (peaks[-1] - peaks[-2]) * bin_edges[peaks[-1]] / (bin_edges[peaks[-1]] + bin_edges[peaks[-2]]))
				
			normalizer = (bin_edges[normalizer_bin_index] + bin_edges[normalizer_bin_index + 1]) / 2.
			# print(normalizer)
			
			image_data /= normalizer
			
		# binarize in reference to normalizer
		if binarizer is not None:
			image_data = np.where(image_data >= binarizer, 1., 0.)

		# use mask
		if use_bet:
			image_data *= mask_data

		input = torch.from_numpy(np.float32(image_data [None, None, :, :, :])).to(device, non_blocking=True)
		prediction, heatmap = model(input)

		print(subject + ' ' + str(prediction[0][0].item()) + ' ' + str(prediction[0][1].item()))
		heatmap = nib.Nifti1Image(np.squeeze(heatmap.cpu().detach().numpy()), image.affine, header=image.header)
		nib.save(heatmap, os.path.join(final_heatmaps_path, '{0}__{1:.3f}_{2:.3f}.nii.gz'.format(subject, prediction[0][0].item(), prediction[0][1].item())))