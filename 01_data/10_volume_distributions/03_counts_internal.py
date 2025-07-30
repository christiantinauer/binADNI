import os
import json
import nibabel as nib
import numpy as np
from scipy.signal import find_peaks

data_base_path = '/mnt/neuro/nas2/Work/binADNI_data_internal_preprocessed'

counts = {}

for group in ['ASPSF', 'ProDem']:
	group_path = os.path.join(data_base_path, group)
	subject_folders = os.listdir(group_path)
	subject_folders.sort()

	counts_group = {}
	counts[group] = counts_group

	for subject_folder in subject_folders:
		print(subject_folder)
		subject_path = os.path.join(group_path, subject_folder)

		counts_subject = {}
		counts_group[subject_folder] = counts_subject

		for binarizer_name, image_name, mask_name in [
			# orig
			('0.0500__orig',	'T1__bias_corrected.nii.gz',												'T1__brain_mask.nii.gz'),
			('0.1375__orig',	'T1__bias_corrected__normalized__bin01375.nii.gz',	'T1__brain_mask.nii.gz'),
			('0.2750__orig',	'T1__bias_corrected__normalized__bin0275.nii.gz',		'T1__brain_mask.nii.gz'),
			('0.4125__orig',	'T1__bias_corrected__normalized__bin04125.nii.gz',	'T1__brain_mask.nii.gz'),

			# nlin
			('0.0500__nlin',	'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',												'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz'),
			('0.1375__nlin',	'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz',	'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz'),
			('0.2750__nlin',	'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz',		'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz'),
			('0.4125__nlin',	'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz',	'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz'),
    ]:
			# image
			image = nib.load(os.path.join(subject_path, image_name))
			# if image.shape != self.input_shape:
			#		print(input_path)
			#		print(image.shape)

			image_data = np.asarray(image.dataobj)
			# image_data = image_data[..., 1]
			# image_data = np.nan_to_num(image_data, copy=False, nan=0., posinf=0., neginf=0.)
			# if np.isnan(np.sum(image_data)):
			#	 print(input_path + ' has NaNs.')
			# if np.max(image_data) == 0.:
			#	 print(input_path + ' max is 0.')

			# mask
			mask = nib.load(os.path.join(subject_path, mask_name))
			# if mask.shape != self.input_shape:
			#	 print(mask_path)
			#	 print(mask.shape)
				
			mask_data = np.asarray(mask.dataobj)
			# if np.isnan(np.sum(mask_data)):
			#	 print(mask_path + ' has NaNs.')
			# if np.max(mask_data) == 0.:
			#	 print(mask_path + ' max is 0.')

			# normalize voxel intensity values to highest peak in histogram of brain voxels (= white matter peak)
			masked_image_data = image_data * mask_data
			
			if binarizer_name.startswith('0.0500'):
				hist, bin_edges = np.histogram(masked_image_data[masked_image_data > 0], bins=196)
				peaks, _ = find_peaks(hist, height=hist.max() / 2, prominence=1, width=4)
				# print(peaks)
				if len(peaks) <= 2:
					normalizer_bin_index = peaks[-1]
				else:
					normalizer_bin_index = (int)(peaks[-2] + (peaks[-1] - peaks[-2]) * bin_edges[peaks[-1]] / (bin_edges[peaks[-1]] + bin_edges[peaks[-2]]))
					
				normalizer = (bin_edges[normalizer_bin_index] + bin_edges[normalizer_bin_index + 1]) / 2.
				# print(normalizer)
							
				normalized_image_data = image_data / normalizer
				binarized_image_data = np.where(normalized_image_data >= .05, 1., 0.)

				image_data = binarized_image_data
				masked_image_data = binarized_image_data * mask_data
			
			counts_subject[binarizer_name] = {
				'full_sum':	int(np.sum(image_data)),
				'full_mean_1': float(np.mean(np.sum(image_data, axis=(1, 2)))),
				'full_mean_2': float(np.mean(np.sum(image_data, axis=(0, 2)))),
				'full_mean_3': float(np.mean(np.sum(image_data, axis=(0, 1)))),


				'bet_sum': int(np.sum(masked_image_data)),
				'bet_mean_1': float(np.mean(np.sum(masked_image_data, axis=(1, 2)))),
				'bet_mean_2': float(np.mean(np.sum(masked_image_data, axis=(0, 2)))),
				'bet_mean_3': float(np.mean(np.sum(masked_image_data, axis=(0, 1)))),
			}

with open('counts_internal.json', 'w') as outfile:
	json.dump(counts, outfile, indent=2)
