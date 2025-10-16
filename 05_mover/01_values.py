import os
import json
import nibabel as nib
import numpy as np
from scipy.signal import find_peaks

data_base_path = '/mnt/neuro/nas2/Work/binADNI/00_data'

stepper = range(1, 101, 3)
groups = {}

for group in ['CN', 'AD']:
	group_values = []
	groups[group] = group_values
	group_path = os.path.join(data_base_path, group)
	subject_folders = os.listdir(group_path)
	subject_folders.sort()

	for subject_folder in subject_folders:
	# for subject_folder in [subject_folders[0], subject_folders[3], subject_folders[7], subject_folders[10], subject_folders[21]]:
	# 	if subject_folder in [
	# 		'003_S_4142__2011-08-31_15_16_18.0__I254890',
	# 		'024_S_4905__2013-08-26_10_28_29.0__I387448',
	# 		'068_S_4968__2013-02-15_16_42_01.0__I360601',
	# 	]:
	# 		continue
		
		print(subject_folder)
		subject_path = os.path.join(group_path, subject_folder)

		for image_name, mask_name in [
			# ('T1__bias_corrected.nii.gz',                   'T1__brain_mask.nii.gz'),
			# ('T1__bias_corrected_@_MNI152_1mm_dof6.nii.gz', 'T1__brain_mask_@_MNI152_1mm_dof6.nii.gz'),
			('T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz'),
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

			mask_data_sum = mask_data.sum()

			# normalize voxel intensity values to highest peak in histogram of brain voxels (= white matter peak)
			masked_image_data = image_data * mask_data
			hist, bin_edges = np.histogram(masked_image_data[masked_image_data > 0], bins=196)
			peaks, _ = find_peaks(hist, height=hist.max() / 2, prominence=1, width=4)
			# print(peaks)
			if len(peaks) <= 2:
				normalizer_bin_index = peaks[-1]
			else:
				# normalizer_bin_index = (int)(peaks[-2] + (peaks[-1] - peaks[-2]) * (bin_edges[peaks[-2]] / bin_edges[peaks[-1]]) / 2.)
				normalizer_bin_index = (int)(peaks[-2] + (peaks[-1] - peaks[-2]) * bin_edges[peaks[-1]] / (bin_edges[peaks[-1]] + bin_edges[peaks[-2]]))
			# print(highest_bin_index)
			normalizer = (bin_edges[normalizer_bin_index] + bin_edges[normalizer_bin_index + 1]) / 2.
			# print(normalizer)
			
			normalized_masked_image_data = masked_image_data / normalizer

			# binarize in reference to normalizer
			divergence = []
			for step in stepper:
				binarizer = step / 100.
				binarized_image_data = np.where(normalized_masked_image_data >= binarizer, 1., 0.)
				binarized_image_data_sum = binarized_image_data.sum()

				divergence.append(binarized_image_data_sum / mask_data_sum)
			
			if divergence[-2] > .7:
				print(peaks)

			# print(divergence)
			group_values.append(divergence)

with open('values.json', 'w') as outfile:
  json.dump(groups, outfile, indent=2)
