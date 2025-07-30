import os
import nibabel as nib
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

data_base_path = '/mnt/neuro/nas2/Work/binADNI_data_internal_preprocessed'

for group in ['ASPSF', 'ProDem']:
	group_path = os.path.join(data_base_path, group)
	subject_folders = os.listdir(group_path)
	subject_folders.sort()

	for subject_folder in subject_folders:
		print(subject_folder)
		subject_path = os.path.join(group_path, subject_folder)

		for image_name, mask_name in [
			('T1__bias_corrected.nii.gz',                   'T1__brain_mask.nii.gz'),
			('T1__bias_corrected_@_MNI152_1mm_dof6.nii.gz', 'T1__brain_mask_@_MNI152_1mm_dof6.nii.gz'),
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

			# normalize voxel intensity values to highest peak in histogram of brain voxels (= white matter peak)
			masked_image_data = image_data * mask_data
			
			plt.clf()
			plt.hist(masked_image_data[masked_image_data > 0], bins=196)
			plt.savefig(os.path.join(subject_path, image_name[0:-7] + '__brain__hist_bins196.png'))
			
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
			# Does need to much space on hard disk...
			# normalized_image = nib.Nifti1Image(normalized_image_data, image.affine, header=image.header)
			# nib.save(normalized_image, os.path.join(subject_path, image_name[0:-7] + '__normalized.nii.gz'))

			# binarize in reference to normalizer
			for binarizer in [
				0.1375,
				0.2750,
				0.4125,
			]:
				binarized_image_data = np.where(normalized_image_data >= binarizer, 1., 0.)
				binarized_image = nib.Nifti1Image(binarized_image_data, image.affine, header=image.header)
				nib.save(binarized_image, os.path.join(subject_path, image_name[0:-7] + '__normalized__bin' + str(binarizer).replace('.', '') + '.nii.gz'))
