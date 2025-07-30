import os
import numpy as np
import nibabel as nib
from scipy.signal import find_peaks

from torch.utils.data import Dataset

class NiftiDataset(Dataset):
	def __init__(self, device, 
							 data_paths, input_names, mask_names,
							 input_shape,
							 use_mask_on_input=False,
							 use_normalize=False,
							 binarizer=None,
							 output_only_categories=False):
		self.device = device
		self.data_paths = data_paths
		self.input_names = input_names
		self.mask_names = mask_names
		self.input_shape = input_shape
		self.use_mask_on_input = use_mask_on_input
		self.use_normalize = use_normalize
		self.binarizer = binarizer
		self.output_only_categories = output_only_categories
		self.filepaths_with_category_and_sample_weighting = []

		for (data_path, class_index) in self.data_paths:
			self.filepaths_with_category_and_sample_weighting.append((
				[os.path.join(data_path, input_name) for input_name in self.input_names],
				[os.path.join(data_path, mask_name) for mask_name in self.mask_names],
				class_index
			))

	def __len__(self):
		return len(self.filepaths_with_category_and_sample_weighting)

	def __getitem__(self, index):
		filepaths_with_category = self.filepaths_with_category_and_sample_weighting[index]
		input_paths, mask_paths, category_index = filepaths_with_category
		
		inputs = []
		masks = []
		for input_path_index in range(0, len(input_paths)):
			input_path = input_paths[input_path_index]
			mask_path = mask_paths[input_path_index]

			# image
			image = nib.load(input_path)
			# if image.shape != self.input_shape:
			#	 print(input_path)
			#	 print(image.shape)
				
			image_data = np.asarray(image.dataobj)
			# image_data = image_data[..., 1]
			# image_data = np.nan_to_num(image_data, copy=False, nan=0., posinf=0., neginf=0.)
			# if np.isnan(np.sum(image_data)):
			#	 print(input_path + ' has NaNs.')
			# if np.max(image_data) == 0.:
			#	 print(input_path + ' max is 0.')

			# mask
			mask = nib.load(mask_path)
			# if mask.shape != self.input_shape:
			#	 print(mask_path)
			#	 print(mask.shape)
				
			mask_data = np.asarray(mask.dataobj)
			# if np.isnan(np.sum(mask_data)):
			#	 print(mask_path + ' has NaNs.')
			# if np.max(mask_data) == 0.:
			#	 print(mask_path + ' max is 0.')

			# normalize voxel intensity values to highest peak in histogram of brain voxels (= white matter peak)
			if self.use_normalize:
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

			# OLD VERSION
			# # normalize voxel intensity values to highest peak in histogram of brain voxels (= white matter peak)
			# if self.use_normalize:
			# 	masked_image_data = image_data * mask_data
			# 	hist, bin_edges = np.histogram(masked_image_data[masked_image_data > 0], bins=196)
			# 	highest_bin_index = np.argmax(hist)
			# 	normalizer = (bin_edges[highest_bin_index] + bin_edges[highest_bin_index + 1]) / 2.

			# 	image_data /= normalizer

			# print(normalizer)
			# print(len(hist))
			# print(highest_bin_index)
			
			# binarize in reference to normalizer
			if self.binarizer is not None:
				image_data = np.where(image_data >= self.binarizer, 1., 0.)

			# use mask
			if self.use_mask_on_input:
				image_data *= mask_data

			inputs.append(image_data)
			masks.append(mask_data)
		
		inputs = np.float32(inputs)
		masks = np.float32(masks)

		if len(inputs.shape) == 5:
			# images and channels to the last
			inputs = np.transpose(inputs, (1, 2 ,3, 0, 4))
			# merge last two dims
			inputs = inputs.reshape(inputs.shape[:-2] + (-1,))
			# combined channels to the first
			inputs = np.transpose(inputs, (3, 0, 1 , 2))

		if len(masks.shape) == 5:
			# masks and channels to the last
			masks = np.transpose(masks, (1, 2 ,3, 0, 4))
			# merge last two dims
			masks = masks.reshape(masks.shape[:-2] + (-1,))
			# combined channels to the first
			masks = np.transpose(masks, (3, 0, 1 , 2))

		# print(inputs.shape)

		# categories = torch.zeros((2,), dtype=torch.float)
		# categories[category_index] = 1.

		if self.output_only_categories:
			return inputs, category_index
		return inputs, masks, category_index
