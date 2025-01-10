import os
import subprocess
import nibabel as nib
import numpy as np
from scipy.signal import find_peaks

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../00_data')

for config in [
  # subject                                           binarizer   bet       numbering
	('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   None,       False,    'A1'),
  ('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   None,       True,     'A2'),
  ('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   0.1375,     False,    'B1'),
  ('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   0.1375,     True,     'B2'),
  ('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   0.275,      False,    'C1'),
  ('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   0.275,      True,     'C2'),
  ('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   0.4125,     False,    'D1'),
  ('CN/002_S_0295__2012-05-10_15_44_50.0__I303066',   0.4125,     True,     'D2'),
]:
  (subject_subpath, binarizer, use_bet, numbering) = config

  image_abs_path = os.path.join(data_base_path, subject_subpath, 'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz')
  mask_abs_path = os.path.join(data_base_path, subject_subpath, 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz')

  params = [
    'fslmaths',
    image_abs_path,
    '-mas',
    mask_abs_path,
    os.path.join(this_file_path, 'brain.nii.gz')
  ]
  subprocess.call(params)

  image = nib.load(image_abs_path)
  image_data = np.asarray(image.dataobj)
  mask = nib.load(mask_abs_path)
  mask_data = np.asarray(mask.dataobj)

  # normalize
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

  # binarizer
  if binarizer is not None:
    threshold = normalizer * binarizer
    # print(threshold)
  else:
    threshold = None

  outputs_cropped = []

  for (hides, voxelLocX, voxelLocY, voxelLocZ) in [
    (['--hidex', '--hidez'], '90', '110', '90'),
    (['--hidex', '--hidey'], '90', '108', '76'),
  ]:
    output_filepath = os.path.join(this_file_path, subject_subpath.replace('/', '__') + f'_{str(use_bet)}_{str(binarizer)}_{voxelLocX}_{voxelLocY}_{voxelLocZ}.png')
    cropped_output_filepath = os.path.join(this_file_path, subject_subpath.replace('/', '__') + f'_{str(use_bet)}_{str(binarizer)}_{voxelLocX}_{voxelLocY}_{voxelLocZ}_cropped.png')
    outputs_cropped.append(cropped_output_filepath)

    fsleyes_rendering_args = [
      'fsleyes',
      'render',
      '--layout',
      'horizontal',
      # '--showColourBar',
      # '--colourBarLocation',
      # 'left',
      '--labelSize',
      '0',
      '--performance',
      '3',
      '--hideCursor',
      '--hideLabels',
      hides[0],
      hides[1],
      '--voxelLoc',
      voxelLocX,
      voxelLocY,
      voxelLocZ,
      '--xcentre',
      '0',
      '0',
      '--ycentre',
      '0',
      '0',
      '--zcentre',
      '0',
      '0',
      '--outfile',
      output_filepath,
      # '--size',
      # '1920',
      # '1080',
      image_abs_path if not use_bet else os.path.join(this_file_path, 'brain.nii.gz'),
      '--cmap',
      'greyscale' if binarizer is None else 'grey',
      '--displayRange',
      str(threshold if binarizer is not None else 0.),
      str(threshold * 1.0001 if binarizer is not None else np.max(image_data)),
    ]

    # print(' '.join(fsleyes_rendering_args))
    subprocess.call(fsleyes_rendering_args)

    crop_args = [
      'convert',
      output_filepath,
      '-crop',
      '490x586+152+6',
      cropped_output_filepath
    ]
    # print(' '.join(crop_args))
    subprocess.call(crop_args)
    
  combine_slices = [
    'convert',
    outputs_cropped[0],
    outputs_cropped[1],
    '+append',
    os.path.join(this_file_path, subject_subpath.replace('/', '__') + '.png'),
  ]
  subprocess.call(combine_slices)

  # add_text_args = [
  #   'convert',
  #   os.path.join(this_file_path, subject_subpath.replace('/', '__') + '.png'),
  #   # '-gravity',
  #   # 'North',
  #   '-pointsize',
  #   '56',
  #   '-fill',
  #   'yellow',
  #   '-draw',
  #   'text 572,70 "' + balanced_acc + '"',
  #   os.path.join(this_file_path, subject_subpath.replace('/', '__') + '.png'),
  # ]
  # print(' '.join(add_text_args))
  # subprocess.call(add_text_args)

  add_text_args = [
    'convert',
    os.path.join(this_file_path, subject_subpath.replace('/', '__') + '.png'),
    # '-gravity',
    # 'North',
    '-pointsize',
    '46',
    '-fill',
    'white',
    '-draw',
    'text 175,70 "' + numbering + '"',
    os.path.join(this_file_path, numbering + '.png'),
  ]
  print(' '.join(add_text_args))
  subprocess.call(add_text_args)

args_c1 = [
  'convert',
  os.path.join(this_file_path, 'A1.png'),
  os.path.join(this_file_path, 'B1.png'),
  os.path.join(this_file_path, 'C1.png'),
  os.path.join(this_file_path, 'D1.png'),
  '-background',
  'white',
  '-splice',
  '0x10+0+0',
  '-append',
  '-chop',
  '0x10+0+0',
  os.path.join(this_file_path, 'column_1.png'),
]
subprocess.call(args_c1)

args_c2 = [
  'convert',
  os.path.join(this_file_path, 'A2.png'),
  os.path.join(this_file_path, 'B2.png'),
  os.path.join(this_file_path, 'C2.png'),
  os.path.join(this_file_path, 'D2.png'),
  '-background',
  'white',
  '-splice',
  '0x10+0+0',
  '-append',
  '-chop',
  '0x10+0+0',
  os.path.join(this_file_path, 'column_2.png'),
]
subprocess.call(args_c2)

r_args = [
  'convert',
  os.path.join(this_file_path, 'column_1.png'),
  os.path.join(this_file_path, 'column_2.png'),
  '-background',
  'white',
  '-splice',
  '10x0+0+0',
  '+append',
  '-chop',
  '10x0+0+0',
  os.path.join(this_file_path, 'final.png'),
]
subprocess.call(r_args)