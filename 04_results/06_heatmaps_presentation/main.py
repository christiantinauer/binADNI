import os
import subprocess
import nibabel as nib
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

this_file_path = os.path.dirname(os.path.realpath(__file__))
target_base_path = '/mnt/neuro/nas2/Work/binADNI/04_results/04_heatmaps_ISMRM2025/heatmaps'
heatmaps_base_path = target_base_path

bootstrap_index = 4
initial_weights_index = 1

for config in [
  # model                         binarizer   bet       n       acc
	('T1_@MNI_nlin',                None,       False,    'A1',   '71.1%'),
  ('T1_@MNI_nlin_bet',            None,       True,     'A2',   '81.6%'),
  ('T1_@MNI_nlin_bin01375',       0.1375,     False,    'B1',   '62.5%'),
  ('T1_@MNI_nlin_bet_bin01375',   0.1375,     True,     'B2',   '78.1%'),
  ('T1_@MNI_nlin_bin0275',        0.275,      False,    'C1',   '72.7%'),
  ('T1_@MNI_nlin_bet_bin0275',    0.275,      True,     'C2',   '79.6%'),
  ('T1_@MNI_nlin_bin04125',       0.4125,     False,    'D1',   '78.0%'),
  ('T1_@MNI_nlin_bet_bin04125',   0.4125,     True,     'D2',   '81.6%'),
]:
  (model_name, binarizer, use_bet, numbering, acc) = config

  # normalize
  masked_image = nib.load('/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')
  masked_image_data = masked_image.get_fdata()
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

  heatmap_abs_path = os.path.join(heatmaps_base_path, model_name, f'bootstrap_index-{bootstrap_index:02d}__initial_weights_index-{initial_weights_index:02d}', 'sum_down_up.nii.gz')

  # mins
  whole_image = nib.load(heatmap_abs_path)
  # print(model_name)
  # d = whole_image.get_fdata()
  # print(np.unravel_index(np.argmax(d), d.shape))
  # continue

  whole_image_data = whole_image.get_fdata().flatten()

  whole_image_data_logical = whole_image_data > 0.
  whole_image_data = whole_image_data[whole_image_data_logical]

  whole_image_sum = np.sum(whole_image_data)
  whole_image_data_normalized = whole_image_data / whole_image_sum

  (n, bins, _) = plt.hist(whole_image_data_normalized,
                          bins=1000,
                          density=False,
                         )

  mean_bins = [(i + j) * 0.5 for i, j in zip(bins[:-1], bins[1:])]

  relevance_sum = .0
  min_i_list = []
  
  scales = [.40]
  scales_index = 0
  
  for bin_index, relevance_in_bin in reversed(list(enumerate(n * mean_bins))):
    relevance_sum = relevance_sum + relevance_in_bin
    if (relevance_sum >= scales[scales_index]):
      min_i_list.append(mean_bins[bin_index] * whole_image_sum)
      scales_index = scales_index + 1
      if scales_index == len(scales):
        break

  # min
  min_i = min_i_list[0]

  # max
  fsl_stats_args = [
    'fslstats',
    heatmap_abs_path,
    '-R',
  ]

  # print(' '.join(fsl_stats_args))
  p = subprocess.Popen(fsl_stats_args, stdout=subprocess.PIPE)
  output = p.stdout.readline()
  # print(output.decode())

  max_i = float(output.decode().strip().split(' ')[1])
  # print(str(max_i))

  outputs_cropped = []

  for (hides, voxelLocX, voxelLocY, voxelLocZ) in [
    (['--hidex', '--hidez'], '90', '110', '90'),
    (['--hidex', '--hidey'], '90', '108', '76'),
  ]:
    output_filepath = os.path.join(this_file_path, model_name + f'_{voxelLocX}_{voxelLocY}_{voxelLocX}.png')
    cropped_output_filepath = os.path.join(this_file_path, model_name + f'_{voxelLocX}_{voxelLocY}_{voxelLocX}_cropped.png')
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
      os.path.join(this_file_path, model_name + f'_{voxelLocX}_{voxelLocY}_{voxelLocX}.png'),
      # '--size',
      # '1920',
      # '1080',
      '/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz' if not use_bet else '/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
      '--cmap',
      'greyscale' if binarizer is None else 'grey',
      '--displayRange',
      str(threshold if not binarizer is None else 0),
      str(threshold * 1.0001 if not binarizer is None else 10098.99),
      heatmap_abs_path,
      '--cmap',
      'hot',
      '--displayRange',
      str(min_i),
      str(max_i),
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
    os.path.join(this_file_path, model_name + '.png'),
  ]
  subprocess.call(combine_slices)

  add_text_args = [
    'convert',
    os.path.join(this_file_path, model_name + '.png'),
    # '-gravity',
    # 'North',
    '-pointsize',
    '56',
    '-fill',
    'yellow',
    '-draw',
    'text 572,70 "' + acc + '"',
    os.path.join(this_file_path, model_name + '.png'),
  ]
  print(' '.join(add_text_args))
  subprocess.call(add_text_args)

  add_text_args = [
    'convert',
    os.path.join(this_file_path, model_name + '.png'),
    # '-gravity',
    # 'North',
    '-pointsize',
    '46',
    '-fill',
    'white',
    '-draw',
    'text 175,70 "' + numbering + '"',
    os.path.join(this_file_path, model_name + '.png'),
  ]
  print(' '.join(add_text_args))
  subprocess.call(add_text_args)

args_c1 = [
  'convert',
  os.path.join(this_file_path, 'T1_@MNI_nlin.png'),
  os.path.join(this_file_path, 'T1_@MNI_nlin_bin01375.png'),
  os.path.join(this_file_path, 'T1_@MNI_nlin_bin0275.png'),
  os.path.join(this_file_path, 'T1_@MNI_nlin_bin04125.png'),
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
  os.path.join(this_file_path, 'T1_@MNI_nlin_bet.png'),
  os.path.join(this_file_path, 'T1_@MNI_nlin_bet_bin01375.png'),
  os.path.join(this_file_path, 'T1_@MNI_nlin_bet_bin0275.png'),
  os.path.join(this_file_path, 'T1_@MNI_nlin_bet_bin04125.png'),
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

preview_args = [
  'convert',
  os.path.join(this_file_path, 'T1_@MNI_nlin_bet.png'),
  os.path.join(this_file_path, 'T1_@MNI_nlin_bet_bin04125.png'),
  '-background',
  'white',
  '-splice',
  '0x10+0+0',
  '-append',
  '-chop',
  '0x10+0+0',
  os.path.join(this_file_path, 'preview.png'),
]
subprocess.call(preview_args)