import os
import subprocess
import numpy as np
import nibabel as nib
from scipy.signal import find_peaks

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, 'xAI_group_means/scale_0.4')
target_base_path = os.path.join(this_file_path, 'xAI_group_heatmaps')

if not os.path.exists(target_base_path):
  os.makedirs(target_base_path)

model_outputs = []

for (model_title, model_name, binarizer) in [
  ('A2', 'A2_heatmaps',   None),
  ('C2', 'C2_heatmaps',   0.275),
  ('D1', 'D1_heatmaps',   0.4125),
]:
  model_path = os.path.join(data_base_path, model_name)

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

  group_files = os.listdir(model_path)
  group_files.sort()

  group_outputs = []

  for group_file_index, group_file in enumerate(group_files):
    group_file_path = os.path.join(model_path, group_file)

    sprite_name = model_name + '_' + group_file[:-7]

    # mins
    whole_image = nib.load(group_file_path)
    whole_image_data = np.abs(whole_image.get_fdata().flatten())

    whole_image_data_logical = whole_image_data > 1e-14
    whole_image_data = whole_image_data[whole_image_data_logical]

    whole_image_sum = np.sum(whole_image_data)
    whole_image_data_normalized = whole_image_data / whole_image_sum

    (n, bins) = np.histogram(whole_image_data_normalized,
                             bins=1000,
                             density=False)

    mean_bins = [(i + j) * 0.5 for i, j in zip(bins[:-1], bins[1:])]

    relevance_sum = .0
    min_i_list = []
    
    scales = [.90]
    scales_index = 0
  
    for bin_index, relevance_in_bin in reversed(list(enumerate(n * mean_bins))):
      relevance_sum = relevance_sum + relevance_in_bin
      if (relevance_sum >= scales[scales_index]):
        min_i_list.append(mean_bins[bin_index] * whole_image_sum)
        scales_index = scales_index + 1
        if scales_index == len(scales):
          break

    # max
    fsl_stats_args = [
      'fslstats',
      group_file_path,
      '-R',
    ]

    # print(' '.join(fsl_stats_args))
    p = subprocess.Popen(fsl_stats_args, stdout=subprocess.PIPE)
    output = p.stdout.readline()
    # print(output.decode())

    o = output.decode().strip().split(' ')
    min_i = float(o[0])
    max_i = float(o[1])
    if min_i * -1 > max_i:
      max_i = min_i * -1
    # print(str(max_i))

    convert_append_outer = [
      'magick',
    ]

    for min_i in min_i_list:
      convert_append_inner = [
        'magick',
      ]
      
      for slice_index in [42, 72, 82, 92, 128]: # [62, 71, 76, 82, 94]: # [27, 73]:
        output_filepath = os.path.join(target_base_path, sprite_name +'_sum_lt_' + str(min_i) + '_slice_' + str(slice_index) + '_from_histo.png')
        cropped_output_filepath = os.path.join(target_base_path, sprite_name +'_sum_lt_' + str(min_i) + '_slice_' + str(slice_index) + '_from_histo_cropped.png')
        convert_append_inner.append(cropped_output_filepath)

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
          '--hidex',
          '--hidey',
          '--voxelLoc',
          '90',
          '108',
          str(slice_index),
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
          '/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz' if model_title == 'D1' else '/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
          '--cmap',
          'greyscale' if binarizer is None else 'grey',
          '--displayRange',
          str(threshold if not binarizer is None else 0),
          str(threshold * 1.0001 if not binarizer is None else 10098.99),
          # '--cmap',
          # 'brain_colours_1hot',
          group_file_path,
          '--cmap',
          'hot', # 'brain_colours_1hot_iso', # 'brain_colours_5redyell_iso',
          # '--negativeCmap',
          # 'blue-lightblue', # 'brain_colours_6bluegrn_iso',
          # '--useNegativeCmap',
          '--displayRange',
          str(min_i),
          str(max_i),
          # # outlines
          # '/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz',
          # '--overlayType',
          # 'label',
          # '--lut',
          # 'graz_plus_basal_ganglia',
          # '--outline',
          # '--outlineWidth',
          # '7',
        ]

        # print(' '.join(fsleyes_rendering_args))
        subprocess.call(fsleyes_rendering_args)

        crop_args = [
          'magick',
          output_filepath,
          '-crop',
          '490x586+152+6',
          cropped_output_filepath
        ]
        # print(' '.join(crop_args))
        subprocess.call(crop_args)
      
      convert_append_inner.append('+append')
      convert_append_inner_output = os.path.join(target_base_path, sprite_name + '_sprite_lt_' + str(min_i) + '_from_histo.png')
      convert_append_inner.append(convert_append_inner_output)

      # print(' '.join(convert_append_inner))
      subprocess.call(convert_append_inner)

      convert_append_outer.append(convert_append_inner_output)
    
    convert_append_outer.append('-append')
    convert_append_outer_output = os.path.join(target_base_path, sprite_name + '.png')
    convert_append_outer.append(convert_append_outer_output)

    print(' '.join(convert_append_outer))
    subprocess.call(convert_append_outer)

    group_outputs.append(convert_append_outer_output)

    rel_index = len('rel_')
    channel_number_index = len('rel_0.351__channel_1')
    create_label_args = [
      'magick',
      '-size',
      '586x46',
      'canvas:black',
      '-pointsize',
      str(46),
      '-fill',
      'white',
      '-draw',
      f'gravity Center font DejaVu-Sans text 0,0 "{model_title}, Group {group_file_index + 1}"',
      '-rotate',
      '-90',
    ]
    create_label_args.append(os.path.join(target_base_path, sprite_name + '__group_index.png'))
    print(' '.join(create_label_args))
    subprocess.call(create_label_args)

    c1_args = [
      'magick',
      os.path.join(target_base_path, sprite_name + '__group_index.png'),
      convert_append_outer_output,
      '-background',
      'white',
      '-splice',
      '10x0+0+0',
      '+append',
      '-chop',
      '10x0+0+0',
      convert_append_outer_output,
    ]
    subprocess.call(c1_args)

  c_f_args = ['magick']
  c_f_args += group_outputs
  c_f_args += [
    '-background',
    'white',
    '-splice',
    '0x10+0+0',
    '-append',
    '-chop',
    '0x10+0+0',
    os.path.join(target_base_path, f'{model_name}.png'),
  ]
  subprocess.call(c_f_args)

  model_outputs.append(os.path.join(target_base_path, f'{model_name}.png'))

c_f_args = ['magick']
c_f_args += model_outputs
c_f_args += [
  '-background',
  'white',
  '-splice',
  '0x30+0+0',
  '-append',
  '-chop',
  '0x30+0+0',
  os.path.join(target_base_path, 'central_image.png'),
]
subprocess.call(c_f_args)
