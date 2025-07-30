import os
import json
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))
source_base_path = '/mnt/neuro/nas2/Work/binADNI/00_data'
target_base_path = '/mnt/neuro/nas2/Work/binADNI/00_data__downsampled'

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/CN_bootstrap.json'), 'r') as infile:
  CN_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/AD_bootstrap.json'), 'r') as infile:
  AD_bootstrap = json.load(infile)

bootstrap_index = 4

for image_names, sub_path in [(CN_bootstrap[bootstrap_index]['test'], 'CN'), (AD_bootstrap[bootstrap_index]['test'], 'AD')]:
  for image_name in image_names:
    for file_name in [
      'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz',
      'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz',
      'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz',
      'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz',
      'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz',
    ]:
      file_source_path = os.path.join(source_base_path, sub_path, image_name, file_name)
      
      file_target_folder = os.path.join(target_base_path, image_name)
      if not os.path.exists(file_target_folder):
        os.makedirs(file_target_folder)
      file_target_path = os.path.join(file_target_folder, file_name)

      params = [
        'flirt',
        '-in',
        file_source_path,
        '-ref',
        file_source_path,
        '-applyisoxfm',
        str(2.0),
        '-out',
        file_target_path,
      ]
      print(' '.join(params))
      subprocess.call(params)
