import os
import json
import nibabel as nib
import numpy as np

for study_name in ['CN', 'AD']:
  excluded_subjects = []
  
  study_path = os.path.join('/mnt/neuro/nas2/Work/binADNI_preprocessed', study_name)
  subject_folders = os.listdir(study_path)
  subject_folders.sort()
  for subject_folder_name in subject_folders:
    subject_folder_path = os.path.join(study_path, subject_folder_name)
    problems = []

    # image file sanity check
    for (image_file_name, volume_index) in [
      # T1
      ('T1.nii.gz', None),
      ('T1__bias_corrected.nii.gz', None),
      ('T1__bias_corrected_@_MNI152_1mm_dof6.nii.gz', None),
      ('T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz', None),

      # Brain masks
      ('T1__brain_mask.nii.gz', None),
      ('T1__brain_mask_@_MNI152_1mm_dof6.nii.gz', None),
      ('T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', None),
    ]:
      image_file = nib.load(os.path.join(subject_folder_path, image_file_name))
      image_file_data = image_file.get_fdata(caching='unchanged')
      if volume_index is not None:
        image_file_data = image_file_data[:, :, :, volume_index]

      # NaNs check
      if np.isnan(image_file_data).any():
        problems.append(image_file_name + ' FOUND NaNs')
      
      # INFs check
      if np.isinf(image_file_data).any():
        problems.append(image_file_name + ' FOUND INFs')

      # image is empty check
      if np.max(image_file_data) == 0.:
        problems.append(image_file_name + ' IS EMPTY')
    
    if len(problems) > 0:
      excluded_subjects.append(subject_folder_name)
      print(subject_folder_name + ' EXCLUDED')
      print('\n'.join(problems))
    else:
      print(subject_folder_name + ' OK')
  
  with open(study_name + '_excludes_image_sanity_checks.json', 'w') as outfile:
    json.dump(excluded_subjects, outfile, indent=2)
