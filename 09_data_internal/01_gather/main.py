import os
import json

STUDIES_BASE_PATH = '/mnt/neuro/nas/STUDIES'

for (class_name, study_name, subject_folder_start) in [
  ('02_CN', 'ASPSF', ''), # all study data is from Graz
  ('02_AD', 'ProDem', ''), # only subjects from Graz
]:
  subjects = []

  subject_folders = os.listdir(os.path.join(STUDIES_BASE_PATH, study_name, 'MRI_DATA/NII'))
  subject_folders.sort()
  for subject_folder_name in subject_folders:
    if not subject_folder_name.startswith(subject_folder_start):
      print(subject_folder_name + ' NOT FROM GRAZ')
      continue
    
    # check if folders exist
    if not os.path.exists(os.path.join(STUDIES_BASE_PATH, study_name, 'MRI_DATA/NII', subject_folder_name, 'data')) or not os.path.exists(os.path.join(STUDIES_BASE_PATH, study_name, 'MRI_DATA/NII', subject_folder_name, 'AddNii')):
      print(subject_folder_name + ' NO DATA/ADDNII FOLDER')
      continue
    
    # T1 file
    T1_file_name = None
    data_contents = os.listdir(os.path.join(STUDIES_BASE_PATH, study_name, 'MRI_DATA/NII', subject_folder_name, 'data'))
    data_contents.sort()
    for image_name in data_contents:
      if image_name.startswith('T1_1mm.M__'):
        T1_file_name = image_name
        break

    # check if needed files are there
    if T1_file_name is None:
      print(subject_folder_name + ' NO T1')
      continue

    # # check if image shape fits (some files have unintendet FoV changes)
    # T1_file_path = os.path.join(STUDIES_BASE_PATH, study_name, 'MRI_DATA/NII', subject_folder_name, 'data', T1_file_name)
    # T1_image = nib.load(T1_file_path)
    # if T1_image.shape != (176, 224, 256):
    #   print(subject_folder_name + ' WRONG T1 SHAPE ' + str(T1_image.shape))
    #   continue
    
    subjects.append(subject_folder_name)
    print(subject_folder_name + ' ADDED')
  
  with open(class_name + '.json', 'w') as outfile:
    json.dump(subjects, outfile, indent=2)
