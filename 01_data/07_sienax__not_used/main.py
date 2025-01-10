import os
import json

# TODO if needed

for study_name in ['CN', 'AD']:
  subject_per_study = {}
  
  study_path = os.path.join('/mnt/data_extern/ADNI_MPRAGE/ADNI_preprocessed', study_name)
  subject_folders = os.listdir(study_path)
  subject_folders.sort()
  for subject_folder_name in subject_folders:
    subject_sienax_folder_path = os.path.join(study_path, subject_folder_name, 'sienax_f0.35/report.sienax')
    if not os.path.exists(subject_sienax_folder_path):
      print('No sienax report for ' + subject_folder_name)
      continue

    with open(subject_sienax_folder_path, 'r') as report:
      report_text = report.read()
      report_lines = report_text.splitlines()
      if len(report_lines) != 25:
        print('Error in ' + subject_sienax_folder_path)
        continue
        
      subject_per_study[subject_folder_name] = {
        'VSCALING': float(report_lines[11].split()[1]), # VSCALING
        'pgrey': float(report_lines[20].split()[2]),    # pgrey (peripheral grey)
        'vcsf': float(report_lines[21].split()[2]),     # vcsf (ventricular CSF)
        'GREY': float(report_lines[22].split()[2]),     # GREY
        'WHITE': float(report_lines[23].split()[2]),    # WHITE
        'BRAIN': float(report_lines[24].split()[2]),    # BRAIN
      }
    
  with open(study_name + '_sienax.json', 'w') as outfile:
    json.dump(subject_per_study, outfile, indent=2)
