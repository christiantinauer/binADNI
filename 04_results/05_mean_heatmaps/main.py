import os
import subprocess

target_base_path = '/mnt/neuro/nas2/Work/binADNI/04_results/04_heatmaps_ISMRM2025/heatmaps'

for model_name in [
  'T1_@MNI_nlin_bin01375',
  'T1_@MNI_nlin_bet_bin01375',
  'T1_@MNI_nlin_bin0275',
  'T1_@MNI_nlin_bet_bin0275',
  'T1_@MNI_nlin_bin04125',
  'T1_@MNI_nlin_bet_bin04125',
  'T1_@MNI_nlin_bet',
  'T1_@MNI_nlin',
]:
  model_path = os.path.join(target_base_path, model_name)

  for run_folder in os.listdir(model_path):
    run_path = os.path.join(model_path, run_folder)

    params = [
      'fslmaths'
    ]
    isfirst = True

    for run_file in os.listdir(run_path):
      if not run_file.endswith('.nii.gz'):
        continue
      
      if run_file in ['sum.nii.gz', 'sum_down.nii.gz', 'sum_down_up.nii.gz']:
        continue

      if not isfirst:
        params.append('-add')
        
      params.append(os.path.join(run_path, run_file))
      isfirst = False

    params.append(os.path.join(run_path, 'sum.nii.gz'))
    print(' '.join(params))
    subprocess.call(params)

    params = [
      'flirt',
      '-in',
      os.path.join(run_path, 'sum.nii.gz'),
      '-ref',
      os.path.join(run_path, 'sum.nii.gz'),
      '-applyisoxfm',
      str(2.0),
      '-out',
      os.path.join(run_path, 'sum_down.nii.gz'),
    ]
    print(' '.join(params))
    subprocess.call(params)

    params = [
      'flirt',
      '-in',
      os.path.join(run_path, 'sum_down.nii.gz'),
      '-ref',
      os.path.join(run_path, 'sum_down.nii.gz'),
      '-applyisoxfm',
      str(1.0),
      '-out',
      os.path.join(run_path, 'sum_down_up.nii.gz'),
    ]
    print(' '.join(params))
    subprocess.call(params)
