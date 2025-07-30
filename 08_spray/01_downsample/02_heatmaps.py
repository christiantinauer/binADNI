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

    for heatmap_file in os.listdir(run_path):
      if not heatmap_file.endswith('.nii.gz'):
        continue
      
      if heatmap_file in ['sum.nii.gz', 'sum_down.nii.gz', 'sum_down_up.nii.gz'] or heatmap_file.endswith('__downsampled.nii.gz'):
        continue

      heatmap_path = os.path.join(run_path, heatmap_file)
      heatmap_downsampled_path = os.path.join(run_path, heatmap_file[:-7] + '__downsampled.nii.gz')
      
      params = [
        'flirt',
        '-in',
        heatmap_path,
        '-ref',
        heatmap_path,
        '-applyisoxfm',
        str(2.0),
        '-out',
        heatmap_downsampled_path,
      ]
      print(' '.join(params))
      subprocess.call(params)
