import os
import json
import numpy as np
import nibabel as nib
from skimage import metrics
import scipy.stats as stats
import matplotlib.pyplot as plt

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = '/mnt/neuro/nas2/Work/binADNI/00_data'
heatmaps_base_path = '/mnt/neuro/nas2/Work/binADNI/04_results/04_heatmaps_ISMRM2025/heatmaps'

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/CN_bootstrap.json'), 'r') as infile:
  CN_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/AD_bootstrap.json'), 'r') as infile:
  AD_bootstrap = json.load(infile)

bootstrap_index = 4
initial_weights_index = 1

image_names = CN_bootstrap[bootstrap_index]['test'] + AD_bootstrap[bootstrap_index]['test']

reference_model_name = 'T1_@MNI_nlin_bet'
reference_model_heatmaps_path = os.path.join(heatmaps_base_path, reference_model_name, f'bootstrap_index-{bootstrap_index:02d}__initial_weights_index-{initial_weights_index:02d}')

reference_model_heatmap_names = {}
for heatmap_name in os.listdir(reference_model_heatmaps_path):
  if heatmap_name.count('__') == 3:
    reference_model_heatmap_names[heatmap_name[0:heatmap_name.rfind('__')]] = heatmap_name

for model_name in [
  'T1_@MNI_nlin_bin01375',
  'T1_@MNI_nlin_bet_bin01375',   
  'T1_@MNI_nlin_bin0275',        
  'T1_@MNI_nlin_bet_bin0275',    
  'T1_@MNI_nlin_bin04125',       
  'T1_@MNI_nlin_bet_bin04125',   
  'T1_@MNI_nlin',                
]:
  print(f'{reference_model_name}_vs_{model_name}')

  model_b_heatmaps_path = os.path.join(heatmaps_base_path, model_name, f'bootstrap_index-{bootstrap_index:02d}__initial_weights_index-{initial_weights_index:02d}')

  model_b_heatmap_names = {}
  for heatmap_name in os.listdir(model_b_heatmaps_path):
    if heatmap_name.count('__') == 3:
      model_b_heatmap_names[heatmap_name[0:heatmap_name.rfind('__')]] = heatmap_name

  metrics_per_subject = {}

  for image_name in image_names:
    print(image_name)

    reference_model_heatmap_path = os.path.join(reference_model_heatmaps_path, reference_model_heatmap_names[image_name])
    model_b_heatmap_path = os.path.join(model_b_heatmaps_path, model_b_heatmap_names[image_name])
    # print(reference_model_heatmap_path)
    # print(model_b_heatmap_path)

    reference_model_heatmap = nib.load(reference_model_heatmap_path)
    reference_model_heatmap_data = reference_model_heatmap.get_fdata()

    model_b_heatmap = nib.load(model_b_heatmap_path)
    model_b_heatmap_data = model_b_heatmap.get_fdata()

    image_bet_mask = nib.load(os.path.join(data_base_path, 'CN' if image_name in CN_bootstrap[bootstrap_index]['test'] else 'AD', image_name, 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz'))
    image_bet_mask_data = image_bet_mask.get_fdata() > 0.5

    subject_metrics = {}
    metrics_per_subject[image_name] = subject_metrics

    # IoU for region where values are in the top 4 deciles
    def min_value(image, scale=.4):
      whole_image_data = image.flatten()

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
      
      scales = [scale]
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
      return min_i

    reference_model_heatmap_data_min_value = min_value(reference_model_heatmap_data, scale=.4)
    model_b_heatmap_data_min_value = min_value(model_b_heatmap_data, scale=.4)

    reference_model_heatmap_data_region = reference_model_heatmap_data >= reference_model_heatmap_data_min_value
    model_b_heatmap_data_region = model_b_heatmap_data >= model_b_heatmap_data_min_value

    subject_metrics['IoU_04'] = (reference_model_heatmap_data_region * model_b_heatmap_data_region).sum() / float((reference_model_heatmap_data_region + model_b_heatmap_data_region).sum())

    reference_model_heatmap_data_min_value = min_value(reference_model_heatmap_data, scale=.1)
    model_b_heatmap_data_min_value = min_value(model_b_heatmap_data, scale=.1)

    reference_model_heatmap_data_region = reference_model_heatmap_data >= reference_model_heatmap_data_min_value
    model_b_heatmap_data_region = model_b_heatmap_data >= model_b_heatmap_data_min_value

    subject_metrics['IoU_01'] = (reference_model_heatmap_data_region * model_b_heatmap_data_region).sum() / float((reference_model_heatmap_data_region + model_b_heatmap_data_region).sum())

    # normalize
    reference_model_heatmap_data = (
      (reference_model_heatmap_data - reference_model_heatmap_data.min()) / (reference_model_heatmap_data.max() - reference_model_heatmap_data.min())
      * 255.
    ).astype(np.uint8)
    model_b_heatmap_data = (
      (model_b_heatmap_data - model_b_heatmap_data.min()) / (model_b_heatmap_data.max() - model_b_heatmap_data.min())
      * 255.
    ).astype(np.uint8)

    # roi
    brain_reference_model_heatmap_data = reference_model_heatmap_data[image_bet_mask_data]
    brain_model_b_heatmap_data = model_b_heatmap_data[image_bet_mask_data]

    subject_metrics['root_mse'] = np.sqrt(metrics.mean_squared_error(brain_reference_model_heatmap_data, brain_model_b_heatmap_data))
    subject_metrics['normalized_root_mse__euclidean'] = metrics.normalized_root_mse(brain_reference_model_heatmap_data, brain_model_b_heatmap_data, normalization='euclidean')
    subject_metrics['normalized_root_mse__minmax'] = metrics.normalized_root_mse(brain_reference_model_heatmap_data, brain_model_b_heatmap_data, normalization='min-max')
    subject_metrics['normalized_root_mse__mean'] = metrics.normalized_root_mse(brain_reference_model_heatmap_data, brain_model_b_heatmap_data, normalization='mean')
    subject_metrics['mssim'] = metrics.structural_similarity(brain_reference_model_heatmap_data, brain_model_b_heatmap_data)
    subject_metrics['person_correlation'] = np.corrcoef(brain_reference_model_heatmap_data.ravel(), brain_model_b_heatmap_data.ravel())[0, 1]
    subject_metrics['spearman_correlation'] = stats.spearmanr(brain_reference_model_heatmap_data.ravel(), brain_model_b_heatmap_data.ravel())[0]
    # subject_metrics['kl_divergence'] = stats.entropy(brain_reference_model_heatmap_data.ravel(), brain_model_b_heatmap_data.ravel())
    subject_metrics['earth_movers_distance'] = stats.wasserstein_distance(brain_reference_model_heatmap_data.ravel(), brain_model_b_heatmap_data.ravel())
    subject_metrics['normalized_mutual_information'] = metrics.normalized_mutual_information(brain_reference_model_heatmap_data, brain_model_b_heatmap_data, bins=256)

  target_path = os.path.join(this_file_path, 'metrics', f'{reference_model_name}__{model_name}')
  if not os.path.exists(target_path):
    os.makedirs(target_path)

  with open(target_path + '/classification__bootstrap_index-{0:02d}__initial_weights_index-{1:02d}.json'.format(bootstrap_index, initial_weights_index), 'w') as outfile:
    json.dump(metrics_per_subject, outfile, indent=2)
