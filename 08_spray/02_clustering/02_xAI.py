import os
import gc
import json
import numpy as np
import nibabel as nib
from scipy.signal import find_peaks
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import seaborn as sns
import alphashape
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

seed = 96
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = '/mnt/neuro/nas2/Work/binADNI/00_data__downsampled'
heatmaps_base_path = '/mnt/neuro/nas2/Work/binADNI/04_results/04_heatmaps_ISMRM2025/heatmaps'

# find the min windowing value for 40% of the top voxels
def min_value(image, scale=.4):
  whole_image_data = image.flatten()

  whole_image_data_logical = whole_image_data > 0.
  whole_image_data = whole_image_data[whole_image_data_logical]

  whole_image_sum = np.sum(whole_image_data)
  whole_image_data_normalized = whole_image_data / whole_image_sum

  (n, bins) = np.histogram(whole_image_data_normalized,
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

  del n
  del bins

  return min_i

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/CN_bootstrap.json'), 'r') as infile:
  CN_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/AD_bootstrap.json'), 'r') as infile:
  AD_bootstrap = json.load(infile)

bootstrap_index = 4

subject_names = []
for image_name in CN_bootstrap[bootstrap_index]['test']:
  subject_names.append(image_name[:10])
print('CN subjects')
print(len(list(set(subject_names))))

subject_names = []
for image_name in AD_bootstrap[bootstrap_index]['test']:
  subject_names.append(image_name[:10])
print('AD subjects')
print(len(list(set(subject_names))))

# fixed values
region_colors = sns.color_palette(palette='bright')
default_k = 3

for scale in [
  # None,
  # .1,
  .4,
]:
  print(scale or 'full')

  plt.clf()
  fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

  for ax, col in zip(axes[0], ['A2', 'C2', 'D1']):
    ax.set_title(col)

  for ax, row in zip(axes[:,0], ['Inputs', 'Heatmaps']):
    ax.set_ylabel(row, size='large')

  fig.tight_layout()

  for model_index, (order_index, clustering_title, inputs) in enumerate([
    # inputs
    # aligned
    # (0,     'inputs_full',               [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', False, True)]),
    ((0, 0),     'A2',                [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', True, True)]),
    # bin1375
    # (4,     'inputs_full_bin01375',      [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', False, False)]),
    # (6,     'inputs_bet_bin01375',       [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin01375.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', True, False)]),
    # bin275
    # (8,     'inputs_full_bin0275',       [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', False, False)]),
    ((0, 1),    'C2',        [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin0275.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', True, False)]),
    # bin4125
    ((0, 2),    'D1',      [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', False, False)]),
    # (14,    'inputs_bet_bin04125',       [(data_base_path, 'T1__bias_corrected_@_MNI152_1mm_nlin__normalized__bin04125.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz', True, False)]),

    # # heatmaps
    # # aligned
    # (1,     'heatmaps_full',             [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
    ((1, 0),     'A2',              [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin_bet/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
    # bin1375
    # (5,     'heatmaps_full_bin01375',    [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin_bin01375/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
    # (7,     'heatmaps_bet_bin01375',     [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin_bet_bin01375/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
    # bin275
    # (9,     'heatmaps_full_bin0275',     [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin_bin0275/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
    ((1, 1),    'C2',      [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin_bet_bin0275/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
    # bin4125
    ((1, 2),    'D1',    [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin_bin04125/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
    # (15,    'heatmaps_bet_bin04125',     [(os.path.join(heatmaps_base_path, 'T1_@MNI_nlin_bet_bin04125/bootstrap_index-04__initial_weights_index-01'), '__downsampled.nii.gz', None, False, False)]),
  ]):
    
    clustering_name = clustering_title + ('_inputs' if order_index[0] == 0 else '_heatmaps')
    print(clustering_name)

    # make output path
    clustering_name = os.path.join('xAI_group_means', f'scale_{'full' if scale is None else str(scale)}', clustering_name)
    if not os.path.exists(clustering_name):
      os.makedirs(clustering_name)

    X = []
    disease_group_index = []
    classification_group_index = []

    # load data and fill X
    for inputs_base_path, file_name_end, mask_name, use_bet, use_normalize in inputs:
      inputs_base_path_list = os.listdir(inputs_base_path)
      inputs_base_path_list.sort()

      for input in inputs_base_path_list:
        input_path = os.path.join(inputs_base_path, input)

        # all in one folder
        if file_name_end.startswith('__'):
          if not input_path.endswith(file_name_end):
            continue
          
          subject_name = input[0:input.rfind('__') - 13]
          file_path = input_path
          mask_path = None

          index_AD_prob = len(input) - len(file_name_end) - 5
          AD_prob = float(input[index_AD_prob:index_AD_prob + 5])                          
          classification_group_index.append(1 if AD_prob > .5 else 0)
        else:
          subject_name = input
          file_path = os.path.join(input_path, file_name_end)
          mask_path = os.path.join(input_path, mask_name) if mask_name is not None else None
          classification_group_index.append(1 if subject_name in AD_bootstrap[bootstrap_index]['test'] else 0)
        
        image = nib.load(file_path)
        # avoid caching!
        image_data = image.get_fdata(caching='unchanged')

        if not mask_path is None:
          mask = nib.load(mask_path)
          # avoid caching!
          mask_data = mask.get_fdata(caching='unchanged')
          
          if use_normalize:
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
          
            image_data /= normalizer

          if use_bet:
            # we have to cope for downsampling where parts of the mask are not 1. anymore...
            image_data[mask_data < .5] = 0.
          
          del mask_data

        if np.any(np.isnan(image_data)):
          print(input)
          print(np.any(np.isnan(image_data)))
          exit()

        if scale is not None and order_index[0] != 0:
          min_val = min_value(image_data, scale=scale)
          image_data[image_data < min_val] = 0.

        X.append(image_data.flatten())
        disease_group_index.append(1 if subject_name in AD_bootstrap[bootstrap_index]['test'] else 0)

        del image_data

    gc.collect()

    X = np.array(X)

    # initial clustering
    clustering = SpectralClustering(
      n_clusters=default_k,
      eigen_solver=None,
      n_components=None,
      random_state=seed,
      n_init=10,
      affinity='nearest_neighbors',
      n_neighbors=12,
      assign_labels='kmeans',
      n_jobs=12,
    ).fit(X)

    A = clustering.affinity_matrix_.toarray()
    D = np.diag(A.sum(axis=1))
    L = D - A 

    # Compute Symmetric Normalized Laplacian
    D_inv_sqrt = np.diag(1. / np.sqrt(D.diagonal()))  # D^(-1/2)
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt  # L_sym = D^(-1/2) L D^(-1/2)

    # eigenvalues
    eigvals_sym, eigvecs_sym = eigh(L_sym)

    # final clustering with k extracted from the max eigenvalue gap of the initial clustering
    k = np.argmax(np.diff(eigvals_sym[:10])) + 1
    clustering = SpectralClustering(
      n_clusters=k,
      eigen_solver=None,
      n_components=None,
      random_state=seed,
      n_init=10,
      affinity='nearest_neighbors',
      n_neighbors=12,
      assign_labels='kmeans',
      n_jobs=12,
    ).fit(X)

    A = clustering.affinity_matrix_.toarray()
    D = np.diag(A.sum(axis=1))
    L = D - A 

    # Compute Symmetric Normalized Laplacian
    D_inv_sqrt = np.diag(1. / np.sqrt(D.diagonal()))  # D^(-1/2)
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt  # L_sym = D^(-1/2) L D^(-1/2)

    # eigenvalues
    eigvals_sym, eigvecs_sym = eigh(L_sym)

    # # plot eigenvalues and eigenvalue gaps
    # axes[0].set_title('Eigenvalues')
    # axes[0].plot(range(1, 11), eigvals_sym[:10], 'o')
    # axes[0].set_xticks(range(1, 11), range(1, 11))
    # axes[1].set_title('Eigenvalue Gaps')
    # axes[1].bar(range(1, 10), np.diff(eigvals_sym[:10]))
    # axes[1].set_xticks(range(1, 10), range(1, 10))

    # distance and t-SNE  
    distance_matrix =  1. / (A + 1e-3) # 1. - A

    X_embedded = TSNE(
      n_components=2,
      perplexity=12,
      early_exaggeration=12.,
      metric='precomputed',
      init='random',
      random_state=seed,
    ).fit_transform(distance_matrix)

    # plot t-SNE embedding with grouping
    row_index, column_index = order_index
    ax = axes[row_index][column_index]

    legend_handles = []

    for group_index in range(k):
      X_embedded_group = X_embedded[np.where(clustering.labels_ == group_index), :][0]
      # print(X_embedded.shape)
      # print(X_embedded_group.shape)

      # compute concave hull
      concave_hull = alphashape.alphashape(X_embedded_group, .08)

      # plot the hull(s)
      if isinstance(concave_hull, Polygon):  # Check if it's a single shape
        x, y = concave_hull.exterior.xy  # Extract boundary points
        legend_group_handle, = ax.fill(x, y, color=region_colors[group_index], alpha=0.3, label='Group ' + str(group_index + 1))
      else:  # If multiple disconnected components exist
        for geom in concave_hull.geoms:
          x, y = geom.exterior.xy
          legend_group_handle, = ax.fill(x, y, color=region_colors[group_index], alpha=0.3, label='Group ' + str(group_index + 1))

      legend_handles.append(legend_group_handle)

      # save mean
      X_group = X[np.where(clustering.labels_ == group_index), :][0]
      X_group_mean = np.mean(X_group, axis=0)
      X_group_mean_image = np.reshape(X_group_mean, (91, 109, 91))

      # save group mean as nifti
      ref_image = nib.load('/opt/fsl/data/standard/MNI152_T1_2mm.nii.gz')
      group_mean_image = nib.Nifti1Image(X_group_mean_image, ref_image.affine, header=ref_image.header)
      nib.save(group_mean_image, os.path.join(clustering_name, 'mean__group_' + str(group_index + 1) + '.nii.gz'))

    disease_group_index = np.array(disease_group_index)
    classification_group_index = np.array(classification_group_index)
    
    ADs_gt = disease_group_index == 1
    # print(ADs_gt)
    NCs_gt = np.logical_not(ADs_gt)
    # print(NCs_gt)
    AD_pr = classification_group_index == 1
    # print(AD_pr)
    NC_pr = np.logical_not(AD_pr)
    # print(NC_pr)
    
    # NC plot
    NC_right_scatter, = ax.plot(X_embedded[np.where(np.logical_and(NCs_gt, NC_pr)), 0][0], X_embedded[np.where(np.logical_and(NCs_gt, NC_pr)), 1][0], marker='x', color='green', label='TN' if row_index != 0 else 'NC', linestyle='None')
    legend_handles.append(NC_right_scatter)

    if row_index != 0:
      NC_wrong_scatter, = ax.plot(X_embedded[np.where(np.logical_and(ADs_gt, NC_pr)), 0][0], X_embedded[np.where(np.logical_and(ADs_gt, NC_pr)), 1][0], marker='o', color='green', markerfacecolor='none', label='FN', linestyle='None')
      legend_handles.append(NC_wrong_scatter)
    
    # AD plot
    AD_right_scatter, = ax.plot(X_embedded[np.where(np.logical_and(ADs_gt, AD_pr)), 0][0], X_embedded[np.where(np.logical_and(ADs_gt, AD_pr)), 1][0], marker='x', color='red', label='TP' if row_index != 0 else 'AD', linestyle='None')
    legend_handles.append(AD_right_scatter)

    if row_index != 0:
      AD_wrong_scatter, = ax.plot(X_embedded[np.where(np.logical_and(NCs_gt, AD_pr)), 0][0], X_embedded[np.where(np.logical_and(NCs_gt, AD_pr)), 1][0], marker='o', color='red', markerfacecolor='none', label='FP', linestyle='None')
      legend_handles.append(AD_wrong_scatter)
    
    # weird y axis auto scale bug, we fix it manually
    dy = (max(X_embedded[:, 1]) - min(X_embedded[:, 1])) * .1
    ax.set_ylim(min(X_embedded[:, 1]) - dy, max(X_embedded[:, 1]) + dy)

    # legends
    ax.legend(handles=legend_handles)

    # savepoint
    plt.savefig(f'figure_2__scale_{'full' if scale is None else str(scale)}.png')

  plt.savefig(f'figure_2__scale_{'full' if scale is None else str(scale)}.png')
  plt.close()
