import os
import json
import subprocess
import numpy as np
import nibabel as nib
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

seed = 90
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

this_file_path = os.path.dirname(os.path.realpath(__file__))
target_base_path = '/mnt/neuro/nas2/Work/binADNI/04_results/04_heatmaps_ISMRM2025/heatmaps'

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/CN_bootstrap.json'), 'r') as infile:
  CN_bootstrap = json.load(infile)

with open(os.path.join(this_file_path, '../../01_data/08_bootstrap/AD_bootstrap.json'), 'r') as infile:
  AD_bootstrap = json.load(infile)

bootstrap_index = 4
initial_weights_index = 1

image_names = CN_bootstrap[bootstrap_index]['test'] + AD_bootstrap[bootstrap_index]['test']

# how many subjects are in the test set?
subject_names = []
for image_name in CN_bootstrap[bootstrap_index]['test']:
  subject_names.append(image_name[:10])
print(len(list(set(subject_names))))

subject_names = []
for image_name in AD_bootstrap[bootstrap_index]['test']:
  subject_names.append(image_name[:10])
print(len(list(set(subject_names))))

# print(subject_names)
# exit()

heatmap_paths = []
X = []
colors = []
n = 292 # 292 is all

for model_heatmaps in [
  'T1_@MNI_nlin/bootstrap_index-04__initial_weights_index-01',
  'T1_@MNI_nlin_bet/bootstrap_index-04__initial_weights_index-01',
  # 'T1_@MNI_nlin_bet_bin0275/bootstrap_index-04__initial_weights_index-01',
  # 'T1_@MNI_nlin_bet_bin01375/bootstrap_index-04__initial_weights_index-01',
  # 'T1_@MNI_nlin_bet_bin04125/bootstrap_index-04__initial_weights_index-01',
]:
  model_heatmaps_path = os.path.join(target_base_path, model_heatmaps)
  heatmaps = os.listdir(model_heatmaps_path)
  heatmaps.sort()

  # only the first n for testing
  for heatmap in heatmaps: #[:n]:
    if not heatmap.endswith('__downsampled.nii.gz'):
      continue

    image_name = heatmap[0:heatmap.rfind('__') - 13]
    # print(image_name)
    # exit()

    colors.append('g' if image_name in CN_bootstrap[bootstrap_index]['test'] else 'r')

    heatmap_path = os.path.join(model_heatmaps_path, heatmap)
    heatmap_paths.append(heatmap_path)

    heatmap_image = nib.load(heatmap_path)
    heatmap_image_data = heatmap_image.get_fdata().flatten()
    # print(heatmap)
    # print(heatmap_image_data.shape)
    X.append(heatmap_image_data)

X = np.array(X)

k = 6 # int(np.log(len(X)))
clustering = SpectralClustering(
  n_clusters=k,
  eigen_solver=None,
  n_components=None,
  random_state=seed,
  n_init=10,
  affinity='nearest_neighbors',
  n_neighbors=10,
  assign_labels='kmeans',
  n_jobs=12,
).fit(X)

# print(clustering.affinity_matrix_)
# exit()

# print(clustering.labels_[:n])
# print(clustering.labels_[n:])


A = clustering.affinity_matrix_.toarray()

D = np.diag(A.sum(axis=1)) 
# print(D.shape)

L = D - A 
# print(L)

# Compute Symmetric Normalized Laplacian
D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))  # D^(-1/2)
L_sym = D_inv_sqrt @ L @ D_inv_sqrt  # L_sym = D^(-1/2) L D^(-1/2)

# print(L_sym.shape)

# Compute Eigenvalues of L_sym
eigvals_sym, eigvecs_sym = eigh(L_sym)

# print(eigvals_sym)

plt.plot(eigvals_sym, 'x')
plt.savefig("test.png")

plt.clf()

plt.plot(eigvals_sym[:60], 'x')
plt.savefig("test_20.png")


distance_matrix = 1. / (A + 1e-12) # 1. - A
# print(distance_matrix)

X_embedded = TSNE(
  n_components=2,
  perplexity=10,
  early_exaggeration=12.,
  metric='precomputed',
  init='random',
  random_state=seed,
).fit_transform(distance_matrix)

plt.clf()

df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'group': clustering.labels_[:]})
# tmp = pd.DataFrame({'x': X_embedded[n:, 0], 'y': X_embedded[n:, 1], 'group': clustering.labels_[n:]})
# df = df.append(tmp)
# plt.scatter(X_embedded[:n, 0], X_embedded[:n, 1], lw=0, s=3000, alpha=0.1, color=colors[:n])
# plt.scatter(X_embedded[n:, 0], X_embedded[n:, 1], lw=0, s=3000, alpha=0.1, color=colors[n:])

fig, ax = plt.subplots(figsize=(8, 6))

# we just need the legend
sns.scatterplot(data=df, x='x', y='y', hue='group', legend=True, palette='bright', linewidth=0, alpha=0.3)
legend = ax.legend_
for collection in ax.collections:
  collection.remove()

# real plot
sns.scatterplot(data=df, x='x', y='y', hue='group', legend=False, palette='bright', s=3000, linewidth=0, alpha=0.02)
# plt.legend(['Group 0', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'])

plt.scatter(X_embedded[:n, 0], X_embedded[:n, 1], marker='o', color=colors[:n], facecolor='none')
plt.scatter(X_embedded[n:, 0], X_embedded[n:, 1], marker='x', color=colors[n:])

plt.savefig('tsne.png')

# vals, vecs = eigh(L)
# print(vals)
# print(vecs)

if not os.path.exists(os.path.join(target_base_path, 'bet_vs_bin_bet_clustering')):
  os.makedirs(os.path.join(target_base_path, 'bet_vs_bin_bet_clustering'))

bet_paths = heatmap_paths[:n]
bet_labels = clustering.labels_[:n]

clusters = [[] for _ in range(k)] # list of lists

for index, cluster_index in enumerate(bet_labels):
  clusters[cluster_index].append(bet_paths[index])

with open('bet.json', 'w') as f:
  json.dump(clusters, f, indent=2)

for index, cluster in enumerate(clusters):
  params = [
    'fslmaths',
  ]
  isfirst = True
  for heatmap_path in cluster:
    if not isfirst:
      params.append('-add')
        
    params.append(heatmap_path)
    isfirst = False

  params.append(os.path.join(target_base_path, 'bet_vs_bin_bet_clustering', f'bet_cluster_{index:02d}_sum.nii.gz'))
  # print(' '.join(params))
  # print(index)
  # print(cluster)
  subprocess.call(params)


bin_bet_paths = heatmap_paths[n:]
bin_bet_labels = clustering.labels_[n:]

clusters = [[] for _ in range(k)] # list of lists

for index, cluster_index in enumerate(bin_bet_labels):
  clusters[cluster_index].append(bin_bet_paths[index])

with open('bin_bet.json', 'w') as f:
  json.dump(clusters, f, indent=2)

for index, cluster in enumerate(clusters):
  params = [
    'fslmaths',
  ]
  isfirst = True
  for heatmap_path in cluster:
    if not isfirst:
      params.append('-add')
        
    params.append(heatmap_path)
    isfirst = False

  params.append(os.path.join(target_base_path, 'bet_vs_bin_bet_clustering', f'bin_bet_cluster_{index:02d}_sum.nii.gz'))
  # print(' '.join(params))
  # print(index)
  # print(cluster)
  subprocess.call(params)

print(bet_labels)
print(bin_bet_labels)
print(bet_labels - bin_bet_labels)
