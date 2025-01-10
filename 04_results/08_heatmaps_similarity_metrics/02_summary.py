import os
import json

import numpy as np

alpha = 5

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, './metrics')

for test_config in [
  ('T1_@MNI_nlin_bet__T1_@MNI_nlin',                ),
  ('T1_@MNI_nlin_bet__T1_@MNI_nlin_bin01375',	      ),
  ('T1_@MNI_nlin_bet__T1_@MNI_nlin_bin0275',		    ),
  ('T1_@MNI_nlin_bet__T1_@MNI_nlin_bin04125',       ),

  ('T1_@MNI_nlin_bet__T1_@MNI_nlin_bet_bin01375',   ),
  ('T1_@MNI_nlin_bet__T1_@MNI_nlin_bet_bin0275',	  ),
  ('T1_@MNI_nlin_bet__T1_@MNI_nlin_bet_bin04125',   ),
]:
  (model_name,) = test_config
  print(model_name)

  rmse_list = []
  nrmse_euclidean_list = []
  nrmse_minmax_list = []
  nrmse_mean_list = []
  nrmse_iqr_list = []
  mssim_list = []
  nmi_list = []

  metrics_path = os.path.join(data_base_path, model_name, 'classification__bootstrap_index-04__initial_weights_index-01.json')
  with open(metrics_path) as infile:
    metrics = json.load(infile)

  IoU_04, IoU_01, root_mse, normalized_root_mse__euclidean, normalized_root_mse__minmax,\
  normalized_root_mse__mean, mssim, person_correlation, spearman_correlation,\
  earth_movers_distance, normalized_mutual_information = zip(*[element.values() for element in metrics.values()])

  IoU_04 = np.array(IoU_04)
  IoU_01 = np.array(IoU_01)
  root_mse = np.array(root_mse)
  normalized_root_mse__euclidean = np.array(normalized_root_mse__euclidean)
  normalized_root_mse__minmax = np.array(normalized_root_mse__minmax)
  normalized_root_mse__mean = np.array(normalized_root_mse__mean)
  mssim = np.array(mssim)
  person_correlation = np.array(person_correlation)
  spearman_correlation = np.array(spearman_correlation)
  earth_movers_distance = np.array(earth_movers_distance)
  normalized_mutual_information = np.array(normalized_mutual_information)

  # nrmse {4:.2f}±{5:.2f} [{6:.2f}, {7:.2f}]\
  # spearman {16:.2f}±{17:.2f} [{18:.2f}, {19:.2f}]\

  print('\
    rmse {0:.2f}±{1:.2f} [{2:.2f}, {3:.2f}]\
    mssim {8:.2f}±{9:.2f} [{10:.2f}, {11:.2f}]\
    pearson {12:.2f}±{13:.2f} [{14:.2f}, {15:.2f}]\
    emd {20:.2f}±{21:.2f} [{22:.2f}, {23:.2f}]\
    mni {24:.2f}±{25:.2f} [{26:.2f}, {27:.2f}]\
    IoU_04 {28:.2f}±{29:.2f} [{30:.2f}, {31:.2f}]\
    IoU_01 {32:.2f}±{33:.2f} [{34:.2f}, {35:.2f}]\
    '.format(
      np.mean(root_mse),                        np.std(root_mse, ddof=1),                         np.percentile(root_mse, alpha / 2.),                        np.percentile(root_mse, 100. - (alpha / 2.)),
      np.mean(normalized_root_mse__minmax),     np.std(normalized_root_mse__minmax, ddof=1),      np.percentile(normalized_root_mse__minmax, alpha / 2.),     np.percentile(normalized_root_mse__minmax, 100. - (alpha / 2.)),
      np.mean(mssim),                           np.std(mssim, ddof=1),                            np.percentile(mssim, alpha / 2.),                           np.percentile(mssim, 100. - (alpha / 2.)),
      np.mean(person_correlation),              np.std(person_correlation, ddof=1),               np.percentile(person_correlation, alpha / 2.),              np.percentile(person_correlation, 100. - (alpha / 2.)),
      np.mean(spearman_correlation),            np.std(spearman_correlation, ddof=1),             np.percentile(spearman_correlation, alpha / 2.),            np.percentile(spearman_correlation, 100. - (alpha / 2.)),
      np.mean(earth_movers_distance),           np.std(earth_movers_distance, ddof=1),            np.percentile(earth_movers_distance, alpha / 2.),           np.percentile(earth_movers_distance, 100. - (alpha / 2.)),
      np.mean(normalized_mutual_information),   np.std(normalized_mutual_information, ddof=1),    np.percentile(normalized_mutual_information, alpha / 2.),   np.percentile(normalized_mutual_information, 100. - (alpha / 2.)),
      np.mean(IoU_04),                          np.std(IoU_04, ddof=1),                           np.percentile(IoU_04, alpha / 2.),                          np.percentile(IoU_04, 100. - (alpha / 2.)),
      np.mean(IoU_01),                          np.std(IoU_01, ddof=1),                           np.percentile(IoU_01, alpha / 2.),                          np.percentile(IoU_01, 100. - (alpha / 2.)),
    )
  )
