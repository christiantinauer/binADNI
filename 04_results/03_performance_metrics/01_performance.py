import os
import json

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import sem

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../00_data')
target_base_path = '../02_performance/classifications'

alpha = 5

ADs = os.listdir(os.path.join(data_base_path, 'AD'))

with open(os.path.join(this_file_path, '../01_weights_selection/trainings_filtered.json'), 'r') as infile:
  selected_weights = json.load(infile)

results = {}

for test_config in [
  ('T1_@MNI_nlin_bin01375',				  ),
  ('T1_@MNI_nlin_bin0275',					),
  ('T1_@MNI_nlin_bin04125',				  ),
  ('T1_@MNI_nlin',									),


  ('T1_@MNI_nlin_bet_bin01375',     ),
  ('T1_@MNI_nlin_bet_bin0275',	    ),
  ('T1_@MNI_nlin_bet_bin04125',     ),
  ('T1_@MNI_nlin_bet',							),
]:
  (model_name,) = test_config
  print(model_name)

  scores_path = os.path.join(target_base_path, model_name)

  bal_acc_list = []
  acc_list = []
  sens_list = []
  spec_list = []
  auc_list = []

  results[model_name] = {
    'bal_acc_list': bal_acc_list,
    'acc_list': acc_list,
    'sens_list': sens_list,
    'spec_list': spec_list,
    'auc_list': auc_list,
  }

  for selected_weight in selected_weights[model_name]:
    if len(selected_weight['selected']) == 0:
      continue

    with open(os.path.join(scores_path, 'classification__bootstrap_index' + selected_weight['selected'][0]['file'][len('T1__bootstrap_index'):len('T1__bootstrap_index-03__initial_weights_index-03')] + '.json')) as infile:
      classifications = json.load(infile)

    selected_threshold = 0.5
    y_true = []
    y_scores = []
    for k, v in classifications.items():
      y_true.append(1. if k in ADs else 0.)
      y_scores.append(1. if v['AD'] >= selected_threshold else 0.)

    (tn, fp, fn, tp) = confusion_matrix(y_true, y_scores).ravel()

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    bal_acc = (sens + spec) / 2.
    acc = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, y_scores)

    # print('bal. acc {0:.2%} sens {1:.2%} spec {2:.2%} auc {3:.2f} acc {4:.2%}'.format(bal_acc, sens, spec, auc, acc))

    bal_acc_list.append(bal_acc)
    acc_list.append(acc)
    sens_list.append(sens)
    spec_list.append(spec)
    auc_list.append(auc)

  bal_acc_list = np.array(bal_acc_list)
  acc_list = np.array(acc_list)
  sens_list = np.array(sens_list)
  spec_list = np.array(spec_list)
  auc_list = np.array(auc_list)

  print('\
    avg acc {0:.2f}±{1:.2%} [{2:.2%}, {3:.2%}]\
    avg sens {4:.2f}±{5:.2%} [{6:.2%}, {7:.2%}]\
    avg spec {8:.2f}±{9:.2%} [{10:.2%}, {11:.2%}]\
    avg auc {12:.2f}±{13:.2f} [{14:.2f}, {15:.2f}]\
    avg bal. acc {16:.2f}±{17:.2%} [{18:.2%}, {19:.2%}]\
    '.format(
      np.mean(acc_list) * 100.,       np.std(acc_list, ddof=1),      np.percentile(acc_list, alpha / 2.),      np.percentile(acc_list, 100. - (alpha / 2.)),
      np.mean(sens_list)* 100.,       np.std(sens_list, ddof=1),     np.percentile(sens_list, alpha / 2.),     np.percentile(sens_list, 100. - (alpha / 2.)),
      np.mean(spec_list)* 100.,       np.std(spec_list, ddof=1),     np.percentile(spec_list, alpha / 2.),     np.percentile(spec_list, 100. - (alpha / 2.)),
      np.mean(auc_list) * 100.,       np.std(auc_list, ddof=1),      np.percentile(auc_list, alpha / 2.),      np.percentile(auc_list, 100. - (alpha / 2.)),
      np.mean(bal_acc_list) * 100.,   np.std(bal_acc_list, ddof=1),  np.percentile(bal_acc_list, alpha / 2.),  np.percentile(bal_acc_list, 100. - (alpha / 2.)),
    )
  )
