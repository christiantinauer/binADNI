import os
import math
import statistics
import json

import scipy.stats as stats
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

# https://gist.github.com/jensdebruijn/13e8eeda85eb8644ac2a4ac4c3b8e732
def corrected_dependent_ttest(data1, data2, test_train_ratio, alpha):
  n = len(data1)
  differences = [(data1[i] - data2[i]) for i in range(n)]
  sd = statistics.stdev(differences)
  divisor = 1 / n * sum(differences)
  denominator = math.sqrt(1 / n + test_train_ratio) * sd
  t_stat = divisor / denominator
  # degrees of freedom
  df = n - 1
  # calculate the critical value
  cv = stats.t.ppf(1.0 - alpha, df)
  # calculate the p-value
  p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
  # return everything
  return t_stat, df, cv, p

def mcnemar_midp(b, c):
  """
  Compute McNemar's test using the "mid-p" variant suggested by:

  M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
  binary matched-pairs data: Mid-p and asymptotic are better than exact 
  conditional. BMC Medical Research Methodology 13: 91.

  `b` is the number of observations correctly labeled by the first---but 
  not the second---system; `c` is the number of observations correctly 
  labeled by the second---but not the first---system.
  """
  n = b + c
  x = min(b, c)
  dist = stats.binom(n, .5)
  p = 2. * dist.cdf(x)
  midp = p - dist.pmf(x)
  return midp

alpha = .05

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../00_data')
target_base_path = '../02_performance/classifications'

ADs = os.listdir(os.path.join(data_base_path, 'AD'))

with open(os.path.join(this_file_path, '../01_weights_selection/trainings_filtered.json'), 'r') as infile:
  selected_weights = json.load(infile)

models = [
  'T1_@MNI_nlin_bin01375'				,
  'T1_@MNI_nlin_bet_bin01375'   ,
  'T1_@MNI_nlin_bin0275'			  ,
  'T1_@MNI_nlin_bet_bin0275'	  ,
  'T1_@MNI_nlin_bin04125'				,
  'T1_@MNI_nlin_bet_bin04125'   ,
  'T1_@MNI_nlin_bet'					  ,
  'T1_@MNI_nlin'								,
]

# only test against best performing model 'T1_@MNI_nlin_bet'
for a in [6]:
  for b in range(len(models)):
    if a == b:
      continue

    model_a = models[a]
    model_b = models[b]

    model_a_scores_path = os.path.join(target_base_path, model_a)
    model_b_scores_path = os.path.join(target_base_path, model_b)

    acc_mcnemar_z_stats = []
    acc_mcnemar_p_values = []
    
    sen_mcnemar_z_stats = []
    sen_mcnemar_p_values = []

    spe_mcnemar_z_stats = []
    spe_mcnemar_p_values = []

    for bootstrap_index in range(1, 11):
      for initial_weights_index in range(1, 4):
        model_a_classifactions_path = os.path.join(model_a_scores_path, f'classification__bootstrap_index-{bootstrap_index:02d}__initial_weights_index-{initial_weights_index:02d}.json')
        model_b_classifactions_path = os.path.join(model_b_scores_path, f'classification__bootstrap_index-{bootstrap_index:02d}__initial_weights_index-{initial_weights_index:02d}.json')

        if not os.path.exists(model_a_classifactions_path) or not os.path.exists(model_b_classifactions_path):
          continue

        # print(model_a_classifactions_path)

        with open(model_a_classifactions_path) as infile:
          model_a_classifications = json.load(infile)

        with open(model_b_classifactions_path) as infile:
          model_b_classifications = json.load(infile)

        selected_threshold = 0.5

        y_true = []
        model_a_y_scores = []
        model_b_y_scores = []
        for k, v in model_a_classifications.items():
          y_true.append(1 if k in ADs else 0)
          model_a_y_scores.append(1 if v['AD'] >= selected_threshold else 0)
          model_b_y_scores.append(1 if model_b_classifications[k]['AD'] >= selected_threshold else 0)

        # https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2902578/#FD7

        model_a_y_scores = np.array(model_a_y_scores)
        model_b_y_scores = np.array(model_b_y_scores)
        y_true = np.array(y_true)

        # accuracy, exact McNemar test
        A_wrong_B_wrong = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores, y_true), np.not_equal(model_b_y_scores, y_true)), 1, 0))
        A_wrong_B_right = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores, y_true), np.equal(model_b_y_scores, y_true)), 1, 0))
        A_right_B_wrong = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores, y_true), np.not_equal(model_b_y_scores, y_true)), 1, 0))
        A_right_B_right = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores, y_true), np.equal(model_b_y_scores, y_true)), 1, 0))
        contingency_table = [[A_wrong_B_wrong, A_wrong_B_right], [A_right_B_wrong, A_right_B_right]]
        result = mcnemar(contingency_table, exact=True)
        acc_mcnemar_z_stats.append(result.statistic)
        acc_mcnemar_p_values.append(result.pvalue)

        # sensitivity, exact McNemar test
        model_a_y_scores_ind = model_a_y_scores[y_true == 1]
        model_b_y_scores_ind = model_b_y_scores[y_true == 1]
        y_true_ind = np.ones_like(model_a_y_scores_ind)

        A_wrong_B_wrong = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_ind, y_true_ind), np.not_equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        A_wrong_B_right = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_ind, y_true_ind), np.equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        A_right_B_wrong = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_ind, y_true_ind), np.not_equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        A_right_B_right = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_ind, y_true_ind), np.equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        contingency_table = [[A_wrong_B_wrong, A_wrong_B_right], [A_right_B_wrong, A_right_B_right]]
        result = mcnemar(contingency_table, exact=True)
        sen_mcnemar_z_stats.append(result.statistic)
        sen_mcnemar_p_values.append(result.pvalue)

        # specificity, exact McNemar test
        model_a_y_scores_noind = model_a_y_scores[y_true == 0]
        model_b_y_scores_noind = model_b_y_scores[y_true == 0]
        y_true_noind = np.zeros_like(model_a_y_scores_noind)

        A_wrong_B_wrong = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_noind, y_true_noind), np.not_equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        A_wrong_B_right = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_noind, y_true_noind), np.equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        A_right_B_wrong = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_noind, y_true_noind), np.not_equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        A_right_B_right = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_noind, y_true_noind), np.equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        contingency_table = [[A_wrong_B_wrong, A_wrong_B_right], [A_right_B_wrong, A_right_B_right]]
        result = mcnemar(contingency_table, exact=True)
        spe_mcnemar_z_stats.append(result.statistic)
        spe_mcnemar_p_values.append(result.pvalue)

    # if np.all(np.array(pvalues) > alpha):
    print(model_a)
    print(model_b)
    # print(pvalues)
   
    # Discrete Bonferroni-Holm Method for Exact McNemar Tests on bootstrap z and p values
    print('acc')
    sorted_pvalue_indices = np.argsort(acc_mcnemar_p_values)
    for i in range(len(sorted_pvalue_indices)):
      adj_pvalues = []
      for j in range(i + 1):
        pvalue_index = sorted_pvalue_indices[j]
        pvalue = acc_mcnemar_p_values[pvalue_index]
        adj_pvalue = pvalue * (len(acc_mcnemar_p_values) - j)
        adj_pvalues.append(adj_pvalue)
      
      p_i = min(1., np.max(adj_pvalues))
      if p_i > alpha:
        break
    
    print(str(i + 1) + ' of ' + str(len(acc_mcnemar_p_values)))
    print(p_i)

    print('sen')
    sorted_pvalue_indices = np.argsort(sen_mcnemar_p_values)
    for i in range(len(sorted_pvalue_indices)):
      adj_pvalues = []
      for j in range(i + 1):
        pvalue_index = sorted_pvalue_indices[j]
        pvalue = sen_mcnemar_p_values[pvalue_index]
        adj_pvalue = pvalue * (len(sen_mcnemar_p_values) - j)
        adj_pvalues.append(adj_pvalue)
      
      p_i = min(1., np.max(adj_pvalues))
      if p_i > alpha:
        break
    
    print(str(i + 1) + ' of ' + str(len(sen_mcnemar_p_values)))
    print(p_i)

    print('spe')
    sorted_pvalue_indices = np.argsort(spe_mcnemar_p_values)
    for i in range(len(sorted_pvalue_indices)):
      adj_pvalues = []
      for j in range(i + 1):
        pvalue_index = sorted_pvalue_indices[j]
        pvalue = spe_mcnemar_p_values[pvalue_index]
        adj_pvalue = pvalue * (len(spe_mcnemar_p_values) - j)
        adj_pvalues.append(adj_pvalue)
      
      p_i = min(1., np.max(adj_pvalues))
      if p_i > alpha:
        break
    
    print(str(i + 1) + ' of ' + str(len(spe_mcnemar_p_values)))
    print(p_i)
