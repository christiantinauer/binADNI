import os
import json
import numpy as np
from scipy import stats

# from https://www.statsmodels.org/dev/_modules/statsmodels/stats/contingency_tables.html#mcnemar
def exact_mcnemar_test(b, c):
  n1, n2 = b, c

  statistic = np.minimum(n1, n2)
  # binom is symmetric with p=0.5
  # SciPy 1.7+ requires int arguments
  int_sum = int(n1 + n2)
  if int_sum != (n1 + n2):
    raise ValueError(
      'exact can only be used with tables containing integers.'
    )
  p_value = stats.binom.cdf(statistic, int_sum, 0.5) * 2
  p_value = np.minimum(p_value, 1)  # limit to 1 if n1==n2
  return statistic, p_value

def exact_mcnemar_p_values(n):
  '''
  Compute all possible p-values for a given number of discordant pairs (n).

  Parameters:
  n: int - Total discordant pairs (b + c).

  Returns:
  list of float - Possible p-values for McNemar's test.
  '''
  p_values = [0.]
  for b in range(n + 1):
    lower_tail = stats.binom.cdf(b, n, 0.5)
    upper_tail = 1 - stats.binom.cdf(b - 1, n, 0.5)
    p_value = 2 * min(lower_tail, upper_tail)
    p_values.append(p_value)
  return sorted(set(p_values))  # Return unique p-values in ascending order

alpha = .05

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = '/mnt/neuro/nas2/Work/binADNI/00_data' # os.path.join(this_file_path, '../../00_data')
target_base_path = '../02_performance/classifications'

ADs = os.listdir(os.path.join(data_base_path, 'AD'))

with open(os.path.join(this_file_path, '../01_weights_selection/trainings_filtered.json'), 'r') as infile:
  selected_weights = json.load(infile)

models = [
  ('T1_@MNI_nlin'								  , 'A1', ),
  ('T1_@MNI_nlin_bet'					    , 'A2', ),
  ('T1_@MNI_nlin_bin01375'				, 'B1', ),
  ('T1_@MNI_nlin_bet_bin01375'    , 'B2', ),
  ('T1_@MNI_nlin_bin0275'			    , 'C1', ),
  ('T1_@MNI_nlin_bet_bin0275'	    , 'C2', ),
  ('T1_@MNI_nlin_bin04125'				, 'D1', ),
  ('T1_@MNI_nlin_bet_bin04125'    , 'D2', ),
]

model_comparisons = []

# only test against best performing model 'T1_@MNI_nlin_bet'
for a in [1]:
  for b in range(len(models)):
    if a == b:
      continue

    model_a, model_a_short = models[a]
    model_b, model_b_short = models[b]

    model_a_scores_path = os.path.join(target_base_path, model_a)
    model_b_scores_path = os.path.join(target_base_path, model_b)

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
        # A_wrong_B_wrong = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores, y_true), np.not_equal(model_b_y_scores, y_true)), 1, 0))
        A_wrong_B_right = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores, y_true), np.equal(model_b_y_scores, y_true)), 1, 0))
        A_right_B_wrong = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores, y_true), np.not_equal(model_b_y_scores, y_true)), 1, 0))
        # A_right_B_right = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores, y_true), np.equal(model_b_y_scores, y_true)), 1, 0))
        statistic, p_value = exact_mcnemar_test(A_wrong_B_right, A_right_B_wrong)
        possible_p_values = exact_mcnemar_p_values(A_wrong_B_right + A_right_B_wrong)
        model_comparisons.append((f'{model_a_short}-{model_b_short}, acc', p_value, possible_p_values, statistic, bootstrap_index, initial_weights_index, np.sum(1. - np.abs(y_true - model_a_y_scores)) / len(y_true), np.sum(1. - np.abs(y_true - model_b_y_scores)) / len(y_true)))

        # sensitivity, exact McNemar test
        model_a_y_scores_ind = model_a_y_scores[y_true == 1]
        model_b_y_scores_ind = model_b_y_scores[y_true == 1]
        y_true_ind = np.ones_like(model_a_y_scores_ind)

        # A_wrong_B_wrong = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_ind, y_true_ind), np.not_equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        A_wrong_B_right = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_ind, y_true_ind), np.equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        A_right_B_wrong = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_ind, y_true_ind), np.not_equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        # A_right_B_right = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_ind, y_true_ind), np.equal(model_b_y_scores_ind, y_true_ind)), 1, 0))
        statistic, p_value = exact_mcnemar_test(A_wrong_B_right, A_right_B_wrong)
        possible_p_values = exact_mcnemar_p_values(A_wrong_B_right + A_right_B_wrong)
        model_comparisons.append((f'{model_a_short}-{model_b_short}, sen', p_value, possible_p_values, statistic, bootstrap_index, initial_weights_index, np.sum(model_a_y_scores_ind) / np.sum(y_true), np.sum(model_b_y_scores_ind) / np.sum(y_true)))

        # specificity, exact McNemar test
        model_a_y_scores_noind = model_a_y_scores[y_true == 0]
        model_b_y_scores_noind = model_b_y_scores[y_true == 0]
        y_true_noind = np.zeros_like(model_a_y_scores_noind)

        # A_wrong_B_wrong = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_noind, y_true_noind), np.not_equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        A_wrong_B_right = np.sum(np.where(np.logical_and(np.not_equal(model_a_y_scores_noind, y_true_noind), np.equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        A_right_B_wrong = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_noind, y_true_noind), np.not_equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        # A_right_B_right = np.sum(np.where(np.logical_and(np.equal(model_a_y_scores_noind, y_true_noind), np.equal(model_b_y_scores_noind, y_true_noind)), 1, 0))
        statistic, p_value = exact_mcnemar_test(A_wrong_B_right, A_right_B_wrong)
        possible_p_values = exact_mcnemar_p_values(A_wrong_B_right + A_right_B_wrong)
        model_comparisons.append((f'{model_a_short}-{model_b_short}, spe', p_value, possible_p_values, statistic, bootstrap_index, initial_weights_index, np.sum(1. - model_a_y_scores_noind) / np.sum(1. - y_true), np.sum(1. - model_b_y_scores_noind) / np.sum(1. - y_true)))

model_comparisons.sort(key=lambda x: x[1])
p_adj_factor = len(model_comparisons)

# Discrete Bonferroni-Holm as in https://pmc.ncbi.nlm.nih.gov/articles/PMC2902578/
discrete_BH_p_values = []
m = len(model_comparisons)
differences_count = {
  'A1': 0,
  'A2': 0,
  'B1': 0,
  'B2': 0,
  'C1': 0,
  'C2': 0,
  'D1': 0,
  'D2': 0,
}
for i in range(m):
  correction_factor = m - i
  min_adjusted = float('inf')
  for j in range(i, m):
    model_comparison = model_comparisons[j]
    p_j = model_comparison[1]
    attainable_j = model_comparison[2]
    adjusted = min([a for a in attainable_j if a >= p_j]) * correction_factor
    min_adjusted = min(min_adjusted, adjusted)
  discrete_BH_p_values.append(min(min_adjusted, 1.0))

for model_comparison_index in range(len(model_comparisons)):
  model_comparison = model_comparisons[model_comparison_index]
  print(f'{model_comparison[0]};{model_comparison[-2]:.2f};{model_comparison[-1]:.2f};{model_comparison[-4]:d};{model_comparison[-3]};{discrete_BH_p_values[model_comparison_index]:.5f}')

  if discrete_BH_p_values[model_comparison_index] > alpha:
    break

  differences_count[model_comparison[0][3:5]] += 1

print(len(model_comparisons))
print(model_comparison_index + 1)
print('(' + str(len(model_comparisons) - model_comparison_index - 1) + ' more)')
print(differences_count)
