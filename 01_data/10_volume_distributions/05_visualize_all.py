import json
import numpy as np
import matplotlib.pyplot as plt

with open('counts_ADNI.json', 'r') as infile:
	counts_ADNI = json.load(infile)

with open('counts_internal.json', 'r') as infile:
	counts_internal = json.load(infile)

with open('../../09_data_internal/08_matching/ASPSF_matched.json', 'r') as infile:
  ASPSF = json.load(infile)

ASPSF = set(ASPSF)

with open('../../09_data_internal/08_matching/ProDem_matched.json', 'r') as infile:
  ProDem = json.load(infile)
ProDem = set(ProDem)

# {
#   "CN": {
#     "002_S_0295__2012-05-10_15_44_50.0__I303066": {
#       "0.1375": {
#         "full": 3732120,
#         "bet": 1911111
#       },
#       "0.2750": {
#         "full": 3221080,
#         "bet": 1771564
#       },
#       "0.4125": {
#         "full": 2674850,
#         "bet": 1576506
#       }
#     },

fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=False, figsize=(9, 16))
fig.suptitle('Volume distribution of subjects', fontsize=16)

ADNI_full_big = []
ADNI_full_small = []
ADNI_small = []
ADNI_bet_big = []

local_full_small = []
local_small = []
local_bet_big = []

for title, row_index, column_index, binarizer_name, preprocess_name, ylim in [
	('A1', 0, 0, '0.0500__nlin', 'full_sum',	[1.8e6, 7.3e6]),
	('A2', 0, 1, '0.0500__nlin', 'bet_sum',		[1.0e6, 2.5e6]),
	('B1', 1, 0, '0.1375__nlin', 'full_sum',	[1.8e6, 7.3e6]),
	('B2', 1, 1, '0.1375__nlin', 'bet_sum',		[1.0e6, 2.5e6]),
	('C1', 2, 0, '0.2750__nlin', 'full_sum',	[1.8e6, 7.3e6]),
	('C2', 2, 1, '0.2750__nlin', 'bet_sum',		[1.0e6, 2.5e6]),
	('D1', 3, 0, '0.4125__nlin', 'full_sum',	[1.8e6, 7.3e6]),
	('D2', 3, 1, '0.4125__nlin', 'bet_sum',		[1.0e6, 2.5e6]),
]:
	all_data = []

	# ADNI
	for group in ['CN', 'AD']:
		print(group)

		group_data = []
		all_data.append(group_data)
  
		counts_group = counts_ADNI[group]
		for subject_name, subject_counts in counts_group.items():
			# if preprocess_name.startswith('full_') and subject_counts[binarizer_name][preprocess_name] > 6e6:
			# 	ADNI_full_big.append(subject_name)
			# 	continue

			if preprocess_name.startswith('full_') and subject_counts[binarizer_name][preprocess_name] < 2e6:
				ADNI_full_small.append(subject_name)
				continue
				
			if subject_counts[binarizer_name][preprocess_name] < 1e6:
				ADNI_small.append(subject_name)
				continue

			if preprocess_name.startswith('bet_') and subject_counts[binarizer_name][preprocess_name] > 2.5e6:
				ADNI_bet_big.append(subject_name)
				continue

			group_data.append(subject_counts[binarizer_name][preprocess_name])
		
		print(np.median(group_data))

	# internal
	for group in ['ASPSF', 'ProDem']:
		print(group)

		group_data = []
		all_data.append(group_data)
  
		counts_group = counts_internal[group]
		for subject_name, subject_counts in counts_group.items():
			if subject_name not in ASPSF and subject_name not in ProDem:
				continue
			
			if preprocess_name.startswith('full_') and subject_counts[binarizer_name][preprocess_name] < 2e6:
				local_full_small.append(subject_name)
				continue
				
			if subject_counts[binarizer_name][preprocess_name] < 1e6:
				local_small.append(subject_name)
				continue

			if preprocess_name.startswith('bet_') and subject_counts[binarizer_name][preprocess_name] > 2.5e6:
				local_bet_big.append(subject_name)
				continue

			group_data.append(subject_counts[binarizer_name][preprocess_name])
		
		print(np.median(group_data))

	ax = axs[row_index, column_index]
	ax.violinplot(all_data, showmeans=False, showmedians=False)
	ax.boxplot(all_data)
	ax.set_ylim(ylim)
	ax.set_title(title)
	ax.set_xticks([1, 2, 3, 4], labels=['NC', 'AD', 'ASPS', 'ProDem'])

plt.tight_layout()
plt.savefig('volume_distributions.png')

print()
print('ADNI_full_big')
print(list(set(ADNI_full_big)))

print()
print('ADNI_full_small')
print(list(set(ADNI_full_small)))

print()
print('ADNI_small')
print(list(set(ADNI_small)))

print()
print('ADNI_bet_big')
print(list(set(ADNI_bet_big)))

print()
print('local_full_small')
print(list(set(local_full_small)))

print()
print('local_small')
print(list(set(local_small)))

print()
print('local_bet_big')
print(list(set(local_bet_big)))
