import json
import matplotlib.pyplot as plt

with open('counts_internal.json', 'r') as infile:
	counts = json.load(infile)

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

plt.cla()

fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=False, figsize=(9, 16))
fig.suptitle('Volume distribution of internal subjects', fontsize=16)

for title, row_index, column_index, binarizer_name, preprocess_name, ylim in [
	('A1', 0, 0, '0.0500__orig', 'full_sum',	[1.0e6, 7.3e6]),
	('A2', 0, 1, '0.0500__orig', 'bet_sum',		[1.0e6, 2.1e6]),
	('B1', 1, 0, '0.1375__orig', 'full_sum',	[1.0e6, 7.3e6]),
	('B2', 1, 1, '0.1375__orig', 'bet_sum',		[1.0e6, 2.1e6]),
	('C1', 2, 0, '0.2750__orig', 'full_sum',	[1.0e6, 7.3e6]),
	('C2', 2, 1, '0.2750__orig', 'bet_sum',		[1.0e6, 2.1e6]),
	('D1', 3, 0, '0.4125__orig', 'full_sum',	[1.0e6, 7.3e6]),
	('D2', 3, 1, '0.4125__orig', 'bet_sum',		[1.0e6, 2.1e6]),
]:
	all_data = []

	for group in ['ASPSF', 'ProDem']:
		group_data = []
		all_data.append(group_data)
  
		counts_group = counts[group]
		for subject_name, subject_counts in counts_group.items():
			# if subject_counts[binarizer_name][preprocess_name] < 1e6:
			# 	print(subject_name)
			# 	continue

			group_data.append(subject_counts[binarizer_name][preprocess_name])

	ax = axs[row_index, column_index]
	ax.violinplot(all_data, showmeans=False, showmedians=False)
	ax.boxplot(all_data)
	# ax.set_ylim(ylim)
	ax.set_title(title)
	ax.set_xticks([1, 2], labels=['NC', 'AD'])

plt.savefig('counts_internal_orig.png')

plt.cla()

fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=False, figsize=(9, 16))
fig.suptitle('Volume distribution of internal subjects', fontsize=16)

for title, row_index, column_index, binarizer_name, preprocess_name, ylim in [
	('A1', 0, 0, '0.0500__nlin', 'full_sum',	[1.0e6, 7.3e6]),
	('A2', 0, 1, '0.0500__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
	('B1', 1, 0, '0.1375__nlin', 'full_sum',	[1.0e6, 7.3e6]),
	('B2', 1, 1, '0.1375__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
	('C1', 2, 0, '0.2750__nlin', 'full_sum',	[1.0e6, 7.3e6]),
	('C2', 2, 1, '0.2750__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
	('D1', 3, 0, '0.4125__nlin', 'full_sum',	[1.0e6, 7.3e6]),
	('D2', 3, 1, '0.4125__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
]:
	all_data = []

	for group in ['ASPSF', 'ProDem']:
		group_data = []
		all_data.append(group_data)
  
		counts_group = counts[group]
		for subject_name, subject_counts in counts_group.items():
			# if subject_counts[binarizer_name][preprocess_name] < 1e6:
			# 	print(subject_name)
			# 	continue

			group_data.append(subject_counts[binarizer_name][preprocess_name])

	ax = axs[row_index, column_index]
	ax.violinplot(all_data, showmeans=False, showmedians=False)
	ax.boxplot(all_data)
	# ax.set_ylim(ylim)
	ax.set_title(title)
	ax.set_xticks([1, 2], labels=['NC', 'AD'])

plt.savefig('counts_internal_nlin.png')

plt.cla()

fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=False, figsize=(9, 16))
fig.suptitle('Volume distribution of ADNI subjects', fontsize=16)

for title, row_index, column_index, binarizer_name, preprocess_name, ylim in [
	('A1', 0, 0, '0.0500__orig', 'bet_sum',		[1.0e6, 7.3e6]),
	('A2', 0, 1, '0.0500__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
	('B1', 1, 0, '0.1375__orig', 'bet_sum',		[1.0e6, 7.3e6]),
	('B2', 1, 1, '0.1375__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
	('C1', 2, 0, '0.2750__orig', 'bet_sum',		[1.0e6, 7.3e6]),
	('C2', 2, 1, '0.2750__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
	('D1', 3, 0, '0.4125__orig', 'bet_sum',		[1.0e6, 7.3e6]),
	('D2', 3, 1, '0.4125__nlin', 'bet_sum',		[1.0e6, 2.1e6]),
]:
	all_data = []

	for group in ['ASPSF', 'ProDem']:
		group_data = []
		all_data.append(group_data)
  
		counts_group = counts[group]
		for subject_name, subject_counts in counts_group.items():
			# if subject_counts[binarizer_name][preprocess_name] < 1e6:
			# 	print(subject_name)
			# 	continue

			group_data.append(subject_counts[binarizer_name][preprocess_name])

	ax = axs[row_index, column_index]
	ax.violinplot(all_data, showmeans=False, showmedians=False)
	ax.boxplot(all_data)
	# ax.set_ylim(ylim)
	ax.set_title(title)
	ax.set_xticks([1, 2], labels=['NC', 'AD'])

plt.savefig('counts_internal_bet_sum_orig_vs_nlin.png')

