import json
import numpy as np

with open('counts_ADNI.json', 'r') as infile:
	counts_ADNI = json.load(infile)

with open('counts_internal.json', 'r') as infile:
	counts_internal = json.load(infile)

binarizer_name = '0.0500__nlin'
preprocess_name = 'bet_sum'

group_data = []
for subject_name, subject_counts in counts_ADNI['CN'].items():
	if subject_counts[binarizer_name][preprocess_name] < 1e6:
		# print(subject_name)
		continue

	group_data.append(subject_counts[binarizer_name][preprocess_name])

CN_median = np.median(group_data) * 1.0

group_data = []
for subject_name, subject_counts in counts_internal['ASPSF'].items():
	if subject_counts[binarizer_name][preprocess_name] < 1e6:
		# print(subject_name)
		continue

	group_data.append(subject_counts[binarizer_name][preprocess_name])

ASPSF_median = np.median(group_data)

print(ASPSF_median)
print(CN_median)
print(str(ASPSF_median / CN_median))


group_data = []
for subject_name, subject_counts in counts_ADNI['AD'].items():
	if subject_counts[binarizer_name][preprocess_name] < 1e6:
		# print(subject_name)
		continue

	group_data.append(subject_counts[binarizer_name][preprocess_name])

AD_median = np.median(group_data) * 1.0

group_data = []
for subject_name, subject_counts in counts_internal['ProDem'].items():
	if subject_counts[binarizer_name][preprocess_name] < 1e6:
		# print(subject_name)
		continue

	group_data.append(subject_counts[binarizer_name][preprocess_name])

ProDem_median = np.median(group_data)

print(AD_median)
print(ProDem_median)
print(str(ProDem_median / AD_median))