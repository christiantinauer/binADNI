import csv
import json
import functools
import numpy as np
import pandas
from psmpy import PsmPy
import matplotlib.pyplot as plt

wanted_visits = [
	# 'ADNI Baseline',
  # 'ADNI Screening',
  # 'ADNI1/GO Month 12',
  # 'ADNI1/GO Month 18',
  # 'ADNI1/GO Month 24',
  # 'ADNI1/GO Month 36',
  # 'ADNI1/GO Month 48',
  # 'ADNI1/GO Month 6',
  # 'ADNI2 Initial Visit-Cont Pt',
  'ADNI2 Month 3 MRI-New Pt',
  'ADNI2 Month 6-New Pt',
  'ADNI2 Screening MRI-New Pt',
  # 'ADNI2 Screening-New Pt',
  # 'ADNI2 Tau-only visit',
  'ADNI2 Year 1 Visit',
  # 'ADNI2 Year 2 Visit',
  # 'ADNI2 Year 3 Visit',
  # 'ADNI2 Year 4 Visit',
  # 'ADNI2 Year 5 Visit',
  # 'ADNI3 Initial Visit-Cont Pt',
  # 'ADNI3 Year 1 Visit',
  # 'ADNI3 Year 2 Visit',
  # 'ADNI3 Year 3 Visit',
  # 'ADNI3 Year 4 Visit',
  # 'ADNI3 Year 5 Visit',
  # 'ADNIGO Month 3 MRI',
  # 'ADNIGO Month 60',
  # 'ADNIGO Month 72',
  # 'ADNIGO Screening MRI',
  # 'No Visit Defined',
  # 'Unscheduled',
]

selected_image_ids =	[]
with open('../01_gather/03_ADNI_selected.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		image_id = row[0]
		selected_image_ids.append(image_id)

ids = []
classes = []
ages = []
sexes = []
image_ids_to_features = {}
subject_ids_to_dementia_scores = {}
with open('../01_gather/00_idaSearch_11_30_2022.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		subject_id = row[0]
		sex = 1 if row[3] == 'M' else 2
		research_group = row[5]
		visit = row[8]
		age = float(row[11])
		cdr = row[12]
		mmse = row[14]
		image_id = row[20]

		if subject_id not in subject_ids_to_dementia_scores:
			subject_id_dementia_scores = {
				'cdr': [],
				'mmse': [],
			}
			subject_ids_to_dementia_scores[subject_id] = subject_id_dementia_scores
		else:
			subject_id_dementia_scores = subject_ids_to_dementia_scores[subject_id]
		
		subject_id_dementia_scores['cdr'].append(cdr)
		subject_id_dementia_scores['mmse'].append(mmse)

		if image_id not in selected_image_ids or visit not in wanted_visits:
			continue

		ids.append(image_id)
		classes.append(1 if research_group == 'AD' else 0)
		ages.append(age)
		sexes.append(sex)
		image_ids_to_features[image_id] = {
			'sex': sex,
			'age': age,
			'subject_id': subject_id,
		}

data = {'id': ids, 'class': classes, 'age': ages, 'sex': sexes}
df = pandas.DataFrame(data=data)

psm = PsmPy(df, treatment='class', indx='id', exclude=[])
psm.logistic_ps(balance=True)

# print(psm.predicted_data)

psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
plt.clf()
psm.plot_match(Title='Side by side matched controls', Ylabel='Count', Xlabel= 'Propensity logit', names = ['NC', 'AD'], colors=['#E69F00', '#56B4E9'], save=True)
plt.clf()
psm.effect_size_plot(title='Standardized Mean differences across covariates before and after matching', before_color='#FCB754', after_color='#3EC8FB', save=True)
plt.clf()

# df_matched = psm.df_matched
# print(df_matched)
df_matched_ids = psm.matched_ids
# print(df_matched_ids)

CN_matched = df_matched_ids['matched_ID'].values.tolist()
# do not sort, so we can later on use the matching and not only the group
# CN_matched.sort()
with open('CN_matched.json', 'w') as outfile:
  json.dump(CN_matched, outfile, indent=2)

AD_matched = df_matched_ids['id'].values.tolist()
# do not sort, so we can later on use the matching and not only the group
# AD_matched.sort()
with open('AD_matched.json', 'w') as outfile:
  json.dump(AD_matched, outfile, indent=2)

CN_matched_ages = list(map(lambda e: image_ids_to_features[e]['age'], CN_matched))
AD_matched_ages = list(map(lambda e: image_ids_to_features[e]['age'], AD_matched))

plt.hist(CN_matched_ages, bins='fd', density=False, alpha=.5)
plt.hist(AD_matched_ages, bins='fd', density=False, alpha=.5)
plt.legend(['CN', 'AD'])
plt.xlabel('age')
plt.ylabel('count')
plt.savefig('age_hists.png')
plt.clf()

print('CN matched age min/max: ' + str(np.min(CN_matched_ages)) + '/' + str(np.max(CN_matched_ages)))
print('AD matched age min/max: ' + str(np.min(AD_matched_ages)) + '/' + str(np.max(AD_matched_ages)))

print('CN matched age median/IQR: ' + str(np.median(CN_matched_ages)) + '/' + str(np.percentile(CN_matched_ages, (25, 75))))
print('AD matched age median/IQR: ' + str(np.median(AD_matched_ages)) + '/' + str(np.percentile(AD_matched_ages, (25, 75))))

print('CN matched age mean/std: ' + str(np.mean(CN_matched_ages)) + '/' + str(np.std(CN_matched_ages)))
print('AD matched age mean/std: ' + str(np.mean(AD_matched_ages)) + '/' + str(np.std(AD_matched_ages)))

CN_matched_sexes = np.array(list(map(lambda e: image_ids_to_features[e]['sex'], CN_matched)))
AD_matched_sexes = np.array(list(map(lambda e: image_ids_to_features[e]['sex'], AD_matched)))

print('CN matched sex m/f: ' + str(np.count_nonzero(CN_matched_sexes == 1)) + '/' + str(np.count_nonzero(CN_matched_sexes == 2)))
print('AD matched sex m/f: ' + str(np.count_nonzero(AD_matched_sexes == 1)) + '/' + str(np.count_nonzero(AD_matched_sexes == 2)))

CN_matched_subjects = list(set(map(lambda e: image_ids_to_features[e]['subject_id'], CN_matched)))
AD_matched_subjects = list(set(map(lambda e: image_ids_to_features[e]['subject_id'], AD_matched)))

print('CN matched subjects/images: ' + str(len(CN_matched_subjects)) + '/' + str(len(CN_matched)))
print('AD matched subjects/images: ' + str(len(AD_matched_subjects)) + '/' + str(len(AD_matched)))

def reducer_sex(a, s):
  a[image_ids_to_features[s]['subject_id']] = image_ids_to_features[s]['sex']
  return a

CN_matched_subject_sexes = np.array(list(functools.reduce(reducer_sex, CN_matched, {}).values()))
AD_matched_subject_sexes = np.array(list(functools.reduce(reducer_sex, AD_matched, {}).values()))

print('CN subjects matched m/f: ' + str(np.count_nonzero(CN_matched_subject_sexes == 1)) + '/' + str(np.count_nonzero(CN_matched_subject_sexes == 2)))
print('AD subjects matched m/f: ' + str(np.count_nonzero(AD_matched_subject_sexes == 1)) + '/' + str(np.count_nonzero(AD_matched_subject_sexes == 2)))

def reducer_dementia_scores_list(a, s):
	return a if a else s

def reducer_cdr(a, s):
	a[s] = functools.reduce(reducer_dementia_scores_list, subject_ids_to_dementia_scores[s]['cdr'], '')
	return a

def reducer_mmse(a, s):
	a[s] = functools.reduce(reducer_dementia_scores_list, subject_ids_to_dementia_scores[s]['mmse'], '')
	return a

CN_matched_subject_cdr = np.array(list(functools.reduce(reducer_cdr, CN_matched_subjects, {}).values()))
AD_matched_subject_cdr = np.array(list(functools.reduce(reducer_cdr, AD_matched_subjects, {}).values()))

print('CN subjects matched cdr 0.0/0.5/1.0/1.5/2.0/empty: ' +\
	str(np.count_nonzero(CN_matched_subject_cdr == '0.0')) + '/' +\
	str(np.count_nonzero(CN_matched_subject_cdr == '0.5')) + '/' +\
	str(np.count_nonzero(CN_matched_subject_cdr == '1.0')) + '/' +\
	str(np.count_nonzero(CN_matched_subject_cdr == '1.5')) + '/' +\
	str(np.count_nonzero(CN_matched_subject_cdr == '2.0')) + '/' +\
	str(np.count_nonzero(CN_matched_subject_cdr == '')))
print('AD subjects matched cdr 0.0/0.5/1.0/1.5/2.0/empty: ' +\
	str(np.count_nonzero(AD_matched_subject_cdr == '0.0')) + '/' +\
	str(np.count_nonzero(AD_matched_subject_cdr == '0.5')) + '/' +\
	str(np.count_nonzero(AD_matched_subject_cdr == '1.0')) + '/' +\
	str(np.count_nonzero(AD_matched_subject_cdr == '1.5')) + '/' +\
	str(np.count_nonzero(AD_matched_subject_cdr == '2.0')) + '/' +\
	str(np.count_nonzero(AD_matched_subject_cdr == '')))

CN_matched_subject_mmse = np.array(list(filter(lambda x: x != '', functools.reduce(reducer_mmse, CN_matched_subjects, {}).values()))).astype(np.float32)
AD_matched_subject_mmse = np.array(list(filter(lambda x: x != '', functools.reduce(reducer_mmse, AD_matched_subjects, {}).values()))).astype(np.float32)

print('CN matched mmse min/max: ' + str(np.min(CN_matched_subject_mmse)) + '/' + str(np.max(CN_matched_subject_mmse)))
print('AD matched mmse min/max: ' + str(np.min(AD_matched_subject_mmse)) + '/' + str(np.max(AD_matched_subject_mmse)))

print('CN matched mmse median/IQR: ' + str(np.median(CN_matched_subject_mmse)) + '/' + str(np.percentile(CN_matched_subject_mmse, (25, 75))))
print('AD matched mmse median/IQR: ' + str(np.median(AD_matched_subject_mmse)) + '/' + str(np.percentile(AD_matched_subject_mmse, (25, 75))))

print('CN matched mmse mean/std: ' + str(np.mean(CN_matched_subject_mmse)) + '/' + str(np.std(CN_matched_subject_mmse)))
print('AD matched mmse mean/std: ' + str(np.mean(AD_matched_subject_mmse)) + '/' + str(np.std(AD_matched_subject_mmse)))

print('CN matched mmse no value count: ' + str(len(CN_matched_subjects) - len(CN_matched_subject_mmse)))
print('AD matched mmse no value count: ' + str(len(AD_matched_subjects) - len(AD_matched_subject_mmse)))
