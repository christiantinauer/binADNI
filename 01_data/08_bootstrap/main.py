import json
import math
from sklearn.utils import resample

with open('../04_rearrange/02_CN.json', 'r') as infile:
	CN = json.load(infile)

with open('../04_rearrange/02_AD.json', 'r') as infile:
	AD = json.load(infile)

# excludes
with open('../06_images_sanity/CN_excludes_preprocess_crashes.json', 'r') as infile:
	CN_excludes_preprocess_crashes = json.load(infile)

with open('../06_images_sanity/AD_excludes_preprocess_crashes.json', 'r') as infile:
	AD_excludes_preprocess_crashes = json.load(infile)

with open('../06_images_sanity/CN_excludes_image_sanity_checks.json', 'r') as infile:
	CN_excludes_image_sanity_checks = json.load(infile)

with open('../06_images_sanity/AD_excludes_image_sanity_checks.json', 'r') as infile:
	AD_excludes_image_sanity_checks = json.load(infile)

# ex excludes
CN_ex_excludes = set(CN) - set(CN_excludes_preprocess_crashes) - set(CN_excludes_image_sanity_checks)
AD_ex_excludes = set(AD) - set(AD_excludes_preprocess_crashes) - set(AD_excludes_image_sanity_checks)

# distinct subject ids
CN_ids_only = list(set(map(lambda e: e.split('__')[0], CN_ex_excludes)))
CN_ids_only.sort()
CN_samples = len(CN_ids_only)

AD_ids_only = list(set(map(lambda e: e.split('__')[0], AD_ex_excludes)))
AD_ids_only.sort()
AD_samples = len(AD_ids_only)

CN_training_samples = math.ceil(CN_samples * .7)
print('CN training ' + str(CN_training_samples))
CN_validation_samples = math.floor(CN_samples * .15)
print('CN validation ' + str(CN_validation_samples))
CN_test_samples = CN_samples - CN_training_samples - CN_validation_samples
print('CN test ' + str(CN_test_samples))

AD_training_samples = math.ceil(AD_samples * .7)
print('AD training ' + str(AD_training_samples))
AD_validation_samples = math.floor(AD_samples * .15)
print('AD validation ' + str(AD_validation_samples))
AD_test_samples = AD_samples - AD_training_samples - AD_validation_samples
print('AD test ' + str(AD_test_samples))

CN_bootstraps = []
AD_bootstraps = []

with open('random_seeds.json', 'r') as infile:
	random_seeds = json.load(infile)

for random_state in random_seeds:
	print('random seed ' + str(random_state))

	# CN
	# train
	CN_ids_only_train = resample(CN_ids_only, replace=False, n_samples=CN_training_samples, random_state=random_state)
	CN_ids_only_oob = list(set(CN_ids_only) - set(CN_ids_only_train))
	CN_ids_only_oob.sort()

	# validation
	CN_ids_only_validation = resample(CN_ids_only_oob, replace=False, n_samples=CN_validation_samples, random_state=random_state)
	CN_ids_only_oob = list(set(CN_ids_only_oob) - set(CN_ids_only_validation))

	# test
	CN_ids_only_test = CN_ids_only_oob

	CN_train = []
	CN_ids_only_train.sort()
	for CN_id in CN_ids_only_train:
		for CN_subject in CN:
			if CN_subject.startswith(CN_id):
				CN_train.append(CN_subject)

	CN_validation = []
	CN_ids_only_validation.sort()
	for CN_id in CN_ids_only_validation:
		for CN_subject in CN:
			if CN_subject.startswith(CN_id):
				CN_validation.append(CN_subject)

	CN_test = []
	CN_ids_only_test.sort()
	for CN_id in CN_ids_only_test:
		for CN_subject in CN:
			if CN_subject.startswith(CN_id):
				CN_test.append(CN_subject)

	# sanity check
	print('CN training images ' + str(len(CN_train)) + ', CN validation images ' + str(len(CN_validation)) + ', CN test images ' + str(len(CN_test)))
	# print(len(np.unique(CN_train + CN_validation + CN_test)))
	# print(len(CN_train) + len(CN_validation) + len(CN_test))
	CN_bootstraps.append({ 'training': CN_train, 'validation': CN_validation, 'test': CN_test })
	
	# AD
	# train
	AD_ids_only_train = resample(AD_ids_only, replace=False, n_samples=AD_training_samples, random_state=random_state)
	AD_ids_only_oob = list(set(AD_ids_only) - set(AD_ids_only_train))
	AD_ids_only_oob.sort()

	# validation
	AD_ids_only_validation = resample(AD_ids_only_oob, replace=False, n_samples=AD_validation_samples, random_state=random_state)
	AD_ids_only_oob = list(set(AD_ids_only_oob) - set(AD_ids_only_validation))

	# test
	AD_ids_only_test = AD_ids_only_oob

	AD_train = []
	AD_ids_only_train.sort()
	for AD_id in AD_ids_only_train:
		for AD_subject in AD:
			if AD_subject.startswith(AD_id):
				AD_train.append(AD_subject)

	AD_validation = []
	AD_ids_only_validation.sort()
	for AD_id in AD_ids_only_validation:
		for AD_subject in AD:
			if AD_subject.startswith(AD_id):
				AD_validation.append(AD_subject)

	AD_test = []
	AD_ids_only_test.sort()
	for AD_id in AD_ids_only_test:
		for AD_subject in AD:
			if AD_subject.startswith(AD_id):
				AD_test.append(AD_subject)

	# sanity check
	print('AD training images ' + str(len(AD_train)) + ', AD validation images ' + str(len(AD_validation)) + ', AD test images ' + str(len(AD_test)))
	# print(len(np.unique(AD_train + AD_validation + AD_test)))
	# print(len(AD_train) + len(AD_validation) + len(AD_test))
	AD_bootstraps.append({ 'training': AD_train, 'validation': AD_validation, 'test': AD_test })

with open('CN_bootstrap.json', 'w') as outfile:
	json.dump(CN_bootstraps, outfile, indent=2)

with open('AD_bootstrap.json', 'w') as outfile:
	json.dump(AD_bootstraps, outfile, indent=2)
