import json

with open('../01_gather/02_CN.json', 'r') as infile:
	CN = json.load(infile)

with open('../01_gather/02_AD.json', 'r') as infile:
	AD = json.load(infile)

# excludes
with open('../03_images_sanity/ASPSF_excludes_preprocess_crashes.json', 'r') as infile:
	CN_excludes_preprocess_crashes = json.load(infile)

with open('../03_images_sanity/ASPSF_excludes_preprocess_crashes.json', 'r') as infile:
	AD_excludes_preprocess_crashes = json.load(infile)

with open('../03_images_sanity/ProDem_excludes_image_sanity_checks.json', 'r') as infile:
	CN_excludes_image_sanity_checks = json.load(infile)

with open('../03_images_sanity/ProDem_excludes_image_sanity_checks.json', 'r') as infile:
	AD_excludes_image_sanity_checks = json.load(infile)

# ex excludes
CN_ex_excludes = set(CN) - set(CN_excludes_preprocess_crashes) - set(CN_excludes_image_sanity_checks)
AD_ex_excludes = set(AD) - set(AD_excludes_preprocess_crashes) - set(AD_excludes_image_sanity_checks)

CN_bootstraps = []
AD_bootstraps = []

CN_test = list(CN_ex_excludes)
CN_test.sort()

AD_test = list(AD_ex_excludes)
AD_test.sort()

CN_bootstraps.append({ 'training': [], 'validation': [], 'test': CN_test })
AD_bootstraps.append({ 'training': [], 'validation': [], 'test': AD_test })

with open('CN_sampling.json', 'w') as outfile:
	json.dump(CN_bootstraps, outfile, indent=2)

with open('AD_sampling.json', 'w') as outfile:
	json.dump(AD_bootstraps, outfile, indent=2)
