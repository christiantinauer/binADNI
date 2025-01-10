import csv
import json

# ADNI2
# DXCHANGE: 
# 1=Stable:NL to NL,
# 2=Stable:MCI to MCI,
# 3=Stable:AD to AD,
# 4=Conv:NL to MCI,
# 5=Conv:MCI to AD,
# 6=Conv:NL to AD,
# 7=Rev:MCI to NL,
# 8=Rev:AD to MCI,
# 9=Rev:AD to NL.

subjects =	{}
subjects_ids_forth_and_back = []
subjects_ids_back_and_forth = []
with open('00_DXSUM_PDXCONV_ADNIALL.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		subject_id = row[3]
		dxchange_string = row[10]
		if dxchange_string == '':
				continue

		dxchange = int(dxchange_string)
		
		# stable
		if dxchange <=3:
				continue

		if subject_id not in subjects:
				dxchanges = []
				subjects[subject_id] = dxchanges
		else:
				dxchanges = subjects[subject_id]

		if dxchange not in dxchanges:
				dxchanges.append(dxchange)
				if len(dxchanges) > 1:
						if	dxchanges[-1] >= 7 and \
								dxchanges[-2] >= 4 and dxchanges[-2] <= 6:
								subjects_ids_forth_and_back.append(subject_id)
						elif	dxchanges[-2] >= 7 and \
									dxchanges[-1] >= 4 and dxchanges[-1] <= 6:
								subjects_ids_back_and_forth.append(subject_id)

with open('01_subjects_changes.json', 'w') as outfile:
	json.dump(subjects, outfile, indent=2)

subjects_forth_and_back = {}
for subject_id in subjects_ids_forth_and_back:
		subjects_forth_and_back[subject_id] = subjects[subject_id]
with open('01_subjects_forth_and_back.json', 'w') as outfile:
	json.dump(subjects_forth_and_back, outfile, indent=2)

subjects_back_and_forth = {}
for subject_id in subjects_ids_back_and_forth:
		subjects_back_and_forth[subject_id] = subjects[subject_id]
with open('01_subjects_back_and_forth.json', 'w') as outfile:
	json.dump(subjects_back_and_forth, outfile, indent=2)

# distinct ids
excluded_subject_ids = list(set(subjects_ids_forth_and_back + subjects_ids_back_and_forth))
excluded_subject_ids.sort()
with open('01_excluded_subject_ids.json', 'w') as outfile:
	json.dump(excluded_subject_ids, outfile, indent=2)

# load the subject ids which have been selected via image ids from pivot ADNI2
with open('../01_gather/04_subject_ids.json', 'r') as infile:
	subject_ids = json.load(infile)

filterd_subject_ids = list(set(subject_ids) - set(excluded_subject_ids))
filterd_subject_ids.sort()
with open('01_subject_ids.json', 'w') as outfile:
	json.dump(filterd_subject_ids, outfile, indent=2)
