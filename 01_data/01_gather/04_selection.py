import csv
import json

image_id_to_subject_id = {}
with open('00_idaSearch_11_30_2022.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		subject_id = row[0]
		image_id = row[20]
		image_id_to_subject_id[image_id] = subject_id

subject_ids =  []
with open('03_ADNI_selected.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		image_id = row[0]
		subject_ids.append(image_id_to_subject_id[image_id])

# distinct
subject_ids = list(set(subject_ids))
subject_ids.sort()
with open('04_subject_ids.json', 'w') as outfile:
	json.dump(subject_ids, outfile, indent=2)
