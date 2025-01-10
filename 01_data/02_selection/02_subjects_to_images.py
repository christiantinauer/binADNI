import csv
import json

selected_image_ids = []
with open('../01_gather/03_ADNI_selected.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		image_id = row[0]
		selected_image_ids.append(image_id)

with open('01_subject_ids.json', 'r') as infile:
	selected_subject_ids = json.load(infile)

download_image_ids = []
with open('../01_gather/00_idaSearch_11_30_2022.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		subject_id = row[0]
		image_id = row[20]

		if subject_id not in selected_subject_ids or image_id not in selected_image_ids:
			continue

		download_image_ids.append(image_id)

with open('01_image_ids.json', 'w') as outfile:
	json.dump(download_image_ids, outfile, indent=2)

with open('01_image_ids.txt', 'w') as outfile:
	outfile.write(','.join(download_image_ids))
