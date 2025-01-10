import os
import csv
import json

data_base_path = '/mnt/neuro/nas2/Work/binADNI/ADNI_download'
data_rearranged_base_path = '/mnt/neuro/nas2/Work/binADNI/ADNI_rearranged'

dcm_2_nii = []

image_id_to_group = {}
groups = {
	'CN': [],
	'AD': [],
}

# with open('01_binADNI_2_01_2024.csv', newline='') as csvfile:
with open('../01_gather/00_idaSearch_11_30_2022.csv', newline='') as csvfile:
	reader = csv.reader(csvfile)
	# skip header
	next(reader, None)
	for row in reader:
		group = row[5] # row[2]	
		image_data_id = 'I' + row[-1] # row[0]
		image_id_to_group[image_data_id] = group

for split_folder in [
	'binADNI',
	'binADNI1',
	'binADNI2',
	'binADNI3',
	'binADNI4',
	'binADNI5',
	'binADNI6',
	'binADNI7',
	'binADNI8',
	'binADNI9',
]:
	subjects_path = os.path.join(data_base_path, split_folder, 'ADNI')
	for subject_folder in os.listdir(subjects_path):
		subject_path = os.path.join(subjects_path, subject_folder)
		for sequence_folder in os.listdir(subject_path):
			sequence_path = os.path.join(subject_path, sequence_folder)
			for date_folder in os.listdir(sequence_path):
				date_path = os.path.join(sequence_path, date_folder)
				for image_folder in os.listdir(date_path):
					image_path = os.path.join(date_path, image_folder)
					destination_subject_folder = subject_folder + '__' + date_folder + '__' + image_folder
					group = image_id_to_group[image_folder]
					groups[group].append(destination_subject_folder)
					dcm_2_nii.append({
						'source': image_path,
						'destination': os.path.join(data_rearranged_base_path, group, destination_subject_folder),
						'filename': 'T1_raw'
					})

# print(len(dcm_2_nii))

groups['CN'].sort()
groups['AD'].sort()

with open('02_dcm2nii.json', 'w') as outfile:
	json.dump(dcm_2_nii, outfile, indent=2)

with open('02_CN.json', 'w') as outfile:
	json.dump(groups['CN'], outfile, indent=2)

with open('02_AD.json', 'w') as outfile:
	json.dump(groups['AD'], outfile, indent=2)
