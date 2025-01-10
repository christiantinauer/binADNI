import json

with open('../03_matching/CN_matched.json', 'r') as infile:
	CN_image_ids = json.load(infile)

with open('../03_matching/AD_matched.json', 'r') as infile:
	AD_image_ids = json.load(infile)

with open('01_image_ids_for_download.txt', 'w') as outfile:
	outfile.write(','.join(CN_image_ids + AD_image_ids))