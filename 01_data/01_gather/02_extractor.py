import csv

rows = []

with open('00_idaSearch_11_30_2022.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
  # skip header
	row = next(reader, None)
	if row is not None:
		row.append('Imaging Protocol Details')
	rows.append(row)

	for row in reader:
		image_id = row[20]

		with open('imaging_protocols/' + image_id + '.html', 'r') as htmlfile:
			lines = htmlfile.readlines()
			imaging_protocol = lines[1510].strip()
			row.append(imaging_protocol)
			rows.append(row)

with open('02_idaSearch_11_30_2022_with_details.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
	writer.writerows(rows)
