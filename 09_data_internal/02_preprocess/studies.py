import json

def get_studies():
	with open('../01_gather/02_CN.json', 'r') as infile:
		CN = json.load(infile)
		# CN = [CN[0], CN[1]]

	with open('../01_gather/02_AD.json', 'r') as infile:
		AD = json.load(infile)
		# AD = [AD[0], AD[1]]

	return {
		'study': [
			'ASPSF',
			'ProDem',
		],
		'subject_id': {
			'ASPSF': CN,
			'ProDem': AD,
		},
	}
