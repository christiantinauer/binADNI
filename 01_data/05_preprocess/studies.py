import json

def get_studies():
	with open('../04_rearrange/02_CN.json', 'r') as infile:
		CN = json.load(infile)
		# CN = [CN[0], CN[1]]

	with open('../04_rearrange/02_AD.json', 'r') as infile:
		AD = json.load(infile)
		# AD = [AD[0], AD[1]]

	return {
		'study': [
			'CN',
			'AD',
		],
		'subject_id': {
			'CN': CN,
			'AD': AD,
		},
	}
