import requests
import csv

cookies = {
	"__utma": "174947263.2087854892.1667484998.1669807157.1669815366.5",
	"__utmb": "174947263.52.10.1669815366",
	"__utmc": "174947263",
	"__utmt": "1",
	"__utmz": "174947263.1669803216.3.3.utmcsr=adni.loni.usc.edu|utmccn=(referral)|utmcmd=referral|utmcct=/",
	"_ga": "GA1.2.2087854892.1667484998",
	"_gid": "GA1.2.1206103460.1669798653",
	"ADV_QUERY": "true",
	"ASSESSMENT_SECTION": "true",
	"https.ida.loni.usc": "134350858.64288.0000",
	"idaCookiePolicy": "true",
	"IMG_TYPE_ORIG_SECTION": "true",
	"IMG_TYPE_POST_PROCESS_SECTION": "false",
	"IMG_TYPE_PRE_PROCESS_SECTION": "false",
	"IS_FORWARD_SORT": "false",
	"JSESSIONID": "CF0E028D629D109E9500C5FE05555B2E",
	"MODALITY_SECTION": "true",
	"PROCESSING_SECTION": "false",
	"PROJECT_SECTION": "true",
	"PROJECT_SPECIFIC_SECTION": "true",
	"PROTOCOL_SECTION": "true",
	"QUALITY_SECTION": "false",
	"SORT_COLUMN": "0",
	"STATUS_SECTION": "true",
	"STUDY_VISIT_SECTION": "true",
	"SUBJECT_SECTION": "true"
}

with open('00_idaSearch_11_30_2022.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	# skip header
	next(reader, None)
	for row in reader:
		image_id = row[20]
		print(image_id)

		response = requests.get('https://ida.loni.usc.edu/pages/access/imageDetail/imageDetails.jsp?project=ADNI&imageId=' + image_id, cookies=cookies)
		with open('imaging_protocols/' + image_id + '.html', 'w') as outfile:
			outfile.write(response.text)
