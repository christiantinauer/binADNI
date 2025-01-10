import os
import json
import subprocess
from multiprocessing.pool import ThreadPool

PROCESSES = 11

with open('02_dcm2nii.json', 'r') as infile:
	image_descriptions = json.load(infile)

# test
# image_descriptions = [image_descriptions[0], image_descriptions[1]]

def run_dcm2niix(image_description, outfile):
	# create destination dir
	os.makedirs(image_description['destination'])
	
	# dcm2niix
	cmd_parts = [
		'dcm2niix',
		'-f',
		image_description['filename'],
		'-o',
		image_description['destination'],
		'-z',
		'y',
		'--terse',
		image_description['source'],
	]
	cmd = ' '.join(cmd_parts)
	print(cmd)
	return subprocess.run(cmd, stdout=outfile, shell=True)

outfile = open('03_protocols.txt', 'w')
pool = ThreadPool(PROCESSES)
results = [pool.apply_async(run_dcm2niix, (image_description, outfile)) for image_description in image_descriptions]

pool.close()
pool.join()

for result in results:
	print(result.get())

outfile.close()
