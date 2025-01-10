import os

from studies import get_studies
from workflow import preprocess

STUDIES_BASE_PATH = '/mnt/neuro/nas2/Work/binADNI/ADNI_rearranged/'

preprocess(
	get_studies(),
	{
		'T1': STUDIES_BASE_PATH + '{study}/{subject_id}/T1_raw.nii.gz',
		'MNI152_1mm_brain_mask': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MNI152_T1_1mm_brain_mask_dil_ero7.nii.gz'),
	},
	'/mnt/neuro/nas2/Work/binADNI_preprocessed',
	11
)
