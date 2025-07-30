from nipype.interfaces.fsl.base import	FSLCommandInputSpec, FSLCommand
from nipype.interfaces.base import traits, File, TraitedSpec

class StandardSpaceRoiInputSpec(FSLCommandInputSpec):
	'''Defines inputs (trait classes) for standard_space_roi'''

	in_file = File(
		exists=True,
		desc='input filename',
		argstr='%s',
		position=0,
		mandatory=True
	)
	out_file = File(
		desc='output filename',
		argstr='%s',
		position=1,
		name_source=['in_file'],
		hash_files=False,
		name_template='%s_ssroi',
	)
	_xor_mask = (
		'mask_fov',
		'mask_mask',
		'mask_none',
	)
	mask_fov = traits.Bool(
		desc='mask output using transformed standard space FOV (default)',
		argstr='-maskFOV',
		xor=_xor_mask,
	)
	mask_mask = File(
		desc='mask output using transformed standard space mask',
		argstr='-maskMASK %s',
		xor=_xor_mask,
	)
	mask_none = traits.Bool(
		desc='do not mask output',
		argstr='-maskNONE',
		xor=_xor_mask,
	)
	_xor_roi = (
		'roi_fov',
		'roi_mask',
		'roi_none',
	)
	roi_fov = traits.Bool(
		desc='cut down input FOV using bounding box of the transformed standard space FOV (default)',
		argstr='-roiFOV',
		xor=_xor_roi,
	)
	roi_mask = File(
		desc='cut down input FOV using nonbackground bounding box of the transformed standard space mask',
		argstr='-roiMASK %s',
		xor=_xor_roi,
	)
	roi_none = traits.Bool(
		desc='do not cut down input FOV',
		argstr='-roiNONE',
		xor=_xor_roi,
	)
	ssref_file = File(
		desc='standard space reference image to use (default /opt/fsl/data/standard/MNI152_T1)',
		argstr='-ssref %s',
	)
	alt_in_file = File(
		desc='alternative input image to apply the ROI to (instead of the one used to register to the reference)',
		argstr='-altinput %s',
		xor=_xor_roi,
	)
	no_cleanup = traits.Bool(
		desc='debug (don\'t delete intermediate files)',
		argstr='-d',
	)
	flirt_options=traits.Str(
		desc='flirt options',
		argstr='%s',
		position=-1,
	)

class StandardSpaceRoiOutputSpec(TraitedSpec):
	out_file = File(
		exists=True,
		desc='path/name of out file',
	)

class StandardSpaceRoi(FSLCommand):
	'''This masks the input and/or reduces its FOV, on the basis of a
	standard space image or mask, that is transformed into the space
	of the input image.
	'''

	_cmd = 'standard_space_roi'
	input_spec = StandardSpaceRoiInputSpec
	output_spec = StandardSpaceRoiOutputSpec

class PairRegInputSpec(FSLCommandInputSpec):
	'''Defines inputs (trait classes) for pairreg'''

	ref_brain_file = File(
		exists=True,
		desc='ref brain filename',
		argstr='%s',
		position=0,
		mandatory=True
	)
	in_brain_file = File(
		exists=True,
		desc='in brain filename',
		argstr='%s',
		position=1,
		mandatory=True
	)
	ref_skull_file = File(
		exists=True,
		desc='ref skull filename',
		argstr='%s',
		position=2,
		mandatory=True
	)
	in_skull_file = File(
		desc='in skull filename',
		argstr='%s',
		position=3,
		mandatory=True
	)
	out_matrix_file = File(
		desc='output matrix filename',
		argstr='%s',
		position=4,
		name_source=['ref_brain_file'],
		keep_extension=True,
		name_template='2%s.mat',
		hash_files=False,
	)
	flirt_options=traits.Str(
		desc='flirt options',
		argstr='%s',
		position=5,
	)

class PairRegOutputSpec(TraitedSpec):
	out_matrix_file = File(
		exists=True,
		desc='path/name of out matrix file',
	)

class PairReg(FSLCommand):
	'''pairreg - perform registration of pairs keeping skull scaling const'''

	_cmd = 'pairreg'
	input_spec = PairRegInputSpec
	output_spec = PairRegOutputSpec
