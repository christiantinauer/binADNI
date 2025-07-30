import os

from nipype.interfaces.utility import IdentityInterface, Function, Select
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.fsl import Reorient2Std, BinaryMaths, UnaryMaths, Threshold, BET, FLIRT, ApplyXFM, FNIRT, ApplyWarp, ApplyMask, FIRST, ExtractROI, ConvertXFM, ErodeImage, DilateImage
from nipype.interfaces.fsl.maths import MathsCommand
from nipype.interfaces.ants import N4BiasFieldCorrection

from nipype_extensions import StandardSpaceRoi, PairReg

# get container
def datasink_container_from_identity(study, subject_id):
	return study + '/' + subject_id

def build_selectfiles_workflow(studies, select_templates, name='selectfiles'):
	# identities
	study_identity_node = Node(IdentityInterface(
		fields=['study'],
	), name=name + '__study_identity')
	study_identity_node.iterables = [
		('study', studies['study']),
	]

	subject_identity_node = Node(IdentityInterface(
		fields=['study', 'subject_id'],
	), name=name + '__subject_identity')
	subject_identity_node.itersource = (name + '__study_identity', 'study')
	subject_identity_node.iterables = [('subject_id', studies['subject_id'])]
	
	# select files
	selectfiles_node = Node(SelectFiles(
		select_templates,
		sort_filelist=True,
		force_lists=['T1'],
	), name=name + '__selectfiles')

	# datasink container
	datasink_container_node = Node(Function(
		input_names=['study', 'subject_id'],
		output_names=['container'],
		function=datasink_container_from_identity,
	), name=name + '__datasink_container')

	# select files subworkflow
	selectfiles_worklow = Workflow(name)
	selectfiles_worklow.connect([
		(study_identity_node, subject_identity_node,				[('study', 'study')]),
		(subject_identity_node, selectfiles_node,						[
																													('study', 'study'),
																													('subject_id', 'subject_id'),
																												]),
		(subject_identity_node, datasink_container_node,		[
																													('study', 'study'),
																													('subject_id', 'subject_id'),
																												]),
	])

	return selectfiles_worklow

def build_datasink_node(target_base_dir, workflow_name):
	return Node(DataSink(
		base_directory=target_base_dir,
		parameterization=False,
		substitutions=[
			('T1_1mm_brain_mask_nlin', ''),
		],
		regexp_substitutions=[
			# T1
			(r'T1/T1_1mm.M__[0-9]+_reoriented.nii.gz', 'T1.nii.gz'),
			(r'T1_bias_corrected/T1_1mm.M__[0-9]+_reoriented_maths.nii.gz', 'T1__bias_corrected.nii.gz'),
			(r'T1_bias_image/T1_1mm.M__[0-9]+_reoriented_ssroi_bias.nii.gz', 'T1__bias_estimation.nii.gz'),

			# T1 @ MNI
			(r'regT1_to_MNI_dof6/T1_1mm.M__[0-9]+_reoriented_maths_flirt.nii.gz', 'T1__bias_corrected_@_MNI152_1mm_dof6.nii.gz'),
			(r'regT1_to_MNI_nlin/T1_1mm.M__[0-9]+_reoriented_maths_warp.nii.gz', 'T1__bias_corrected_@_MNI152_1mm_nlin.nii.gz'),

			# T1 registration files
			(r'regMatrix_dof6/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_masked_flirt.mat', 'T1_to_MNI152_1mm_dof6.mat'),
			(r'regMatrix_dof12/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_masked_flirt.mat', 'T1_to_MNI152_1mm_dof12.mat'),
			(r'regFieldFile_nlin/T1_1mm.M__[0-9]+_reoriented_maths_field.nii.gz', 'T1_1mm_to_MNI152_warp.nii.gz'),
			(r'regFieldCoeffFile_nlin/T1_1mm.M__[0-9]+_reoriented_maths_fieldwarp.nii.gz', 'T1_to_MNI152_warpcoeff.nii.gz'),

			# # brain masks
			(r'stdmasked_brain_mask/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_mask_masked.nii.gz', 'T1__brain_mask.nii.gz'),
			(r'regBrainMask_to_MNI_dof6/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_masked_flirt_bin.nii.gz', 'T1__brain_mask_@_MNI152_1mm_dof6.nii.gz'),
			(r'regBrainMask_to_MNI_nlin/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_masked_warp_bin.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin.nii.gz'),

			# # eroded brain masks
			(r'regT1_erode/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_mask_masked_ero.nii.gz', 'T1__brain_mask_ero5.nii.gz'),
			(r'regT1_brainmask_erode_dof6/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_masked_flirt_bin_ero.nii.gz', 'T1__brain_mask_@_MNI152_1mm_dof6_ero5.nii.gz'),
			(r'regT1_brainmask_erode_nlin/T1_1mm.M__[0-9]+_reoriented_maths_ssroi_brain_masked_warp_bin_ero.nii.gz', 'T1__brain_mask_@_MNI152_1mm_nlin_ero5.nii.gz'),
			
			(r'T1_1mm_to_MNI152_warp.nii.gz', 'T1_to_MNI152_warp.nii.gz'),
		]
	), name=workflow_name + '__datasink')

def preprocess(studies, select_templates, target_base_dir, processor_count=1):
	# preprocess workflow
	preprocess_workflow = Workflow(
		name='preprocess',
		base_dir=os.path.join(target_base_dir, 'tmp')
	)

	# selectfiles subworkflow
	selectfiles_workflow = build_selectfiles_workflow(
		studies, select_templates,
	)

	# datasink
	datasink_node = build_datasink_node(target_base_dir, 'T1')

	# select first T1
	select_first_T1_node = Node(Select(
		index=[0],
	), name='T1__select_first_T1')

	# fslreorient2std <input> <output>
	reorient2std_node = Node(Reorient2Std(), name='T1__reorient2std')

	# # fslroi <input> <output> 0 160 0 240 0 256
	# roi_node = Node(ExtractROI(
	# 	x_min=0,
	# 	x_size=160,
	# 	y_min=0,
	# 	y_size=240,
	# 	z_min=0,
	# 	z_size=256,
	# ), name='T1__roi')

	# P: first T1
	preprocess_workflow.connect([
		(selectfiles_workflow, select_first_T1_node,	[('selectfiles__selectfiles.T1', 'inlist')]),
		(select_first_T1_node, reorient2std_node,			[('out', 'in_file')]),
		# (reorient2std_node, roi_node,									[('out_file', 'in_file')]),
		(selectfiles_workflow, datasink_node,					[('selectfiles__datasink_container.container', 'container')]),
	])

	# standard_space_roi <input> <output> -maskMASK /opt/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil -roiNONE
	initial_bet_node = Node(StandardSpaceRoi(
		mask_mask='/opt/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz',
		roi_none=True,
	), name='T1__initial_bet_node')

	# N4BiasFieldCorrection -d 3 -i <input> -o [ <output>, <bias_field> ] -r
	bias_field_estimation_node = Node(N4BiasFieldCorrection(
		output_image='T1_bias_corrected.nii.gz',
		rescale_intensities=True,
		save_bias=True,
	), name='T1__bias_field_estimation_node')

	# fslmaths <input> -div <bias_field> <output>
	bias_field_correction_node = Node(BinaryMaths(
		operation='div',
	), name='T1__bias_field_correction_node')

	# standard_space_roi <input> <output> -maskMASK /opt/fsl/data/standard/MNI152_T1_2mm -roiNONE
	no_neck_node = Node(StandardSpaceRoi(
		mask_mask='/opt/fsl/data/standard/MNI152_T1_2mm.nii.gz',
		roi_none=True,
	), name='T1__no_neck_node')

	# bet <input> <output> -m -s -f 0.5 -R
	bet_node = Node(BET(
		frac=0.5,
		mask=True,
		robust=True,
		skull=True,
	), name='T1__bet')

	# pairreg <ref_brain> <in_brain> <ref_skull> <in_skull> <output_matrix>
	pairreg_node = Node(PairReg(
		ref_brain_file='/opt/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz',
		ref_skull_file='/opt/fsl/data/standard/MNI152_T1_2mm_skull.nii.gz',
	), name='T1__pairreg')

	# convert_xfm -omat <output> -inverse <T1_to_Std>
	invert_node = Node(ConvertXFM(
		invert_xfm=True,
	), name='T1__invert_node')

	# flirt -applyxfm -in <input> -ref <reference> -init <dof6_matrix> -out <output>
	apply_std_brain_mask_to_T1 = Node(FLIRT(
		apply_xfm=True,
		in_file='/opt/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz',
	), name='T1__apply_std_brain_mask_to_T1')

	# fslmaths <input> -thr 0.5 <output>
	threshold_brain_mask_node = Node(Threshold(
		thresh=.5,
	), name='T1__threshold_brain_mask_node')

	# fslmaths <input> -bin <output>
	final_brain_mask_node = Node(UnaryMaths(
		operation='bin',
	), name='T1__final_brain_mask_node')

	# fslmaths <input> -mas <mask> <output>
	correct_final_brain_mask_node = Node(ApplyMask(
	), name='T1__correct_final_brain_mask_node')

	# fslmaths <input> -mas <mask> <output>
	apply_final_brain_mask_node = Node(ApplyMask(
	), name='T1__apply_final_brain_mask_node')

	# P: bias field correction and brain extraction
	# we use N4 instead of N3
	# based on FSL SIENAX https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/SIENA/UserGuide
	preprocess_workflow.connect([
		# bias field correction
		(reorient2std_node, initial_bet_node,														[('out_file', 'in_file')]),
		(initial_bet_node, bias_field_estimation_node,									[('out_file', 'input_image')]),
		(reorient2std_node, bias_field_correction_node,									[('out_file', 'in_file')]),
		(bias_field_estimation_node, bias_field_correction_node,				[('bias_image', 'operand_file')]),
		
		# bet
		(bias_field_correction_node, no_neck_node,											[('out_file', 'in_file')]),
		(no_neck_node, bet_node,																				[('out_file', 'in_file')]),
		(bet_node, pairreg_node,																				[('out_file', 'in_brain_file'), ('skull_file', 'in_skull_file')]),
		(pairreg_node, invert_node,																		 	[('out_matrix_file', 'in_file')]),

		# standard brain mask bet
		(bet_node, apply_std_brain_mask_to_T1,													[('out_file', 'reference')]),
		(invert_node, apply_std_brain_mask_to_T1,											 	[('out_file', 'in_matrix_file')]),
		(apply_std_brain_mask_to_T1, threshold_brain_mask_node,				 	[('out_file', 'in_file')]),
		(threshold_brain_mask_node, final_brain_mask_node,							[('out_file', 'in_file')]),
		(final_brain_mask_node, correct_final_brain_mask_node,					[('out_file', 'mask_file')]),
		(bet_node, correct_final_brain_mask_node,											 	[('mask_file', 'in_file')]),
		(correct_final_brain_mask_node, apply_final_brain_mask_node,		[('out_file', 'mask_file')]),
		(bet_node, apply_final_brain_mask_node,												 	[('out_file', 'in_file')]),

		# datasink
		(reorient2std_node, datasink_node,													 		[('out_file', 'T1')]),
		# (initial_bet_node, datasink_node,																[('out_file', 'T1_initial_bet')]),
		(bias_field_estimation_node, datasink_node,										 	[('bias_image', 'T1_bias_image')]),
		(bias_field_correction_node, datasink_node,										 	[('out_file', 'T1_bias_corrected')]),
		# (bet_node, datasink_node,																				[('out_file', 'T1_brain')]),
		# (bet_node, datasink_node,																				[('mask_file', 'T1_brain_mask')]),
		# (pairreg_node, datasink_node,																		[('out_matrix_file', 'T1_pairreg')]),
		(correct_final_brain_mask_node, datasink_node,									[('out_file', 'stdmasked_brain_mask')]),		
		# (apply_final_brain_mask_node, datasink_node,										[('out_file', 'stdmasked_brain')]),
	])

	# flirt -cost corratio -dof 6 -in <input> -ref <reference> -out <output> -omat <outputmatrix>
	reg_to_MNI_dof6_node = Node(FLIRT(
		cost='corratio',
		dof=6,
		reference='/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
	), name='T1__reg_to_MNI_dof6')

	# flirt -cost corratio -dof 12 -in <input> -ref <reference> -out <output> -omat <outputmatrix>
	reg_to_MNI_dof12_node = Node(FLIRT(
		cost='corratio',
		dof=12,
		reference='/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
	), name='T1__reg_to_MNI_dof12')

	# fnirt --in=<input> --ref=<reference> --config=T1_2_MNI152_2mm --aff=<dof12_matrix> --iout=<output> --fout=<field_output>
	reg_to_MNI_nlin_node = Node(FNIRT(
		config_file='T1_2_MNI152_2mm',
		field_file=True,
		fieldcoeff_file=True,
		ref_file='/opt/fsl/data/standard/MNI152_T1_2mm.nii.gz',
	), name='T1__reg_to_MNI_nlin')

	# applywarp -i <input> -o <output> -r <reference> -w <field_output>
	apply_to_MNI_nlin_node = Node(ApplyWarp(
		ref_file='/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz',
	), name='T1__apply_to_MNI_nlin')

	# fslmaths <input> -bin <output>
	final_dof6_brain_mask_node = Node(UnaryMaths(
		operation='bin',
	), name='T1__final_dof6_brain_mask')

	# fslmaths <input> -bin <output>
	final_nlin_brain_mask_node = Node(UnaryMaths(
		operation='bin',
	), name='T1__final_nlin_brain_mask')

	# P: reg T1 to MNI with dof6, dof12 and nlin
	preprocess_workflow.connect([
		# reg to MNI
		(apply_final_brain_mask_node, reg_to_MNI_dof6_node,		 	[('out_file', 'in_file')]),
		(apply_final_brain_mask_node, reg_to_MNI_dof12_node,		[('out_file', 'in_file')]),
		(reg_to_MNI_dof12_node, reg_to_MNI_nlin_node,					 	[('out_matrix_file', 'affine_file')]),
		(bias_field_correction_node, reg_to_MNI_nlin_node,			[('out_file', 'in_file')]),
		(apply_final_brain_mask_node, apply_to_MNI_nlin_node,		[('out_file', 'in_file')]),
		(reg_to_MNI_nlin_node, apply_to_MNI_nlin_node,					[('field_file', 'field_file')]),

		# dof6, nlin brain mask
		(reg_to_MNI_dof6_node, final_dof6_brain_mask_node,			[('out_file', 'in_file')]),
		(apply_to_MNI_nlin_node, final_nlin_brain_mask_node,		[('out_file', 'in_file')]),

		# datasink
		(selectfiles_workflow, datasink_node,									 	[('selectfiles__selectfiles.MNI152_1mm_brain_mask', 'T1_1mm_brain_mask_nlin')]),
		# (reg_to_MNI_dof6_node, datasink_node,										[('out_file', 'reg_dof6')]),
		(reg_to_MNI_dof6_node, datasink_node,									 	[('out_matrix_file', 'regMatrix_dof6')]),
		# (reg_to_MNI_dof12_node, datasink_node,									[('out_file', 'reg_dof12')]),
		(reg_to_MNI_dof12_node, datasink_node,									[('out_matrix_file', 'regMatrix_dof12')]),
		# (reg_to_MNI_nlin_node, datasink_node,									 	[('warped_file', 'reg_nlin')]),
		(reg_to_MNI_nlin_node, datasink_node,									 	[('field_file', 'regFieldFile_nlin')]),
		(reg_to_MNI_nlin_node, datasink_node,									 	[('fieldcoeff_file', 'regFieldCoeffFile_nlin')]),
		# (apply_to_MNI_nlin_node, datasink_node,									[('out_file', 'regT1_nlin_1mm')]),
		(final_dof6_brain_mask_node, datasink_node,						 	[('out_file', 'regBrainMask_to_MNI_dof6')]),
		(final_nlin_brain_mask_node, datasink_node,						 	[('out_file', 'regBrainMask_to_MNI_nlin')]),
	])	

	# fslmath -ero -kernel boxv 5
	erode_node = Node(ErodeImage(
		kernel_shape='boxv',
		kernel_size=5,
	), name='T1__erode')

	# fslmath -ero -kernel boxv 5
	erode_dof6_node = Node(ErodeImage(
		kernel_shape='boxv',
		kernel_size=5,
	), name='T1__erode_dof6')

	# fslmath -ero -kernel boxv 5
	erode_nlin_node = Node(ErodeImage(
		kernel_shape='boxv',
		kernel_size=5,
	), name='T1__erode_nlin')

	# P: erode brain masks
	preprocess_workflow.connect([
		(correct_final_brain_mask_node, erode_node,			[('out_file', 'in_file')]),
		(final_dof6_brain_mask_node, erode_dof6_node,		[('out_file', 'in_file')]),
		(final_nlin_brain_mask_node, erode_nlin_node,		[('out_file', 'in_file')]),

		(erode_node, datasink_node,											[('out_file', 'regT1_erode')]),
		(erode_dof6_node, datasink_node,								[('out_file', 'regT1_brainmask_erode_dof6')]),
		(erode_nlin_node, datasink_node,								[('out_file', 'regT1_brainmask_erode_nlin')]),
	])

	# flirt -applyxfm -in <input> -ref <reference> -init <dof6_matrix> -out <output>
	apply_T1_to_MNI_dof6_node = Node(ApplyXFM(
		apply_xfm=True,
		reference='/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
	), name='T1__apply_T1_to_MNI_dof6')

	# applywarp -i <input> -o <output> -r <reference> -w <field_output>
	apply_T1_to_MNI_nlin_node = Node(ApplyWarp(
		ref_file='/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz',
	), name='T1__apply_T1_to_MNI_nlin')

	# P: transform corrected T1 to MNI dof6 and nlin
	preprocess_workflow.connect([		
		(bias_field_correction_node, apply_T1_to_MNI_dof6_node,										[('out_file', 'in_file')]),
		(reg_to_MNI_dof6_node, apply_T1_to_MNI_dof6_node,													[('out_matrix_file', 'in_matrix_file')]),

		(bias_field_correction_node, apply_T1_to_MNI_nlin_node,										[('out_file', 'in_file')]),
		(reg_to_MNI_nlin_node, apply_T1_to_MNI_nlin_node,													[('field_file', 'field_file')]),

		# datasink
		(apply_T1_to_MNI_dof6_node, datasink_node,																[('out_file', 'regT1_to_MNI_dof6')]),
		(apply_T1_to_MNI_nlin_node, datasink_node,																[('out_file', 'regT1_to_MNI_nlin')]),
	])

	preprocess_workflow.write_graph(dotfilename='./graphs/preprocess.dot', graph2use='orig', simple_form=True)
	preprocess_workflow.run('MultiProc', plugin_args={'n_procs': processor_count})
