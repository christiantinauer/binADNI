import os

import torch

from model import Model
from dataset import NiftiDataset
from constraints import BiasNegConstraint
from callbacks import ReduceLRAndResetStateDictOnPlateau
from losses import RelevanceGuidedLoss

def training(last_saved_model, bootstrap_index, initial_weights_index, start_epoch, training_config, training_data_paths, validation_data_paths, device, epochs, batch_size, final_params_destination):
	(lr, rf, image_name, mask_name, use_bet, use_normalize, binarizer, use_rg, input_shape, first_linear_in_features, dtd_unflattened_dim, _, initial_weights) = training_config

	# model
	model = Model(first_linear_in_features=first_linear_in_features, with_softmax=False, with_dtd=use_rg, dtd_unflattened_dim=dtd_unflattened_dim)
	model = model.to(device)

	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# load states
	if last_saved_model is not None:
		loaded_state = torch.load(last_saved_model)
		model_state_dict = loaded_state['model_state_dict']
		model.load_state_dict(model_state_dict)
		optimizer_state_dict = loaded_state['optimizer_state_dict']
		optimizer.load_state_dict(optimizer_state_dict)
	else:
		model_state_dict = torch.load(initial_weights.format(initial_weights_index))
		model.load_state_dict(model_state_dict, strict=False)
	
	# datasets
	training_set = NiftiDataset(
		device,
		training_data_paths,
		[
			image_name,
		],
		[
			mask_name,
		],
		input_shape,
		use_mask_on_input=use_bet,
		use_normalize=use_normalize,
		binarizer=binarizer,
		output_only_categories=not use_rg,
	)

	validation_set = NiftiDataset(
		device,
		validation_data_paths,
		[
			image_name,
		],
		[
			mask_name,
		],
		input_shape,
		use_mask_on_input=use_bet,
		use_normalize=use_normalize,
		binarizer=binarizer,
		output_only_categories=not use_rg,
	)

	# constraints
	biasNegConstraint = BiasNegConstraint()

	# losses
	ce_loss_fn = torch.nn.CrossEntropyLoss()
	# rg_loss_fn = RelevanceGuidedLoss()

	def train_loop(dataloader):
		model.train()
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		training_loss, ce_training_loss, correct = 0, 0, 0

		# Here, we use enumerate(training_loader) instead of
		# iter(training_loader) so that we can track the batch
		# index and do some intra-epoch reporting
		for batch, (inputs, labels) in enumerate(dataloader):
			inputs = inputs.to(device, non_blocking=True)
			# masks = masks.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			# Prediction and loss
			pred = model(inputs)
			ce_loss = ce_loss_fn(pred, labels)
			# rg_loss = rg_loss_fn(heatmaps, masks)
			loss = ce_loss #+ rg_loss

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Constraints
			model.apply(biasNegConstraint)

			# Losses
			training_loss += loss.item()
			ce_training_loss += ce_loss.item()
			# rg_training_loss += rg_loss.item()
			correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

			if batch % 5 == 0:
				loss, current = loss.item(), (batch + 1) * len(inputs)
				print(f'loss: {loss:>7f}, ce_loss: {ce_loss:>7f}	[{current:>3d}/{size:>3d}]')
	
		training_loss /= num_batches
		ce_training_loss /= num_batches
		# rg_training_loss /= num_batches
		correct /= size
		return training_loss, ce_training_loss, correct

	def validation_loop(dataloader):
		# Set the model to evaluation mode - important for batch normalization and dropout layers
		model.eval()
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		validation_loss, ce_validation_loss, correct = 0, 0, 0

		# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
		# also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
		with torch.no_grad():
			for inputs, labels in dataloader:
				inputs = inputs.to(device, non_blocking=True)
				# masks = masks.to(device, non_blocking=True)
				labels = labels.to(device, non_blocking=True)

				pred = model(inputs)
				ce_loss = ce_loss_fn(pred, labels)
				# rg_loss = rg_loss_fn(heatmaps, masks)
				loss = ce_loss # + rg_loss

				validation_loss += loss.item()
				ce_validation_loss += ce_loss.item()
				# rg_validation_loss += rg_loss.item()
				correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

		validation_loss /= num_batches
		ce_validation_loss /= num_batches
		# rg_validation_loss /= num_batches
		correct /= size
		return validation_loss, ce_validation_loss, correct

	train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)
	validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)

	callback = ReduceLRAndResetStateDictOnPlateau(
		factor=rf,
		patience=5,
		min_delta=1e-4,
		cooldown=0,
		min_lr=1e-6,
		verbose=1,
	)

	callback.on_train_begin(model)
	for t in range(start_epoch, epochs):
		print(f'Epoch {t+1}\n-------------------------------')
		training_loss, training_ce_loss, training_correct = train_loop(train_loader)
		print(f'Training Error: \n Accuracy: {(100*training_correct):>0.1f}%, Avg loss: {training_loss:>8f}, Avg CE loss: {training_ce_loss:>8f}\n')
		validation_loss, validation_ce_loss, validation_correct = validation_loop(validation_loader)
		print(f'Validation Error: \n Accuracy: {(100*validation_correct):>0.1f}%, Avg loss: {validation_loss:>8f}, Avg CE loss: {validation_ce_loss:>8f}\n')
		
		# Save
		torch.save(
			{
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
			},
			os.path.join(final_params_destination, f'T1__bootstrap_index-{bootstrap_index:02d}__initial_weights_index-{initial_weights_index:02d}__{t+1:03d}-tcl-{training_loss:.3f}-vcl-{validation_loss:.3f}-tca-{training_correct:.3f}-vca-{validation_correct:.3f}.pth')
		)

		# Callback
		callback.on_epoch_end(t, model, optimizer, validation_loss)
