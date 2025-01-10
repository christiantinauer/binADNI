from torch import inf

class ReduceLRAndResetStateDictOnPlateau():
	def __init__(self,
							 factor=0.1,
							 patience=10,
							 verbose=0,
							 min_delta=1e-4,
							 cooldown=0,
							 min_lr=0):
		if factor >= 1.0:
			raise ValueError(f'ReduceLROnPlateau does not support a factor >= 1.0. Got {factor}')
		self.factor = factor
		self.min_lr = min_lr
		self.min_delta = min_delta
		self.patience = patience
		self.verbose = verbose
		self.cooldown = cooldown
		self.cooldown_counter = 0  # Cooldown counter.
		self.wait = 0
		self.best = 0
		self.state_dict = None
		self._reset()

	def _reset(self):
		"""Resets wait counter and cooldown counter.
		"""
		self.best = inf
		self.cooldown_counter = 0
		self.wait = 0

	def on_train_begin(self, model):
		self._reset()
		self.state_dict = model.state_dict()

	def on_epoch_end(self, epoch, model, optimizer, validation_loss):
		if self.in_cooldown():
			self.cooldown_counter -= 1
			self.wait = 0

		if validation_loss < self.best - self.min_delta:
			self.best = validation_loss
			self.state_dict = model.state_dict()
			self.wait = 0
		elif not self.in_cooldown():
			self.wait += 1
			if self.wait >= self.patience:
				for i, param_group in enumerate(optimizer.param_groups):
					old_lr = float(param_group['lr'])
					new_lr = max(old_lr * self.factor, self.min_lr)
					param_group['lr'] = new_lr
					if self.verbose:
						epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
						print('Epoch {}: reducing learning rate'
									' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
					if self.state_dict is not None:
						model.load_state_dict(self.state_dict)
						if self.verbose > 0:
							print('\nEpoch %05d: Reset state dict.' % (epoch + 1))
					self.cooldown_counter = self.cooldown
					self.wait = 0

	def in_cooldown(self):
		return self.cooldown_counter > 0
