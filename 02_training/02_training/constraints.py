class BiasNegConstraint(object):
	def __call__(self, module):
		if hasattr(module, 'bias'):
			module.bias.data.mul_(module.bias.data <= 0)