# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

import torch
from torch import nn

class Crypto_Net(nn.Module):
	def __init__(self, n_input, n_hidden, n_output):
		# Very important to super so you can override forward later
		#XXX some nets are written this Python 2 way, not sure why
		#super(Crypto_Net, self).__init__()
		super().__init__()
		# Define the architecture
		self.net = nn.Sequential(
			# Layers 1 and 2 with 95 inputs, 20 outputs
			nn.Linear(n_input, n_hidden),
			# Activation function
			nn.LeakyReLU(),
			# Layers 2 and 3 with 20 inputs 5 outputs
			nn.Linear(n_hidden, n_output)
			)

	def forward(self, volumes_sample):
		x = self.net(volumes_sample)
		return x


