# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

import torch
from torch import nn
from parameters import p

class Crypto_Net(nn.Module):
	def __init__(self, 
		n_input, n_output, n_hidden1=None, n_hidden2=None, 
		n_hidden3=None, n_hidden4=None):

		super().__init__()
		if p['type'] == 'simple':
			# Define the architecture
			self.net = nn.Sequential(
				# Layers 1 and 2 with 95 inputs, 20 outputs
				nn.Linear(n_input, n_hidden1),
				# Activation function
				nn.LeakyReLU(),
				# Layers 2 and 3 with 20 inputs 5 outputs
				nn.Linear(n_hidden1, n_output)
				)

		if p['type'] == 'deep':
			self.net = nn.Sequential(
				nn.Linear(n_input, n_hidden1),
				nn.LeakyReLU(),
				nn.Linear(n_hidden1, n_hidden2),
				nn.LeakyReLU(),
				nn.Linear(n_hidden2, n_hidden3),
				nn.LeakyReLU(),
				nn.Linear(n_hidden3, n_hidden4),
				nn.LeakyReLU(),
				nn.Linear(n_hidden4, n_output)
				)


	def forward(self, volumes_sample):
		x = self.net(volumes_sample)
		return x


