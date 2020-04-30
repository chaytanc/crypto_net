# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

import torch
from torch import nn
from parameters import p

class Crypto_Net(nn.Module):
	def __init__(self, p):
		n_input = p['n_input']
		n_hidden1 = p['n_hidden1']
		n_hidden2 = p['n_hidden2']
		n_hidden3 = p['n_hidden3']
		n_hidden4 = p['n_hidden4']
		n_output = p['n_output']

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
				nn.Dropout(p['dropout_prob']),
				nn.Linear(n_hidden1, n_hidden2),
				nn.LeakyReLU(),
				nn.Dropout(p['dropout_prob']),
				nn.Linear(n_hidden2, n_hidden3),
				nn.LeakyReLU(),
				nn.Dropout(p['dropout_prob']),
				nn.Linear(n_hidden3, n_hidden4),
				nn.LeakyReLU(),
				nn.Linear(n_hidden4, n_output)
				)

		if p['type'] == 'auto':
			self.net = self.make_net(p)


	def forward(self, volumes_sample):
		x = self.net(volumes_sample)
		return x

	#XXX need a way to specify the size of each layer such that the number of
	# sizes specified is determined by the number of layers specified
	#XXX test
	def make_net(p):
		''' This function constructs a pytorch nn.Sequential model based on
		the number of layers and layer sizes specified in parameters.
		'''

		n_layers = p['total_hidden_layers']
		modules = []
		# Append input layer
		modules.append(nn.Linear(p['n_input'], p['n_hidden']))
		modules.append(nn.LeakyReLU())
		modules.append(nn.Dropout(p['dropout_prob']))
		# Append hidden layers
		# Ex: 3 hidden layers n_hidden1, n_hidden2, n_hidden3, n_layers = 3
		# We want to loop through n_hidden1 and n_hidden2 but not n_hidden3
		# which we account for after the loop ends
		for x in (range(n_layers) - 1):
			hidden_size = p['n_hidden{}'.format(x + 1)]
			next_hidden_size = p['n_hidden{}'.format(x + 2)]
			modules.append(nn.Linear(hidden_size, next_hidden_size))
			modules.append(nn.LeakyReLU())
			modules.append(nn.Dropout(p['dropout_prob']))
		# Append output layers
		last_hidden = p['n_hidden{}'.format(n_layers)]
		modules.append(nn.Linear(last_hidden, p['n_output']))
		sequential = nn.Sequential(*modules)
		return sequential


