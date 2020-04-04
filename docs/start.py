# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

from nn import Crypto_Net
from data_processor import Data_Processor
import torch
import torch.nn, torch.optim
from torch.utils.data import DataLoader, Dataset
from parameters import p
import logging

# Constructs layers, weights, biases, activation funcs etc...
model = Crypto_Net(p['n_input'], p['n_hidden'], p['n_output'])

# Loss func and method of gradient descent
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = p['lr'])

def setup_logger(logging_level):
	''' Args: logger supports levels DEBUG, INFO, WARNING, ERROR, CRITICAL.
	logger_level should be passed in in the format logging.LEVEL '''

	logging.basicConfig(level=logging_level)
	logger = logging.getLogger(__name__)
	return logger
log = setup_logger(logging.DEBUG)
 
def initialize():
	data_path = './all_currencies.csv'
	logging_level = logging.INFO
	dp = Data_Processor(data_path, logging_level)
	samples_dict = dp.main(
		p['sample_and_label_size'], p['label_size'], 
		p['sample_size'], p['train_fraction'])
	train_samples = samples_dict['train_samples']
	train_labels = samples_dict['train_labels']
	test_samples = samples_dict['test_samples']
	test_labels = samples_dict['test_labels']

def train(p, model, criterion, optimizer, samples, labels):
	'''
	Args:
		p: dict containing params subject to change like epoch number
		model: made in Crypto_Net, architechture/neurons to pass data through
		criterion: this is a loss function; the type of error calculation
		optimizer: this is the type of gradient descent (ie stochastic...)
	NOTE: criterion, optimizer, train_loader all are pytorch objects
	'''
	# epoch is one full pass through dataset
	for epoch in range(p['epochs']):
		for i, (sample, target) in enumerate(zip(samples, labels)):
			# Forward pass for each batch of volumes stored in train_loader
			model_output = model(sample)
			loss = criterion(model_output, target)
			# Backpropagation of error
			optimizer.zero_grad()
			# computes new grad
			loss.backward()
			# update weights
			optimizer.step()
			log.debug('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				.format(
					epoch+1, p['epochs'], 
					i+1, len(samples), loss.item()
			))

			#XXX mess w/ the 100 param to print sufficiently, make a hyperparam
			if (i+1) % 100 == 0:
				log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
					.format(
						epoch+1, p['epochs'], 
						i+1, len(samples), loss.item()
				))

def test(p, model, criterion, test_samples, test_labels):
	'''
	This func gets runs some unused data through and shows the average error.
	Args:
	'''
	i = 0
	every_fifth_i = 0
	total_error = 0
	# No gradient computed for testing since we aren't updating weights
	with torch.no_grad():
		for sample, label in zip(test_samples, test_labels):
			model_output = model(sample)
			loss = criterion(model_output, label)
			log.debug(
				'Step [{}/{}], ' +\
				'Loss: {:.4f}'
				.format(
					i+1, len(test_samples), loss.item()
				)
			)
			if (i + 1) % 5:
				error = get_accuracy(model_output, label)
				total_error += error
				avg_acc = get_avg_acc(total_error, every_fifth_i)
				log.info(
					'Step [{}/{}], ' +\
					'Loss: {:.4f}, Avg Accuracy: {}'
					.format(
						i+1, len(test_samples), loss.item(), avg_acc
					)
				)
				every_fifth_i += 1
			i += 1

def get_accuracy(model_output, target):
	error = model_output - target
	return error

def get_avg_acc(total_error, i):
	average_error = total_error / (i + 1)
	return average_error

if __name__ == "__main__":
	initialize()
	train(p, model, criterion, optimizer, train_samples, train_labels)
	test(p, test_samples, test_labels)

#---------------------Other method of loading data---------------------
#class Data(Dataset):
#	def __init__(self):
#		self.dp = Data_Processor('./all_currencies.csv')
#		self.volumes = self.dp.main()
#		self.mapp = self.dp.map_indices(self.volumes) 
#
#	def __len__(self):
#		length = self.dp.get_length(self.volumes)
#		return length
#
#	def __getitem__(self, ind):
#		lst_ind, num_ind = self.mapp[ind]
#		# item is the volume of a cryptocurrency stock at a given index
#		item = self.volumes[lst_ind][num_ind]
#		return item, label
#
#def get_data(data, labels):
#	#train_data = Data()
#	#test_data = 
#
#	train_loader = DataLoader(dataset=train_data, batch_size=95, shuffle=True)
#	test_loader = DataLoader(dataset=test_data, batch_size=95, shuffle=True)
		
#XXX train function if using Dataset and train_loader etc
#def train(p, model, criterion, optimizer, train_loader):
#	'''
#	Args:
#		p: dict containing params subject to change like epoch number
#		model: made in Crypto_Net, architechture/neurons to pass data through
#		criterion: this is a loss function; the type of error calculation
#		optimizer: this is the type of gradient descent (ie stochastic...)
#		train_loader: training data as loaded by pytorch
#	NOTE: criterion, optimizer, train_loader all are pytorch objects
#	'''
#	# epoch is one full pass through dataset
#	for epoch in range(p['epochs']):
#		# data represents company volumes
#		#XXX should def print train_loader to understand how it shapes the data
#		#XXX experiment
#		#for i, (data, target) in enumerate(train_loader):
#		for data, target in train_loader:
#			# Forward pass for each batch of volumes stored in train_loader
#			model_output = model(data)
#			loss = criterion(model_output, target)
#			# Backpropagation of error
#			optimizer.zero_grad()
#			# computes new grad
#			loss.backward()
#			# update weights
#			optimizer.step()
#
#			if (i+1) % 100 == 0:
#				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
#					epoch+1, num_epochs, i+1, total_step, loss.item()))
#
#def test(p, test_loader):
#	'''
#	This func gets runs some unused data through and shows the average error.
#	Args:
#		test_loader: pytorch object for testing data
#	'''
#	i = 0
#	j = 0
#	total_error = 0
#	# No gradient computed for testing since we aren't updating weights
#	with torch.no_grad():
#		for data, target in test_loader:
#			model_output = model(data)
#			if (i + 1) % 5:
#				error = get_accuracy(model_output, target)
#				total_error += error
#				avg_acc = get_avg_acc(total_error, j)
#				print('AVERAGE NETWORK ERROR, in dollars: {}'.format(avg_acc))
#				j += 1
#			i += 1
			

