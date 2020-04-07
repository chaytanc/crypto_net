# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

from nn import Crypto_Net
from data_processor import Data_Processor
import torch
from torch import autograd
import torch.nn, torch.optim
from torch.utils.data import DataLoader, Dataset
from parameters import p
import logging

# Constructs layers, weights, biases, activation funcs etc...
model = Crypto_Net(p['n_input'], p['n_hidden'], p['n_output'])

# Loss func and method of gradient descent
#criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr = p['lr'])

def setup_logger(logging_level):
	''' Args: logger supports levels DEBUG, INFO, WARNING, ERROR, CRITICAL.
	logger_level should be passed in in the format logging.LEVEL '''

	logging.basicConfig(level=logging_level)
	logger = logging.getLogger(__name__)
	return logger
 
log = setup_logger(logging.DEBUG)

#---------------------Other method of loading data---------------------
class Data(Dataset):
	def __init__(self, data_path, logging_level, train_or_test):
		dp = Data_Processor(data_path, logging_level)
		samples_dict = dp.main(
			p['sample_and_label_size'], p['label_size'], 
			p['sample_size'], p['train_fraction'])

		self.samples = samples_dict['{}_samples'.format(train_or_test)]
		self.labels = samples_dict['{}_labels'.format(train_or_test)]

	def __getitem__(self, ind):
		sample = self.samples[ind]
		label = self.labels[ind]
		return sample, label

	def __len__(self):
		length = len(self.samples)
		return length

def get_data(data_path, logging_level):
	train_data = Data(data_path, logging_level, 'train')
	test_data =  Data(data_path, logging_level, 'test')
	train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
	test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
	return (train_loader, test_loader)
		
def train(p, model, criterion, optimizer, train_loader):
	'''
	Args:
		p: dict containing params subject to change like epoch number
		model: made in Crypto_Net, architechture/neurons to pass data through
		criterion: this is a loss function; the type of error calculation
		optimizer: this is the type of gradient descent (ie stochastic...)
	NOTE: criterion, optimizer, train_loader all are pytorch objects
	'''
	with autograd.detect_anomaly():
		# epoch is one full pass through dataset
		for epoch in range(p['epochs']):
			for i, (sample, target) in enumerate(train_loader):
				# Forward pass for each batch of volumes stored in train_loader
				model_output = model(sample)
				loss = criterion(model_output, target)
				# Backpropagation of error
				optimizer.zero_grad()
				# computes new grad
				loss.backward(retain_graph=True)
				# update weights
				optimizer.step()
	#			log.debug('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
	#				.format(
	#					epoch+1, p['epochs'], 
	#					i+1, len(train_loader), loss.item()
	#			))

				#import pdb; pdb.set_trace()
				#XXX mess w/ the 100 param, make a hyperparam
				if (i+1) % 100 == 0:
					log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
						.format(
							epoch+1, p['epochs'], 
							i+1, len(train_loader), loss.item()
					))

def test(p, model, criterion, test_loader):
	'''
	This func gets runs some unused data through and shows the average error.
	Args:
	'''
	i = 0
	every_fifth_i = 0
	total_error = 0
	# No gradient computed for testing since we aren't updating weights
	with torch.no_grad():
		for sample, label in test_loader:
			model_output = model(sample)
			loss = criterion(model_output, label)
			log.debug(
				'Step [{}/{}], ' +\
				'Loss: {:.4f}'
				.format(
					i+1, len(test_loader), loss.item()
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
						i+1, len(test_loader), loss.item(), avg_acc
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
	data_path = './all_currencies.csv'
	logging_level = logging.INFO
	train_loader, test_loader = get_data(data_path, logging_level)
	train(p, model, criterion, optimizer, train_loader)
	test(p, model, criterion, test_loader)

#-----------------------TRAINING, NO OVERLOADING DATASET ---------------------
#def train(p, model, criterion, optimizer, samples, labels):
#	'''
#	Args:
#		p: dict containing params subject to change like epoch number
#		model: made in Crypto_Net, architechture/neurons to pass data through
#		criterion: this is a loss function; the type of error calculation
#		optimizer: this is the type of gradient descent (ie stochastic...)
#	NOTE: criterion, optimizer, train_loader all are pytorch objects
#	'''
#	# epoch is one full pass through dataset
#	for epoch in range(p['epochs']):
#		for i, (sample, target) in enumerate(zip(samples, labels)):
#			# Forward pass for each batch of volumes stored in train_loader
#			model_output = model(sample)
#			loss = criterion(model_output, target)
#			# Backpropagation of error
#			optimizer.zero_grad()
#			# computes new grad
#			loss.backward()
#			# update weights
#			optimizer.step()
#			log.debug('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#				.format(
#					epoch+1, p['epochs'], 
#					i+1, len(samples), loss.item()
#			))
#
#			#XXX mess w/ the 100 param to print sufficiently, make a hyperparam
#			if (i+1) % 100 == 0:
#				log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#					.format(
#						epoch+1, p['epochs'], 
#						i+1, len(samples), loss.item()
#				))
#
#def test(p, model, criterion, test_samples, test_labels):
#	'''
#	This func gets runs some unused data through and shows the average error.
#	Args:
#	'''
#	i = 0
#	every_fifth_i = 0
#	total_error = 0
#	# No gradient computed for testing since we aren't updating weights
#	with torch.no_grad():
#		for sample, label in zip(test_samples, test_labels):
#			model_output = model(sample)
#			loss = criterion(model_output, label)
#			log.debug(
#				'Step [{}/{}], ' +\
#				'Loss: {:.4f}'
#				.format(
#					i+1, len(test_samples), loss.item()
#				)
#			)
#			if (i + 1) % 5:
#				error = get_accuracy(model_output, label)
#				total_error += error
#				avg_acc = get_avg_acc(total_error, every_fifth_i)
#				log.info(
#					'Step [{}/{}], ' +\
#					'Loss: {:.4f}, Avg Accuracy: {}'
#					.format(
#						i+1, len(test_samples), loss.item(), avg_acc
#					)
#				)
#				every_fifth_i += 1
#			i += 1
#
#def get_accuracy(model_output, target):
#	error = model_output - target
#	return error
#
#def get_avg_acc(total_error, i):
#	average_error = total_error / (i + 1)
#	return average_error
#
#if __name__ == "__main__":
#	data_path = './all_currencies.csv'
#	logging_level = logging.INFO
#	dp = Data_Processor(data_path, logging_level)
#	samples_dict = dp.main(
#		p['sample_and_label_size'], p['label_size'], 
#		p['sample_size'], p['train_fraction'])
#	train_samples = samples_dict['train_samples']
#	train_labels = samples_dict['train_labels']
#	test_samples = samples_dict['test_samples']
#	test_labels = samples_dict['test_labels']
#	train(p, model, criterion, optimizer, train_samples, train_labels)
#	test(p, model, criterion, test_samples, test_labels)
#


#------------------------------------OLD----------------------------------
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

