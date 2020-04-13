# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

from nn import Crypto_Net
from data_processor import Data_Processor
import torch
from torch import autograd
import torch.nn, torch.optim
from torch.utils.data import DataLoader, Dataset
from parameters import p
import pandas as pd
import logging
import sys
import os
from datetime import datetime
import json

# Constructs layers, weights, biases, activation funcs etc...
if p['type'] == 'simple':
	model = Crypto_Net(p, p['n_input'], p['n_output'], p['n_hidden1'])
else:
	model = Crypto_Net(p, 
		p['n_input'], p['n_output'], 
		p['n_hidden1'], p['n_hidden2'], p['n_hidden3'], p['n_hidden4']
		)

# Loss func and method of gradient descent
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = p['lr'])

def setup_logger(logging_level, log_file):
	''' Args: logger supports levels DEBUG, INFO, WARNING, ERROR, CRITICAL.
	logger_level should be passed in in the format logging.LEVEL '''
#	logger = logging.getLogger(__name__)
#	c_handler = logging.StreamHandler(stream = sys.stdout)
#	f_handler = logging.FileHandler(log_file, mode = 'w')
#	c_format = logging.Formatter('%(levelname)s - %(funcName)s: %(message)s')
#	f_format = logging.Formatter(
#		'%(asctime)s - %(funcName)s - %(filename)s: %(message)s')
#	c_handler.setFormatter(c_format)
#	f_handler.setFormatter(f_format)
#	c_handler.setLevel(logging_level)
#	f_handler.setLevel(logging_level)
#	logger.addHandler(c_handler)
#	logger.addHandler(f_handler)
#	logger.setLevel(logging_level)

	logging.basicConfig(level = logging_level)
	logger = logging.getLogger(__name__)
	return logger
 
log = setup_logger(logging.DEBUG, p['log_file'])

#---------------------Other method of loading data---------------------
class Data(Dataset):
	def __init__(self, p, data_path, logging_level, train_or_test):
		dp = Data_Processor(data_path, logging_level)
		samples_dict = dp.main(
			p, p['sample_and_label_size'], p['label_size'], 
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

def get_data(p, data_path, logging_level):
	train_data = Data(p, data_path, logging_level, 'train')
	test_data =  Data(p, data_path, logging_level, 'test')
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
				loss.backward()
				# update weights
				optimizer.step()

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
	total_loss = 0
	# No gradient computed for testing since we aren't updating weights
	with torch.no_grad():
		for sample, label in test_loader:
			model_output = model(sample)
			loss = criterion(model_output, label)
			total_loss += loss
			if (i + 1) % 5:
				error = get_accuracy(model_output, label)
				total_error += error
				avg_err = get_avg_error(total_error, every_fifth_i)
				log.info(
					'Step [{}/{}], \n '.format(i+1, len(test_loader)) +\
					'Loss: {:.4f}, \n Avg Error: {}'.format(
						loss.item(), avg_err)
				)
				every_fifth_i += 1
			i += 1

	avg_loss = total_loss / len(test_loader)
	write_stats_to_file(p, p['stats_output_file'], avg_loss, avg_err, model)

def get_accuracy(model_output, target):
	error = model_output - target
	return error

def get_avg_error(total_error, i):
	average_error = total_error / (i + 1)
	return average_error

#XXX need to write the stats if it's the last iteration -- not every time
# or can avg the loss and just write that at the end
# also append model type and date/time and learning rate
def write_stats_to_file(p, output_filename, avg_loss, avg_err, model):
	stat_summary = {}
	#XXX named_parameters or just parameters?
	#for name, param in model.named_parameters():
	#name, param = model.state_dict().items()
	# Converts the dict of params into a string so that pandas doesn't screw up
	#str_param = json.dumps(param)

	# These are all wrapped in lists so that I can use orient='columns' which
	# expects an iterable
	stat_summary['Date'] = [datetime.now()]
	stat_summary['Average Loss'] = [avg_loss]
	stat_summary['Average Error'] = [avg_err]
	stat_summary['Learning Rate'] = [p['lr']]
	stat_summary['Model Type'] = [p['type']]
	#stat_summary['Name'] = str(name)
	#stat_summary['Parameters'] = str_param
	# orient='columns' is default
	df = pd.DataFrame.from_dict(stat_summary, orient='columns')

	# If file already exists, need to append, else need to create new file
	if os.path.exists(output_filename) and p['overwrite_stats'] == False:
		with open(output_filename, 'a') as f:
			df.to_csv(f, header=False)
	else:
		with open(output_filename, 'w') as f:
			df.to_csv(f, header=True)

if __name__ == "__main__":
	data_path = './all_currencies.csv'
	logging_level = logging.INFO
	train_loader, test_loader = get_data(p, data_path, logging_level)
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
#				avg_acc = get_avg_error(total_error, every_fifth_i)
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
#def get_avg_error(total_error, i):
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

