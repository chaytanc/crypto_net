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
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


# Constructs layers, weights, biases, activation funcs etc...
model = Crypto_Net(p)

#XXX should be parameters
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
	def __init__(self, p, samples_dict, train_or_test):

		#dp = Data_Processor(data_path, logging_level)
		#samples_dict = dp.main(
			#p, p['sample_and_label_size'], p['label_size'], 
			#p['sample_size'], p['train_fraction'])

		self.samples = samples_dict['{}_samples'.format(train_or_test)]
		self.labels = samples_dict['{}_labels'.format(train_or_test)]

	def __getitem__(self, ind):
		sample = self.samples[ind]
		label = self.labels[ind]
		return sample, label

	def __len__(self):
		length = len(self.samples)
		return length

def get_data(p, samples_dict):
	train_data = Data(p, samples_dict, 'train')
	test_data =  Data(p, samples_dict, 'test')
	train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
	test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
	return (train_loader, test_loader)
		
################################ TRAIN #####################################
def train(p, model, criterion, optimizer, train_loader):
	'''
	Args:
		p: dict containing params subject to change like epoch number
		model: made in Crypto_Net, architechture/neurons to pass data through
		criterion: this is a loss function; the type of error calculation
		optimizer: this is the type of gradient descent (ie stochastic...)
	NOTE: criterion, optimizer, train_loader all are pytorch objects
	'''
	# Used to compute avg_err
	total_error = 0
	# Used to compute avg_loss
	total_loss = 0
	# Used to make pyplots
	total_loss_lst = []
	avg_err_lst = []
	# Used to determine how often to output loss stats
	ip_ratio = p['train_iter_to_print_ratio']
	# Used to plot loss on tensorboard
	running_loss = 0

	with autograd.detect_anomaly():
		# epoch is one full pass through dataset
		for epoch in range(p['epochs']):
			for i, (sample, label) in enumerate(train_loader):
				# Forward pass for each batch of volumes stored in train_loader
				model_output = model(sample)
				loss = criterion(model_output, label)
				# Backpropagation of error
				optimizer.zero_grad()
				# computes new grad
				loss.backward()
				# update weights
				optimizer.step()

				# Calculate metrics, make plots, and log.info for loss
				# Since total_loss append is not within the ip_ratio loop
				# we should use 1 as the ip_ratio since it is updated
				# every iteration
				total_loss_lst.append(loss)
				running_loss += loss.item()
				# Every 100 (ip_ratio) iterations:
				if (i+1) % ip_ratio == 0:

					log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
						.format(
							epoch+1, p['epochs'], 
							i+1, len(train_loader), loss.item()
							)
						)

					# All of this is using the model_output which is a tensor
					# with five output predicitons. It is NOT a scalar
					error = get_accuracy(model_output, label)
					total_error += error
					avg_err = get_avg_error(total_error, ip_ratio)
					# Flatten the output tensor into a scalar of avgs
					avg_avg_err = get_tensor_avg(avg_err)
					avg_err_lst.append(avg_avg_err)

		#write_tensorboard(p, total_loss_lst, train_loader)
		loss_plot(p, total_loss_lst, avg_err_lst)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ TRAIN ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

############################## TEST ##################################
def test(p, model, criterion, test_loader):
	'''
	This func gets runs some unused data through and shows the average error.
	Args:
	'''
	# Used to print and compute loss stats every nth iteration of testing
	n = 5
	every_nth_i = 0
	# Used to compute avg_err
	total_error = 0
	# Used to compute avg_loss
	total_loss = 0
	# Used to make pyplots
	total_loss_lst = []
	avg_err_lst = []

	model.eval()
	# No gradient computed for testing since we aren't updating weights
	with torch.no_grad():
		for i, (sample, label) in enumerate(test_loader):
			model_output = model(sample)
			loss = criterion(model_output, label)
			total_loss += loss
			total_loss_lst.append(loss)
			if (i + 1) % n:
				error = get_accuracy(model_output, label)
				total_error += error
				avg_err = get_avg_error(total_error, every_nth_i)
				# Flatten the output tensor into a scalar of avgs
				avg_avg_err = get_tensor_avg(avg_err)
				avg_err_lst.append(avg_avg_err)
				log.info(
					'Step [{}/{}], \n '.format(i+1, len(test_loader)) +\
					'Loss: {:.4f}, \n Avg Error: {}'.format(
						loss, avg_err)
				)
				every_nth_i += 1

	avg_loss = total_loss / len(test_loader)
	write_stats_to_file(p, avg_loss, avg_err, model)
	#XXX working on it
	#loss_plot(p, total_loss_lst, avg_err_lst, 
		#test_loader, every_nth_i=n, loss_nth_i=1, test=True)

	loss_plot(p, total_loss_lst, avg_err_lst)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ TEST ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def get_accuracy(model_output, target):
	error = model_output - target
	return error

def get_avg_error(total_error, i):
	average_error = total_error / (i + 1)
	return average_error

def get_tensor_avg(tensor):
	total = 0
	length = 0
	# Have to loop more than expected since tensors are like [[1,2,3]] w/
	# extra lst to wrap it in
	if len(tensor.size()) != 0: 
		for i, lst in enumerate(tensor):
			#nump = tensor.numpy()
			#import pdb; pdb.set_trace()
			#if len(nump[i]) != 0: 
			if len(tensor[i].size()) != 0: 
				for scalar in lst:
					total += scalar
					length += 1
	try:
		avg = total / length 
	except ZeroDivisionError:
		log.critical("CRITICAL: Tensor \n {} \n is empty, cannot be averaged." +
			" Check calls to get_tensor_avg in start.py. \n")
		avg = 0
	return avg

def write_stats_to_file(p, avg_loss, avg_err, model):
	#XXX used model to print out weights/biases, since removed b/c not useful?
	output_filename = p['stats_output_file']
	stat_summary = {}
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

def write_tensorboard(p, total_loss_lst, train_loader):
	if p['visualize'] == True:
		# Clears the tensorboard directory before writing to it 
		# again so that the graph doesnt bug out
		for f in os.listdir(p['tensorboard_dir']):
			os.remove(p['tensorboard_dir'] + '/' + f)

		w = SummaryWriter(p['tensorboard_dir'])
		for loss in total_loss_lst:
			for i in range(len(train_loader)):
				#XXX check that len(trainloader) is the correct arg for this
				w.add_scalar('training_loss', loss, i)

		#w.add_scalar('Loss/test', loss, i)	
		#w.add_scalar('Avg_Error/test', avg_err, i)	
		w.close()

def plot_data(p, total_lst):
	# There should be one loss calculated for each sample in data_loader
	
	i_lst = [x for x in range(len(total_lst))]
	plt.grid()
	plt.plot(i_lst, total_lst)
	plt.show()

def loss_plot(p, total_loss_lst, avg_err_lst):
	if p['visualize'] == True:
		plt.title("Loss with Respect to Iterations")
		plt.xlabel("Iterations")
		plt.ylabel("Loss")
		#XXX trying to get logscale on y for loss
		plt.yscale("log")
		plot_data(p, total_loss_lst) 
		plt.title("Average Error with Respect to Iterations")
		plt.xlabel("Iterations")
		plt.ylabel("Average Error")
		plot_data(p, avg_err_lst)

if __name__ == "__main__":
	data_path = './all_currencies.csv'
	logging_level = logging.INFO
	if p['load_model'] == True:
		train_loader, test_loader = get_data(p, data_path, logging_level)
		# Loads the state_dict then feeds into model
		model.load_state_dict(
			torch.load(p['model_filename'])
			)
		test(p, model, criterion, test_loader)

	else:
		train_loader, test_loader = get_data(p, data_path, logging_level)
		train(p, model, criterion, optimizer, train_loader)
		test(p, model, criterion, test_loader)
		torch.save(model.state_dict(), p['model_filename'])

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
#	every_nth_i = 0
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
#				avg_acc = get_avg_error(total_error, every_nth_i)
#				log.info(
#					'Step [{}/{}], ' +\
#					'Loss: {:.4f}, Avg Accuracy: {}'
#					.format(
#						i+1, len(test_samples), loss.item(), avg_acc
#					)
#				)
#				every_nth_i += 1
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

