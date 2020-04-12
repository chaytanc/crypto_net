# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

# Sources: 
#XXX pytorch documentation
# Dataset came from user taniaj on Kaggle.com:
# https://www.kaggle.com/taniaj/cryptocurrency-market-history-coinmarketcap/version/9
# Info about __name__: 
# https://www.freecodecamp.org/news/whats-in-a-python-s-name-506262fe61e8/

# Purpose: Write a neural network to predict cryptocurr total stock volume
#   after 3-6 months based on past volumes
#Psuedo-code
# Get data
	# Pre-process by getting relevant data
	# Overwrite torch.utils.data Dataset funcs __getitem__ and __len__
	# so you can use DataLoader(params...) to load data
# Label data
# Divide data into test and train
# Make neural net structure
# Train
# Test

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import logging
import random
import pickle
from parameters import p
import sys
import math

def setup_logger(logging_level, log_file):
	''' Args: logger supports levels DEBUG, INFO, WARNING, ERROR, CRITICAL.
	logger_level should be passed in in the format logging.LEVEL '''

	# __name__ describes how the script is called, which is currently __main__
	# If the file/func was imported then the name would be nameScript
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

class Data_Processor():
	def __init__(self, data_path, logging_level):
		# logging_level format is, for example, logging.DEBUG
		self.log = setup_logger(logging_level, p['log_file'])
		self.data_path = data_path
		
	# Assumes the file all_currencies.csv (cited in Sources) is in the working
	# directory
	def get_df(self):
		''' Reads in the csv file with cryptocurrency prices stock prices '''
		if os.path.exists(self.data_path):
			self.log.info(' Reading file %s', self.data_path)
			df = pd.read_csv(self.data_path)
		else:
			self.log.critical(' Failed to read data file %s, setting'
				+ ' df = 0', self.data_path)
			df = 0
		return df

	# We need to separate the data based on different crypto companies
	def get_segmented_data(self, df):
		''' This should return a list with each element corresponding to the
		number of rows of data for a company. Ex: crypto_rows = [1049, 1100]
		corresponds to two currencies, the first with 1049 rows of data in the
		csv.
		'''
		self.log.info(' Logging rows where crypto currency changes company')
		# The first column is the row numbers, which resets to 0 for each new
		# cryptocurrency/company
		row_numbers = df[df.columns[0]]
		last_row = -1
		crypto_rows = []
		# iterate through each row in the csv
		i = 1
		for row in row_numbers:

			# If the last row number is not less than the current row then log
			# a company change. Also if the absolute iterations reaches the 
			# total rows in the file (aka it's the last entry) log that.
			if not (last_row < row):
				crypto_rows.append(i) 
				self.log.info(' Different currency found;' +\
					' last currency took %s rows of data', i)
				# Reset i, company row iteration counter
				i = 1
				last_row = -1

			i += 1
			last_row = row
			#self.log.debug('CRYPTO_ROWS: {}'.format(crypto_rows))

		self.log.info(' Appended final company rows to crypto_rows.')
		# Have to subtract the last increment since its the end of the file
		crypto_rows.append(i - 1)
		# Accounts for the column headers in the first row
		crypto_rows[0] = crypto_rows[0] - 1	
		return crypto_rows

	def get_volumes(self, df, crypto_rows):
		''' Should return a list of lists with one list per crypto containing
		all the volumes of that currency in the list. Ex: [[127, 126, 124],
		[992, 889, 923]] is two cryptocurrencies with 3 volumes logged for each
		'''

		self.log.info(' Getting the all the volumes and putting them into a' +\
			' separate list for each different currency.')
		all_currency_vol = []
		individual_currency_vol = []
		# abs_iterations increments with each new company, i increments
		# with each volume added to a company, representing which row is 
		# being iterated over
		i = 0 
		company_i = 0
		# iterate through all volumes
		for volume in df[df.columns[7]]:

			# Checks if the row number is equal to the row at which the a new 
			# currency is logged (as stored in crypto_rows) 
			if(i == crypto_rows[company_i]):

				self.log.info(" Appending indiv. company volumes to list.")
				all_currency_vol.append(individual_currency_vol)
				# Clears the individual currency volume list when a new
				# company is found THEN appends the next volume
				individual_currency_vol = []
				individual_currency_vol.append(volume)
				i = 1
				company_i += 1

			else:
				individual_currency_vol.append(volume)
				self.log.info(" Appending volume to indiv. company list.")
				i += 1

		self.log.info(" Appending indiv. company volumes to list.")
		all_currency_vol.append(individual_currency_vol)
		return all_currency_vol
	
	#XXX make all_currency_vol and volumes be the same words
	def clean_data(self, volumes, sample_and_label_size):
		''' This function will trim each company's data so that it is divisible
		by your batchsize and label size combined. 
		EX: 95 rows for input and 5 rows for labels = 100 sample_and_label_size
		If len(data) = 101, this function removes the last row.
		'''

		for i, company in enumerate(volumes):	
			length = len(company)
			# sample size is 95, labels use 5, so 100
			remainder = length % sample_and_label_size
			# subtract remainder number of volumes off company
			del company[length - remainder : length]

		inds_to_delete = []
		for i, company in enumerate(volumes):	
			# Removes any now empty/useless lists
			new_len = len(company)
			if new_len == 0:
				inds_to_delete.append(i)

		for ind in sorted(inds_to_delete, reverse=True):
			del volumes[ind]

		return volumes

	#XXX IT IS MISLEADING to only return labels since it ALSO changes volumes
	# (passed by reference)
	def label_data(self, volumes, sample_and_label_size, label_size):
		''' Volumes will be passed in the format which has been separated by
		company into lists which are all evenly divisible by 100. (clean_data)
		***volumes is also modified by this function***
		'''
		# make list of lable indices that we will delete from
		inds_to_delete = []
		# make labels list
		labels = []
		# Iterate through companies
		for lst, company in enumerate(volumes):
			# make individual label for the comp
			label = []
			# Iterate through rows
			for row, data in enumerate(company):
				# batch and label size is 100 in my case
				# If it is the 100th item (99th index) it will use remove 
				# label_size amount of indices from the end and append them to
				# label which will append to labels
				if ((row + 1) % sample_and_label_size) == 0:
					for each in range(label_size):
						# Fill labels list with last five datapoints of that 100
						label.append(data - each)	
						inds_to_delete.append((lst, row - each))
			# Append the ind comp label list to the labels list
			labels.append(label)

		# Need to sort it so that we delete the larger indices FIRST so that
		# the order of the other indices is not screwed up as we delete
		for lst, row in sorted(inds_to_delete, reverse=True):
			# Delete label indices from the volumes
			del volumes[lst][row]

		assert(len(volumes) == len(labels))
		return labels

	# Need to get a list of 95 volumes, turn it to a tensor and pass it into
	# model as a param so that it goes through forward
	def get_samples(self, two_d_array, size):
		''' 
		This basically just resizes an array to a 2d array with elements 
		of a given size.

		Args:
			two_d_array: any list with two dims, i.e. [[4], [1, 2]]
			size: the size of the second dimension of two_d_array that you WANT
		'''
		samples = []
		sample = []
		# Iterate through companies
		for lst, company in enumerate(two_d_array):
			# Iterate through volumes in each company list
			for row, volume in enumerate(company):
				sample.append(volume)
				if ((row + 1) % size) == 0:
					samples.append(sample)
					sample = []
		return samples

	#XXX shorten this?
	def get_train_test_samples(self, samples, labels, train_fraction):
		'''
		Args:
			samples: format returned by get_samples()
				Ex: [[1, 2, 3], [4, 5, 6]] if the input tensor size is 3. 
			labels: format returned by label_data()
				Ex: [[16], [10]] if the label_size is 1
			train_fraction: must be a fraction of one representing the percent
				of the data which will be used for training. 
				Ex: .8 = 80% training data, 20% testing.
		'''
		train_test_samples = {}
		n_samples = len(samples)
		samples_inds = list(enumerate(samples))
		self.log.debug("SAMPLES_INDS: {}".format(samples_inds))
		# Divide into x% train, x% test
		n_train = int(n_samples * train_fraction)
		n_test = int(n_samples * (1 - train_fraction))
		train_samples = []
		train_labels = []
		test_samples = []
		test_labels = []
		# Randomly sample n training samples and n test samples
		# train_ind = [(ind, val)] format
		train_set = random.sample(samples_inds, n_train)
		self.log.debug("TRAIN_SET: {}".format(train_set))
		train_inds = [train_ind for train_ind, train in train_set]

		# test_ind is whatever train_inds aren't
		#test_inds = [ind for ind, sample in enumerate(samples), 
		#	if ind not in train_inds]
		test_inds = []
		for ind, sample in enumerate(samples):
			if ind not in train_inds:
				test_inds.append(ind)

		#XXX FOR TESTING:
		self.log.debug("LEN SAMPLES, LEN LABELS: {}, {}".format(
			len(samples), len(labels)))
		assert(len(samples) == len(labels))

		self.log.debug("TRAIN_INDS: {}".format(train_inds))
		self.log.debug("TEST_INDS: {}".format(test_inds))
		for ind in train_inds:
			self.log.debug("TRAIN IND: {}".format(ind))
			train_samples.append(samples[ind])
			train_labels.append(labels[ind])
		
		for ind in test_inds:
			test_samples.append(samples[ind])
			test_labels.append(labels[ind])

		tr_samples = self.to_tensor(train_samples)
		tr_labels = self.to_tensor(train_labels)
		te_samples = self.to_tensor(test_samples)
		te_labels = self.to_tensor(test_labels)

		tr_samples, tr_labels = self.delete_empty_labels(tr_samples, tr_labels)
		te_samples, te_labels = self.delete_empty_labels(te_samples, te_labels)

		train_test_samples['train_samples'] = tr_samples
		train_test_samples['train_labels'] = tr_labels
		train_test_samples['test_samples'] = te_samples
		train_test_samples['test_labels'] = te_labels

		return train_test_samples

	def to_tensor(self, array):
		new_array = []
		for sample in array:
			# Check if index of samples is one of the indices in train_inds
			tensor = torch.tensor(
				sample, requires_grad=True, dtype=torch.float32)
			new_array.append(tensor)
		return new_array

	def delete_empty_labels(self, samples, labels):
		''' Inputs must be tensors! '''
		length = len(labels[0])
		inds_to_delete = []
		for i, label in enumerate(labels):
			if all(torch.isnan(label)):
				inds_to_delete.append(i)	
	
		for ind in sorted(inds_to_delete, reverse=True):
			del samples[ind]
			del labels[ind]
		return (samples, labels)

	def average_nans(self, lst):
		#''' Input must be after labels w/ all nans were deleted. '''
		''' Input must be lst -- not jagged! This function loops through
		all samples and replaces nans with the average of the funciton. If
		they're all nans then.... #XXX
		 '''
		# turn into a numpy array
		for sample in lst:
			arr = np.asarray(sample)
			# find average value of the lst not including nans 
			len_excluding_nans = (len(arr) - np.isnan(arr).sum())
			average = np.nansum(arr) / len_excluding_nans
			# replace nans with avg
			for i, sample_val in enumerate(sample):
				if math.isnan(sample_val):
					sample[i] = average
		self.find_nans(lst)
		return lst

	def find_nans(self, lst):
		for j, sample in enumerate(lst):
			for i, sample_val in enumerate(sample):
				if math.isnan(sample_val):
					self.log.debug("FIND NAN FUNC")
					#import pdb; pdb.set_trace()
					sample[i] = 1.0

	def replace_zeroes(self, lst):
		for j, sample in enumerate(lst):
			for i, sample_val in enumerate(sample):
				if sample_val == 0:
					lst[j][i] = 1.0
		return lst
					
	def map_indices(self, volumes):
		'''
		This maps a one dimensional index as the key and the 
		corresponding two dimensional indices as the value.
		See the tests for examples.
		'''
		mapp = {}
		# mapp_index will increment once every time a entry is made, mapping 
		# every element of lists within volumes to a different dict index
		mapp_index = 0
		for lst_ind, lst in enumerate(volumes):
			for num_ind, num in enumerate(lst):
				mapp[mapp_index] = (lst_ind, num_ind)
				mapp_index += 1
		return mapp	

	def get_length(self, volumes):
		''' This function gets the the total elements of a 2d array. '''
		total_length = 0
		for comp in volumes:
			for volume in comp:
				total_length += 1	
		return total_length

	def see_data(self, volumes, out_path=None):
		try:
			volumes.to_csv(out_path)
		except ValueError as e:
			self.log.info(" Tried to convert data to csv, no path to csv input")

		#XXX matplotlib no work, ask noah later
		#table = plt.table(volumes)
		#plt.figure(table)
		#plt.show(table)
		#self.log.info(volumes.head())

	def save_train_test_samples(self, train_test_samples, name):
		'''
		Args:
			train_test_samples: This can actually be any python object you
				want to save, not just train_test_samples
			name: How you want to name the file you save to object to. Should
				end in .pkl

		pkl or pickle is a Python model which can turn a Python object
		to a byte stream. '''

		# 'wb' means writing to the file in binary, which pickle does
		with open(name, 'wb') as f:
			pickle.dump(train_test_samples, f, pickle.DEFAULT_PROTOCOL)
			f.close()

	def load_train_test_samples(self, name):
		''' name should end in .pkl '''
		try:
			with open (name, 'rb') as f:
				obj = pickle.load(f)
				f.close()
		except FileNotFoundError as e:
			self.log.critical(
				'You have not processed and saved the data yet!' +\
				'\n Try changing the "load_processing" parameter to False ' +\
				' in parameters.py! \n'
				)
		return obj
				
	# Chains together all the functions necessary to process the data
	def main(
		self, sample_and_label_size, label_size, 
		sample_size, train_fraction):

		if p['load_processing'] == True:
			samples_dict = self.load_train_test_samples(p['obj_filename'])
		else:
			df = self.get_df()
			crypto_rows = self.get_segmented_data(df)
			volumes = self.get_volumes(df, crypto_rows)
			cleaner_volumes = self.clean_data(volumes, sample_and_label_size)
			labels = self.label_data(
				cleaner_volumes, sample_and_label_size, label_size)
			labels_samples = self.get_samples(labels, label_size)
			samples = self.get_samples(cleaner_volumes, sample_size)
			labels_samp = self.average_nans(labels_samples)
			samp = self.average_nans(samples)
			samples_dict = self.get_train_test_samples(
				samp, labels_samp, train_fraction)
			self.save_train_test_samples(samples_dict, p['obj_filename'])
				
		return samples_dict
		
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

if __name__ == '__main__':
	pass
	#data_loader = Data_Processor('./all_currencies.csv')
	#data_loader.main()

