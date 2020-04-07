# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

# Sources:
# https://docs.python.org/3/library/unittest.html

import torch
import pandas
import unittest
import sys
sys.path.append('../docs')
import data_processor
import nn
import logging
import start

def setup_logger(logger_level):
	''' Args: logger supports levels DEBUG, INFO, WARNING, ERROR, CRITICAL.
	logger_level should be passed in in the format logging.LEVEL '''

	logging.basicConfig(level=logger_level)
	logger = logging.getLogger(__name__)
	return logger

class Test_NN(unittest.TestCase):

	def setUp(self):
		self.dp = data_processor.Data_Processor(
			'./test_data.csv', logging.DEBUG)
		self.df = self.dp.get_df()
		self.log = setup_logger(logging.DEBUG)

	def test_get_segmented_data_100(self):
		''' 
		This test uses the fake data made in ./test_data.csv, loaded above.
		'''
		self.log.debug("\n TEST 100 \n")
		# Make sure data is segmented based on when the the row counter in the
		# first column goes back to zero and then that company's number of rows
		# is appended to list which is returned
		rows_lst = self.dp.get_segmented_data(self.df)
		# Notice that rows_lst does not depend on the row number in the csv
		# being correct, but rather relies on an iteration counter from 
		# the loop so that the number of rows is always accurate
		assert rows_lst == [1, 2]

	def test_get_volumes_200(self):
		self.log.debug("\n TEST 200 \n")
		rows_lst = self.dp.get_segmented_data(self.df)
		# assumes the that volume column is the 8th col (df.col[7]) of the set
		volumes = self.dp.get_volumes(self.df, rows_lst)
		assert volumes == [[6], [100, 101]]

	def test_clean_data_300(self):
		self.log.debug("\n TEST 300 \n")
		fake_volumes = [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11]]
		sample_and_label_size = 3
		cleaned = self.dp.clean_data(fake_volumes, sample_and_label_size)
		assert(cleaned == [[0, 1, 2,], [6, 7, 8, 9, 10, 11]])

	def test_label_data_400(self):
		self.log.debug("\n TEST 400 \n")
		fake_volumes = [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11]]
		sample_and_label_size = 3
		label_size = 1
		# Cleaned data will look like: [[0, 1, 2], [6, 7, 8, 9, 10, 11]] 
		cleaned = self.dp.clean_data(fake_volumes, sample_and_label_size)
		labels = self.dp.label_data(
			cleaned, sample_and_label_size, label_size)
		assert(labels == [[2], [8, 11]])
		assert(fake_volumes == [[0, 1], [6, 7, 9, 10]])

	def test_map_indices_500(self):
		self.log.debug("\n TEST 500 \n")
		fake_volumes = [[10, 11], [12]]
		mapp = self.dp.map_indices(fake_volumes)
		assert(mapp == {0:(0, 0), 1:(0, 1), 2:(1, 0)})
		lst_ind, num_ind = mapp[1]
		assert(lst_ind == 0 and num_ind == 1)

	def test_get_length_600(self):
		self.log.debug("\n TEST 600 \n")
		fake_volumes = [[10, 11], [12]]
		length = self.dp.get_length(fake_volumes)
		assert(length == 3)

	def test_data_length_700(self):
		self.log.debug("\n TEST 700 \n")
		# assert that there is the correct ratio of training data to labels
		pass

	def test_get_samples_800(self):
		self.log.debug("\n TEST 800 \n")
		fake_data = [[0, 1], [2], [3, 4, 5]]	
		size = 1
		samples = self.dp.get_samples(fake_data, size)	
		self.log.debug("SAMPLES TEST: {}".format(samples))
		assert(samples == [[0], [1], [2], [3], [4], [5]])

	#XXX Make fake_data be processed already rather than calling 
	# the processing funcs
	def test_get_train_test_samples_900(self):
		self.log.debug("\n TEST 900 \n")
		fake_volumes = [[10, 11, 12, 13], [14, 15], [16, 17, 18, 19, 20]]
		sample_and_label_size = 2
		label_size = 1
		# Cleaned data will look like: 
		# [[10, 11, 12, 13], [14, 15], [16, 17, 18, 19]] 
		cleaned = self.dp.clean_data(fake_volumes, sample_and_label_size)
		labels = self.dp.label_data(
			fake_volumes, sample_and_label_size, label_size)
		assert(labels == [[11, 13], [15], [17, 19]])
		# Labels will look like:  [[11, 13], [15], [17, 19]]
		# Cleaned now looks like: [[10, 12], [14], [16, 18]]
		labels_samples = self.dp.get_samples(labels, 1)
		assert(labels_samples == [[11], [13], [15], [17], [19]])
		samples = self.dp.get_samples(cleaned, 1)
		# Labels now looks like:  [[11], [13], [15], [17], [19]]
		# Samples now looks like: [[10], [12], [14], [16], [18]]
		# .6 should use  2/3 of samples for training
		samples_dict = self.dp.get_train_test_samples(
			samples, labels_samples, .6)
		self.log.debug("SAMPLES_DICT TEST: {}".format(samples_dict))
		assert(len(samples_dict['train_samples']) == 3)
		assert(len(samples_dict['train_labels']) == 3)
		assert(len(samples_dict['test_samples']) == 2)
		assert(len(samples_dict['test_labels']) == 2)

	def test_train_1000(self):
		self.log.debug("\n TEST 1000 \n")
		fake_samples = [[0, 1, 2], [4, 5, 6]]
		fake_labels = [[3], [7]] 
		samples_dict = self.dp.get_train_test_samples(
			fake_samples, fake_labels, 1)	
		fake_p = {'n_input' : 3, 'n_hidden' : 20, 'n_output' : 1, 
			'lr' : 0.001, 'epochs' : 25, 'sample_and_label_size' : 4, 
			'label_size' : 1, 'sample_size' : 3, 'train_fraction' : 1
		}
		
		model = nn.Crypto_Net(
			fake_p['n_input'], fake_p['n_hidden'], fake_p['n_output'])
		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr = fake_p['lr'])
		train_samples = samples_dict['train_samples']
		train_labels = samples_dict['train_labels']

		try:
			start.train(
				fake_p, model, criterion, optimizer, 
				train_samples, train_labels
				)
			# Success!
			self.log.debug("TRAIN SUCCESS!")
			status = True
			assert(True)
		except Exception as e:
			self.log.debug("ERROR: {}".format(e))
			status = False
			assert(False)

		return status

	def test_test_1100(self):
		self.log.debug("\n TEST 1100 \n")
		fake_samples = [[10, 11, 12], [14, 15, 16]]
		fake_labels = [[13], [17]] 
		fake_p = {'n_input' : 3, 'n_hidden' : 20, 'n_output' : 1, 
			'lr' : 0.001, 'epochs' : 25, 'sample_and_label_size' : 4, 
			'label_size' : 1, 'sample_size' : 3, 'train_fraction' : 0
		}
		model = nn.Crypto_Net(
			fake_p['n_input'], fake_p['n_hidden'], fake_p['n_output'])
		criterion = torch.nn.MSELoss()
		samples_dict = self.dp.get_train_test_samples(
			fake_samples, fake_labels, fake_p['train_fraction'])	
		test_samples = samples_dict['test_samples']
		test_labels = samples_dict['test_labels']

		try:
			start.test(fake_p, model, criterion, test_samples, test_labels)
			self.log.debug("TEST SUCCESS!")
			status = True
			assert(True)

		except Exception as e:
			self.log.debug("ERROR: {}".format(e))
			status = False
			assert(False)

		return status

			
	def tearDown(self):
		pass
		
if __name__ == '__main__':
	unittest.main()

