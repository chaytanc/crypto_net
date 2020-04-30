# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

# Sources:
# https://docs.python.org/3/library/unittest.html

import torch
import pandas
import numpy as np
import unittest
import sys
sys.path.append('../docs')
import data_processor
import nn
import logging
import start
from fake_parameters import fake_p
from nn import Crypto_Net
import traceback
import functools

def setup_logger(logger_level):
	''' Args: logger supports levels DEBUG, INFO, WARNING, ERROR, CRITICAL.
	logger_level should be passed in in the format logging.LEVEL '''

	logging.basicConfig(level=logger_level)
	logger = logging.getLogger(__name__)
	return logger

class Test_NN(unittest.TestCase):

	def disabled(self):
		def _decorator(f):
			 print(str(f) + ' has been disabled')
		return _decorator

#	def debug_on(*exceptions):
#		if not exceptions:
#			exceptions = (AssertionError, )
#		def decorator(f):
#			@functools.wraps(f)
#			def wrapper(*args, **kwargs):
#				try:
#					return f(*args, **kwargs)
#				except exceptions:
#					info = sys.exc_info()
#					traceback.print_exception(*info) 
#					pdb.post_mortem(info[2])
#			return wrapper
#		return decorator

	def setUp(self):
		self.dp = data_processor.Data_Processor(
			'./test_data.csv', logging.DEBUG)
		self.df = self.dp.get_df()
		self.log = setup_logger(logging.DEBUG)

	#@disabled
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
		assert rows_lst == [2, 2]

	#@disabled
	def test_get_volumes_200(self):
		self.log.debug("\n TEST 200 \n")
		rows_lst = self.dp.get_segmented_data(self.df)
		# assumes the that volume column is the 8th col (df.col[7]) of the set
		volumes = self.dp.get_volumes(self.df, rows_lst)
		assert volumes == [[6, 8], [100, 101]]

	#@disabled
	def test_clean_data_300(self):
		self.log.debug("\n TEST 300 \n")
		fake_volumes = [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11]]
		sample_and_label_size = 3
		cleaned = self.dp.clean_data(fake_volumes, sample_and_label_size)
		assert(cleaned == [[0, 1, 2], [6, 7, 8, 9, 10, 11]])

	#@disabled
	def test_label_data_400(self):
		self.log.debug("\n TEST 400 \n")
		fake_volumes = [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11]]
		sample_and_label_size = 3
		label_size = 2
		# Cleaned data will look like: [[0, 1, 2], [6, 7, 8, 9, 10, 11]] 
		cleaned = self.dp.clean_data(fake_volumes, sample_and_label_size)
		labels = self.dp.label_data(
			cleaned, sample_and_label_size, label_size)
		assert(labels == [[2, 1], [8, 7, 11, 10]])
		assert(fake_volumes == [[0], [6, 9]])

	#@disabled
	def test_map_indices_500(self):
		self.log.debug("\n TEST 500 \n")
		fake_volumes = [[10, 11], [12]]
		mapp = self.dp.map_indices(fake_volumes)
		assert(mapp == {0:(0, 0), 1:(0, 1), 2:(1, 0)})
		lst_ind, num_ind = mapp[1]
		assert(lst_ind == 0 and num_ind == 1)

	#@disabled
	def test_get_length_600(self):
		self.log.debug("\n TEST 600 \n")
		fake_volumes = [[10, 11], [12]]
		length = self.dp.get_length(fake_volumes)
		assert(length == 3)

	#@disabled
	def test_data_length_700(self):
		self.log.debug("\n TEST 700 \n")
		# assert that there is the correct ratio of training data to labels
		pass

	#@disabled
	def test_get_samples_800(self):
		self.log.debug("\n TEST 800 \n")
		fake_data = [[0, 1, 2, 3], [4, 5], [6, 7]]	
		size = 2
		samples = self.dp.get_samples(fake_data, size)	
		self.log.debug("SAMPLES TEST: {}".format(samples))
		assert(samples == [[0, 1], [2, 3], [4, 5], [6, 7]])

	#@disabled
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

	#@disabled
	def test_train_1000(self):
		self.log.debug("\n TEST 1000 \n")

		data_path = './test_volumes.csv'
		dp = data_processor.Data_Processor(data_path, logging.DEBUG)
		fake_p['k_auto'] = True
		fake_p['kmeans'] = True

		samples_dict = dp.main(
			fake_p, fake_p['sample_and_label_size'], fake_p['label_size'], 
			fake_p['sample_size'], fake_p['train_fraction']
			)

		model = Crypto_Net(fake_p)
		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr = fake_p['lr'])

		if fake_p['kmeans'] == True:
			# If kmeans is true, then samples_dict is actually a list of
			# multiple samples_dicts: one for each cluster
			for key, sd in samples_dict.items():
				train_loader, test_loader = start.get_data(fake_p, sd)
				self.try_method(start.train, fake_p, model, 
					criterion, optimizer, train_loader)	

		else:
			train_loader, test_loader = start.get_data(fake_p, samples_dict)
			self.log.debug("TRAIN_LOADER: {}".format(train_loader))
			self.try_method(start.train, fake_p, model, 
				criterion, optimizer, train_loader)	

	#@disabled
	def test_test_1100(self):
		self.log.debug("\n TEST 1100 \n")

		data_path = './test_volumes.csv'
		dp = data_processor.Data_Processor(data_path, logging.DEBUG)
		fake_p['k_auto'] = False
		fake_p['kmeans'] = True

		samples_dict = dp.main(
			fake_p, fake_p['sample_and_label_size'], fake_p['label_size'], 
			fake_p['sample_size'], fake_p['train_fraction']
			)

		model = Crypto_Net(fake_p)
		criterion = torch.nn.MSELoss()
		if fake_p['kmeans'] == True:
			# If kmeans is true, then samples_dict is actually a list of
			# multiple samples_dicts: one for each cluster
			for key, sd in samples_dict.items():
				train_loader, test_loader = start.get_data(fake_p, sd)
				self.try_method(start.test, fake_p, model, 
					criterion, test_loader)	

		else:
			train_loader, test_loader = start.get_data(fake_p, samples_dict)
			self.log.debug("TEST_LOADER: {}".format(test_loader))
			self.try_method(start.test, fake_p, model, 
				criterion, test_loader)	

	def try_method(self, method, *args):
		try:
			#start.test(fake_p, model, criterion, test_loader)
			method(*args)
			self.log.debug("TEST SUCCESS!")

		except Exception as e:
			self.log.debug("TEST FAIL, ERROR: {}".format(e))
			trace = traceback.format_exc()
			self.log.debug("STACK TRACE {}:".format(trace))
			assert(False)

	#@disabled
	def test_replace_zeroes_1200(self):
		self.log.debug("\n TEST 1200 \n")
		fake_samples = [[10, 0, 12], [14, 15, 16]]
		fake_samp = self.dp.replace_zeroes(fake_samples)	
		self.log.debug(fake_samp)
		assert(fake_samp == [[10, 1.0, 12], [14, 15, 16]])

	#@disabled
	def test_average_nans_1300(self):
		self.log.debug("\n TEST 1300 \n")
		fake_samples = [[float('nan'), 4.0, float('nan')], 
			[float('nan'), 8.0, 9.0]]
		fake_samples = self.dp.average_nans(fake_samples)
		self.log.debug("AVGD NaNs: {}".format(fake_samples))
		assert(fake_samples == [[4.0, 4.0, 4.0], [8.5, 8.0, 9.0]])

	@disabled
	def test_loss_plot_1400(self):
		''' Don't over-think shit - Officer kenny '''
		self.log.debug("\n TEST 1400 \n")
		# Fake params epochs is 3, data_loader will be 2 long, therefore
		# number of loss items recorded must be 6
		total_loss_lst = [1, 2, 3, 4, 5, 6]
		avg_err_lst = []

		# Our ip_ratio is 2 so overall, avg_acc will be iterated 3 times
		ip_ratio = fake_p['train_iter_to_print_ratio']

		# Arbitrary list that is 2 long, since the only reason we pass in
		# data_loader is so that we can use len(data_loader)
		data_loader = [21, 22]

		# Fake output from get_avg_err()
		avg_err_tensor = torch.rand(5)

		for i in range(fake_p['epochs'] * len(data_loader)):
			if (i+1) % ip_ratio == 0: 
				avg_err = start.get_tensor_avg(avg_err_tensor)
				avg_err_lst.append(avg_err)
		start.loss_plot(fake_p, total_loss_lst, avg_err_lst)

	@disabled
	def test_get_cluster_inds_1500(self):
		self.log.debug("\n TEST 1500 \n")
		k = 2
		# Elbow should be at 2 clusters
		zipped_volumes = [(70, 300), (74, 150), (69, 100), 
				(1000, 8), (2000, 5), (1500, 4)]
		cluster_indices = self.dp.get_cluster_inds(k, zipped_volumes)
		self.log.debug("CLUSTER_INDICES: {}".format(cluster_indices))
		# Means that the first three datapoints passed to kmeans are in 
		# cluster 0 and datapoints indexed 3-5 (or vice versa since clusters
		# are initialized randomly)
		assert(all(cluster_indices) == all(np.asarray([0, 0, 0, 1, 1, 1])) or
				(all(cluster_indices) == all(np.asarray([1, 1, 1, 0, 0, 0])))
			)

	@disabled
	def test_get_zipped_volumes_and_devs_1540(self):
		self.log.debug("\n TEST 1540 \n")
		#NOTE std deviations use Bessel's correction
		# Has been processed by get_segmented_data and each inner list
		# represents one company's volumes
		fake_volumes = [[0, 1, 2], [3, 4, 5]]
		zipped_volumes = self.dp.get_zipped_volumes_and_devs(fake_volumes)
		# [Average volume, std dev]
		assert(zipped_volumes == [[1., 1.], [4.0, 1.0]])

	@disabled
	def test_get_slopes_1550(self):
		self.log.debug("\n TEST 1550 \n")
		x = [0, 1, 2, 3]
		y = [2, 4, 8, 16]
		slopes = self.dp.get_slopes(x, y)
		assert(slopes == [(0, 2), (1, 4), (2, 8)])

	@disabled
	def test_get_auto_k_1600(self):
		self.log.debug("\n TEST 1600 \n")
		slopes = [(0, 2), (1, 4), (2, 8)]
		self.log.debug("SLOPES: {}".format(slopes))
		k = self.dp.get_auto_k(slopes)
		self.log.debug('k IN get_auto_k(): {}'.format(k))
		# Smallest slope is the first element in slopes, (0, 2) w slope of 2
		assert(k == 1)
		
	@disabled
	def test_group_volumes_by_cluster_1700(self):
		self.log.debug("\n TEST 1700 \n")
		k = 2
		fake_volumes = (1000, 2000, 3000, 1, 2, 3)
		fake_cluster_inds = np.asarray([1, 1, 1, 0, 0, 0])
		vol_clusters = self.dp.group_volumes_by_cluster(
			fake_volumes, fake_cluster_inds, k)
		assert(all(vol_clusters) == all([[1000, 2000, 3000], [1, 2, 3]]))

	@disabled
	def test_k_cluster_1700(self):
		self.log.debug("\n TEST 1700 \n")
		fake_p['k_auto'] = True
		# Format (volume, dev)
		zipped_volumes = [(70, 300), (74, 150), (69, 100), 
				(1000, 8), (2000, 5), (1500, 4)]
		vol_clusters = self.dp.k_cluster(fake_p, zipped_volumes)
		assert(
			all(vol_clusters) == all([[70, 74, 69], [1000, 2000, 1500]]) \
			or all([[1000, 2000, 1500], [70, 74, 69]])
			)

	def tearDown(self):
		pass
		
if __name__ == '__main__':
	unittest.main()

