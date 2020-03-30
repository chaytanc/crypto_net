# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

# Sources:
# https://docs.python.org/3/library/unittest.html

import torch
import pandas
import unittest
import sys
sys.path.append('../docs')
import nn

class Test_NN(unittest.TestCase):

	def setUp(self):
		self.dp = nn.Data_Processor('./test_data.csv')
		self.df = self.dp.get_df()

	def test_get_segmented_data_100(self):
		''' 
		This test uses the fake data made in ./test_data.csv, loaded above.
		'''
		# Make sure data is segmented based on when the the row counter in the
		# first column goes back to zero and then that company's number of rows
		# is appended to list which is returned
		rows_lst = self.dp.get_segmented_data(self.df)
		# Notice that rows_lst does not depend on the row number in the csv
		# being correct, but rather relies on an iteration counter from 
		# the loop so that the number of rows is always accurate
		assert rows_lst == [1, 2]

	def test_get_volumes_200(self):
		rows_lst = self.dp.get_segmented_data(self.df)
		# assumes the that volume column is the 8th col (df.col[7]) of the set
		volumes = self.dp.get_volumes(self.df, rows_lst)
		assert volumes == [[6], [100, 101]]

	def test_clean_data_300(self):
		fake_volumes = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]
		batch_and_label_size = 3
		cleaned = self.dp.clean_data(fake_volumes, batch_and_label_size)
		print('CLEANED: {}'.format(cleaned))
		assert(cleaned == [[0, 1, 2,], [6, 7, 8]])

	def test_label_data_400(self):
		fake_volumes = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]
		batch_and_label_size = 3
		label_size = 1
		# Cleaned data will look like: [[0, 1, 2], [6, 7, 8]] 
		cleaned = self.dp.clean_data(fake_volumes, batch_and_label_size)
		labels = self.dp.label_data(
			fake_volumes, batch_and_label_size, label_size)
		self.dp.log.debug("LABELS: {}".format(labels))
		assert(labels == [[2], [8]])
		assert(fake_volumes == [[0, 1], [6, 7]])

	def test_map_indices_500(self):
		fake_volumes = [[10, 11], [12]]
		mapp = self.dp.map_indices(fake_volumes)
		assert(mapp == {0:(0, 0), 1:(0, 1), 2:(1, 0)})
		lst_ind, num_ind = mapp[1]
		assert(lst_ind == 0 and num_ind == 1)

	def test_get_length_600(self):
		fake_volumes = [[10, 11], [12]]
		length = self.dp.get_length(fake_volumes)
		assert(length == 3)
		
	def tearDown(self):
		pass
		
if __name__ == '__main__':
	unittest.main()

