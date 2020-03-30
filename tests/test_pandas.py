# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

import pandas as pd
import logging

# __name__ describes how the script is called, which is currently __main__
# If the file/func was imported then the name would be nameScript
def setup_logger(logger_level):
	''' Args: logger supports levels DEBUG, INFO, WARNING, ERROR, CRITICAL.
	logger_level should be passed in in the format logging.LEVEL '''

	logging.basicConfig(level=logger_level)
	logger = logging.getLogger(__name__)
	return logger

def read():
	print('\n READ \n')
	df = pd.read_csv('./test_data.csv')
	log.info(df.head())
	return df

def create_rows():
	print('\n CREATE \n')
	# With labels as rows
	df = pd.DataFrame([
	['jose', 'lydia', 'mr. olympian'], 
	[11, 124, 48], [593, 12]],
	index=['name', 'x', 'y'])
	log.info(df.head())
	return df

def create_cols():
	print('\n CREATE \n')
	# With labels as rows
	df = pd.DataFrame([
	['jose', 11, 593], 
	['lydia', 124, 12], ['Mr.Olympian', 48]],
	columns=['name', 'x', 'y'])
	log.info(df.head())
	return df

def select(df):
	print('\n SELECT \n')
	try:
		# .loc uses labels
		name_column = df['name']
		log.info(name_column)
	except KeyError as e:
		log.critical(e)
	# .iloc uses indexes/numbers, got the first row
	name_column = df.iloc[0]
	log.info(name_column)

# Splicing is inclusive
def splice(df):
	print('\n SPLICE \n')
	cols = df[['name', 'x']]
	log.info(cols)
	cols = df.loc['name', 'x']
	log.info(cols)

if __name__ == '__main__':
	log = setup_logger(logging.INFO)
	#df = read()
	#select(df)
	#splice(df)
	df = create_rows()
	select(df)
	df = create_cols()
	select(df)

