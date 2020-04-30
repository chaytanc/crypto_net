# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:
import pandas as pd

def slices(path, out_path):
	df = pd.read_csv(path)
	#XXX may need '0'
	new = df.iloc[0:200]
	print(new)
	new.to_csv(out_path)
	more = df.iloc[5000:5100]
	more.to_csv(out_path, mode='a', header=False)

slices('../docs/all_currencies.csv', 'test_volumes.csv')
