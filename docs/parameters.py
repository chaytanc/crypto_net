# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

# Parameters
#XXX need criterion, optim etc to be params, need to 
# condense sample_size and n_output etc...
p = {
		'n_input' : 95, 'n_hidden1' : 20, 'n_hidden2' : 20, 'n_hidden3' : 15, 
		'n_hidden4' : 10, 'n_output' : 5, 'lr' : 0.0001, 'epochs' : 7, 
		'sample_and_label_size' : 100, 
		'label_size' : 5, 'sample_size' : 95, 'train_fraction' : 0.9,
		'log_file' : './log.log', 'type' : 'deep', 'load_processing' : True,
		'obj_filename' : 'samples_dict.pkl'
	}


