# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

# Parameters
#XXX need criterion, optim etc to be params, need to 
# condense sample_size and n_output etc...
fake_p = {
		'n_input' : 5, 'n_hidden1' : 5, 'n_hidden2' : 5, 'n_hidden3' : 5, 
		'n_hidden4' : 3, 'n_output' : 2, 'lr' : 0.0001, 'epochs' : 3, 
		'sample_and_label_size' : 7, 
		'label_size' : 2, 'sample_size' : 5, 'train_fraction' : 0.8,
		'log_file' : './log.log', 'type' : 'simple', 'load_processing' : False,
		'obj_filename' : 'samples_dict.pkl', 
		'stats_output_file' : './fake_stats.csv', 'overwrite_stats' : True,
		'dropout_prob' : 0.2, 'tensorboard_dir' : './fake_tb',
		'visualize' : True, 'load_model' : False, 
		'model_filename' : 'model.pkl', 'train_iter_to_print_ratio' : 2,
		'kmeans' : True, 'k_auto' : False, 'cluster_range' : range(1, 3)
	}


