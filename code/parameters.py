# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

from hyperopt import hp
from torch import nn

# If you want to optimize hyperparams, set to True! Run as normal? Set to False
bayesian_optimization = True

# Parameters
#XXX need criterion, optim etc to be params, need to 
# condense sample_size and n_output etc...
p = {
		'n_input' : 95, 'n_hidden1' : 20, 'n_hidden2' : 20, 'n_hidden3' : 15, 
		'n_hidden4' : 10, 'n_output' : 5, 'lr' : 0.0001, 'epochs' : 7, 
		'sample_and_label_size' : 100, 
		'label_size' : 5, 'sample_size' : 95, 'train_fraction' : 0.9,
		'log_file' : './log.log', 'type' : 'deep', 
		'obj_filename' : 'samples_dict.pkl', 
		'stats_output_file' : './stats.csv', 'overwrite_stats' : False,
		'dropout_prob' : 0.2, 'tensorboard_dir' : './tb', 
		'load_processing' : True,
		'visualize' : True, 'load_model' : True, 
		'model_filename' : 'model.pkl', 'train_iter_to_print_ratio' : 100,
		'kmeans' : True, 'k_auto' : False, 'cluster_range' : range(1, 13),
	}

# Hyperopt parameters
hp_p = {
		#XXX finna need me some AWS
		'hidden_layer_range' : (2, 15),
		# Number of layers if not using a range
		#'total_hidden_layers' : 5,
		# N Neurons per layer
		'hidden_node_range' : (6, 30),
		# That's gonna be painful...XXX AWS?
		#XXX if it ends up selecting the high end of this range then you need
		# more epochs still...
		'epochs_range' : (4, 200),
		'lr_range' : (0.000000001, 0.5),
		'dropout_prob_range' : (0.0, 0.9)	
		}

layers_p = {
		'n_input' : 95, 
		'n_output' : 5, 
		'sample_and_label_size' : 100, 'label_size' : 5, 'sample_size' : 95, 

		'train_fraction' : 0.9, 
		'type' : 'auto', 
		'load_processing' : True,
		'log_file' : './log.log', 
		'obj_filename' : 'samples_dict.pkl', 
		'stats_output_file' : './stats.csv', 
		'overwrite_stats' : False,
		'tensorboard_dir' : './tb',
		'visualize' : True,
		'load_model' : False,
		'model_filename' : 'model.pkl',
		'train_iter_to_print_ratio' : 100,
		'kmeans' : True, 'k_auto' : False,
		'cluster_range' : range(1, 13),

		# quniform rounds for discrete vals and if q=1 then it rounds normally
		'n_hidden_layers' : hp.quniform(
			label = 'n_hidden_layers',
			low = hp_p['hidden_layer_range'][0],
			high = hp_p['hidden_layer_range'][1],
			q = 1
			),
		}

def make_hiddens(hp_p, new_p):
	''' This updates new_p so that the number of total layers and the 
		neurons per layer are adjustable based on hp_p '''

	# Need to run n_hidden_layers as a separate optimization because we can't
	# access the on the fly samplings of the number of hidden layers and use 
	# that as an int -- it is many integers
	#for x in range(new_p['n_hidden_layers']):
	for x in range(new_p['n_hidden_layers']):
		# Favors lower number of neurons
		# Has the + 1 so that it doesn't start at 0
		name = 'n_hidden{}'.format(x + 1)
		low, high = hp_p['hidden_node_range']
		# Fills w/ hyperopt's Apply objects
		new_p[name] = hp.qloguniform(
			label = name, 
			low = low,
			high = high,
			q = 1)

#NOTE for bayesian hyperparams
#XXX need to make ranges for new optional ??
#XXX can just make new_p an extension of layers_p so that we don't repeat
new_p = {
		'n_input' : 95, 
		'n_output' : 5, 
		'sample_and_label_size' : 100, 'label_size' : 5, 'sample_size' : 95, 

		'train_fraction' : 0.9, 
		'type' : 'auto', 
		'load_processing' : True,
		'log_file' : './log.log', 
		'obj_filename' : 'samples_dict.pkl', 
		'stats_output_file' : './stats.csv', 
		'overwrite_stats' : False,
		'tensorboard_dir' : './tb',
		'visualize' : True,
		'load_model' : False,
		'model_filename' : 'model.pkl',
		'train_iter_to_print_ratio' : 100,
		'kmeans' : True, 'k_auto' : False,
		'cluster_range' : range(1, 13),

		#NOTE loguniform skews right, centered toward low (left) numbers
		#NOTE you can also use hp.normal or qnormal etc... for unbounded but
		# centered distributions whereas anything w uniform is bounded
		
		# quniform rounds for discrete vals and if q=1 then it rounds normally
#		'n_hidden_layers' : hp.quniform(
#			label = 'n_hidden_layers',
#			low = hp_p['hidden_layer_range'][0],
#			high = hp_p['hidden_layer_range'][1],
#			q = 1
#			),
		'epochs' : hp.quniform(
			label = 'epochs', 
			low = hp_p['epochs_range'][0],
			high = hp_p['epochs_range'][1],
			q = 1
			), 
		'lr' : hp.uniform(
			'lr', 
			*hp_p['lr_range']
			), 
		'dropout_prob' : hp.uniform(
			'dropout_prob',
		   *hp_p['dropout_prob_range']
		   ),
	}

# This updates new_p with the n_hiddens which are the number of neurons each
# hidden layer has. This is done based on the number of layers made, so it
# had to be done after new_p was constructed
#XXX we may want make_hiddens to be a function outside of parameters because
# we may want to update new_p from start.py
#XXX made this called in start.py after we have optimized n_hidden_layers
#make_hiddens(hp_p, new_p)

