# vim: set noexpandtab sw=4 ts=4 noautoindent fileencoding=utf-8:

from hyperopt import hp

# Parameters
#XXX need criterion, optim etc to be params, need to 
# condense sample_size and n_output etc...
p = {
		'n_input' : 95, 'n_hidden1' : 20, 'n_hidden2' : 20, 'n_hidden3' : 15, 
		'n_hidden4' : 10, 'n_output' : 5, 'lr' : 0.0001, 'epochs' : 7, 
		'sample_and_label_size' : 100, 
		'label_size' : 5, 'sample_size' : 95, 'train_fraction' : 0.9,
		'log_file' : './log.log', 'type' : 'deep', 'load_processing' : True,
		'obj_filename' : 'samples_dict.pkl', 
		'stats_output_file' : './stats.csv', 'overwrite_stats' : False,
		'dropout_prob' : 0.2, 'tensorboard_dir' : './tb', 
		'visualize' : True, 'load_model' : True, 
		'model_filename' : 'model.pkl', 'train_iter_to_print_ratio' : 100,
		'kmeans' : True, 'k_auto' : False, 'cluster_range' : range(1, 13),
	}

# Hyperopt parameters
hp_p = {
		'total_hidden_layers' : 5,
		'n_hidden_range' : (6, 30),
		# That's gonna be painful...XXX AWS?
		'epochs_range' : (4, 200),
		'lr_range' : (0.000000001, 0.5),
		'dropout_prob_range' : (0.0, 0.9)	
		#XXX batch size???
		}

def make_hiddens(hp_p, new_p):
	for x in range(hp_p['total_hidden_layers']):
		# Favors lower number of neurons
		name = 'n_hidden{}'.format(x)
		new_p[name] = hp.loguniform(name, *hp_p['n_hidden_range'])

#XXX for bayesian hyperparams
new_p = {
		'n_input' : 95, 'n_output' : 5, 
		'sample_and_label_size' : 100, 'label_size' : 5, 'sample_size' : 95, 
		'train_fraction' : 0.9, 
		'type' : 'auto', 

		'lr' : hp.uniform('lr', *hp_p['lr_range']), 
		'epochs' : hp.uniform('epochs', *hp_p['epochs_range']), 
		'dropout_prob' : hp.uniform(
			'dropout_prob', *hp_p['dropout_prob_range']),

		'load_processing' : True,
		'log_file' : './log.log', 
		'obj_filename' : 'samples_dict.pkl', 
		'stats_output_file' : './stats.csv', 
		'overwrite_stats' : False,
		'tensorboard_dir' : './tb',
		'visualize' : True,
		'load_model' : True,
		'model_filename' : 'model.pkl',
		'train_iter_to_print_ratio' : 100,
		'kmeans' : True, 'k_auto' : False,
		#XXX optimize
		'cluster_range' : range(1, 13),
	}

make_hiddens(hp_p, new_p)

