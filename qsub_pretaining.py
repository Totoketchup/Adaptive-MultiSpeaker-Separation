#!/usr/bin/env python
import subprocess
from itertools import product

_batch_size = [32]
_learning = [0.001]
_epochs = [20]

_window_size = [1024, 2048]
_max_pool = [64, 128, 256]
_filters = [256, 512]
_with_max_pool = [True]
_with_average_pool = [False]

_beta = [0.0, 1e-3]
_sparsity = [1e-3]
_regularization = [0.0, 1e-3]
_overlap_coef = [0.0, 1e-3]
_loss = ['l2']

_nb_speakers = [2]
_men = [True]
_women = [True]
_no_random_picking = [False]

print len(list(product(_batch_size, _learning, _epochs, _window_size, _max_pool, _filters, 
	_with_max_pool, _with_average_pool, _beta, _sparsity, _regularization, 
	_overlap_coef, _loss, _nb_speakers, _men, _women, _no_random_picking)))


i = 0 
for prod in product(_batch_size, _learning, _epochs, _window_size, _max_pool, _filters, 
	_with_max_pool, _with_average_pool, _beta, _sparsity, _regularization, 
	_overlap_coef, _loss, _nb_speakers, _men, _women, _no_random_picking):

	b, a, e, w, m, f, w_m, w_a, bt, s, r, o_c, l, n, me, wo, nr = prod
	
	if w_m and w_a:
		continue
	if ((me and not wo) and not nr) or ((not me and wo) and not nr):
		continue

	i += 1


	qsub_command = """qsub -v BATCH_SIZE={0},
								LEARNING_RATE={1},
								EPOCHS={2},
								WINDOW_SIZE={3},
								MAX_POOL={4},
								FILTERS={5},
								WITH_MAX_POOL={6},
								WITH_AVERAGE_POOL={7},
								BETA={8},
								SPARSITY={9},
								REGULARIZATION={10},
								OVERLAP_COEF={11} 
								LOSS={12},
								NB_SPEAKERS={13},
								MEN={14},
								WOMEN={15},
								NO_RANDOM_PICKING={16} job_pretraining.pbs"""\
								.format(b, a, e, w, m, f, 
									'--with_max_pool' if w_m else ' ', '--with_average_pool' if w_a else ' ', 
									bt, s, r, o_c, l, n, 
									'--men' if me else '' , '--women' if wo else ' ',
									'--no_random_picking' if nr else ' ')

	print qsub_command 

	# Comment the following 3 lines when testing to prevent jobs from being submitted
	exit_status = subprocess.call(qsub_command, shell=True)
	if exit_status is 1:  # Check to make sure the job submitted
		print "Job {0} failed to submit".format(qsub_command)

print i