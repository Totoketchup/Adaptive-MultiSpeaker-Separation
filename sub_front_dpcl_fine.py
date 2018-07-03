#!/usr/bin/env python
import subprocess
from itertools import product

_learning = [1e-3, 1e-4, 1e-5]

_bk = [10.]

_model = [ ('ancient-dream-2393', True, True, False)]

_train = ['front back prediction enhance', 'prediction enhance', 'enhance', 'prediction']

i = 0 
for prod in product(_learning, _bk, _model, _train):

	l, bk, m, tr = prod

	mod = m[0]

	g = '\"'
	if m[1]:
		g += '--men ' 
	if m[2]:
		g += '--women ' 
	if m[3]:
		g += '--no_random_picking ' 
	g+='\"'
	tr2 = ''
	for t in tr.split(" "):
		tr2 += t[0]

	qsub_command = """qsub -v alpha={0},
								model="{1}",
								bk={2},
								train="{3}",
								train2={4},
								gender={5} 
								front_dpcl_finetuning.pbs""".format(l, mod, bk, tr, tr2, g)

	qsub_command = qsub_command.replace("\n", "")
	qsub_command = qsub_command.replace("\t", "")

	print qsub_command 
	# Comment the following 3 lines when testing to prevent jobs from being submitted
	exit_status = subprocess.call(qsub_command, shell=True)
	if exit_status is 1:  # Check to make sure the job submitted
		print "Job {0} failed to submit".format(qsub_command)
	break
print i