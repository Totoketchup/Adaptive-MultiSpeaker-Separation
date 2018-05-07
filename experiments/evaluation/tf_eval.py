# coding: utf-8
from utils.trainer import *
from models.L41 import L41Model

if __name__ == '__main__':
	p = MyArgs()

	p.parser.add_argument('--model_folder', help='Path to the Model folder to load', required=True) 
	p.select_inferencer()
	p.add_adapt_args()
	p.add_separator_args()
	args = p.get_args()

	# Switch on different model Inferencer:
	if args.model == 'front_L41':
		inferencer = Front_Separator_Inference
	elif args.model == 'STFT_L41':
		inferencer = STFT_inference
	elif args.model == 'front_L41_enhance':
		inferencer = Front_Separator_Enhanced_Inference
	elif args.model == 'pretraining':
		inferencer = Pretrained_Inference

	inferencer = inferencer(L41Model, 'inference', **vars(args))

	sdr = 0.0

	i = 0
	for _, _, sdr_ in inferencer.sdr_improvement():
		sdr += sdr_*args.batch_size
		i += 1
		print sdr/float(i*args.batch_size), sdr_


	print 'SDR =', sdr/float(i*args.batch_size)