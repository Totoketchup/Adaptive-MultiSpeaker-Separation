# coding: utf-8
from utils.trainer import MyArgs, STFT_DPCL_Trainer

if __name__ == '__main__':
	p = MyArgs()
	#Preprocess arguments
	p.parser.add_argument(
		'--window_size', type=int, help='Size of the window for STFT', required=False, default=512)
	p.parser.add_argument(
		'--hop_size', type=int, help='Hop size for the STFT', required=False, default=256)
	#Network arguments
	p.parser.add_argument(
		'--layer_size', type=int, help='Size of hidden layers in BLSTM', required=False, default=600)
	p.parser.add_argument(
		'--embedding_size', type=int, help='Size of the embedding output', required=False, default=40)
	p.parser.add_argument(
		'--nonlinearity', help='Nonlinearity used', required=False, default='logistic')
	p.parser.add_argument(
		'--normalize', help='Normalization of the embedded space', action="store_false")

	args = p.get_args()

	trainer = STFT_DPCL_Trainer(**vars(args))
	trainer.build_model()
	trainer.train()