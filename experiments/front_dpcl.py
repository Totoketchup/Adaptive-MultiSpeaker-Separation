from experiments.trainer import MyArgs, Front_Separator_Trainer
from models.dpcl import DPCL

if __name__ == '__main__':
	p = MyArgs()


	# Adapt model to load + params
	p.parser.add_argument(
		'--model_folder', help='Path to Adapt folder to load', required=True)

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

	trainer = Front_Separator_Trainer(DPCL, 'front_DPCL', pretraining=False, **vars(args))
	trainer.build_model()
	trainer.train()