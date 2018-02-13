# coding: utf-8
from utils.trainer import MyArgs, Adapt_Pretrainer

if __name__ == '__main__':
	p = MyArgs()
	
	#Preprocess arguments
	p.parser.add_argument(
		'--window_size', type=int, help='Size of the 1D Conv width', required=False, default=1024)
	p.parser.add_argument(
		'--filters', type=int, help='Number of filters/bases for the 1D Conv', required=False, default=512)
	p.parser.add_argument(
		'--max_pool', type=int, help='Max Pooling size', required=False, default=512)

	#Loss arguments
	p.parser.add_argument(
		'--regularization', type=float, help='Coefficient for L2 regularization', required=False, default=1e-4)
	p.parser.add_argument(
		'--beta', type=float, help='Coefficient for Sparsity constraint', required=False, default=1e-2)
	p.parser.add_argument(
		'--sparsity', type=float, help='Average Sparsity constraint', required=False, default=0.01)
	p.parser.add_argument(
		'--overlap_coef', type=float, help='Coefficient for Overlapping loss', required=False, default=1e-4)

	args = p.get_args()

	trainer = Adapt_Pretrainer(pretraining=True, **vars(args))
	trainer.build_model()
	trainer.train()

#Network arguments
	