# coding: utf-8
from utils.trainer import MyArgs, Front_Separator_Enhance_Trainer
from models.dpcl import DPCL

if __name__ == '__main__':
	p = MyArgs()


	# Adapt model to load + params
	p.parser.add_argument(
		'--model_folder', help='Path to Adapt folder to load', required=True)

	p.add_separator_args()
	p.add_enhance_layer_args()
	args = p.get_args()

	trainer = Front_Separator_Enhance_Trainer(DPCL, 'front_DPCL_enhance', pretraining=False, **vars(args))
	trainer.build_model()
	trainer.train()