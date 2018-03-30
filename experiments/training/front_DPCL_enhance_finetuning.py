# coding: utf-8
from utils.trainer import MyArgs, Front_Separator_Enhance_Finetuning_Trainer
from models.dpcl import DPCL

if __name__ == '__main__':
	p = MyArgs()

	# Adapt model to load + params
	p.parser.add_argument(
		'--model_folder', help='Path to model folder to load', required=True)

	p.add_adapt_args()
	p.add_separator_args()
	p.add_enhance_layer_args()
	
	args = p.get_args()

	trainer = Front_Separator_Enhance_Finetuning_Trainer(DPCL, 'front_DPCL_enhance_finetuning', pretraining=False, **vars(args))
	trainer.train()