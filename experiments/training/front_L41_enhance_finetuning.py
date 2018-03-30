# coding: utf-8
from utils.trainer import MyArgs, Front_Separator_Enhance_Finetuning_Trainer
from models.L41 import L41Model

if __name__ == '__main__':
	p = MyArgs()

	# Adapt model to load + params
	p.parser.add_argument(
		'--model_folder', help='Path to model folder to load', required=True)
	
	p.add_adapt_args()
	p.add_separator_args()
	p.add_enhance_layer_args()
	
	args = p.get_args()

	trainer = Front_Separator_Enhance_Finetuning_Trainer(L41Model, 'front_L41_enhance_finetuning', pretraining=False, **vars(args))
	trainer.train()