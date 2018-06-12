# coding: utf-8
from utils.trainer import MyArgs, Front_Separator_Trainer
from models.SC_V2 import L41ModelV2

if __name__ == '__main__':
	p = MyArgs()

	# Adapt model to load + params
	p.parser.add_argument(
		'--model_folder', help='Path to Adapt folder to load', required=True)

	p.parser.add_argument(
		'--model_previous', help='Path to previous folder to load', required=False, default=None)

	p.add_separator_args()
	args = p.get_args()

	trainer = Front_Separator_Trainer(L41ModelV2, 'front_DANet_SCE', pretraining=False, **vars(args))
	trainer.train()