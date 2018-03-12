# coding: utf-8
from utils.trainer import MyArgs, Front_Separator_Trainer
from models.focus import FocusModel

if __name__ == '__main__':
	p = MyArgs()

	p.parser.add_argument(
		'--model_folder', help='Path to Adapt folder to load', required=True)

	p.add_separator_args()
	
	args = p.get_args()

	trainer = Front_Separator_Trainer(FocusModel, 'front_focus', pretraining=False, **vars(args))
	trainer.build_model()
	trainer.train()