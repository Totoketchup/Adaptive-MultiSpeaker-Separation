# coding: utf-8
from utils.trainer import MyArgs, STFT_Separator_FineTune_Trainer
from models.L41 import L41Model
if __name__ == '__main__':
	p = MyArgs()

	p.parser.add_argument('--model_folder', help='Path to the Model folder to load', required=True)	
	p.add_stft_args()
	p.add_finetuning_args()
	p.add_separator_args()
	
	args = p.get_args()

	trainer = STFT_Separator_FineTune_Trainer(L41Model, 'STFT_L41_finetuning', **vars(args))
	trainer.train()