# coding: utf-8
from utils.trainer import MyArgs, STFT_Separator_Trainer
from models.SC_V2 import L41ModelV2

if __name__ == '__main__':
	p = MyArgs()

	p.parser.add_argument('--model_folder', help='Path to the Model folder to load', required=False, default=None)		
	
	p.add_stft_args()
	p.add_separator_args()
	
	args = p.get_args()

	trainer = STFT_Separator_Trainer(L41ModelV2, 'STFT_DANet_SCE', **vars(args))
	trainer.train()