# coding: utf-8
from utils.trainer import MyArgs, STFT_Separator_Trainer
from models.L41 import L41Model

if __name__ == '__main__':
	p = MyArgs()
	
	p.add_stft_args()
	p.add_separator_args()
	
	args = p.get_args()

	trainer = STFT_Separator_Trainer(L41Model, 'STFT_L41', **vars(args))
	trainer.train()