# coding: utf-8
from utils.trainer import MyArgs, STFT_Separator_Trainer
from models.dpcl import DPCL

if __name__ == '__main__':
	p = MyArgs()

	p.add_stft_args()
	p.add_separator_args()
	
	args = p.get_args()

	trainer = STFT_Separator_Trainer(DPCL, 'STFT_DPCL',**vars(args))
	trainer.build_model()
	trainer.train()