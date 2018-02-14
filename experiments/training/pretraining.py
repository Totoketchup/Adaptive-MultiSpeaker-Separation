# coding: utf-8
from utils.trainer import MyArgs, Adapt_Pretrainer

if __name__ == '__main__':
	p = MyArgs()
	
	#Preprocess arguments
	p.add_adapt_args()
	args = p.get_args()

	trainer = Adapt_Pretrainer(pretraining=True, **vars(args))
	trainer.build_model()
	trainer.train()

#Network arguments
	