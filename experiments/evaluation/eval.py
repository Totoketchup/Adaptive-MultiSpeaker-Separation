# coding: utf-8
from utils.trainer import MyArgs, Front_Separator_Inference, STFT_inference, STFT_finetuned_inference
from models.L41 import L41Model
from utils.bss_eval import bss_eval_sources
import numpy as np

if __name__ == '__main__':
    p = MyArgs()

    p.parser.add_argument('--model_folder', help='Path to the Model folder to load', required=True) 
    p.select_inferencer()
    p.add_adapt_args()
    p.add_separator_args()    
    args = p.get_args()

    # Switch on different model Inferencer:
    if args.model == 'front_L41':
        inferencer = Front_Separator_Inference
    elif args.model == 'STFT_L41':
        inferencer = STFT_inference

    inferencer = inferencer(L41Model, 'inference', **vars(args))

    sdr = 0.0
    sir = 0.0
    sar = 0.0
    i = 0
    for mix, non_mix, separated in inferencer.inference():
        for m, n_m, s in zip(list(mix), list(non_mix), list(separated)):
            print m.shape, n_m.shape, s.shape
            mix_stack = np.array([m, m])
            
            no_separation = bss_eval_sources(n_m, mix_stack)
            separation = bss_eval_sources(n_m, s)

            sdr_ = np.mean(separation[0] - no_separation[0])
            sir_ = np.mean(separation[1] - no_separation[1])
            sar_ = np.mean(separation[2] - no_separation[2])
            print sdr_, sir_, sar_
            sdr += sdr_
            sir += sir_
            sar += sar_
            
            i += 1

    sdr /= float(i) 
    sir /= float(i) 
    sar /= float(i)
    print sdr, sir, sar