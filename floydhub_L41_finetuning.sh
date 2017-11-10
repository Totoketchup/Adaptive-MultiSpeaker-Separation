floyd init das

floyd run \
--data totoketchup/datasets/audio_norm_raw_16k/2:/h5py_files \
--data totoketchup/projects/das/134/output:/model \
--env tensorflow-1.4:py2 \
--tensorboard --gpu \
"python -m experiments.L41_finetuning.py"