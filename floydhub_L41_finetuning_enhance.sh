floyd init das

floyd run \
--data totoketchup/datasets/audio_norm_raw_16k/3:/h5py_files \
--data totoketchup/projects/das/237/output:/model2 \
--data totoketchup/projects/das/203/output:/model1 \
--env tensorflow-1.4:py2 \
--tensorboard --gpu \
-m 'finetuning with 237, enhance' \
"python -m experiments.L41_finetuning_enhance.py"