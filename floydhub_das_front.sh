floyd init das

floyd run \
--data totoketchup/datasets/audio_norm_raw_16k/2:/h5py_files \
--data totoketchup/projects/das/179/output:/model \
--env tensorflow-1.4:py2 \
--tensorboard --gpu \
-m 'DAS train front' \
"python -m experiments.das_front"