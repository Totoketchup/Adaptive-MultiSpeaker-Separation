floyd init das

floyd run \
--data totoketchup/datasets/audio_norm_raw_16k/3:/h5py_files \
--data totoketchup/projects/das/221/output:/model \
--env tensorflow-1.4:py2 \
--tensorboard --gpu \
-m 'enhance L41 with job 221 (3 BLSTM, no reg)' \
"python -m experiments.L41_enhance"
