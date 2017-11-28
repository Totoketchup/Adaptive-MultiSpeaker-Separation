floyd init das

floyd run --data totoketchup/datasets/audio_norm_raw_16k/3:/h5py_files \
 --env tensorflow-1.4:py2 \
 -m "pretraining with 'SAAAME' convolutions" \
 --tensorboard --gpu \
 "python -m experiments.pretraining"