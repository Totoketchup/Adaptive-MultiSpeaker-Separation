floyd init das

floyd run \
--data totoketchup/datasets/audio_norm_raw_16k/2:/h5py_files \
--data totoketchup/projects/das/70/output:/model \
--env tensorflow-1.3:py2 \
--tensorboard --gpu \
"pip install --upgrade tensorflow-gpu==1.4rc1 && python -m experiments.dpcl_front"