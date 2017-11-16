floyd init das

floyd run \
--data totoketchup/datasets/audio_norm_raw_16k/2:/h5py_files \
--data totoketchup/projects/das/196/output:/model \
--env tensorflow-1.4:py2 \
--tensorboard --gpu \
-m "L41 front training" \
"python -m experiments.L41_front"

# THE 70 is 512 / 256 not same and good converged  'green-sound-9629'
# The 81th is 256 / 128 and the same
# The 83th is 512 / 256 and the same 'still-lab-4999