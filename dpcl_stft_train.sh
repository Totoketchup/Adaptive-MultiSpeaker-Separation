#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc nvcr-tensorflow-1712_g1 
#$ -e err_l41_stft_train.o
#$ -o std_l41_stft_train.o

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL 
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

. /fefs/opt/dgx/env_set/common_env_set.sh

export PATH="/home/anthony/anaconda2/bin:$PATH"
export LD_LIBRARY_PATH="/home/anthony/anaconda2/lib:$LD_LIBRARY_PATH"
export PATH="/home/anthony/anaconda2/lib:$PATH"
export LIBRARY_PATH="/home/anthony/anaconda2/lib:$LIBRARY_PATH"
export PATH="/home/anthony/.local/bin:$PATH"

python -m experiments.DPCL_STFT_training --dataset h5py_files/train-clean-100-8-s.h5 \
--chunk_size 20480 --nb_speakers 2 --no_random_picking --epochs 10 --batch_size 124 \
--learning_rate 0.001 --window_size 512 --hop_size 256 --layer_size 600 --embedding_size 40 > dpcl_stft_train.txt


