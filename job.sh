#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc nvcr-tensorflow-1712_g1 
#$ -e err_pretraining.o
#$ -o std_pretraining.o

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL 
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL


. /fefs/opt/dgx/env_set/common_env_set.sh
#. /fefs/opt/dgx/env_set/nvcr-tensorflow-1712.sh

export PATH="/home/anthony/anaconda2/bin:$PATH"
export LD_LIBRARY_PATH="/home/anthony/anaconda2/lib:$LD_LIBRARY_PATH"
export PATH="/home/anthony/anaconda2/lib:$PATH"
export LIBRARY_PATH="/home/anthony/anaconda2/lib:$LIBRARY_PATH"
export PATH="/home/anthony/.local/bin:$PATH"



#printenv
#pip install --user -r requirements.txt
#pip install tensorflow-gpu
#nvidia-smi
python -m experiments.pretraining
#python test_tf.py
#pip uninstall tensorflow
#pip uninstall tensorflow-gpu
#pip install --user tensorflow-gpu
#conda remove tensorflow
#conda install tensorflow-gpu
