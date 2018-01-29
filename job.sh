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
export PATH="/home/anthony/anaconda2/bin:$PATH"
export LD_LIBRARY_PATH="/home/anthony/anaconda2/lib:$LD_LIBRARY_PATH"

. /fefs/opt/dgx/env_set/common_env_set.sh

pip install --user -r requirements.txt
python -m experiments.pretraining