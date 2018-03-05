# Adaptive and Focus Layer for Multi-Speaker Separation Problem

This repository contains the implementation of an Adaptive Layer and Focus Layer for the Multi-Speaker Separation Problem.
The Adaptive Layer consists in a sparse linear AutoEncoder replacing the use of Spectograms and dealing directly with raw audio files.
This AutoEncoder is added around the current following state of the art architectures:
* Deep Clustering Model (with and without enhancing layer)
* L41 (Magnolia) Model (with and without enhancing layer)

These are compared to traditional STFT approaches with the following architectures:
* Deep Clustering Model (with and without enhancing layer)
* L41 (Magnolia) Model (with and without enhancing layer)
* Dense Model (TODO) with 2 speakers
* PIT Model (TODO) with 2 speakers

The Focus Layer is in constuction.

## Training

## Adapt Layer
## Pretraining

```
python -m experiments.training.pretraining --men --women --loss sdr+l2 --separation mask --learning_rate 0.001 --nb_speakers 2 --batch_size 4 --filters 256 --max_pool 256 --beta 0.0 --regularization 0.0 --overlap_coef 1.0 --no_random_picking
```

#### Arguments:

##### Related to Data:

* --men: Use men voices
* --women: Use women voices
* --nb_speakers: Number of mixed speakers
* --no_random_picking: Regular mixing (Man + Female + Man + Female etc...) instead of random mixing

##### Related to Hyperparams for the Adapt Layer:

Architecture:

* --filters: Number of AutoEncoder Bases
* --max_pool: Max pooling size along time axis

Coefficients:

* --beta: Coefficient for Sparsity loss : KL divergence
* --regularization: Coefficient for L2 reg on weights
* --overlap_coef: Coefficient for the overlapping loss
* --loss: l2 / sdr / l2+sdr

##### Related to Hyperparams for Training:

* --batch_size (int)
* --learning_rate (float)

#### Current requirements:
```
scipy==1.0.0
tqdm==4.19.4
SoundFile==0.9.0.post1
matplotlib==2.1.0
numpy==1.12.0
tensorflow_gpu==1.4.0
librosa==0.5.1
haikunator==2.1.0
h5py==2.7.0
scikit_learn==0.19.1
tensorflow==1.5.0rc1
```



