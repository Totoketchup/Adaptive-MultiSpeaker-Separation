#!/usr/bin/env python 
# -*- coding: utf-8 -*-

'''
Source separation algorithms attempt to extract recordings of individual
sources from a recording of a mixture of sources.  Evaluation methods for
source separation compare the extracted sources from reference sources and
attempt to measure the perceptual quality of the separation.

See also the bss_eval MATLAB toolbox:
    http://bass-db.gforge.inria.fr/bss_eval/

Conventions
-----------

An audio signal is expected to be in the format of a 1-dimensional array where
the entries are the samples of the audio signal.  When providing a group of
estimated or reference sources, they should be provided in a 2-dimensional
array, where the first dimension corresponds to the source number and the
second corresponds to the samples.

Metrics
-------

* :func:`mir_eval.separation.bss_eval_sources`: Computes the bss_eval_sources
  metrics from bss_eval, which optionally optimally match the estimated sources
  to the reference sources and measure the distortion and artifacts present in
  the estimated sources as well as the interference between them.

* :func:`mir_eval.separation.bss_eval_sources_framewise`: Computes the
  bss_eval_sources metrics on a frame-by-frame basis.

* :func:`mir_eval.separation.bss_eval_images`: Computes the bss_eval_images
  metrics from bss_eval, which includes the metrics in
  :func:`mir_eval.separation.bss_eval_sources` plus the image to spatial
  distortion ratio.

* :func:`mir_eval.separation.bss_eval_images_framewise`: Computes the
  bss_eval_images metrics on a frame-by-frame basis.

References
----------
  .. [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.


This code is licensed under the MIT License:
The MIT License (MIT)

Copyright (c) 2014 Colin Raffel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Please see http://craffel.github.io/mir_eval/ for more information
'''

import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import collections
import itertools
import warnings
import tensorflow as tf
#from . import util
import cupy as cp

# The maximum allowable number of sources (prevents insane computational load)
MAX_SOURCES = 100

def validate(reference_sources, estimated_sources):
    """Checks that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources

    """

    if reference_sources.shape != estimated_sources.shape:
        raise ValueError('The shape of estimated sources and the true '
                         'sources should match.  reference_sources.shape '
                         '= {}, estimated_sources.shape '
                         '= {}'.format(reference_sources.shape,
                                       estimated_sources.shape))

    if reference_sources.ndim > 3 or estimated_sources.ndim > 3:
        raise ValueError('The number of dimensions is too high (must be less '
                         'than 3). reference_sources.ndim = {}, '
                         'estimated_sources.ndim '
                         '= {}'.format(reference_sources.ndim,
                                       estimated_sources.ndim))

    if reference_sources.size == 0:
        warnings.warn("reference_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif _any_source_silent(reference_sources):
        raise ValueError('All the reference sources should be non-silent (not '
                         'all-zeros), but at least one of the reference '
                         'sources is all 0s, which introduces ambiguity to the'
                         ' evaluation. (Otherwise we can add infinitely many '
                         'all-zero sources.)')

    if estimated_sources.size == 0:
        warnings.warn("estimated_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif _any_source_silent(estimated_sources):
        raise ValueError('All the estimated sources should be non-silent (not '
                         'all-zeros), but at least one of the estimated '
                         'sources is all 0s. Since we require each reference '
                         'source to be non-silent, having a silent estimated '
                         'source will result in an underdetermined system.')

    if (estimated_sources.shape[0] > MAX_SOURCES or
            reference_sources.shape[0] > MAX_SOURCES):
        raise ValueError('The supplied matrices should be of shape (nsrc,'
                         ' nsampl) but reference_sources.shape[0] = {} and '
                         'estimated_sources.shape[0] = {} which is greater '
                         'than mir_eval.separation.MAX_SOURCES = {}.  To '
                         'override this check, set '
                         'mir_eval.separation.MAX_SOURCES to a '
                         'larger value.'.format(reference_sources.shape[0],
                                                estimated_sources.shape[0],
                                                MAX_SOURCES))


def _any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return np.any(np.all(np.sum(
        sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))


def bss_eval_sources(reference_sources, estimated_sources,
                     compute_permutation=True):
    """
    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Passing ``False`` for ``compute_permutation`` will improve the computation
    performance of the evaluation; however, it is not always appropriate and
    is not the way that the BSS_EVAL Matlab toolbox computes bss_eval_sources.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_sources(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have same shape as
        reference_sources)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``). Note: ``perm`` will be ``[0, 1, ...,
        nsrc-1]`` if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
        Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
        Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
        (2007-2010): Achievements and remaining challenges", Signal Processing,
        92, pp. 1928-1936, 2012.

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = estimated_sources.shape[0]

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt(reference_sources,
                                        estimated_sources[jest],
                                        jtrue, 512)
                sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], sir[idx], sar[idx], np.asarray(popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                _bss_decomp_mtifilt(reference_sources,
                                    estimated_sources[j],
                                    j, 512)
            sdr[j], sir[j], sar[j] = \
                _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (sdr, sir, sar, popt)

def _bss_decomp_mtifilt(reference_sources, estimated_source, j, flen):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """
    nsampl = estimated_source.size
    # decomposition
    # true source image
    s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))

    # spatial (or filtering) distortion
    e_spat = _project(reference_sources[j, np.newaxis, :], estimated_source,
                      flen) - s_true

    # interference
    e_interf = _project(reference_sources,
                        estimated_source, flen) - s_true - e_spat

    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:nsampl] += estimated_source

    return (s_true, e_spat, e_interf, e_artif)

def _project(reference_sources, estimated_source, flen):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nsrc, flen - 1))))
    

    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))

    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)
    # np.save('sf64', sf)
    # exit()
    
    # inner products between delayed versions of reference_sources
    G = np.zeros((nsrc * flen, nsrc * flen))
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * np.conj(sf[j])            
            ssf = np.real(scipy.fftpack.ifft(ssf))
           
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                          r=ssf[:flen])

            G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T

    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros(nsrc * flen)
    for i in range(nsrc):
        ssef = sf[i] * np.conj(sef)
        ssef = np.real(scipy.fftpack.ifft(ssef))
        D[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    # Computing projection
    # Distortion filters
    
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order='F')

    # Filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc):
        sproj += fftconvolve(C[:, i], reference_sources[i])[:nsampl + flen - 1]
    return sproj

def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db(np.sum(s_filt**2), np.sum((e_interf + e_artif)**2))
    sir = _safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_filt + e_interf)**2), np.sum(e_artif**2))
    return (sdr, sir, sar)

def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.
    """
    if den == 0:
        return np.Inf
    return 10 * np.log10(num / den)



def bss_eval_sources_tf(reference_sources, estimated_sources,
                     compute_permutation=True, nsrc=2):
    """
    """

    # make sure the input is of shape (nsrc, nsampl)
    nsampl = tf.shape(estimated_sources)[-1]
    estimated_sources = tf.cast(tf.reshape(estimated_sources, [nsrc, nsampl]), tf.float64)
    reference_sources = tf.cast(tf.reshape(reference_sources, [nsrc, nsampl]), tf.float64)

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = [[[]]*nsrc]*nsrc
        sir = [[[]]*nsrc]*nsrc
        sar = [[[]]*nsrc]*nsrc
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt_tf(reference_sources,
                                        estimated_sources[jest],
                                        jtrue, 512, nsrc)
                sdr[jest][jtrue], sir[jest][jtrue], sar[jest][jtrue] = \
                    _bss_source_crit_tf(s_true, e_spat, e_interf, e_artif)

        return sdr       
        sdr = tf.stack(sdr)
        sir = tf.stack(sir)
        sar = tf.stack(sar)

        return sdr, sir, sar

        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], sir[idx], sar[idx], np.asarray(popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                _bss_decomp_mtifilt(reference_sources,
                                    estimated_sources[j],
                                    j, 512)
            sdr[j], sir[j], sar[j] = \
                _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (sdr, sir, sar, popt)



def _bss_decomp_mtifilt_tf(reference_sources, estimated_source, j, flen, nsrc):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """

    # decomposition
    # true source image

    s_true = tf.concat([reference_sources[j], tf.zeros([flen - 1], dtype=tf.float64)], 0)

    # spatial (or filtering) distortion
    e_spat = _project_tf(tf.expand_dims(reference_sources[j, :], 0), estimated_source, flen, 1) - s_true
    
    # interference
    e_interf = _project_tf(reference_sources, estimated_source, flen, nsrc) - s_true - e_spat
    
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    
    e_artif += tf.concat([estimated_source, tf.zeros([flen - 1], dtype=tf.float64),], 0)

    return (s_true, e_spat, e_interf, e_artif)

def toeplitz_tf(col, row):
    L = tf.shape(col)[0]
    R = tf.shape(row)[0]
    r = tf.range(L)

    def concat(i):
        begin = tf.clip_by_value(i-R+1, 0, L)
        size = tf.clip_by_value(i+1, 1, R)
        left = tf.reverse(tf.slice(col,[begin], [size]), [0])

        size = tf.clip_by_value(R-i-1, 0, R-i)
        right = tf.slice(row, [1], [size])
        return tf.concat([left, right], 0)

    out = tf.map_fn(lambda i: tf.cast(concat(i), dtype=tf.float64), r, dtype=tf.float64)
    return out



def _project_tf(reference_sources, estimated_source, flen, nsrc):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    # nsrc = tf.shape(reference_sources)[0]
    nsampl = tf.shape(reference_sources)[1]
    typ = reference_sources.dtype

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = tf.concat([reference_sources, tf.zeros([nsrc, flen - 1], dtype=typ)], 1)

    estimated_source = tf.concat([estimated_source, tf.zeros([flen - 1], dtype=typ)], 0)

    top = tf.cast(nsampl + flen - 1, dtype=tf.float64)
    two = tf.constant(2.,tf.float64)
    n_fft = tf.cast(tf.pow(two, tf.ceil(tf.log(top)/tf.log(two))), tf.int32)

    pad_reference_sources = tf.pad(reference_sources, [[0,0],[0,n_fft-tf.shape(reference_sources)[-1]]])
    pad_estimated_source = tf.pad(estimated_source, [[0,n_fft-tf.shape(reference_sources)[-1]]])

    sf = tf.spectral.fft(tf.cast(tf.complex(pad_reference_sources,tf.constant(0.,tf.float64)),tf.complex64))
    sef = tf.spectral.fft(tf.cast(tf.complex(pad_estimated_source,tf.constant(0.,tf.float64)),tf.complex64))

    # inner products between delayed versions of reference_sources
    G = tf.zeros([nsrc * flen, nsrc * flen], dtype=tf.float64)
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * tf.conj(sf[j])
            ssf = tf.real(tf.ifft(ssf))
            ss = toeplitz_tf(tf.concat([tf.reshape(ssf[0],[1]), tf.reverse(ssf[-flen+1:],[0])],0), ssf[:flen])
            
            paddings = tf.constant([[j * flen, nsrc * flen - (j+1) * flen],
                [i * flen, nsrc * flen - (i+1) * flen]])

            G +=  tf.pad(tf.transpose(ss), paddings, "CONSTANT")

            if i != j: 
                paddings = tf.constant([[i * flen, nsrc * flen - (i+1) * flen],
                    [j * flen, nsrc * flen - (j+1) * flen]])
                G += tf.pad(ss, paddings, "CONSTANT")

    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = tf.zeros([nsrc * flen], dtype=tf.float64)
    for i in range(nsrc):
        ssef = sf[i] * tf.conj(sef)
        ssef = tf.cast(tf.real(tf.ifft(ssef)), tf.float64)
        paddings = tf.constant([[i * flen, nsrc * flen - (i+1) * flen]])
        conc = tf.concat([tf.reshape(ssef[0], [1]), tf.reverse(ssef[-flen+1:],[0])],0)
        D += tf.pad(conc, paddings, "CONSTANT")

    # Computing projection
    # Distortion filters

    s = tf.linalg.solve(G, tf.expand_dims(D,1))
    C = tf.reshape(s, [flen, nsrc])

    # Filtering
    sproj = tf.zeros([nsampl + flen - 1], dtype=tf.float64)

    for i in range(nsrc):
        r = tf.pad(reference_sources[i], [[C[:, i].shape[0]-1, C[:, i].shape[0]-1]], "CONSTANT")
        data   = tf.reshape(r, [1, int(r.shape[0]), 1], name='data')
        kernel = tf.reshape(C[:, i], [int(C[:, i].shape[0]), 1, 1], name='kernel')
        conv = tf.reshape(tf.nn.conv1d(data, kernel, 1, 'VALID'), [-1])
        sproj += conv[:nsampl + flen - 1]
    return sproj

def _bss_source_crit_tf(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db_tf(tf.reduce_sum(tf.square(s_filt)), tf.reduce_sum(tf.square(e_interf + e_artif)))
    sir = _safe_db_tf(tf.reduce_sum(tf.square(s_filt)), tf.reduce_sum(tf.square(e_interf)))
    sar = _safe_db_tf(tf.reduce_sum(tf.square(s_filt + e_interf)), tf.reduce_sum(tf.square(e_artif)))
    return (sdr, sir, sar)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10., dtype=numerator.dtype))
    return numerator / denominator

def _safe_db_tf(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.
    """
    return 10.0 * log10(num / (den + 1e-12))


def bss_eval_sources_cupy(reference_sources, estimated_sources,
                     compute_permutation=True, nsrc=2):
    """
    

    """

    # make sure the input is of shape (nsrc, nsampl)
    nsampl = estimated_sources.shape[-1]
    estimated_sources = cp.reshape(cp.array(estimated_sources, dtype=cp.float64), [nsrc, nsampl])
    reference_sources = cp.reshape(cp.array(reference_sources, dtype=cp.float64), [nsrc, nsampl])

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = cp.empty((nsrc, nsrc))
        sir = cp.empty((nsrc, nsrc))
        sar = cp.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt_cupy(reference_sources,
                                        estimated_sources[jest],
                                        jtrue, 512, nsrc)
                sdr[jest,jtrue], sir[jest,jtrue], sar[jest,jtrue] = \
                    _bss_source_crit_cupy(s_true, e_spat, e_interf, e_artif)
        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(cp.asnumpy(sir)[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (cp.asnumpy(sdr)[idx], cp.asnumpy(sir)[idx], cp.asnumpy(sar)[idx], np.asarray(popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                _bss_decomp_mtifilt(reference_sources,
                                    estimated_sources[j],
                                    j, 512)
            sdr[j], sir[j], sar[j] = \
                _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (sdr, sir, sar, popt)



def _bss_decomp_mtifilt_cupy(reference_sources, estimated_source, j, flen, nsrc):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """

    # decomposition
    # true source image

    s_true = cp.concatenate((reference_sources[j], cp.zeros([flen - 1], dtype=cp.float64)), 0)

    # spatial (or filtering) distortion
    e_spat = _project_cupy(cp.expand_dims(reference_sources[j, :], 0), estimated_source, flen, 1) - s_true

    # interference
    e_interf = _project_cupy(reference_sources, estimated_source, flen, nsrc) - s_true - e_spat

    # artifacts
    e_artif = -s_true - e_spat - e_interf
    
    e_artif += cp.concatenate((estimated_source, cp.zeros([flen - 1], dtype=cp.float64)), 0)

    return (s_true, e_spat, e_interf, e_artif)

def toeplitz_cupy(c, r):
    vals = cp.concatenate((r[-1:0:-1], c))
    a, b = cp.ogrid[0:len(c), len(r) - 1:-1:-1]
    indx = a + b
    return vals[indx]



def _project_cupy(reference_sources, estimated_source, flen, nsrc):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    # nsrc = tf.shape(reference_sources)[0]
    nsampl = reference_sources.shape[1]
    typ = reference_sources.dtype

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = cp.concatenate((reference_sources, cp.zeros([nsrc, flen - 1], dtype=typ)), 1)

    estimated_source = cp.concatenate((estimated_source, cp.zeros([flen - 1], dtype=typ)), 0)

    n_fft = cp.power(2., cp.ceil(cp.log2(nsampl + flen - 1))).astype('i')

    sf = cp.fft.fft(reference_sources, n=int(n_fft), axis=1)
    sef = cp.fft.fft(estimated_source, n=int(n_fft))

    # inner products between delayed versions of reference_sources
    G = cp.empty([nsrc * flen, nsrc * flen])
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * cp.conj(sf[j])
            ssf = cp.real(cp.fft.ifft(ssf))
            ss = toeplitz_cupy(cp.concatenate((cp.reshape(ssf[0],[1]), ssf[-1:-flen:-1]),0), ssf[:flen])
            G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = cp.transpose(ss)

    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = cp.empty([nsrc * flen])
    for i in range(nsrc):
        ssef = sf[i] * cp.conj(sef)
        ssef = cp.real(cp.fft.ifft(ssef))
        conc = cp.concatenate([cp.reshape(ssef[0], [1]), cp.flip(ssef[-flen+1:],0)], 0)
        D[i * flen: (i+1) * flen] = conc
    
    # Computing projection
    # Distortion filters

    s = cp.linalg.solve(G, cp.expand_dims(D,1))
    if nsrc == 2:
        C = cp.concatenate((s[:flen],s[flen:]),1)
    else:
        C = cp.reshape(s,(flen, nsrc))

    # Filtering
    sproj = cp.zeros([nsampl + flen - 1], dtype=cp.float64)

    for i in range(nsrc):
        fshape = C[:, i].shape[0] + reference_sources[i].shape[0] - 1 
        fft1 = cp.fft.rfftn(C[:, i], (fshape,))
        fft2 = cp.fft.rfftn(reference_sources[i], (fshape,))
        ifft = cp.fft.irfftn(fft1 * fft2, (fshape,))
        sproj += ifft[:nsampl + flen - 1]
    return sproj

def _bss_source_crit_cupy(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db_cupy(cp.sum(cp.square(s_filt)), cp.sum(cp.square(e_interf + e_artif)))
    sir = _safe_db_cupy(cp.sum(cp.square(s_filt)), cp.sum(cp.square(e_interf)))
    sar = _safe_db_cupy(cp.sum(cp.square(s_filt + e_interf)), cp.sum(cp.square(e_artif)))
    return (sdr, sir, sar)

def _safe_db_cupy(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.
    """
    return 10.0 * cp.log10(num / (den + 1e-12))




if __name__=="__main__":
    # Simple demo
    np.random.seed(0)
    ts = np.linspace(0,5,10000)
    srcs = np.array([np.sin(ts*600),
                     np.cos(320*ts+0.01)])
    recons = srcs[::-1] + np.random.randn(*srcs.shape)*5
    # srcs = srcs.astype(np.float32)
    # recons = recons.astype(np.float32)

    # print bss_eval_sources(srcs, recons, )
    print bss_eval_sources_cupy(srcs, recons)
    # print bss_eval_sources_cupy(srcs, recons)
    # print bss_eval_sources_cupy(srcs, recons)
    # # with tf.Session() as sess:
    #     z = bss_eval_sources_tf(srcs, recons, nsrc=2)
    #     print sess.run(z)

    #     ref_source = tf.get_default_graph().get_tensor_by_name("concat_1:0")
    #     sf_ = tf.get_default_graph().get_tensor_by_name("FFT:0")
    #     G_ = tf.get_default_graph().get_tensor_by_name("add_1:0")
    #     D_ = tf.get_default_graph().get_tensor_by_name("add_2:0")
    #     solve_ = tf.get_default_graph().get_tensor_by_name("MatrixSolve:0")
    #     # solve_ = tf.get_default_graph().get_tensor_by_name("matrix_solve_ls/cholesky_solve/MatrixTriangularSolve_1:0")

    #     solve_ = sess.run(solve_)
    #     solve_t = np.load('solve.npy')
    #     print np.amax(np.abs(solve_t - solve_))
    #     print np.mean(np.abs(solve_t - solve_))
    #     print solve_.shape
    #     print np.sum(np.abs(solve_ - solve_t))


        # D_ = sess.run(D_)
        # D_t = np.load('Dtrue.npy')
        # print np.amax(np.abs(D_t - D_))
        # print np.mean(np.abs(D_t - D_))
        # print D_.shape
        # print np.sum(np.abs(D_ - D_t))

        # res = sess.run(z)
        # print res

        # G_ = sess.run(G_)
        # G_t = np.load('Gtrue.npy')
        # print np.mean(np.abs(G_t - G_))
        # print G_.shape
        # print np.sum(np.abs(G_ - G_t))

 # 132.41732881  156.43356713  167.00013129  163.50050324  146.97019664
 #  118.63418311   79.3042768    33.58955628  -15.12519248  -62.51050633
 # -104.08168082 -136.33431998 -155.66246597 -160.79545752 -151.69389258
 # -129.01531038  -94.87828513  -51.66565495   -4.25240363   43.43126893
 #   86.44413459  119.95704999  142.89777788  153.37550551  149.54522462
 #  133.5755465   104.74182882   66.55903068   22.35356269  -22.85681464
 #  -65.02145255 -101.10557399 -127.71448516 -143.18723617 -146.7635675
 # -137.14354823 -114.88259857  -81.33819346  -39.53651128    5.82647031
 #   50.71832371   90.650099    121.79146289  141.14312203  149.08681799
 #  143.40594139  124.59793111   93.81094515   54.94845522   10.02667629
 #  -36.1488173   -79.88483926 -116.53630054 -143.37077393 -158.38891665
 # -159.56058581 -146.49495781 -120.54831055  -83.86658732  -40.24125905
 #    6.63564222   52.44207975   93.32223661  125.40825682  145.34558507
 #  152.52769419  146.01602555  125.7062975    94.68356713   54.93525855
 #   10.59665073  -33.96273359  -75.11800445 -108.60576165 -132.85729541
 # -144.76633009 -143.94712224 -130.62477621 -105.86396847  -71.7694084
 #  -31.22226056   11.38030397   53.40202731   90.78319696  118.95720828
 #  137.15714093  143.96755663  138.33390913  119.94305759   89.90747448
 #   52.17219083    9.26158554  -34.00630951  -74.33389329 -107.62558728
 # -131.22944451 -142.78102222 -141.60827754 -126.71392686 -100.35710993
 #  -64.67997847  -21.94818893   22.1518392    63.76041942  100.5111892
 #  127.87292858  145.25297361  149.70546115  140.62620609  120.38673518
 #   90.50720111   53.85169294   13.08600546  -29.23838843  -67.75181952
 # -100.35287621 -123.5487458  -135.23296067 -135.06193656 -122.64083453
 #  -98.93604911  -66.41990343  -28.89375324   11.10064889   50.59182237
 #   84.86032163  111.17929074  127.05012003  132.43495853  126.19404719
 #  108.7715518    81.07594394   46.40650757    7.19596473  -32.5825981
 #  -69.71920114 -100.07295443 -120.94857576 -130.90877387 -128.67773073
 # -115.31454067  -92.23703443  -60.36370491  -22.71069405   17.61836725
 #   58.02858485   94.02627836  121.25775449  138.44468155  142.56412908
 #  133.70122954  112.82765065   82.80952712   44.82638346    2.17808095
 #  -40.93466192  -81.02720078 -113.48482837 -136.72881192 -148.73010238
 # -147.99535594 -134.61486382 -108.32925069  -72.03807054  -29.45748342
 #   14.87408207   58.2662888    95.80242874  123.8593201   141.59815306
 #  146.97027388  139.90071789  120.70861754   91.46419107   53.91923796
 #   11.22379661  -31.780023    -72.12383479 -106.14121844 -130.66918365
 # -143.26005806 -143.03987319 -130.67018722 -106.86807413  -74.74252342
 #  -35.53139876    5.89279775   46.07101822   82.04711477  110.09246635
 #  129.00942189  135.39689389  129.57916894  111.87632894   83.19321236
 #   47.09204295    6.39669512  -34.68110882  -72.41402149 -102.60455675
 # -123.00312666 -132.74932822 -130.84573918 -116.61880974  -91.54965835
 #  -57.8354944   -19.77389857   20.22036766   58.19086718   91.37859351
 #  116.61869358  132.2671659   136.58124593  129.44923357  110.75501267
 #   81.86455402   45.64506884    5.52778502  -34.71201594  -70.41560615
 #  -99.80451093 -120.84560622 -131.3508426  -130.50176624 -117.86281082
 #  -95.8008647   -65.37494012  -29.27696914    9.29227045   46.38511518
 #   78.89787691  103.23189458  117.77966792  121.23723012  113.89539035
 #   95.52045749   68.94453528   37.01296792    0.52354379  -35.73538402
 #  -68.32422633  -95.05881516 -113.36318665 -121.40572078 -118.5321355
 # -104.98039481  -82.75249212  -52.1548772   -15.96597136   21.45096976
 #   56.34691376   86.75055074  109.26134998  122.2502667   124.23710604
 #  115.59136728   97.03502897   69.33387085   34.5229468    -4.45353545
 #  -42.86014147  -78.36634193 -107.42687437 -127.52027016 -136.54306894
 # -134.8065076  -121.20414734  -96.73851051  -63.82943347  -25.62174862
 #   14.75068306   53.90334161   86.51201464  112.50975088  128.70088839
 #  133.07478903  125.44525458  107.06320172   79.2516116    43.39821537
 #    5.15398756  -33.33125912  -67.94121091  -96.59559836 -115.42630007
 # -123.50806511 -119.95496725 -106.09659033  -81.87829608  -49.81819713
 #  -12.46887803   25.63978091   61.13723621   92.44663571  114.7335056
 #  126.50844735  127.43315254  116.77907501   95.76121847   65.66175543
 #   29.61900974  -10.13417865  -48.28075824  -82.44071978 -109.8643051
 # -127.26823936 -133.72058785 -129.11333443 -113.814453    -87.93214101
 #  -54.23085738  -15.65575342   25.10993888   63.16315176   95.37417931
 #  119.0981163   132.51889223  134.3680403   124.33768744  103.44975243
 #   73.52692395   36.85556231   -3.3761203   -42.97056245  -79.12435429
 # -108.81416906 -128.93969813 -137.66412303 -132.83269243 -116.67696307
 #  -89.71162301  -54.50166598  -15.37049707   25.32023486   62.76144062
 #   92.728313    114.47548299  124.67476956  123.73945021  111.11063924
 #   89.33776881   60.05889649   25.03742603  -12.68570939  -49.99882482
 #  -83.05861673 -108.765381   -124.82424232 -129.44890195 -121.9395215
 # -103.07219264  -75.16901727  -41.10808471   -2.27227971   36.64896019
 #   72.18541458  101.21606255  120.33428837  128.60036768  125.2304869
 #  111.12728945   87.54551393   56.12878612   20.18044669  -17.48758441
 #  -53.76636438  -85.35124706 -109.66265971 -124.6761437  -128.9936546
 # -122.74736707 -105.66287866  -80.01010107  -46.97809495  -10.13619841
 #   27.43513327   62.68274191   92.59201497  113.23204021  123.6169785
 #  123.51422102  111.46042714   89.81390523   58.99083029   22.51253482
 #  -15.70298892  -53.21987512  -86.27455388 -112.19555447 -126.82648266
 # -129.75026827 -120.48929036 -100.36793522  -71.29846231  -36.66791896
 #    1.16471255   38.49062748   72.87963018  101.37423723  121.57042847
 #  131.25369948  128.65889778  113.67618912   88.72846745   55.78192383
 #   16.51295134  -24.31637458  -62.77935583  -95.86953825 -120.74946993
 # -135.53122795 -139.23197568 -130.14766767 -109.69634792  -79.60866878
 #  -41.35790669    1.07068898   43.75030579   82.77784365  115.5207185
 #  139.2045762   151.27163186  150.4551287   135.2532567   108.33359652
 #   71.94004655   29.43646184  -14.90627172  -59.339115    -98.65150839
 # -128.88215364 -147.81614869 -154.37580791 -146.67644415 -125.80699251
 #  -94.41763276  -54.59601272   -9.63768959   35.92799193   78.47817908
 #  113.7712606   138.37035691  149.63159861  147.19824576  131.4769416
 #  104.30674586   67.56800428   24.92588233  -20.60164128  -64.7878612
 # -103.23495317 -133.0138571  -150.30576097 -153.73057973 -143.05009282
 # -119.14870539  -83.14070628  -39.69660999    6.2872889    52.12161629
 #   93.55421515  126.58886695  147.97450703  156.08489255  150.52114878
 #  131.32970336  101.89841614   62.81091108   17.60573801  -30.61372502
 #  -76.08653742 -116.13479267 -146.34590349 -163.89628749 -167.4980287
 # -155.39677233 -129.11935109  -90.81775732  -45.2022977     4.86216119
 #   54.55878747   99.64753133  136.29578176  160.13149278  169.36248987
 #  163.97728374  142.42978846  109.1599562    66.61979516   18.29822273
 #  -31.294262    -78.43752629 -118.49751526 -147.02671062 -161.35747587
 # -159.3905177  -142.95570103 -113.89227057  -74.74469664  -29.34827307
 #   18.75146171   65.4799087   106.36378781  138.36593688  157.7887981
 #  162.78076415  152.71234187  127.82185875   90.33848735   43.81569049
 #   -6.14026283  -55.31357765

# [ 3.14911157e+00 -1.53655089e+00  4.12366043e+00 -4.95763732e+00
#  -1.17330795e+01  2.29963226e+01 -1.58362696e+01  6.51152742e+00
#   1.65284588e+00 -4.31771890e+00  8.23509424e+00 -8.99802730e+00
#   1.02456199e+00  4.23638002e+00 -3.15741028e+00  7.39528040e+00
#  -1.38293131e+01  1.17190850e+01 -7.94021699e+00 -4.32884080e+00
#   2.16470322e+01 -1.39537639e+01 -8.58411383e+00  2.40781292e+01
#  -3.35296682e+01  2.67692157e+01 -1.17724495e+01  1.04554344e+01
#  -7.54347557e+00 -4.67022136e+00  7.96829287e+00 -6.95039532e+00
#  -7.14954452e-02  1.15651050e+01 -6.60189199e+00  3.81373275e+00
#  -3.69005197e+00 -6.52346195e+00  8.33053384e+00 -4.77331193e+00
#   2.17554157e+00 -2.79300912e+00  1.97489117e+01 -2.90115737e+01
#   1.48424435e+01 -7.53515628e+00  1.50262380e+01 -2.24363772e+01
#   1.94209825e+01 -1.25366511e+01  1.05751009e+01 -1.04864156e+01
#   1.02483501e+00  7.34989314e+00 -3.51823494e+00 -3.37220940e+00
#   3.56824045e+00 -6.02928199e+00  5.92802820e+00 -3.72654095e+00
#   3.17624170e+00 -3.31867033e+00 -3.38868049e+00  1.29534077e+01
#  -1.14206038e+01 -3.33659268e+00  1.58953890e+01 -1.75728521e+01
#   1.18566873e+01 -1.76792796e+00 -5.91062447e+00  9.71903872e+00
#  -1.72469369e+01  2.02350218e+01 -1.42309648e+01  5.18230204e+00
#   1.17332906e+00 -5.96290136e-01  1.46369100e+00 -8.70400260e+00
#   1.56867913e+01 -1.10981062e+01 -7.86338202e+00  2.28759121e+01
#  -1.29793673e+01 -2.24715489e+00 -2.11068549e+00  2.78946690e-01
#   1.42886452e+01 -1.84479966e+01  1.65607323e+01 -1.39146157e+01
#   1.00375495e+01 -7.44420910e+00  5.82515881e+00 -6.45214170e+00
#   1.30959491e+01 -1.64022937e+01  9.32903375e+00  6.72240027e+00
#  -2.15341132e+01  1.48496923e+01  9.56742394e+00 -2.23644744e+01
#   2.76129831e+01 -2.66084902e+01  1.00830906e+01  1.29203033e+01
#  -1.45395054e+01  6.50244100e+00 -5.02300530e+00 -5.07050529e+00
#   2.23115965e+01 -2.47324471e+01  1.71007102e+01 -5.01663185e+00
#  -5.33179871e+00  8.65874912e+00 -2.53509220e+00 -2.38196630e+00
#  -5.04964975e+00  1.22551051e+01 -1.36536110e+00 -1.37542386e+01
#   1.28466268e+01 -6.82865226e+00  1.32844722e+01 -1.62929285e+01
#   6.71065717e+00 -6.05138273e+00  1.23940813e+01 -1.31290005e+01
#   1.03651901e+01 -8.17049316e+00  9.85731579e+00 -6.45668639e+00
#  -2.67362604e+00  7.35123810e+00 -9.89959782e+00  4.24013089e+00
#   1.00596102e+01 -1.05979419e+01  3.73535282e+00  6.80745524e+00
#  -1.28929213e+01 -2.04893795e+00  1.89546174e+01 -2.25074540e+01
#   1.66464336e+01 -5.19285446e+00  8.78680572e+00 -1.95881788e+01
#   1.16804387e+01  2.47545222e+00 -7.37761754e+00  1.27315823e+01
#  -1.92097828e+01  1.05039332e+01  2.05872889e+00 -5.29042906e+00
#   1.26281945e+01 -1.55412990e+01  3.18020396e+00 -4.32583912e+00
#   1.64177834e+01 -1.87501141e+01  5.54085383e+00  1.45598508e+01
#  -1.74902530e+01  9.25875716e+00 -6.60912805e+00  6.92621155e+00
#  -9.52675268e+00  4.52233347e+00  9.79492923e+00 -1.52167672e+01
#   8.45647890e+00 -8.45920898e-01  1.43000953e+00 -3.16721334e+00
#  -3.46188817e+00  8.96836519e+00 -1.36699594e+01  2.24577638e+01
#  -2.53465356e+01  1.36320507e+01  1.92325774e+00 -1.04596488e+01
#   1.69002885e+01 -2.56389814e+01  2.21244063e+01 -1.00177075e+01
#  -3.95638275e+00  1.32365644e+01 -1.25850954e+01  8.86999538e+00
#  -5.52311625e+00  7.34081656e+00 -8.84452619e+00 -3.03993733e+00
#   7.22904908e+00  5.04911528e+00 -8.20812632e+00  4.50800908e+00
#  -1.19022484e+01  1.85626310e+01 -1.30001508e+01  9.57582708e+00
#  -7.74154831e+00  7.78344411e+00 -7.98734911e+00  6.62941019e+00
#  -8.03794797e+00  3.70964822e+00  3.64551967e+00 -9.90095062e-01
#  -8.66178707e-01  8.76653104e+00 -1.85417278e+01  7.36883695e+00
#   5.67414425e+00 -4.69480742e+00  6.65093545e+00 -1.52069487e+01
#   1.74005665e+01 -8.68895378e+00  1.52193409e+00 -4.75652000e+00
#   6.25830481e+00 -8.64886549e+00  1.00237285e+01 -6.38120382e+00
#   6.30641719e+00 -1.32584312e+01  1.63020425e+01 -4.38487593e+00
#  -2.02699007e+01  2.84542647e+01 -1.19902505e+01 -5.64603761e+00
#   7.06582277e+00 -6.07152258e-01 -1.91028418e+00  1.87683253e+00
#  -8.29207535e+00  1.96310711e+01 -1.32264627e+01 -5.97939980e+00
#   4.25948004e+00  1.19273860e+01 -1.52821418e+01  1.01564919e+01
#  -6.95300653e+00  7.42560540e+00 -4.14468079e+00 -5.61861753e+00
#   3.64459019e+00 -5.29215982e-01  1.17888356e+01 -1.95292612e+01
#   1.28978652e+01 -5.65167722e+00  4.60041273e+00 -1.31117985e+01
#   1.86593030e+01 -9.55884420e+00 -1.95908997e+00 -4.40776360e-02
#   2.98676773e+00  1.50280939e+00 -1.92340082e+01  3.80268564e+01
#  -3.08866429e+01  5.59442272e+00  4.22064511e+00  2.63370114e+00
#  -5.29026336e+00 -8.33511682e+00  2.92625172e+01 -3.12724340e+01
#   1.96812169e+01 -1.62690321e+01  2.04461157e+01 -1.75907726e+01
#   1.03374847e+01 -1.15330124e+01  1.91876338e+01 -1.44312265e+01
#   8.28043985e+00 -1.21208186e+01  8.29073610e+00  1.33505815e+01
#  -2.90359582e+01  2.09899507e+01  3.32133656e-02 -1.03663536e+01
#   8.57337620e+00 -8.26428278e+00  9.79478467e+00 -1.42366085e+01
#   2.33094741e+01 -2.29519977e+01  6.72318430e+00  8.19120682e+00
#  -1.14004808e+01  2.01854639e+00  1.64859136e+00  9.49834165e+00
#  -1.35288727e+01  4.50903653e+00  6.36168544e+00 -1.61264225e+01
#   1.27387448e+01 -2.07675572e+00  1.08088824e+00 -2.51539742e+00
#  -3.01590026e-01  2.13459754e+00 -1.00875833e+00 -2.36416683e+00
#   1.57790625e+00  5.79671714e+00 -1.04538389e+01  4.21769010e+00
#   4.35074387e+00 -5.12894829e+00  1.31755712e+01 -2.60766304e+01
#   2.32397218e+01 -9.24473022e+00 -8.26373533e+00  1.90133212e+01
#  -1.86286311e+01  1.02882234e+00  2.20971668e+01 -2.90358295e+01
#   2.41773596e+01 -1.90594776e+01  1.81603388e+01 -1.36110394e+01
#  -2.59935987e+00  5.83334586e+00 -2.91034706e+00  5.72441473e+00
#  -3.27866863e+00 -1.00617822e+00  2.61825484e+00 -3.59397256e-01
#  -1.97087619e+00 -3.21674329e+00 -8.63909780e-03  1.69141443e+01
#  -2.31038476e+01  1.14361977e+01 -2.00260512e-01 -7.66417511e+00
#   1.25397082e+01 -8.07305849e+00  5.98863238e+00 -4.31300595e+00
#  -3.40834280e+00  8.06592142e+00 -7.21352370e+00  1.67040255e+00
#   2.25889961e+00 -2.57005947e+00  5.42205912e-01  1.78844093e+00
#  -5.95281284e+00  1.14367773e+01 -1.45722356e+01  1.60866118e+01
#  -1.48859402e+01  7.48102278e+00 -4.16714601e-01  3.61942050e-01
#  -1.16551543e+01  1.58654070e+01 -1.00519572e+00 -1.77725342e+01
#   2.40637129e+01 -2.47531869e+01  1.86469986e+01 -1.74359624e+00
#  -1.26400870e+01  1.19904748e+01 -8.79506679e+00  1.84033504e+01
#  -2.21437494e+01  1.15644268e+01 -6.61460252e+00  4.40329811e+00
#  -6.82857545e+00  1.23567458e+01 -1.03563978e+01  1.05469946e+01
#  -6.76831576e+00  1.80305759e+00 -3.25059176e+00 -3.17403613e+00
#   2.71742016e+00  1.03511572e+01 -9.64272100e+00 -8.28520079e+00
#   1.90367309e+01 -8.75527864e+00 -4.15251155e+00  2.78153675e+00
#  -1.85741787e+00 -1.89506473e+00  1.36361186e+01 -1.61238452e+01
#   5.69866016e+00  8.76091001e+00 -1.36228973e+01  5.14024783e+00
#  -1.00460768e+00  7.53567852e+00 -4.77905467e+00 -3.32662449e+00
#   4.26516543e+00 -1.27147046e+01  2.29522561e+01 -1.33318568e+01
#   3.06467716e+00  5.11585319e+00 -2.24232887e+01  2.72548253e+01
#  -8.40829829e+00 -5.13262470e+00 -2.23074326e+00  1.48687848e+01
#  -1.32314646e+01 -2.39092321e+00  1.09141851e+01 -3.83071866e+00
#  -5.84721863e+00  8.24538343e+00 -7.42377096e+00  2.19009507e+00
#  -2.73142245e+00  8.04401969e+00 -4.83323999e+00  3.23745235e+00
#  -8.19593001e+00  8.65184627e+00 -1.01597863e+01  7.30713010e+00
#   1.90969838e+00 -8.70095918e+00  1.37994663e+01 -1.09318312e+01
#   2.32300277e+00 -2.06304088e-01  9.22718984e+00 -1.80613523e+01
#   3.13705120e+00  1.77728613e+01 -1.31478395e+01  8.76809629e-01
#  -6.44933171e-01  3.89929725e+00  7.32022265e-01 -7.78487460e+00
#   1.95120849e+01 -3.00953747e+01  1.95863101e+01 -1.21462368e+01
#   2.04706853e+01 -2.48168021e+01  1.87111276e+01 -7.15674298e+00
#  -3.95813949e+00  1.40353199e+01 -1.55234552e+01  8.28315108e+00
#  -1.36891006e+01  2.05101108e+01 -1.38945833e+01  5.49333941e+00
#   7.12964099e-01 -1.01965293e+01  1.03739638e+01  4.94659095e+00
#  -2.48964573e+01  3.73243240e+01 -2.59031098e+01  4.44607838e+00
#   4.06781963e+00 -7.67140448e+00  8.35940835e+00  2.92416434e+00
#  -6.11197141e+00  8.64708995e+00 -1.87058131e+01  1.12465116e+01
#   1.99184638e+00 -3.84352631e+00  6.46187097e+00 -2.25047086e+00
#  -4.24025865e+00  8.40423434e+00 -1.16964492e+01  6.50976402e+00
#  -9.74047645e-01 -4.14297829e+00  4.60511449e+00 -1.10855929e+00
#   9.92070177e+00 -1.31261171e+01  1.91478967e+00  1.09201095e+00]
