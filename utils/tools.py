# -*- coding: utf-8 -*-
from __future__ import print_function 
import sys
# Returns the string of remaining training time
def getETA(batchTime, nbBatch, batchIndex, nbEpoch, epoch):
    seconds = int(batchTime*(nbBatch-batchIndex-1) + batchTime*nbBatch*(nbEpoch-epoch-1))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh%02dm%02ds" % (h, m, s)


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r{0} |{1}| {2}{3} {4}\r'.format(prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()