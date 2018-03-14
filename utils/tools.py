#coding: utf-8

# Returns the string of remaining training time
def getETA(batchTime, nbBatch, batchIndex, nbEpoch, epoch):
	seconds = int(batchTime*(nbBatch-batchIndex) + batchTime*nbBatch*(nbEpoch-epoch))
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	return "%dh%02dm%02ds" % (h, m, s)

