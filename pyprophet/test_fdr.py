#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
	profile
except:
	profile = lambda x: x

from optimized import count_num_positives
from stats import get_error_stat_from_null, get_error_table_using_percentile_positives_new
from stats import lookup_s_and_q_values_from_error_table, find_nearest_matches
from stats import to_one_dim_array
import logging
import numpy as np


def nonParamQvaluesOLD(target_scores, decoy_scores):
	ts = np.sort(target_scores)[::-1] # first sort asc, then reverse
	ds = np.sort(decoy_scores)[::-1]
	
	qvalues = np.ones(len(ts))
	i_t = 0	# index in target/decoy score array
	i_d = 0
	td_ratio = float(len(ts)) / len(ds)
	maxq = 1.0 / (len(ts) * len(ds))
	
	while i_t < len(ts) and i_d < len(ds):
		if ts[i_t] > ds[i_d]:
			maxq = max( maxq, ((float(i_d) / (i_t+1)) * td_ratio ) )
			qvalues[i_t] = maxq
			i_t += 1
		else:
			i_d += 1
	
	i_d = len(ds)-1
	i_t = len(ts)-1
	frac_negs = []
	while i_d >= len(ds)/2:#0 and ds[i_d] < d_mean:
		#print i_t, i_d, len(ts), len(ds), ts[i_t], ds[i_d]
		if ts[i_t] < ds[i_d]:
			frac_negs.append(float(len(ts) - i_t) / (len(ds) - (i_d)))
			i_t -= 1
		else:
			i_d -= 1
	
	if len(frac_negs) == 0:
		est_frac_neg = 0.0
	else:
		est_frac_neg = np.median(frac_negs) * td_ratio
	
	est_num_neg = int(est_frac_neg * len(ts))
	
	#print est_num_neg, est_frac_neg, qvalues
	qvalues = qvalues * est_frac_neg
	
	TP = [-1.0 for x in ts]
	FP = [-1.0 for x in ts]
	TN = [-1.0 for x in ts]
	FN = [-1.0 for x in ts]
	sens = [-1.0 for x in ts]
	spec = [-1.0 for x in ts]
	#print len(TP), len(FP), qvalues
	for i in range(len(ts)):
		TP[i] = int(qvalues[i] * (i+1))
		FP[i] = (i+1) - TP[i]
		TN[i] = int(est_num_neg - FP[i])
		FN[i] = (len(ts) - (i+1)) - TN[i]
		sens[i] = float(TP[i]) / (TP[i]+FN[i])
		spec[i] = float(TN[i]) / (TN[i]+FP[i])
	
	df = pd.DataFrame(dict(
		score=ts,
		qvalue=qvalues,
		TP=TP,
		FP=FP,
		TN=TN,
		FN=FN,
		sens=sens,
		spec=spec
	))
	
	return df, est_frac_neg, frac_negs

df, frac_neg, frac_negs = nonParamQvalues(ts, ds)

randn = np.random.randn
n = 1000
ts = np.concatenate((randn(n) + 3.0, randn(n) - 1.0))
ds = randn(2*n) - 1.0

plt.figure()
plt.hist([ts, ds], 20, color=['w', 'r'], label=['target', 'decoy'], histtype='bar')
plt.legend(loc=2)
plt.show()

def getFracNegs(n, t):
	ts = np.concatenate((randn(n) + t, randn(n) - 1.0))
	ds = randn(2*n) - 1.0
	return nonParamQvalues(ts, ds)

plt.figure()
for i in range(10):
	df, frac_neg, frac_negs = getFracNegs(10000, 3)
	nq01 = len(df[df["qvalue"] < 0.01])
	plt.plot(frac_negs, label="nq01: %d" % nq01)

plt.legend()
plt.show()



def bootstrapFracNegs(n, t, num_iter):
	ts = np.concatenate((randn(n) + t, randn(n) - 1.0))
	ds = randn(2*n) - 1.0
	iters = []
	targets = []
	decoys = []
	for i in range(num_iter):
		T = np.random.choice(ts, size=n)
		D = np.random.choice(ds, size=n)
		iters.append(nonParamQvalues(T, D))
		targets.append(T)
		decoys.append(D)
	return iters, targets, decoys

n = 1000
x, targets, decoys = bootstrapFracNegs(n, 3, 10)
plt.figure()
plt.plot([0, n/5], [0.5, 0.5], color="k")
X = []
for temp in x:
	df, frac_neg, frac_negs = temp
	X.append(np.array(frac_negs))
	nq01 = len(df[df["qvalue"] < 0.01])
	plt.plot(frac_negs, label="nq01: %d" % nq01, color="r")

meanFracNegs = np.mean([temp[:n/5] for temp in X], axis=0)
print "median of mean:", np.median(meanFracNegs)
print "medians of bootstraps:", ["%.3f" % np.median(temp) for temp in X]
plt.plot(meanFracNegs, label="mean", color="b")
plt.axis([0, n/5, 0, 1])
plt.show()


plt.figure()
for t, d in zip(targets, decoys):
	plt.hist([t, d], 20, color=['k', 'r'], label=['target', 'decoy'], histtype='step')

plt.show()




n = 1000
t = 3
ts = np.concatenate((randn(n) + t, randn(n) - 1.0))
ds = randn(2*n) - 1.0
et = NonParametricErrorTable(None, ts, ds)
