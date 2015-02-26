#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
	profile
except:
	profile = lambda x: x

from stats import pnorm, mean_and_std_dev, to_one_dim_array
import numpy as np


def getNullModel(x):
	if x == "normal":
		return NormalNullModel()
	elif x == "log-normal":
		return LogNormalNullModel()
	elif x == "non-param":
		return NonParamNullModel()
	else:
		raise Exception("unknown null-model '%s'" % x)
	


class NullModel(object):
	def pvalues(self, target_scores, decoy_scores):
		raise NotImplementedError

class NormalNullModel(NullModel):

	def pvalues(self, target_scores, decoy_scores):
		mu, nu = mean_and_std_dev(decoy_scores)
		return 1.0 - pnorm(target_scores, mu, nu)



class NonParamNullModel(NullModel):

	def pvalues(self, target_scores, decoy_scores):
		ts = target_scores
		ds = np.sort(to_one_dim_array(decoy_scores))
		pvalues = np.ones(len(ts))

		i_t = 0
		i_d = 0
		while i_t < len(ts) and i_d < len(ds):
			if ts[i_t] < ds[i_d]:
				pvalues[i_t] = 1.0 - float(i_d) / len(ds)
				i_t += 1
			else:
				i_d += 1

		while i_t < len(ts):
			pvalues[i_t] = 1.0 / len(ds)
			i_t += 1

		return pvalues


class LogNormalNullModel(NullModel):

	def pvalues(self, target_scores, decoy_scores):
		#print "LOG NORMAL NULL MODEL:"
		corr = - np.min(decoy_scores) + 0.0001
		mu, nu = mean_and_std_dev(np.log(decoy_scores + corr))
		#print mu, nu, np.min(decoy_scores)
		return 1.0 - pnorm(np.log(target_scores + corr), mu, nu)
