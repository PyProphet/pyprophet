#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
	profile
except:
	profile = lambda x: x

from stats import *




def fastPow(f, i):
	if i == 0: 
		return 1
	elif i % 2 == 0:
		return fastPow(f * f, i / 2)
	else:
		return fastPow(f*f, i/2) * f

def getFDRCalc(x):
	if x == "mProph":
		return MProphFDRCalc()
	elif x == "storey":
		return StoreyFDRCalc()
	else:
		raise Exception("unknown fdr-calculation '%s'" % x)


class FDRCalc(object):
	def calc(self, pvalues, lamb):
		""" returns DataFrame with 'pvalue svalue qvalue percentile_positive FDR' columns"""
		raise NotImplementedError


class MProphFDRCalc(FDRCalc):
	def calc(self, pvalues, lamb):
		return get_error_table_from_pvalues_new(pvalues, lamb)
	
	
class StoreyFDRCalc(FDRCalc):
	
	def calc(self, pvalues, lamb):
		""" meaning pvalues presorted i descending order"""
		
		m = len(pvalues)
		pi0 = (pvalues > lamb).sum() / ((1 - lamb)*m)
		
		pFDR = np.ones(m)
		#print "pFDR    y        Pr     fastPow"
		for i in range(m):
			y = pvalues[i]
			Pr = max(1, m - i) / float(m)
			pFDR[i] = (pi0 * y) / (Pr * (1 - fastPow(1-y, m)))
		#	print pFDR[i], y, Pr, fastPow(1-y, m)
		
		
		num_null = pi0*m
		num_alt = m - num_null
		num_negs = np.array(range(m))
		num_pos = m - num_negs
		pp = num_pos / float(m)
		
		qvalues = np.ones(m)
		qvalues[0] = pFDR[0]
		for i in range(m-1):
			qvalues[i+1] = min(qvalues[i], pFDR[i+1])
		
		sens = ((1.0 - qvalues) * num_pos) / num_alt
		sens[sens > 1.0] = 1.0
		
		df = pd.DataFrame(dict(
			pvalue=pvalues,
			qvalue=qvalues,
			FDR=pFDR,
			percentile_positive=pp,
			sens=sens
		))
		
		df["svalue"] = df.sens[::-1].cummax()[::-1]
		
		return df, num_null, m





def getStatCalc(x):
	if x == "mProph":
		return MProphStatCalc()
	elif x == "jt":
		return JTStatCalc()
	else:
		raise Exception("unknown stats-calculation '%s'" % x)

class StatCalc(object):
	def calc(self, df, scores, num_null):
		""" returns DataFrame with columns 'TP FP TN FN sens spec svalue' in addition to df columns"""
		raise NotImplementedError


class MProphStatCalc(object):
	def calc(self, df, scores, num_null):
		return get_error_table_using_percentile_positives_new(
			df, scores, num_null)


class JTStatCalc(object):
	def calc(self, df, scores, num_null):
		scores = np.sort(scores)[::-1]
		x = self.resample(df, scores, "cutoff")
		return self.appendStats(x, num_null)
	
	
	def resample(self, df, scores, col):
		
		#print "RESAMPLING:"
		#print df[col].min(), df[col].max(), len(df)
		#print np.min(scores), np.max(scores), len(scores)
	
		def closestAbove(xs):
			pos = xs[xs >= 0]
			if len(pos) == 0:
				return None
			else:
				return pos.idxmin()
		
		def sample(s):
			high = closestAbove(df[col] - s)
			low = closestAbove(s - df[col])
			if high is None:
				return df.loc[low]
			elif low is None:
				return df.loc[high]
			elif high == low:
				return df.loc[high]
			else:
				k = (s - df[col][low]) / (df[col][high] - df[col][low])
				return (df.loc[low] * (1-k) + df.loc[high] * k).values
		
		X = np.array([sample(s) for s in scores])
		return pd.DataFrame(X, columns=df.columns)
	
	
	def appendStats(self, df, est_num_null):
		""" df need to be sorted in ascending pvalue order. """
		qvalues = df['qvalue']
		n = len(qvalues)
		TP = np.ones(n)
		FP = np.ones(n)
		TN = np.ones(n)
		FN = np.ones(n)
		sens = np.ones(n)
		spec = np.ones(n)
		
		for i in range(n):
			FP[i] = int(qvalues[i] * (i+1))
			TP[i] = (i+1) - FP[i]
			TN[i] = int(est_num_null - FP[i])
			FN[i] = (n - (i+1)) - TN[i]
			sens[i] = float(TP[i]) / (TP[i]+FN[i])
			spec[i] = float(TN[i]) / (TN[i]+FP[i])
		
		#df['svalue']=spec
		df['TP']=TP
		df['FP']=FP
		df['TN']=TN
		df['FN']=FN
		df['sens']=sens
		df['spec']=spec
		return df





def getStatSampler(x):
	if x == "mProph":
		return MProphStatSampler()
	else:
		raise Exception("unknown stat-sampler '%s'" % x)


class StatSampler(object):
	def summary_table(self, df, qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
		""" summary error table for some typical q-values """
		raise NotImplementedError
	
	def final_table(self, df, num_cut_offs=51):
		""" create artificial cutoff sample points from given range of cutoff
		values in df, number of sample points is 'num_cut_offs'"""
		raise NotImplementedError

		
class MProphStatSampler(StatSampler):
	def summary_table(self, df, qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
		qvalues = to_one_dim_array(qvalues)
		# find best matching fows in df for given qvalues:
		ix = find_nearest_matches(df.qvalue.values, qvalues)
		# extract sub table
		df_sub = df.iloc[ix]
		# remove duplicate hits, mark them with None / NAN:
		for i_sub, (i0, i1) in enumerate(zip(ix, ix[1:])):
			if i1 == i0:
				df_sub.iloc[i_sub + 1, :] = None
		# attach q values column
		df_sub.qvalue = qvalues
		# remove old index from original df:
		df_sub.reset_index(inplace=True, drop=True)
		return df_sub
	
	def final_table(self, df, num_cut_offs=51):
		cutoffs = df.cutoff.values
		min_ = min(cutoffs)
		max_ = max(cutoffs)
		# extend max_ and min_ by 5 % of full range
		margin = (max_ - min_) * 0.05
		sampled_cutoffs = np.linspace(min_ - margin, max_ + margin, num_cut_offs)

		# find best matching row index for each sampled cut off:
		ix = find_nearest_matches(df.cutoff.values, sampled_cutoffs)
	
		# create sub dataframe:
		sampled_df = df.iloc[ix]
		sampled_df.cutoff = sampled_cutoffs
		# remove 'old' index from input df:
		sampled_df.reset_index(inplace=True, drop=True)

		return sampled_df


	
