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
from stats import to_one_dim_array, mean_and_std_dev, pnorm, get_error_table_from_pvalues_new
import logging
import numpy as np
import pandas as pd

class IErrorTable(object):
	
	def summary_table(self):
		raise NotImplementedError
	
	def final_table(self):
		raise NotImplementedError
	
	def enrich(self, input_table, experiment):
		raise NotImplementedError




class FlexibleErrorTable(IErrorTable):
	
	
	def __init__(self, scores, ref_target_scores, ref_decoy_scores, lambda_, null_model, fdr_calc, stat_calc, stat_sampler):
		self.null_model = null_model
		self.fdr = fdr_calc
		self.stat = stat_calc
		self.stat_sampler = stat_sampler
		
		decoy_scores = to_one_dim_array(ref_decoy_scores)
		target_scores = to_one_dim_array(ref_target_scores)
		target_scores = np.sort(target_scores[~np.isnan(target_scores)])

		target_pvalues = self.null_model.pvalues(target_scores, decoy_scores)
		
		FDR_table, num_null, num_total = self.fdr.calc(target_pvalues, lambda_)
		FDR_table["cutoff"] = target_scores

		self.est_null_rel_size = float(num_null) / num_total
		self.df = self.stat.calc(FDR_table, scores, self.est_null_rel_size * len(scores))
		

	
	def summary_table(self):
		return self.stat_sampler.summary_table(self.df)
	
	def final_table(self):
		return self.stat_sampler.final_table(self.df)

	def enrich(self, input_table, experiment):
		s_values, q_values = lookup_s_and_q_values_from_error_table(experiment["d_score"],
																	self.df)
		experiment["m_score"] = q_values
		experiment["s_value"] = s_values
		logging.info("mean m_score = %e, std_dev m_score = %e" % (np.mean(q_values),
					 np.std(q_values, ddof=1)))
		logging.info("mean s_value = %e, std_dev s_value = %e" % (np.mean(s_values),
					 np.std(s_values, ddof=1)))
		experiment.add_peak_group_rank()

		scored_table = input_table.join(experiment[["d_score", "m_score", "peak_group_rank"]])
		return scored_table
	



class OldNonParamErrorTable(IErrorTable):
	
	def __init__(self, scores, ref_target_scores, ref_decoy_scores):
		df, est_num_null, num_target = self.nonParamQvalues(ref_target_scores, ref_decoy_scores)
		self.est_null_rel_size = float(est_num_null) / num_target
		self.df = self.resample(df, scores, 'cutoff')
		self.appendStats(self.df, self.est_null_rel_size * len(scores))
	
	def nonParamQvalues(self, target_scores, decoy_scores, num_bootstrap_iter=10):
		#print target_scores
		ts = np.sort(to_one_dim_array(target_scores))[::-1] # first sort asc, then reverse
		ds = np.sort(to_one_dim_array(decoy_scores))[::-1]
		
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
		
		def estimateNullRelSize(ts, ds):
			ts = np.sort(ts)[::-1]
			ds = np.sort(ds)[::-1]
			i_d = len(ds)-1
			i_t = len(ts)-1
			frac_negs = []
			while i_d >= len(ds)/2:#0 and ds[i_d] < d_mean:
				if ts[i_t] < ds[i_d]:
					frac_negs.append(float(len(ts) - i_t) / (len(ds) - (i_d)))
					i_t -= 1
				else:
					i_d -= 1
			
			if len(frac_negs) == 0:
				return 0.0
			else:
				return np.mean(frac_negs) * td_ratio
		
		estimates = []
		for i in range(num_bootstrap_iter):
			T = np.random.choice(ts, size=len(ts)/2)
			D = np.random.choice(ds, size=len(ds)/2)
			estimates.append(estimateNullRelSize(T, D))
		
		est_null_rel_size = np.mean(estimates)
		est_num_null = int(est_null_rel_size * len(ds))
		
		print est_num_null, est_null_rel_size#, qvalues
		qvalues = qvalues * est_null_rel_size
		
		df = pd.DataFrame(dict(
			cutoff=ts,
			qvalue=qvalues
		))
		
		return df, est_num_null, len(ts)
	
	
	def summary_table(self, qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
		df = self.df
		qvalues = to_one_dim_array(qvalues)
		ix = find_nearest_matches(df.qvalue.values, qvalues)
		df_sub = df.iloc[ix]
		for i_sub, (i0, i1) in enumerate(zip(ix, ix[1:])):
			if i1 == i0:
				df_sub.iloc[i_sub + 1, :] = None
		df_sub.qvalue = qvalues
		df_sub.reset_index(inplace=True, drop=True)
		return df_sub


	def resample(self, df, scores, col):
		def closestAbove(xs):
			pos = xs[xs > 0]
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
			else:
				k = (df[col][low] - s) / (df[col][high] - s)
				return (df.loc[low] * (1-k) + df.loc[high] * k).values
		
		X = np.array([sample(s) for s in scores])
		return pd.DataFrame(X, columns=df.columns)
	
	
	def appendStats(self, df, est_num_null):
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
		
		df['svalue']=spec
		df['TP']=TP
		df['FP']=FP
		df['TN']=TN
		df['FN']=FN
		df['sens']=sens
		df['spec']=spec
	
	
	
	def final_table(self, num_cut_offs=51):
		df = self.df
		cutoffs = df.cutoff.values
		min_ = min(cutoffs)
		max_ = max(cutoffs)
		margin = (max_ - min_) * 0.05
		sampled_cutoffs = np.linspace(min_ - margin, max_ + margin, num_cut_offs)
		ix = find_nearest_matches(df.cutoff, sampled_cutoffs)
		sampled_df = df.iloc[ix]
		sampled_df.cutoff = sampled_cutoffs
		sampled_df.reset_index(inplace=True, drop=True)
		return sampled_df
	
	
	def enrich(self, input_table, experiment):
		s_values, q_values = lookup_s_and_q_values_from_error_table(experiment["d_score"],
																	self.df)
		experiment["m_score"] = q_values
		experiment["s_value"] = s_values
		logging.info("mean m_score = %e, std_dev m_score = %e" % (np.mean(q_values),
					 np.std(q_values, ddof=1)))
		logging.info("mean s_value = %e, std_dev s_value = %e" % (np.mean(s_values),
					 np.std(s_values, ddof=1)))
		experiment.add_peak_group_rank()

		scored_table = input_table.join(experiment[["d_score", "m_score", "peak_group_rank"]])
		return scored_table   

