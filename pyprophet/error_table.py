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
	
	
	def __init__(self, 	scores, ref_target_scores, ref_decoy_scores, lambda_, 
						null_model, fdr_calc, stat_calc, stat_sampler, decoysMissing=0.0):
		self.null_model = null_model
		self.fdr = fdr_calc
		self.stat = stat_calc
		self.stat_sampler = stat_sampler
		
		decoy_scores = to_one_dim_array(ref_decoy_scores)
		target_scores = to_one_dim_array(ref_target_scores)
		target_scores = np.sort(target_scores[~np.isnan(target_scores)])

		target_pvalues = self.null_model.pvalues(target_scores, decoy_scores) * (1.0 - decoysMissing)
		
		FDR_table, num_null, num_total = self.fdr.calc(target_pvalues, lambda_)
		FDR_table["cutoff"] = target_scores

		self.num_null = num_null
		self.num_total = num_total
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
	



