# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    profile = lambda x: x

import pandas as pd
import numpy as np

from stats import (lookup_s_and_q_values_from_error_table, calculate_final_statistics,
                   mean_and_std_dev, final_err_table, summary_err_table, pnorm, find_cutoff, posterior_pg_prob)
from stats import posterior_chromatogram_hypotheses_fast
from config import CONFIG
from misc import nice_time

from data_handling import (prepare_data_table, Experiment)
from classifiers import (LDALearner, ConsensusPredictor, LinearPredictor)
from semi_supervised import (AbstractSemiSupervisedTeacher, StandardSemiSupervisedTeacher)

from error_table import FlexibleErrorTable
from null_model import *
from fdr_calc import *  

import multiprocessing

import logging

import time

def unwrap_self_for_multiprocessing((inst, method_name, args),):
	""" You can not call methods with multiprocessing, but free functions,
		If you want to call  inst.method(arg0, arg1),

			unwrap_self_for_multiprocessing(inst, "method", (arg0, arg1))

		does the trick.
	"""
	return getattr(inst, method_name)(*args)


class HolyGostQuery(object):

	""" HolyGhostQuery assembles the unsupervised methods.
		See below how PyProphet parameterises this class.
	"""

	def __init__(self, semi_supervised_teacher):
		assert isinstance(semi_supervised_teacher,
						  AbstractSemiSupervisedTeacher)
		self.semi_supervised_teacher = semi_supervised_teacher



	@profile
	def process_csv(self, path, delim=",", loaded_scorer=None, loaded_weights=None, p_score=False):
		start_at = time.time()

		logging.info("read %s" % path)

		table = pd.read_csv(path, delim, na_values=["NA", "NaN", "infinite"])

		if loaded_scorer is not None:
			logging.info("apply scorer to  %s" % path)
			result_tables, clf_scores, data_for_persistence, trained_weights = self.apply_loaded_scorer(table, loaded_scorer)
			data_for_persistence = None
		else:
			logging.info("learn and apply scorer to %s" % path)
			result_tables, clf_scores, data_for_persistence, trained_weights = self.tutor_and_apply_classifier(table, p_score, loaded_weights)
		
		logging.info("processing %s finished" % path)
		logging.info("time needed: %s" % (nice_time(time.time() - start_at)))
		return result_tables, clf_scores, data_for_persistence, trained_weights


	@profile
	def tutor_and_apply_classifier(self, table, p_score=False, loaded_weights=None):

		prepared_table, score_columns = prepare_data_table(table)

		experiment = Experiment(prepared_table)

		is_test = CONFIG.get("is_test", False)

		if is_test:  # for reliable results
			experiment.df.sort("tg_id", ascending=True, inplace=True)

		experiment.log_summary()

		all_test_target_scores = []
		all_test_decoy_scores = []
		clfs = []
		
		train_frac	 = CONFIG.get("train.fraction")
		is_test	 = CONFIG.get("is_test", False)
		neval		 = CONFIG.get("xeval.num_iter")
		fraction = CONFIG.get("xeval.fraction")
		teacher	 = self.semi_supervised_teacher # inst
		num_processes = CONFIG.get("num_processes")
		
		# reserve part of experiment for testing and FDR calc.
		experiment.split_train_test(train_frac, is_test)
		train_exp = experiment
		test_exp = None
		if train_frac < 0.99:
			train_exp, test_exp = experiment.get_train_and_test_peaks()
		
		xval_type = CONFIG.get("xval.type")
		if xval_type == "split":
			train_exp.set_xval_sets(neval, is_test)
			xval_sets = xval_sets(neval, int(fraction * neval + 0.5))
				
		if loaded_weights == None:
			logging.info("start %d cross evals using %d processes" % (neval, num_processes))
			if num_processes == 1:
				for k in range(neval):
					if xval_type == "split":
						train_exp.train_on_xval_sets(xval_sets[k])
					else:
						train_exp.split_train_test(fraction, is_test)
					(ttt_scores, ttd_scores, clf) = teacher.tutor_randomized(train_exp)
					all_test_target_scores.extend(ttt_scores)
					all_test_decoy_scores.extend(ttd_scores)
					clfs.append(clf)
			else:
				pool = multiprocessing.Pool(processes=num_processes)
				while neval:
					remaining = max(0, neval - num_processes)
					todo = neval - remaining
					neval -= todo
					args = ((teacher, "tutor_randomized", (train_exp, )), ) * todo
					res = pool.map(unwrap_self_for_multiprocessing, args)
					top_test_target_scores = [ti for r in res for ti in r[0]]
					top_test_decoy_scores = [ti for r in res for ti in r[1]]
					clfs.extend([r[2] for r in res])
					all_test_target_scores.extend(top_test_target_scores)
					all_test_decoy_scores.extend(top_test_decoy_scores)

			logging.info("finished cross evals")
		else:
			logging.info("start application of pretrained weights")
			loaded_clf = LinearPredictor(loaded_weights)
			clfs.append(loaded_clf)
			clf_scores = loaded_clf.score(experiment, True)
			experiment.set_and_rerank("classifier_score", clf_scores)
			all_test_target_scores.extend(experiment.get_top_target_peaks()["classifier_score"])
			all_test_decoy_scores.extend(experiment.get_top_decoy_peaks()["classifier_score"])
			logging.info("finished pretrained scoring")
			

		final_classifier = ConsensusPredictor(clfs)
		# TODO: How to solve this for general (non-linear) predictors?
		# ... maybe just ignore for now
		loaded_weights = final_classifier.get_coefs()

		d = {}
		d["tg_id"] = experiment.df.tg_num_id.values
		d["decoy"] = experiment.df.is_decoy.values
		for i in range(len(clfs)):
			s = clfs[i].score(experiment, True)
			experiment.set_and_rerank("classifier_score", s)
			d["clf%d_score" % i] = s.flatten()
			d["clf%d_rank1" % i] = experiment.df.is_top_peak.values

		for c in score_columns:
			d[c] = table[c]

		results, res_dict, data_for_persistence = self.apply_classifier(final_classifier, experiment, test_exp,
															 all_test_target_scores,
															 all_test_decoy_scores, table, p_score=p_score)
		logging.info("calculated scoring and statistics")
		return  results, pd.DataFrame(d), data_for_persistence + (score_columns,), loaded_weights



	@profile
	def apply_loaded_scorer(self, table, loaded_scorer):

		final_classifier, mu, nu, df_raw_stat, num_null, num_total, loaded_score_columns = loaded_scorer

		prepared_table, __ = prepare_data_table(table, loaded_score_columns=loaded_score_columns)

		experiment = Experiment(prepared_table)

		final_score = final_classifier.score(experiment, True)
		experiment["d_score"] = (final_score - mu) / nu

		scored_table = self.enrich_table_with_results(table, experiment, df_raw_stat)

		trained_weights = final_classifier.get_coefs()

		return (None, None, scored_table), None, None, trained_weights



	@profile
	def enrich_table_with_results(self, table, experiment, df_raw_stat):
		s_values, q_values = lookup_s_and_q_values_from_error_table(experiment["d_score"],
																	df_raw_stat)
		experiment["m_score"] = q_values
		experiment["s_value"] = s_values
		logging.info("mean m_score = %e, std_dev m_score = %e" % (np.mean(q_values),
					 np.std(q_values, ddof=1)))
		logging.info("mean s_value = %e, std_dev s_value = %e" % (np.mean(s_values),
					 np.std(s_values, ddof=1)))
		experiment.add_peak_group_rank()

		scored_table = table.join(experiment[["d_score", "m_score", "peak_group_rank"]])
		return scored_table



	@profile
	def apply_classifier(self, final_classifier, experiment, test_exp, all_test_target_scores,
						 all_test_decoy_scores, table, p_score=False):

		lambda_ = CONFIG.get("final_statistics.lambda")

		mu, nu, final_score = self.calculate_params_for_d_score(final_classifier, experiment)
		experiment["d_score"] = (final_score - mu) / nu

		if (CONFIG.get("final_statistics.fdr_all_pg")):
			all_tt_scores = experiment.get_top_target_peaks()["d_score"]
		else:
			all_tt_scores = experiment.get_top_target_peaks()["d_score"]

		is_test	 = CONFIG.get("is_test", False)
		
		if is_test:
			d = {
				'pyProph':FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NormalNullModel(),
									MProphFDRCalc(),
									MProphStatCalc(),
									MProphStatSampler()
							),
				'nonParam':FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NonParamNullModel(),
									MProphFDRCalc(),
									MProphStatCalc(),
									MProphStatSampler()
							),
				'logNormal':FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									LogNormalNullModel(),
									MProphFDRCalc(),
									MProphStatCalc(),
									MProphStatSampler()
							),
				'nonParam-storey':FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NonParamNullModel(),
									StoreyFDRCalc(),
									MProphStatCalc(),
									MProphStatSampler()
							),
				'nonParam-storey-jt':FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NonParamNullModel(),
									StoreyFDRCalc(),
									JTStatCalc(),
									MProphStatSampler()
							)
				}
			
			d["res"] = d["pyProph"]
		
			if test_exp is not None:
				muT, nuT, final_scoreT = self.calculate_params_for_d_score(final_classifier, test_exp)
				test_exp["d_score"] = (final_scoreT - muT) / nuT
				d['true_pyProph'] = FlexibleErrorTable(
						all_tt_scores, 
						test_exp.get_top_target_peaks()["d_score"],
						test_exp.get_top_decoy_peaks()["d_score"],
						lambda_,
						NormalNullModel(),
						MProphFDRCalc(),
						MProphStatCalc(),
						MProphStatSampler()
					)
				d['true_nonParam'] = FlexibleErrorTable(
						all_tt_scores, 
						test_exp.get_top_target_peaks()["d_score"],
						test_exp.get_top_decoy_peaks()["d_score"],
						lambda_,
						NonParamNullModel(),
						MProphFDRCalc(),
						MProphStatCalc(),
						MProphStatSampler()
					)
				d['true_logNormal'] = FlexibleErrorTable(
						all_tt_scores,
						test_exp.get_top_target_peaks()["d_score"],
						test_exp.get_top_decoy_peaks()["d_score"],
						lambda_,
						LogNormalNullModel(),
						MProphFDRCalc(),
						MProphStatCalc(),
						MProphStatSampler()
					)
		else:
			 null_model 	= getNullModel(CONFIG.get("final_statistics.null_model"))
			 fdr_calc 		= getFDRCalc(CONFIG.get("final_statistics.fdr_calc"))
			 stat_calc 		= getStatCalc(CONFIG.get("final_statistics.stat_calc"))
			 stat_sampler 	= getStatSampler(CONFIG.get("final_statistics.stat_sampler"))
			 decoys_missing	= CONFIG.get("decoy.missing", 0.0)
			 d = dict(
			 		res = FlexibleErrorTable(
						all_tt_scores,
						all_test_target_scores,
						all_test_decoy_scores,
						lambda_,
						null_model,
						fdr_calc,
						stat_calc,
						stat_sampler,
						decoys_missing
					)
			 	)

		def getRes(et):
			return (et.summary_table(), et.final_table(), et.enrich(table, experiment))

		et = d["res"]
		sum_tab, fin_tab, score_tab = getRes(et)

		if CONFIG.get("compute.probabilities"):
			logging.info( "" )
			logging.info( "Posterior Probability estimation:" )
			logging.info( "Estimated number of null %0.2f out of a total of %s. " % (et.num_null, et.num_total) )

			# Note that num_null and num_total are the sum of the
			# cross-validated statistics computed before, therefore the total
			# number of data points selected will be 
			#	len(data) /  xeval.fraction * xeval.num_iter
			# 
			prior_chrom_null = et.num_null * 1.0 / et.num_total
			number_true_chromatograms = (1.0-prior_chrom_null) * len(experiment.get_top_target_peaks().df)
			number_target_pg = len( Experiment(experiment.df[(experiment.df.is_decoy == False) ]).df )
			prior_peakgroup_true = number_true_chromatograms / number_target_pg

			logging.info( "Prior for a peakgroup: %s" % (number_true_chromatograms / number_target_pg))
			logging.info( "Prior for a chromatogram: %s" % str(1-prior_chrom_null) )
			logging.info( "Estimated number of true chromatograms: %s out of %s" % (
				number_true_chromatograms, len(experiment.get_top_target_peaks().df)) )
			logging.info( "Number of target data: %s" % len( Experiment(experiment.df[(experiment.df.is_decoy == False) ]).df ) )

			# pg_score = posterior probability for each peakgroup
			# h_score = posterior probability for the hypothesis that this peakgroup is true (and all other false)
			# h0_score = posterior probability for the hypothesis that no peakgroup is true

			pp_pg_pvalues = posterior_pg_prob(experiment, prior_peakgroup_true, lambda_=lambda_)
			experiment.df[ "pg_score"]  = pp_pg_pvalues
			score_tab = score_tab.join(experiment[["pg_score"]])

			allhypothesis, h0 = posterior_chromatogram_hypotheses_fast(experiment, prior_chrom_null)
			experiment.df[ "h_score"]  = allhypothesis
			experiment.df[ "h0_score"]  = h0
			score_tab = score_tab.join(experiment[["h_score", "h0_score"]])

		needed_to_persist = (final_classifier, mu, nu,
					et.df.loc[:, ["svalue", "qvalue", "cutoff"]], et.num_null, et.num_total)
		return (sum_tab, fin_tab, score_tab), d, needed_to_persist


	@profile
	def calculate_params_for_d_score(self, classifier, experiment):
		score = classifier.score(experiment, True)
		experiment.set_and_rerank("classifier_score", score)

		if (CONFIG.get("final_statistics.fdr_all_pg")):
			td_scores = experiment.get_decoy_peaks()["classifier_score"]
		else:
			td_scores = experiment.get_top_decoy_peaks()["classifier_score"]
		
		#mu = 0#mu = np.median(td_scores)
		#nu = 1#nu = np.percentile(td_scores, 95.0)
				
		mu, nu = mean_and_std_dev(td_scores)
		return mu, nu, score


@profile
def PyProphet():
	return HolyGostQuery(StandardSemiSupervisedTeacher(LDALearner))
