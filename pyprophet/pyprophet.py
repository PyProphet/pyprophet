# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
	profile  # ignore
except:
	profile = lambda x: x

import pandas as pd
import numpy as np

from stats import (lookup_s_and_q_values_from_error_table, calculate_final_statistics,
				   mean_and_std_dev, final_err_table, summary_err_table)
from config import CONFIG

from data_handling import (prepare_data_table, Experiment)
from classifiers import (LDALearner, ConsensusPredictor)
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
	def process_csv(self, path, delim=",", loaded_scorer=None):
		start_at = time.time()

		logging.info("read %s" % path)

		table = pd.read_csv(path, delim, na_values=["NA", "NaN", "infinite"])

		if loaded_scorer is not None:
			logging.info("apply scorer to  %s" % path)
			result_tables, clf_scores, data_for_persistence = self.apply_loaded_scorer(table, loaded_scorer)
			data_for_persistence = None
		else:
			logging.info("learn and apply scorer to %s" % path)
			result_tables, clf_scores, data_for_persistence = self.tutor_and_apply_classifier(table)
		logging.info("processing %s finished" % path)

		needed = time.time() - start_at
		hours = int(needed / 3600)
		needed -= hours * 3600

		minutes = int(needed / 60)
		needed -= minutes * 60

		logging.info("time needed: %02d:%02d:%.1f" % (hours, minutes, needed))
		return result_tables, clf_scores, data_for_persistence


	@profile
	def tutor_and_apply_classifier(self, table):

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
		teacher	 = self.semi_supervised_teacher
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
		final_classifier = ConsensusPredictor(clfs)
		
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

		results, data_for_persistence = self.apply_classifier(final_classifier, experiment, test_exp,
															 all_test_target_scores,
															 all_test_decoy_scores, table)
		logging.info("calculated scoring and statistics")
		return  results, pd.DataFrame(d), data_for_persistence + (score_columns,)



	@profile
	def apply_loaded_scorer(self, table, loaded_scorer):

		final_classifier, mu, nu, df_raw_stat, loaded_score_columns = loaded_scorer

		prepared_table, __ = prepare_data_table(table, loaded_score_columns=loaded_score_columns)

		experiment = Experiment(prepared_table)

		final_score = final_classifier.score(experiment, True)
		experiment["d_score"] = (final_score - mu) / nu

		scored_table = self.enrich_table_with_results(table, experiment, df_raw_stat)

		return {'pyProph':(None, None, scored_table)}, None, None



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
						 all_test_decoy_scores, table):

		lambda_ = CONFIG.get("final_statistics.lambda")

		mu, nu, final_score = self.calculate_params_for_d_score(final_classifier, experiment)
		experiment["d_score"] = (final_score - mu) / nu

		all_tt_scores = experiment.get_top_target_peaks()["d_score"]
		
		def getRes(et):
			return (et.summary_table(), et.final_table(), et.enrich(table, experiment))
		
		is_test	 = CONFIG.get("is_test", False)
		
		if CONFIG.get("is_test", False):
			d = {
				'pyProph-flex':getRes(FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NormalNullModel(),
									PyProphFDRCalc(),
									PyProphStatCalc(),
									PyProphStatSampler()
							)), 
				'nonParam-flex':getRes(FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NonParamNullModel(),
									PyProphFDRCalc(),
									PyProphStatCalc(),
									PyProphStatSampler()
							)),
				'logNormal-flex':getRes(FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									LogNormalNullModel(),
									PyProphFDRCalc(),
									PyProphStatCalc(),
									PyProphStatSampler()
							)),
				'nonParam-storey':getRes(FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NonParamNullModel(),
									StoreyFDRCalc(),
									PyProphStatCalc(),
									PyProphStatSampler()
							)),
				'nonParam-storey-jt':getRes(FlexibleErrorTable(
									all_tt_scores,
									all_test_target_scores,
									all_test_decoy_scores,
									lambda_,
									NonParamNullModel(),
									StoreyFDRCalc(),
									JTStatCalc(),
									PyProphStatSampler()
							))
				}
			
			d["res"] = d["nonParam-flex"]
		
			if test_exp is not None:
				muT, nuT, final_scoreT = self.calculate_params_for_d_score(final_classifier, test_exp)
				test_exp["d_score"] = (final_scoreT - muT) / nuT
				d['true_pyProph'] = getRes(FlexibleErrorTable(
						all_tt_scores, 
						test_exp.get_top_target_peaks()["d_score"],
						test_exp.get_top_decoy_peaks()["d_score"],
						lambda_,
						NormalNullModel(),
						PyProphFDRCalc(),
						PyProphStatCalc(),
						PyProphStatSampler()
					))
				d['true_nonParam'] = getRes(FlexibleErrorTable(
						all_tt_scores, 
						test_exp.get_top_target_peaks()["d_score"],
						test_exp.get_top_decoy_peaks()["d_score"],
						lambda_,
						NonParamNullModel(),
						PyProphFDRCalc(),
						PyProphStatCalc(),
						PyProphStatSampler()
					))
				d['true_logNormal'] = getRes(FlexibleErrorTable(
						all_tt_scores,
						test_exp.get_top_target_peaks()["d_score"],
						test_exp.get_top_decoy_peaks()["d_score"],
						lambda_,
						LogNormalNullModel(),
						PyProphFDRCalc(),
						PyProphStatCalc(),
						PyProphStatSampler()
					))
		else:
			 null_model 	= getNullModel(CONFIG.get("final_statistics.null_model"))
			 fdr_calc 		= getFDRCalc(CONFIG.get("final_statistics.fdr_calc"))
			 stat_calc 		= getStatCalc(CONFIG.get("final_statistics.stat_calc"))
			 stat_sampler 	= getStatSampler(CONFIG.get("final_statistics.stat_sampler"))
			 decoys_missing	= CONFIG.get("decoy.missing", 0.0)
			 d = dict(
			 		res = getRes(FlexibleErrorTable(
						all_tt_scores,
						all_test_target_scores,
						all_test_decoy_scores,
						lambda_,
						null_model,
						fdr_calc,
						stat_calc,
						stat_sampler,
						decoys_missing
					))
			 	)


		needed_to_persist = (final_classifier, mu, nu)
							 #error_table.df_raw_stat.loc[:, ["svalue", "qvalue", "cutoff"]])
		return d, needed_to_persist

	@profile
	def calculate_params_for_d_score(self, classifier, experiment):
		score = classifier.score(experiment, True)
		experiment.set_and_rerank("classifier_score", score)
		td_scores = experiment.get_top_decoy_peaks()["classifier_score"]
		
		mu = 0#mu = np.median(td_scores)
		nu = 1#nu = np.percentile(td_scores, 95.0)
				
		#mu, nu = mean_and_std_dev(td_scores)
		return mu, nu, score


@profile
def PyProphet():
	return HolyGostQuery(StandardSemiSupervisedTeacher(LDALearner()))
