# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    profile = lambda x: x

from data_handling import Experiment
from classifiers import AbstractLearner
from scaler import ShiftDivScaler
from config import CONFIG

import numpy as np
from stats import mean_and_std_dev, find_cutoff


import logging


class AbstractSemiSupervisedTeacher(object):

    def start_semi_supervised_learning(self, train):
        raise NotImplementedError()

    def iter_semi_supervised_learning(self, train):
        raise NotImplementedError()

    def new_student(self):
        raise NotImplementedError()

    @profile
    def tutor_randomized(self, experiment):
        assert isinstance(experiment, Experiment)

        num_iter = CONFIG.get("semi_supervised_learner.num_iter")
        logging.info("start learn_randomized")
        lesson = experiment.get_train_peaks()
        lesson.rank_by("main_score")

        #print "num train:", len(lesson.df)
        # initial semi-supervised learning
        student = self.new_student()
        clf_scores = self.start_semi_supervised_learning(lesson, student)
        lesson.set_and_rerank("classifier_score", clf_scores)

        # semi supervised iteration
        for inner in range(num_iter):
            clf_scores = self.iter_semi_supervised_learning(lesson, student)
            lesson.set_and_rerank("classifier_score", clf_scores)

        # after semi-supervised iterations classify full dataset
        clf_scores = student.score(experiment, True)
        experiment.set_and_rerank("classifier_score", clf_scores)

        td_scores = experiment.get_top_decoy_peaks()["classifier_score"]

#        mid = np.median(td_scores)
#        p95 = np.percentile(td_scores, 95.0)
#        print mid, p95

        mu, nu = mean_and_std_dev(td_scores)
        student.post_scaler = ShiftDivScaler(mu, nu)
        experiment["classifier_score"] = student.post_scaler.scale(experiment["classifier_score"])
        experiment.rank_by("classifier_score")

        top_test_peaks = experiment.get_top_test_peaks()

        top_test_target_scores = top_test_peaks.get_target_peaks()["classifier_score"]
        top_test_decoy_scores = top_test_peaks.get_decoy_peaks()["classifier_score"]

        logging.info("end learn_randomized")

        return top_test_target_scores, top_test_decoy_scores, student


class StandardSemiSupervisedTeacher(AbstractSemiSupervisedTeacher):

    def __init__(self, create_inner_learner):
        #assert isinstance(inner_learner, AbstractLearner)
        self.create_inner_learner = create_inner_learner

    def new_student(self):
        return self.create_inner_learner()

    def select_train_peaks(self, train, sel_column, fdr, lambda_):
        assert isinstance(train, Experiment)
        assert isinstance(sel_column, basestring)
        assert isinstance(fdr, float)

        tt_peaks = train.get_top_target_peaks()
        tt_scores = tt_peaks[sel_column]
        td_peaks = train.get_top_decoy_peaks()
        td_scores = td_peaks[sel_column]

        # find cutoff fdr from scores and only use best target peaks:
        cutoff = find_cutoff(tt_scores, td_scores, lambda_, fdr)
        best_target_peaks = tt_peaks.filter_(tt_scores >= cutoff)
        return td_peaks, best_target_peaks

    def start_semi_supervised_learning(self, lesson, student):
        fdr = CONFIG.get("semi_supervised_learner.initial_fdr")
        lambda_ = CONFIG.get("semi_supervised_learner.initial_lambda")
        td_peaks, bt_peaks = self.select_train_peaks( lesson, "main_score", fdr, lambda_)
        student.learn(td_peaks, bt_peaks, False)
        clf_scores = student.score(lesson, False)
        clf_scores -= np.mean(clf_scores)
        return clf_scores

    @profile
    def iter_semi_supervised_learning(self, lesson, student):
        fdr = CONFIG.get("semi_supervised_learner.iteration_fdr")
        lambda_ = CONFIG.get("semi_supervised_learner.iteration_lambda")
        td_peaks, bt_peaks = self.select_train_peaks( lesson, "classifier_score", fdr, lambda_)
        student.learn(td_peaks, bt_peaks, True)
        clf_scores = student.score(lesson, True)
        return clf_scores
