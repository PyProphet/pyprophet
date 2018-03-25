import numpy as np
import click

from .data_handling import Experiment
from .classifiers import AbstractLearner
from .stats import mean_and_std_dev, find_cutoff

try:
    profile
except NameError:
    profile = lambda x: x


class AbstractSemiSupervisedLearner(object):

    def __init__(self, xeval_num_iter, xeval_fraction, is_test):
        self.xeval_num_iter = xeval_num_iter
        self.xeval_fraction = xeval_fraction
        self.is_test = is_test

    def start_semi_supervised_learning(self, train):
        raise NotImplementedError()

    def iter_semi_supervised_learning(self, train):
        raise NotImplementedError()

    def averaged_learner(self, params):
        raise NotImplementedError()

    def score(self, df, params):
        raise NotImplementedError()

    @profile
    def learn_randomized(self, experiment):
        assert isinstance(experiment, Experiment)

        click.echo("      learn on cross-validation fold")

        experiment.split_for_xval(self.xeval_fraction, self.is_test)
        train = experiment.get_train_peaks()

        train.rank_by("main_score")

        params, clf_scores = self.start_semi_supervised_learning(train)

        train.set_and_rerank("classifier_score", clf_scores)

        # semi supervised iteration:
        for inner in range(self.xeval_num_iter):
            params, clf_scores = self.iter_semi_supervised_learning(train)
            train.set_and_rerank("classifier_score", clf_scores)

        # after semi supervised iteration: classify full dataset
        clf_scores = self.score(experiment, params)
        mu, nu = mean_and_std_dev(clf_scores)
        experiment.set_and_rerank("classifier_score", clf_scores)

        td_scores = experiment.get_top_decoy_peaks()["classifier_score"]

        mu, nu = mean_and_std_dev(td_scores)
        experiment["classifier_score"] = (experiment["classifier_score"] - mu) / nu
        experiment.rank_by("classifier_score")

        top_test_peaks = experiment.get_top_test_peaks()

        top_test_target_scores = top_test_peaks.get_target_peaks()["classifier_score"]
        top_test_decoy_scores = top_test_peaks.get_decoy_peaks()["classifier_score"]

        return top_test_target_scores, top_test_decoy_scores, params


class StandardSemiSupervisedLearner(AbstractSemiSupervisedLearner):

    def __init__(self, inner_learner, xeval_num_iter, xeval_fraction, is_test, initial_fdr, iteration_fdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, use_pemp, use_pfdr):
        assert isinstance(inner_learner, AbstractLearner)
        AbstractSemiSupervisedLearner.__init__(self, xeval_num_iter, xeval_fraction, is_test)
        self.inner_learner = inner_learner
        self.initial_fdr = initial_fdr
        self.iteration_fdr = iteration_fdr
        self.pi0_lambda = pi0_lambda
        self.pi0_method = pi0_method
        self.pi0_smooth_df = pi0_smooth_df
        self.pi0_smooth_log_pi0 = pi0_smooth_log_pi0
        self.use_pemp = use_pemp
        self.use_pfdr = use_pfdr

    def select_train_peaks(self, train, sel_column, fdr, lambda_, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, use_pemp, use_pfdr):
        assert isinstance(train, Experiment)
        assert isinstance(sel_column, str)
        assert isinstance(fdr, float)

        tt_peaks = train.get_top_target_peaks()
        tt_scores = tt_peaks[sel_column]
        td_peaks = train.get_top_decoy_peaks()
        td_scores = td_peaks[sel_column]

        # find cutoff fdr from scores and only use best target peaks:
        cutoff = find_cutoff(tt_scores, td_scores, fdr, lambda_, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, use_pemp, use_pfdr)
        best_target_peaks = tt_peaks.filter_(tt_scores >= cutoff)
        return td_peaks, best_target_peaks

    def start_semi_supervised_learning(self, train):
        td_peaks, bt_peaks = self.select_train_peaks(train, "main_score", self.initial_fdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.use_pemp, self.use_pfdr)
        model = self.inner_learner.learn(td_peaks, bt_peaks, False)
        w = model.get_parameters()
        clf_scores = model.score(train, False)
        clf_scores -= np.mean(clf_scores)
        return w, clf_scores

    @profile
    def iter_semi_supervised_learning(self, train):
        td_peaks, bt_peaks = self.select_train_peaks(train, "classifier_score", self.iteration_fdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.use_pemp, self.use_pfdr)

        model = self.inner_learner.learn(td_peaks, bt_peaks, True)
        w = model.get_parameters()
        clf_scores = model.score(train, True)
        return w, clf_scores

    def averaged_learner(self, params):
        return self.inner_learner.averaged_learner(params)

    def score(self, df, params):
        self.inner_learner.set_parameters(params)
        return self.inner_learner.score(df, True)
