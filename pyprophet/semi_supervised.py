import numpy as np
import click

from .data_handling import Experiment
from .classifiers import AbstractLearner, XGBLearner
from .stats import mean_and_std_dev, find_cutoff

try:
    profile
except NameError:
    profile = lambda x: x


class AbstractSemiSupervisedLearner(object):

    def __init__(self, xeval_fraction, xeval_num_iter, test):
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.test = test

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

        click.echo("Info: Learning on cross-validation fold.")

        experiment.split_for_xval(self.xeval_fraction, self.test)
        train = experiment.get_train_peaks()

        train.rank_by("main_score")

        params, clf_scores = self.start_semi_supervised_learning(train)

        train.set_and_rerank("classifier_score", clf_scores)

        # semi supervised iteration:
        for inner in range(self.xeval_num_iter):
            # # tune first iteration of semi-supervised learning
            # if inner == 0:
            #     params, clf_scores = self.tune_semi_supervised_learning(train)
            # else:
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

    def learn_final(self, experiment):
        assert isinstance(experiment, Experiment)

        click.echo("Info: Learning on cross-validated scores.")

        experiment.rank_by("classifier_score")

        params, clf_scores = self.tune_semi_supervised_learning(experiment)
        experiment.set_and_rerank("classifier_score", clf_scores)

        # after semi supervised iteration: classify full dataset
        clf_scores = self.score(experiment, params)
        mu, nu = mean_and_std_dev(clf_scores)
        experiment.set_and_rerank("classifier_score", clf_scores)

        td_scores = experiment.get_top_decoy_peaks()["classifier_score"]

        mu, nu = mean_and_std_dev(td_scores)
        experiment["classifier_score"] = (experiment["classifier_score"] - mu) / nu
        experiment.rank_by("classifier_score")

        return params


class StandardSemiSupervisedLearner(AbstractSemiSupervisedLearner):

    def __init__(self, inner_learner, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, test):
        assert isinstance(inner_learner, AbstractLearner)
        AbstractSemiSupervisedLearner.__init__(self, xeval_fraction, xeval_num_iter, test)
        self.inner_learner = inner_learner
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.ss_initial_fdr = ss_initial_fdr
        self.ss_iteration_fdr = ss_iteration_fdr
        self.parametric = parametric
        self.pfdr = pfdr
        self.pi0_lambda = pi0_lambda
        self.pi0_method = pi0_method
        self.pi0_smooth_df = pi0_smooth_df
        self.pi0_smooth_log_pi0 = pi0_smooth_log_pi0

    def select_train_peaks(self, train, sel_column, cutoff_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0):
        assert isinstance(train, Experiment)
        assert isinstance(sel_column, str)
        assert isinstance(cutoff_fdr, float)

        tt_peaks = train.get_top_target_peaks()
        tt_scores = tt_peaks[sel_column]
        td_peaks = train.get_top_decoy_peaks()
        td_scores = td_peaks[sel_column]

        # find cutoff fdr from scores and only use best target peaks:
        cutoff = find_cutoff(tt_scores, td_scores, cutoff_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0)
        best_target_peaks = tt_peaks.filter_(tt_scores >= cutoff)
        return td_peaks, best_target_peaks

    def start_semi_supervised_learning(self, train):
        td_peaks, bt_peaks = self.select_train_peaks(train, "main_score", self.ss_initial_fdr, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0)
        model = self.inner_learner.learn(td_peaks, bt_peaks, False)
        w = model.get_parameters()
        clf_scores = model.score(train, False)
        clf_scores -= np.mean(clf_scores)
        return w, clf_scores

    @profile
    def iter_semi_supervised_learning(self, train):
        td_peaks, bt_peaks = self.select_train_peaks(train, "classifier_score", self.ss_iteration_fdr, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0)

        model = self.inner_learner.learn(td_peaks, bt_peaks, True)
        w = model.get_parameters()
        clf_scores = model.score(train, True)
        return w, clf_scores

    def tune_semi_supervised_learning(self, train):
        td_peaks, bt_peaks = self.select_train_peaks(train, "classifier_score", self.ss_iteration_fdr, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0)

        if isinstance(self.inner_learner, XGBLearner) and self.inner_learner.xgb_hyperparams['autotune']:
            self.inner_learner.tune(td_peaks, bt_peaks, True)

        model = self.inner_learner.learn(td_peaks, bt_peaks, True)
        w = model.get_parameters()
        clf_scores = model.score(train, True)
        return w, clf_scores

    def averaged_learner(self, params):
        return self.inner_learner.averaged_learner(params)

    def set_learner(self, model):
        return self.inner_learner.set_parameters(model)

    def score(self, df, params):
        self.inner_learner.set_parameters(params)
        return self.inner_learner.score(df, True)
