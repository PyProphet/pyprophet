import numpy as np
import click

from .data_handling import Experiment, update_chosen_main_score_in_table
from .classifiers import AbstractLearner, XGBLearner
from .stats import find_cutoff

try:
    profile
except NameError:
    profile = lambda x: x


class AbstractSemiSupervisedLearner(object):

    def __init__(self, xeval_fraction, xeval_num_iter, test):
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.test = test

    def start_semi_supervised_learning(self, train, score_columns, working_thread_number):
        raise NotImplementedError()

    def iter_semi_supervised_learning(self, train):
        raise NotImplementedError()

    def averaged_learner(self, params):
        raise NotImplementedError()

    def score(self, df, params):
        raise NotImplementedError()

    @profile
    def learn_randomized(self, experiment, score_columns, working_thread_number):
        assert isinstance(experiment, Experiment)

        click.echo("Info: Learning on cross-validation fold.")

        experiment.split_for_xval(self.xeval_fraction, self.test)
        train = experiment.get_train_peaks()

        train.rank_by("main_score")

        params, clf_scores, use_as_main_score = self.start_semi_supervised_learning(train, score_columns, working_thread_number)
        
        # Get current main score column name
        old_main_score_column = [col for col in score_columns if  'main' in col][0]
        # Only Update if chosen main score column has changed
        if use_as_main_score != old_main_score_column and self.ss_use_dynamic_main_score:
            train, _ = update_chosen_main_score_in_table(train, score_columns, use_as_main_score)
            train.rank_by("main_score")
            experiment, score_columns = update_chosen_main_score_in_table(experiment, score_columns, use_as_main_score)

        train.set_and_rerank("classifier_score", clf_scores)

        # semi supervised iteration:
        for inner in range(self.xeval_num_iter):
            # # tune first iteration of semi-supervised learning
            # if inner == 0:
            #     params, clf_scores = self.tune_semi_supervised_learning(train)
            # else:
            params, clf_scores = self.iter_semi_supervised_learning(train, score_columns, working_thread_number)
            train.set_and_rerank("classifier_score", clf_scores)

        # after semi supervised iteration: classify full dataset
        clf_scores = self.score(experiment, params)
        experiment.set_and_rerank("classifier_score", clf_scores)

        experiment.normalize_score_by_decoys('classifier_score')
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
        experiment.set_and_rerank("classifier_score", clf_scores)

        experiment.normalize_score_by_decoys('classifier_score')
        experiment.rank_by("classifier_score")

        return params


class StandardSemiSupervisedLearner(AbstractSemiSupervisedLearner):

    def __init__(self, inner_learner, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, test, main_score_selection_report, outfile, level, ss_use_dynamic_main_score):
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
        self.main_score_selection_report = main_score_selection_report
        self.outfile = outfile
        self.level = level
        self.ss_use_dynamic_main_score = ss_use_dynamic_main_score

    def select_train_peaks(self, train, sel_column, cutoff_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, mapper=None, main_score_selection_report=False, outfile=None, level=None, working_thread_number=None):
        assert isinstance(train, Experiment)
        assert isinstance(sel_column, str)
        assert isinstance(cutoff_fdr, float)

        tt_peaks = train.get_top_target_peaks()
        tt_scores = tt_peaks[sel_column]
        td_peaks = train.get_top_decoy_peaks()
        td_scores = td_peaks[sel_column]

        # find cutoff fdr from scores and only use best target peaks:
        cutoff = find_cutoff(tt_scores, td_scores, cutoff_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, sel_column, mapper, main_score_selection_report, outfile, level, working_thread_number)
        best_target_peaks = tt_peaks.filter_(tt_scores >= cutoff)
        return td_peaks, best_target_peaks

    def get_delta_td_bt_feature_size(self, train, col, mapper, working_thread_number):
        '''
        Get the the difference in feature size based on which column is used in select_train_peaks for top decoy features and best target features
        '''
        assert isinstance(train, Experiment)
        assert isinstance(col, str)
        
        # Try catch exception when using a feature column that cannot generate a valid pi0 estimation due to imbalance of number of top decoys to best targets
        try:
            td_peaks, bt_peaks = self.select_train_peaks(train, col, self.ss_initial_fdr, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, mapper, self.main_score_selection_report, self.outfile, self.level, working_thread_number)
            return abs(td_peaks.df.shape[0] - bt_peaks.df.shape[0])
        except:
            # Return highest possible value if select_train_peaks fails to run due to not being able to compute pi0 estimation
            return float('inf')

    def start_semi_supervised_learning(self, train, score_columns, working_thread_number):
        # Get tables aliased score variable name
        df_column_score_alias = [col for col in train.df.columns if col not in ['tg_id', 'tg_num_id', 'is_decoy', 'is_top_peak', 'is_train', 'classifier_score']]
        # Generate column alias name to score feature name
        mapper = {alias_col : col for alias_col, col in zip(df_column_score_alias, score_columns)}
        if isinstance(self.inner_learner, XGBLearner):
            # dynamic selection of main score seems to only benefit the XBGLearner, the LDALearner performs worse when we apply this
            # Use the min() function to find the column with the smallest delta value
            use_as_main_col_alias = min(df_column_score_alias, key=lambda x: self.get_delta_td_bt_feature_size(train, x, mapper, working_thread_number))
        else:
            use_as_main_col_alias = 'main_score'

        td_peaks, bt_peaks = self.select_train_peaks(train, use_as_main_col_alias, self.ss_initial_fdr, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, mapper, self.main_score_selection_report, self.outfile, self.level, working_thread_number)
        model = self.inner_learner.learn(td_peaks, bt_peaks, False)
        w = model.get_parameters()
        clf_scores = model.score(train, False)
        clf_scores -= np.mean(clf_scores)

        return w, clf_scores, mapper[use_as_main_col_alias]

    @profile
    def iter_semi_supervised_learning(self, train, score_columns, working_thread_number):
        # Get tables aliased score variable name
        df_column_score_alias = [col for col in train.df.columns if col not in ['tg_id', 'tg_num_id', 'is_decoy', 'is_top_peak', 'is_train']]
        # Generate column alias name to score feature name
        mapper = {alias_col : col for alias_col, col in zip(df_column_score_alias, score_columns + ('classifier_score',))}
        td_peaks, bt_peaks = self.select_train_peaks(train, "classifier_score", self.ss_iteration_fdr, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, mapper, self.main_score_selection_report, self.outfile, self.level, working_thread_number)

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
