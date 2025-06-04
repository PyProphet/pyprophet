"""
This module implements semi-supervised learning for statistical scoring and error estimation
in targeted proteomics and glycoproteomics data analysis.

It provides abstract and concrete implementations of semi-supervised learners, enabling
iterative learning and scoring workflows. The learners support dynamic main score selection,
cross-validation, and parameter tuning.

Classes:
    - AbstractSemiSupervisedLearner: Base class for semi-supervised learning workflows.
    - StandardSemiSupervisedLearner: Implements a standard semi-supervised learning workflow.

Functions:
    - profile: A no-op decorator for profiling (used if no profiler is available).
"""

import numpy as np
from loguru import logger

from .._config import RunnerIOConfig
from ..stats import find_cutoff
from .classifiers import AbstractLearner, SVMLearner, XGBLearner
from .data_handling import Experiment, update_chosen_main_score_in_table

try:
    profile
except NameError:
    profile = lambda x: x


class AbstractSemiSupervisedLearner(object):
    """
    Abstract base class for semi-supervised learning workflows.

    Attributes:
        xeval_fraction (float): Fraction of data used for cross-validation.
        xeval_num_iter (int): Number of iterations for cross-validation.
        test (bool): Whether to enable testing mode.
    """

    def __init__(self, xeval_fraction, xeval_num_iter, test):
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.test = test

    def start_semi_supervised_learning(
        self, train, score_columns, working_thread_number
    ):
        """
        Abstract method to start the semi-supervised learning process.

        Args:
            train (Experiment): Training data.
            score_columns (list): List of score column names.
            working_thread_number (int): Number of threads to use.
        """
        raise NotImplementedError()

    def iter_semi_supervised_learning(self, train):
        """
        Abstract method for iterative semi-supervised learning.

        Args:
            train (Experiment): Training data.
        """
        raise NotImplementedError()

    def averaged_learner(self, params, **kwargs):
        """
        Abstract method to create an averaged learner from multiple parameter sets.

        Args:
            params (list): List of parameter sets.
            kwargs: Additional arguments.
        """
        raise NotImplementedError()

    def score(self, df, params):
        """
        Abstract method to score the given data using the trained model.

        Args:
            df (pd.DataFrame): Input data.
            params (dict): Model parameters.
        """
        raise NotImplementedError()

    @profile
    def learn_randomized(self, experiment, score_columns, working_thread_number):
        """
        Performs randomized semi-supervised learning with cross-validation.

        Args:
            experiment (Experiment): The experiment data.
            score_columns (list): List of score column names.
            working_thread_number (int): Number of threads to use.

        Returns:
            tuple: Target scores, decoy scores, and model parameters.
        """
        assert isinstance(experiment, Experiment)

        logger.info("Learning on cross-validation fold.")

        experiment.split_for_xval(self.xeval_fraction, self.test)
        train = experiment.get_train_peaks()

        train.rank_by("main_score")

        params, clf_scores, use_as_main_score = self.start_semi_supervised_learning(
            train, score_columns, working_thread_number
        )

        # Get current main score column name
        old_main_score_column = [col for col in score_columns if "main" in col][0]
        # Only Update if chosen main score column has changed
        if (
            use_as_main_score != old_main_score_column
            and self.ss_use_dynamic_main_score
        ):
            train, _ = update_chosen_main_score_in_table(
                train, score_columns, use_as_main_score
            )
            train.rank_by("main_score")
            experiment, score_columns = update_chosen_main_score_in_table(
                experiment, score_columns, use_as_main_score
            )

        train.set_and_rerank("classifier_score", clf_scores)

        # semi supervised iteration:
        for inner in range(self.xeval_num_iter):
            # # tune first iteration of semi-supervised learning
            # if inner == 0:
            #     params, clf_scores = self.tune_semi_supervised_learning(train)
            # else:
            params, clf_scores = self.iter_semi_supervised_learning(
                train, score_columns, working_thread_number
            )
            train.set_and_rerank("classifier_score", clf_scores)

        # after semi supervised iteration: classify full dataset
        clf_scores = self.score(experiment, params)
        experiment.set_and_rerank("classifier_score", clf_scores)

        experiment.normalize_score_by_decoys("classifier_score")
        experiment.rank_by("classifier_score")

        top_test_peaks = experiment.get_top_test_peaks()

        top_test_target_scores = top_test_peaks.get_target_peaks()["classifier_score"]
        top_test_decoy_scores = top_test_peaks.get_decoy_peaks()["classifier_score"]

        return top_test_target_scores, top_test_decoy_scores, params

    def learn_final(self, experiment):
        """
        Performs final learning on cross-validated scores.

        Args:
            experiment (Experiment): The experiment data.

        Returns:
            dict: Final model parameters.
        """
        assert isinstance(experiment, Experiment)

        logger.info("Learning on cross-validated scores.")

        experiment.rank_by("classifier_score")

        params, clf_scores = self.tune_semi_supervised_learning(experiment)
        experiment.set_and_rerank("classifier_score", clf_scores)

        # after semi supervised iteration: classify full dataset
        clf_scores = self.score(experiment, params)
        experiment.set_and_rerank("classifier_score", clf_scores)

        experiment.normalize_score_by_decoys("classifier_score")
        experiment.rank_by("classifier_score")

        return params


class StandardSemiSupervisedLearner(AbstractSemiSupervisedLearner):
    """
    Implements a standard semi-supervised learning workflow.

    Attributes:
        inner_learner (AbstractLearner): The base learner used for training.
        ss_initial_fdr (float): Initial FDR threshold for training.
        ss_iteration_fdr (float): FDR threshold for iterative learning.
        parametric (bool): Whether to use parametric FDR estimation.
        pfdr (bool): Whether to use pFDR estimation.
        pi0_lambda (list): Lambda values for pi0 estimation.
        pi0_method (str): Method for pi0 estimation.
        pi0_smooth_df (int): Degrees of freedom for pi0 smoothing.
        pi0_smooth_log_pi0 (bool): Whether to log-transform pi0 values.
        ss_use_dynamic_main_score (bool): Whether to dynamically select the main score.
    """

    def __init__(
        self,
        inner_learner,
        xeval_fraction,
        xeval_num_iter,
        ss_initial_fdr,
        ss_iteration_fdr,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        test,
        main_score_selection_report,
        outfile,
        level,
        ss_use_dynamic_main_score,
    ):
        assert isinstance(inner_learner, AbstractLearner)
        AbstractSemiSupervisedLearner.__init__(
            self, xeval_fraction, xeval_num_iter, test
        )
        self.inner_learner = inner_learner
        self.autotune = inner_learner.autotune
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

    @classmethod
    def from_config(cls, config: RunnerIOConfig, base_learner):
        """
        Creates a StandardSemiSupervisedLearner instance from a configuration object.

        Args:
            config (RunnerIOConfig): The configuration object.
            base_learner (AbstractLearner): The base learner used for training.

        Returns:
            StandardSemiSupervisedLearner: The initialized learner.
        """
        rc = config.runner
        return cls(
            base_learner,
            rc.xeval_fraction,
            rc.xeval_num_iter,
            rc.ss_initial_fdr,
            rc.ss_iteration_fdr,
            rc.error_estimation_config.parametric,
            rc.error_estimation_config.pfdr,
            rc.error_estimation_config.pi0_lambda,
            rc.error_estimation_config.pi0_method,
            rc.error_estimation_config.pi0_smooth_df,
            rc.error_estimation_config.pi0_smooth_log_pi0,
            rc.test,
            rc.main_score_selection_report,
            config.outfile,
            config.level,
            rc.ss_use_dynamic_main_score,
        )

    def select_train_peaks(
        self,
        train,
        sel_column,
        cutoff_fdr,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        mapper=None,
        main_score_selection_report=False,
        outfile=None,
        level=None,
        working_thread_number=None,
    ):
        """
        Selects the best target peaks and top decoy peaks based on FDR thresholds.

        Args:
            train (Experiment): Training data.
            sel_column (str): Column used for selection.
            cutoff_fdr (float): FDR threshold for selection.
            parametric (bool): Whether to use parametric FDR estimation.
            pfdr (bool): Whether to use pFDR estimation.
            pi0_lambda (list): Lambda values for pi0 estimation.
            pi0_method (str): Method for pi0 estimation.
            pi0_smooth_df (int): Degrees of freedom for pi0 smoothing.
            pi0_smooth_log_pi0 (bool): Whether to log-transform pi0 values.
            mapper (dict, optional): Mapping of column aliases to feature names.
            main_score_selection_report (bool, optional): Whether to generate a score selection report.
            outfile (str, optional): Path to the output file.
            level (str, optional): Analysis level (e.g., peptide, protein).
            working_thread_number (int, optional): Number of threads to use.

        Returns:
            tuple: Top decoy peaks and best target peaks.
        """

        assert isinstance(train, Experiment)
        assert isinstance(sel_column, str)
        assert isinstance(cutoff_fdr, float)

        tt_peaks = train.get_top_target_peaks()
        tt_scores = tt_peaks[sel_column]
        td_peaks = train.get_top_decoy_peaks()
        td_scores = td_peaks[sel_column]

        # find cutoff fdr from scores and only use best target peaks:
        cutoff = find_cutoff(
            tt_scores,
            td_scores,
            cutoff_fdr,
            parametric,
            pfdr,
            pi0_lambda,
            pi0_method,
            pi0_smooth_df,
            pi0_smooth_log_pi0,
            sel_column,
            mapper,
            main_score_selection_report,
            outfile,
            level,
            working_thread_number,
        )
        best_target_peaks = tt_peaks.filter_(tt_scores >= cutoff)
        return td_peaks, best_target_peaks

    def get_delta_td_bt_feature_size(self, train, col, mapper, working_thread_number):
        """
        Calculates the difference in feature size between top decoy peaks and best target peaks.

        Args:
            train (Experiment): Training data.
            col (str): Column used for selection.
            mapper (dict): Mapping of column aliases to feature names.
            working_thread_number (int): Number of threads to use.

        Returns:
            int: The absolute difference in feature size.
        """
        assert isinstance(train, Experiment)
        assert isinstance(col, str)

        # Try catch exception when using a feature column that cannot generate a valid pi0 estimation due to imbalance of number of top decoys to best targets
        try:
            td_peaks, bt_peaks = self.select_train_peaks(
                train,
                col,
                self.ss_initial_fdr,
                self.parametric,
                self.pfdr,
                self.pi0_lambda,
                self.pi0_method,
                self.pi0_smooth_df,
                self.pi0_smooth_log_pi0,
                mapper,
                self.main_score_selection_report,
                self.outfile,
                self.level,
                working_thread_number,
            )
            return abs(td_peaks.df.shape[0] - bt_peaks.df.shape[0])
        except:
            # Return highest possible value if select_train_peaks fails to run due to not being able to compute pi0 estimation
            return float("inf")

    def start_semi_supervised_learning(
        self, train, score_columns, working_thread_number
    ):
        """
        Starts the semi-supervised learning process.

        Args:
            train (Experiment): Training data.
            score_columns (list): List of score column names.
            working_thread_number (int): Number of threads to use.

        Returns:
            tuple: Model parameters, classifier scores, and selected main score column.
        """
        # Get tables aliased score variable name
        df_column_score_alias = [
            col
            for col in train.df.columns
            if col
            not in [
                "tg_id",
                "tg_num_id",
                "is_decoy",
                "is_top_peak",
                "is_train",
                "classifier_score",
            ]
        ]
        # Generate column alias name to score feature name
        mapper = {
            alias_col: col
            for alias_col, col in zip(df_column_score_alias, score_columns)
        }
        if isinstance(self.inner_learner, XGBLearner):
            # dynamic selection of main score seems to only benefit the XBGLearner, the LDALearner performs worse when we apply this
            # Use the min() function to find the column with the smallest delta value
            use_as_main_col_alias = min(
                df_column_score_alias,
                key=lambda x: self.get_delta_td_bt_feature_size(
                    train, x, mapper, working_thread_number
                ),
            )
        else:
            use_as_main_col_alias = "main_score"

        td_peaks, bt_peaks = self.select_train_peaks(
            train,
            use_as_main_col_alias,
            self.ss_initial_fdr,
            self.parametric,
            self.pfdr,
            self.pi0_lambda,
            self.pi0_method,
            self.pi0_smooth_df,
            self.pi0_smooth_log_pi0,
            mapper,
            self.main_score_selection_report,
            self.outfile,
            self.level,
            working_thread_number,
        )
        model = self.inner_learner.learn(td_peaks, bt_peaks, False)
        w = model.get_parameters()
        clf_scores = model.score(train, False)
        clf_scores -= np.mean(clf_scores)

        return w, clf_scores, mapper[use_as_main_col_alias]

    @profile
    def iter_semi_supervised_learning(
        self, train, score_columns, working_thread_number
    ):
        """
        Performs iterative semi-supervised learning.

        Args:
            train (Experiment): Training data.
            score_columns (list): List of score column names.
            working_thread_number (int): Number of threads to use.

        Returns:
            tuple: Model parameters and classifier scores.
        """
        # Get tables aliased score variable name
        df_column_score_alias = [
            col
            for col in train.df.columns
            if col not in ["tg_id", "tg_num_id", "is_decoy", "is_top_peak", "is_train"]
        ]
        # Generate column alias name to score feature name
        mapper = {
            alias_col: col
            for alias_col, col in zip(
                df_column_score_alias, score_columns + ("classifier_score",)
            )
        }
        td_peaks, bt_peaks = self.select_train_peaks(
            train,
            "classifier_score",
            self.ss_iteration_fdr,
            self.parametric,
            self.pfdr,
            self.pi0_lambda,
            self.pi0_method,
            self.pi0_smooth_df,
            self.pi0_smooth_log_pi0,
            mapper,
            self.main_score_selection_report,
            self.outfile,
            self.level,
            working_thread_number,
        )

        model = self.inner_learner.learn(td_peaks, bt_peaks, True)
        w = model.get_parameters()
        clf_scores = model.score(train, True)
        return w, clf_scores

    def tune_semi_supervised_learning(self, train):
        """
        Tunes the semi-supervised learning model.

        Args:
            train (Experiment): Training data.

        Returns:
            tuple: Model parameters and classifier scores.
        """
        td_peaks, bt_peaks = self.select_train_peaks(
            train,
            "classifier_score",
            self.ss_iteration_fdr,
            self.parametric,
            self.pfdr,
            self.pi0_lambda,
            self.pi0_method,
            self.pi0_smooth_df,
            self.pi0_smooth_log_pi0,
        )

        if isinstance(self.inner_learner, XGBLearner) and self.inner_learner.autotune:
            self.inner_learner.tune(td_peaks, bt_peaks, True)
        elif isinstance(self.inner_learner, SVMLearner) and self.inner_learner.autotune:
            self.inner_learner.tune(td_peaks, bt_peaks, True)

        model = self.inner_learner.learn(td_peaks, bt_peaks, True)
        w = model.get_parameters()
        clf_scores = model.score(train, True)
        return w, clf_scores

    def averaged_learner(self, params, **kwargs):
        """
        Creates an averaged learner from multiple parameter sets.

        Args:
            params (list): List of parameter sets.
            kwargs: Additional arguments.

        Returns:
            AbstractLearner: The averaged learner.
        """
        return self.inner_learner.averaged_learner(params, **kwargs)

    def set_learner(self, model):
        """
        Sets the parameters of the inner learner.

        Args:
            model (object): The model parameters.
        """
        logger.trace(f"Setting inner learner parmeters from : {model}")
        return self.inner_learner.set_parameters(model)

    def score(self, df, params):
        """
        Scores the given data using the trained model.

        Args:
            df (pd.DataFrame): Input data.
            params (dict): Model parameters.

        Returns:
            np.ndarray: Classifier scores.
        """
        self.inner_learner.set_parameters(params)
        return self.inner_learner.score(df, True)
