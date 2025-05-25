from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union
import os
from hyperopt import hp
import numpy as np

from ._base import BaseIOConfig


@dataclass
class ErrorEstimationConfig:
    """
    Configuration for global and local FDR (false discovery rate) estimation.

    Attributes:
        parametric (bool): Whether to use parametric estimation of p-values.
        pfdr (bool): Whether to compute positive FDR (pFDR) instead of traditional FDR.
        pi0_lambda (Any): Lambda range or fixed value for pi0 estimation (e.g., [0.1, 0.5, 0.05] or [0.4, 0.0, 0.0]).
        pi0_method (str): Method to estimate pi0; either 'smoother' or 'bootstrap'.
        pi0_smooth_df (int): Degrees of freedom for smoothing function in pi0 estimation.
        pi0_smooth_log_pi0 (bool): Whether to apply smoothing on log(pi0) estimates.
        lfdr_truncate (bool): If True, truncate local FDR values above 1 to 1.
        lfdr_monotone (bool): If True, enforce monotonic increase of local FDR values.
        lfdr_transformation (str): Transformation of p-values; either 'probit' or 'logit'.
        lfdr_adj (float): Smoothing bandwidth adjustment factor in local FDR estimation.
        lfdr_eps (float): Threshold for trimming empirical p-value distribution tails.
    """

    # Global FDR & pi0
    parametric: bool = False
    pfdr: bool = False
    pi0_lambda: Union[float, List[float]] = (0.1, 0.5, 0.05)
    pi0_method: str = "bootstrap"
    pi0_smooth_df: int = 3
    pi0_smooth_log_pi0: bool = False

    # Local FDR
    lfdr_truncate: bool = True
    lfdr_monotone: bool = True
    lfdr_transformation: str = "probit"
    lfdr_adj: float = 1.5
    lfdr_eps: float = np.power(10.0, -8)


@dataclass
class RunnerConfig:
    """
    Configuration for scoring, classifier setup, learning parameters, and optional features.

    Attributes:
        classifier (str): Classifier type used for semi-supervised learning ('LDA', 'SVM' or 'XGBoost').
        autotune (bool): Whether to autotune hyperparameters for the classifier (XGBoost / SVM)
        ss_main_score (str): Starting main score for semi-supervised learning (can be 'auto').
        main_score_selection_report (bool): Whether to generate a report for main score selection.

        xgb_hyperparams (bool): Whether to autotune XGBoost hyperparameters.
        xgb_params (dict): Default XGBoost parameters for training.
        xgb_params_space (dict): Search space for XGBoost hyperparameter optimization.

        xeval_fraction (float): Fraction of data used in each cross-validation iteration.
        xeval_num_iter (int): Number of cross-validation iterations.

        ss_initial_fdr (float): Initial FDR threshold for target selection.
        ss_iteration_fdr (float): FDR threshold used in subsequent learning iterations.
        ss_num_iter (int): Number of semi-supervised training iterations.
        ss_score_filter (bool): Whether to filter features based on score set or profile.
        ss_scale_features (bool): Whether to scale features before training.
        ss_use_dynamic_main_score (bool): Automatically determined during `__post_init__`.

        group_id (str): Column used to group PSMs for learning and statistics.
        error_estimation_config (ErrorEstimationConfig): Settings for global and local error estimation.

        ipf_max_peakgroup_rank (int): Max rank of peak groups considered in IPF.
        ipf_max_peakgroup_pep (float): Max PEP for peak group consideration in IPF.
        ipf_max_transition_isotope_overlap (float): Max isotope overlap for transition selection in IPF.
        ipf_min_transition_sn (float): Min log S/N for transition selection in IPF.

        glyco (bool): Whether glycopeptide-specific scoring is enabled.
        density_estimator (str): Score density estimation method ('kde' or 'gmm').
        grid_size (int): Number of grid cutoffs used for local FDR calculation.

        add_alignment_features (bool): Whether to add chromatographic alignment features.
        tric_chromprob (bool): Whether to compute chromatogram probabilities (for TRIC).
        threads (int): Number of CPU threads to use; -1 means all CPUs.
        test (bool): Whether to enable test mode with deterministic behavior.
        color_palette (str): Color palette used in PDF report rendering.
    """

    # Scoring / classifier options
    classifier: str = "LDA"
    autotune: bool = False
    ss_main_score: str = "auto"
    main_score_selection_report: bool = False

    # XGBoost-related hyperparameters
    xgb_hyperparams: bool = False
    xgb_params: dict = field(default_factory=dict)
    xgb_params_space: dict = field(default_factory=dict)

    # Cross-validation settings
    xeval_fraction: float = 0.5
    xeval_num_iter: int = 10

    # Semi-supervised settings
    ss_initial_fdr: float = 0.15
    ss_iteration_fdr: float = 0.05
    ss_num_iter: int = 10
    ss_score_filter: bool = (
        False  # Derived from whether ss_score_filter string is empty
    )
    ss_scale_features: bool = False
    ss_use_dynamic_main_score: bool = field(init=False)

    # Grouping & statistical options
    group_id: str = "group_id"
    error_estimation_config: ErrorEstimationConfig = field(
        default_factory=ErrorEstimationConfig
    )

    # IPF options
    ipf_max_peakgroup_rank: int = 1
    ipf_max_peakgroup_pep: float = 0.7
    ipf_max_transition_isotope_overlap: float = 0.5
    ipf_min_transition_sn: float = 0.0

    # Glyco options
    glyco: bool = False
    density_estimator: str = "gmm"
    grid_size: int = 256

    # Miscellaneous
    add_alignment_features: bool = False
    tric_chromprob: bool = False
    threads: int = 1
    test: bool = False
    color_palette: str = "normal"

    def __post_init__(self):
        # Check for auto main score selection
        if self.ss_main_score == "auto":
            # Set starting default main score
            self.ss_main_score = "var_xcorr_shape"
            self.ss_use_dynamic_main_score = True
        else:
            self.ss_use_dynamic_main_score = False


@dataclass
class RunnerIOConfig(BaseIOConfig):
    """
    Wrapper configuration class for I/O and runner parameters.

    Attributes:
        infile (str): Input file path (.osw, .parquet, or .tsv).
        outfile (str): Output file path (same format as input).
        level (str): Scoring level ('ms1', 'ms2', 'ms1ms2', 'transition', or 'alignment').
        context (str): Optional scoring context (e.g. 'experiment-wide', not commonly used).
        prefix (str): Derived from `outfile`, used as prefix for output artifacts.
        runner (RunnerConfig): All scoring and learning configuration settings.
        extra_writes (dict): Dictionary of named output paths (e.g., report, weights, summary).
    """

    runner: RunnerConfig
    extra_writes: dict = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.extra_writes = dict(self._extra_writes())

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "infile": self.infile,
            "outfile": self.outfile,
            "subsample_ratio": self.subsample_ratio,
            "level": self.level,
            "prefix": self.prefix,
            **vars(self.runner),
        }

    @classmethod
    def from_cli_args(
        cls,
        infile,
        outfile,
        subsample_ratio,
        level,
        context,
        classifier,
        autotune,
        xeval_fraction,
        xeval_num_iter,
        ss_initial_fdr,
        ss_iteration_fdr,
        ss_num_iter,
        ss_main_score,
        ss_score_filter,
        ss_scale_features,
        group_id,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
        ipf_max_peakgroup_rank,
        ipf_max_peakgroup_pep,
        ipf_max_transition_isotope_overlap,
        ipf_min_transition_sn,
        add_alignment_features,
        glyco,
        density_estimator,
        grid_size,
        tric_chromprob,
        threads,
        test,
        color_palette,
        main_score_selection_report,
    ):
        """
        Creates a configuration object from command-line arguments.
        """
        xgb_hyperparams = {
            "autotune": autotune,
            "autotune_num_rounds": 10,
            "num_boost_round": 100,
            "early_stopping_rounds": 10,
            "test_size": 0.33,
        }

        xgb_params = {
            "eta": 0.3,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1,
            "colsample_bytree": 1,
            "colsample_bylevel": 1,
            "colsample_bynode": 1,
            "lambda": 1,
            "alpha": 0,
            "scale_pos_weight": 1,
            "verbosity": 0,
            "objective": "binary:logitraw",
            "nthread": 1,
            "eval_metric": "auc",
        }

        if test:
            xgb_params["tree_method"] = "exact"

        xgb_params_space = {
            "eta": hp.uniform("eta", 0.0, 0.3),
            "gamma": hp.uniform("gamma", 0.0, 0.5),
            "max_depth": hp.quniform("max_depth", 2, 8, 1),
            "min_child_weight": hp.quniform("min_child_weight", 1, 5, 1),
            "subsample": 1,
            "colsample_bytree": 1,
            "colsample_bylevel": 1,
            "colsample_bynode": 1,
            "lambda": hp.uniform("lambda", 0.0, 1.0),
            "alpha": hp.uniform("alpha", 0.0, 1.0),
            "objective": "binary:logitraw",
            "nthread": 1,
            "eval_metric": "auc",
            "scale_pos_weight": 1.0,
            "verbosity": 0,
        }

        error_estimation_config = ErrorEstimationConfig(
            parametric=parametric,
            pfdr=pfdr,
            pi0_lambda=pi0_lambda,
            pi0_method=pi0_method,
            pi0_smooth_df=pi0_smooth_df,
            pi0_smooth_log_pi0=pi0_smooth_log_pi0,
            lfdr_truncate=lfdr_truncate,
            lfdr_monotone=lfdr_monotone,
            lfdr_transformation=lfdr_transformation,
            lfdr_adj=lfdr_adj,
            lfdr_eps=lfdr_eps,
        )

        runner_config = RunnerConfig(
            classifier=classifier,
            autotune=autotune,
            ss_main_score=ss_main_score,
            main_score_selection_report=main_score_selection_report,
            xgb_hyperparams=xgb_hyperparams,
            xgb_params=xgb_params,
            xgb_params_space=xgb_params_space,
            xeval_fraction=xeval_fraction,
            xeval_num_iter=xeval_num_iter,
            ss_initial_fdr=ss_initial_fdr,
            ss_iteration_fdr=ss_iteration_fdr,
            ss_num_iter=ss_num_iter,
            ss_score_filter=ss_score_filter,
            ss_scale_features=ss_scale_features,
            group_id=group_id,
            error_estimation_config=error_estimation_config,
            ipf_max_peakgroup_rank=ipf_max_peakgroup_rank,
            ipf_max_peakgroup_pep=ipf_max_peakgroup_pep,
            ipf_max_transition_isotope_overlap=ipf_max_transition_isotope_overlap,
            ipf_min_transition_sn=ipf_min_transition_sn,
            add_alignment_features=add_alignment_features,
            glyco=glyco,
            density_estimator=density_estimator,
            grid_size=grid_size,
            tric_chromprob=tric_chromprob,
            threads=threads,
            test=test,
            color_palette=color_palette,
        )

        return cls(
            infile=infile,
            outfile=outfile,
            subsample_ratio=subsample_ratio,
            context=context,
            level=level,
            runner=runner_config,
        )

    def _extra_writes(self):
        """
        Generates paths for various output files based on the prefix provided.

        Yields:
            Tuple[str, str]: A tuple containing the name of the output file type and the corresponding file path.
        """
        yield "output_path", os.path.join(self.prefix + "_scored.tsv")
        yield "summ_stat_path", os.path.join(self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_full_stat.csv")
        yield "trained_weights_path", os.path.join(self.prefix + "_weights.csv")
        yield "trained_model_path_ms1", os.path.join(self.prefix + "_ms1_model.bin")
        yield "trained_model_path_ms1ms2", os.path.join(
            self.prefix + "_ms1ms2_model.bin"
        )
        yield "trained_model_path_ms2", os.path.join(self.prefix + "_ms2_model.bin")
        yield "trained_model_path_transition", os.path.join(
            self.prefix + "_transition_model.bin"
        )
        yield "trained_model_path_alignment", os.path.join(
            self.prefix + "_alignment_model.bin"
        )
        yield "report_path", os.path.join(self.prefix + "_report.pdf")


@dataclass
class IPFIOConfig(BaseIOConfig):
    ipf_ms1_scoring: bool = True
    ipf_ms2_scoring: bool = True
    ipf_h0: bool = True
    ipf_grouped_fdr: bool = False
    ipf_max_precursor_pep: float = 0.7
    ipf_max_peakgroup_pep: float = 0.7
    ipf_max_precursor_peakgroup_pep: float = 0.4
    ipf_max_transition_pep: float = 0.6
    propagate_signal_across_runs: bool = False
    ipf_max_alignment_pep: float = 0.7
    across_run_confidence_threshold: float = 0.5

    @classmethod
    def from_cli_args(
        cls,
        infile,
        outfile,
        subsample_ratio,
        level,
        context,
        ipf_ms1_scoring,
        ipf_ms2_scoring,
        ipf_h0,
        ipf_grouped_fdr,
        ipf_max_precursor_pep,
        ipf_max_peakgroup_pep,
        ipf_max_precursor_peakgroup_pep,
        ipf_max_transition_pep,
        propagate_signal_across_runs,
        ipf_max_alignment_pep,
        across_run_confidence_threshold,
    ):
        """
        Creates a configuration object from command-line arguments.
        """
        return cls(
            infile=infile,
            outfile=outfile,
            subsample_ratio=subsample_ratio,
            level=level,
            context=context,
            ipf_ms1_scoring=ipf_ms1_scoring,
            ipf_ms2_scoring=ipf_ms2_scoring,
            ipf_h0=ipf_h0,
            ipf_grouped_fdr=ipf_grouped_fdr,
            ipf_max_precursor_pep=ipf_max_precursor_pep,
            ipf_max_peakgroup_pep=ipf_max_peakgroup_pep,
            ipf_max_precursor_peakgroup_pep=ipf_max_precursor_peakgroup_pep,
            ipf_max_transition_pep=ipf_max_transition_pep,
            propagate_signal_across_runs=propagate_signal_across_runs,
            ipf_max_alignment_pep=ipf_max_alignment_pep,
            across_run_confidence_threshold=across_run_confidence_threshold,
        )


@dataclass
class LevelContextIOConfig(BaseIOConfig):
    # level: Literal["peptide", "glycopeptide", "protein", "gene"] = "peptide"
    context_fdr: Literal["global", "experiment-wide", "run-specific"] = "global"
    error_estimation_config: ErrorEstimationConfig = field(
        default_factory=ErrorEstimationConfig
    )
    color_palette: Literal["normal", "protan", "deutran", "tritan"] = "normal"

    # Glycopeptide-specific options
    density_estimator: Literal["kde", "gmm"] = "gmm"
    grid_size: int = 256

    @classmethod
    def from_cli_args(
        cls,
        infile,
        outfile,
        subsample_ratio,
        level,
        context,  # context of module
        context_fdr,
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
        color_palette,
        density_estimator,
        grid_size,
    ):
        """
        Creates a configuration object from command-line arguments.
        """

        error_estimation_config = ErrorEstimationConfig(
            parametric=parametric,
            pfdr=pfdr,
            pi0_lambda=pi0_lambda,
            pi0_method=pi0_method,
            pi0_smooth_df=pi0_smooth_df,
            pi0_smooth_log_pi0=pi0_smooth_log_pi0,
            lfdr_truncate=lfdr_truncate,
            lfdr_monotone=lfdr_monotone,
            lfdr_transformation=lfdr_transformation,
            lfdr_adj=lfdr_adj,
            lfdr_eps=lfdr_eps,
        )

        return cls(
            infile=infile,
            outfile=outfile,
            subsample_ratio=subsample_ratio,
            level=level,
            context=context,
            context_fdr=context_fdr,
            error_estimation_config=error_estimation_config,
            color_palette=color_palette,
            density_estimator=density_estimator,
            grid_size=grid_size,
        )
