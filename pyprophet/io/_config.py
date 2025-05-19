from dataclasses import dataclass, field
from typing import Any, Dict
import os
from ._base import BaseIOConfig


@dataclass
class RunnerConfig:
    # Scoring / classifier options
    classifier: str
    ss_main_score: str
    main_score_selection_report: bool

    # XGBoost-related hyperparameters
    xgb_hyperparams: bool
    xgb_params: dict
    xgb_params_space: dict

    # Cross-validation settings
    xeval_fraction: float
    xeval_num_iter: int

    # Semi-supervised settings
    ss_initial_fdr: float
    ss_iteration_fdr: float
    ss_num_iter: int
    ss_score_filter: bool
    ss_use_dynamic_main_score: bool = field(init=False)

    # Grouping & statistical options
    group_id: str
    parametric: bool
    pfdr: bool

    # Pi0 estimation
    pi0_lambda: Any  # Typically float or List[float]
    pi0_method: str
    pi0_smooth_df: int
    pi0_smooth_log_pi0: bool

    # Local FDR config
    lfdr_truncate: bool
    lfdr_monotone: bool
    lfdr_transformation: str
    lfdr_adj: bool
    lfdr_eps: float

    # IPF options
    ipf_max_peakgroup_rank: int
    ipf_max_peakgroup_pep: float
    ipf_max_transition_isotope_overlap: float
    ipf_min_transition_sn: float
    add_alignment_features: bool

    # Glyco options
    glyco: bool
    density_estimator: str
    grid_size: int

    # Miscellaneous
    tric_chromprob: bool
    threads: int
    test: bool
    color_palette: str

    def __post_init__(self):
        # Check for auto main score selection
        if self.ss_main_score == "auto":
            # Set starting default main score
            self.ss_main_score = "var_xcorr_shape"
            self.use_dynamic_main_score = True
        else:
            self.use_dynamic_main_score = False


@dataclass
class RunnerIOConfig(BaseIOConfig):
    runner: RunnerConfig

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "infile": self.infile,
            "outfile": self.outfile,
            "level": self.level,
            "prefix": self.prefix,
            **vars(self.runner),
        }

    def extra_writes(self):
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
        yield "report_path", os.path.join(self.prefix + "_report.pdf")


@dataclass
class IPFIOConfig(BaseIOConfig):
    ipf_filter_decoys: bool
    min_ipf_confidence: float


@dataclass
class LevelContextIOConfig(BaseIOConfig):
    context_level: str
    aggregation_method: str
