import os
import click
from loguru import logger


from .util import (
    AdvancedHelpCommand,
    shared_statistics_options,
    write_logfile,
    transform_threads,
    measure_memory_usage_and_time,
    memray_profile,
)
from .._config import RunnerIOConfig
from ..scoring.runner import PyProphetLearner, PyProphetWeightApplier


# PyProphet semi-supervised learning and scoring
@click.command(name="score", cls=AdvancedHelpCommand)
# File handling
@click.option(
    "--in",
    "infile",
    required=True,
    type=click.Path(exists=True),
    help="PyProphet input file. Valid formats are .osw, .parquet and .tsv.",
)
@click.option(
    "--out",
    "outfile",
    type=click.Path(exists=False),
    help="PyProphet output file. Valid formats are .osw, .parquet and .tsv. Must be the same format as input file.",
)
@click.option(
    "--subsample_ratio",
    default=1.0,
    show_default=True,
    type=float,
    help="Subsampling ratio for large data. Use <1.0 to subsample precursors for semi-supervised learning, the learned weights will then be applied to the full data set.",
)
# Semi-supervised learning
@click.option(
    "--classifier",
    default="LDA",
    show_default=True,
    type=click.Choice(["LDA", "SVM", "XGBoost"]),
    help='Either a "LDA", "SVM" or "XGBoost" classifier is used for semi-supervised learning.',
)
@click.option(
    "--autotune/--no-autotune",
    default=False,
    show_default=True,
    help="Autotune hyperparameters for XGBoost/SVM.",
    hidden=True,
)
@click.option(
    "--apply_weights",
    type=click.Path(exists=True),
    help="Apply PyProphet score weights file (*.csv/*.bin) instead of semi-supervised learning.",
)
@click.option(
    "--xeval_fraction",
    default=0.5,
    show_default=True,
    type=float,
    help="Data fraction used for cross-validation of semi-supervised learning step.",
    hidden=True,
)
@click.option(
    "--xeval_num_iter",
    default=10,
    show_default=True,
    type=int,
    help="Number of iterations for cross-validation of semi-supervised learning step.",
)
@click.option(
    "--ss_initial_fdr",
    default=0.15,
    show_default=True,
    type=float,
    help="Initial FDR cutoff for best scoring targets.",
    hidden=True,
)
@click.option(
    "--ss_iteration_fdr",
    default=0.05,
    show_default=True,
    type=float,
    help="Iteration FDR cutoff for best scoring targets.",
    hidden=True,
)
@click.option(
    "--ss_num_iter",
    default=10,
    show_default=True,
    type=int,
    help="Number of iterations for semi-supervised learning step.",
)
@click.option(
    "--ss_main_score",
    default="auto",
    show_default=True,
    type=str,
    help='Main score to start semi-supervised-learning. Default is set to auto, meaning each iteration of learning a dynamic main score selection process will occur. If you want to have a set starting main score for each learning iteration, you can set a specifc score, i.e. "var_xcorr_shape"',
    hidden=True,
)
@click.option(
    "--ss_score_filter",
    default="",
    help='Specify scores which should used for scoring. In addition specific predefined profiles can be used. For example for metabolomis data use "metabolomics".  Please specify any additional input as follows: "var_ms1_xcorr_coelution,var_library_corr,var_xcorr_coelution,etc."',
    hidden=True,
)
@click.option(
    "--ss_scale_features/--no-ss_scale_features",
    default=False,
    show_default=True,
    help="Scale / standardize features to unit variance before semi-supervised learning.",
)
# Statistics
@click.option(
    "--group_id",
    default="group_id",
    show_default=True,
    type=str,
    help="Group identifier for calculation of statistics.",
    hidden=True,
)
@shared_statistics_options
# OpenSWATH options
@click.option(
    "--level",
    default="ms2",
    show_default=True,
    type=click.Choice(["ms1", "ms2", "ms1ms2", "transition", "alignment"]),
    help='Either "ms1", "ms2", "ms1ms2", "transition", or "alignment"; the data level selected for scoring. "ms1ms2 integrates both MS1- and MS2-level scores and can be used instead of "ms2"-level results."',
)
@click.option(
    "--add_alignment_features/--no-add_alignment_features",
    default=False,
    show_default=True,
    help="Add alignment features to scoring.",
    hidden=True,
)
# IPF options
@click.option(
    "--ipf_max_peakgroup_rank",
    default=1,
    show_default=True,
    type=int,
    help="Assess transitions only for candidate peak groups until maximum peak group rank.",
    hidden=True,
)
@click.option(
    "--ipf_max_peakgroup_pep",
    default=0.7,
    show_default=True,
    type=float,
    help="Assess transitions only for candidate peak groups until maximum posterior error probability.",
    hidden=True,
)
@click.option(
    "--ipf_max_transition_isotope_overlap",
    default=0.5,
    show_default=True,
    type=float,
    help="Maximum isotope overlap to consider transitions in IPF.",
    hidden=True,
)
@click.option(
    "--ipf_min_transition_sn",
    default=0,
    show_default=True,
    type=float,
    help="Minimum log signal-to-noise level to consider transitions in IPF. Set -1 to disable this filter.",
    hidden=True,
)
# Glyco/GproDIA Options
@click.option(
    "--glyco/--no-glyco",
    default=False,
    show_default=True,
    help="Whether glycopeptide scoring should be enabled.",
    hidden=True,
)
@click.option(
    "--density_estimator",
    default="gmm",
    show_default=True,
    type=click.Choice(["kde", "gmm"]),
    help='Either kernel density estimation ("kde") or Gaussian mixture model ("gmm") is used for score density estimation.',
    hidden=True,
)
@click.option(
    "--grid_size",
    default=256,
    show_default=True,
    type=int,
    help="Number of d-score cutoffs to build grid coordinates for local FDR calculation.",
    hidden=True,
)
# TRIC
@click.option(
    "--tric_chromprob/--no-tric_chromprob",
    default=False,
    show_default=True,
    help="Whether chromatogram probabilities for TRIC should be computed.",
    hidden=True,
)
# Visualization
@click.option(
    "--color_palette",
    default="normal",
    show_default=True,
    type=click.Choice(["normal", "protan", "deutran", "tritan"]),
    help="Color palette to use in reports.",
    hidden=True,
)
@click.option(
    "--main_score_selection_report/--no-main_score_selection_report",
    default=False,
    show_default=True,
    help="Generate a report for main score selection process.",
    hidden=True,
)
# Processing
@click.option(
    "--threads",
    default=1,
    show_default=True,
    type=int,
    help="Number of threads used for semi-supervised learning. -1 means all available CPUs.",
    callback=transform_threads,
)
@click.option(
    "--test/--no-test",
    default=False,
    show_default=True,
    help="Run in test mode with fixed seed.",
    hidden=True,
)
@click.option(
    "--profile",
    is_flag=True,
    help="Enable memory allocation tracking and profiling. Requires memrary to be installed.",
)
@click.pass_context
@memray_profile()
@measure_memory_usage_and_time
@logger.catch(reraise=True)
def score(
    ctx,
    infile,
    outfile,
    subsample_ratio,
    classifier,
    autotune,
    apply_weights,
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
    level,
    add_alignment_features,
    ipf_max_peakgroup_rank,
    ipf_max_peakgroup_pep,
    ipf_max_transition_isotope_overlap,
    ipf_min_transition_sn,
    glyco,
    density_estimator,
    grid_size,
    tric_chromprob,
    color_palette,
    main_score_selection_report,
    threads,
    test,
    profile,  # NOQA: F841 unused variable, but used in decorator
):
    """
    Conduct semi-supervised learning and error-rate estimation for MS1, MS2 and transition-level data.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    config = RunnerIOConfig.from_cli_args(
        infile,
        outfile,
        subsample_ratio,
        level,
        "score_learn",
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
    )

    write_logfile(
        ctx.obj["LOG_LEVEL"],
        f"{config.prefix}_pyp_score_{level}.log",
        ctx.obj["LOG_HEADER"],
    )

    # Validate file type and subsample ratio, subsample_ratio is currently only applicateble for "parquet_split", "parquet_split_multi". If this combination is not met, throw warning and set subsample_ratio to 1.0
    if (
        config.file_type not in ["parquet", "parquet_split", "parquet_split_multi"]
        and subsample_ratio < 1.0
    ):
        logger.warning(
            "Semi-supervised learning on a subset of the data, and then applying the weights to the full data is currently only supported for `parquet_split` and `parquet_split_multi` files.\nFor `osw`, you need to manually subsample the osw using the `subsample` module.\nSetting subsample_ratio to 1.0.",
        )
        config.subsample_ratio = 1.0

    if not apply_weights:
        if config.subsample_ratio < 1.0:
            logger.info(
                f"Conducting {level} semi-supervised learning on {config.subsample_ratio * 100}% of the data.",
            )
            weights_path = PyProphetLearner(config).run()
            # Apply weights from subsampled result to full infile

            logger.info(
                f"Info: Applying {level} weights from {weights_path} to the full data set.",
            )
            config.subsample_ratio = 1.0
            config.context = "score_apply"
            if config.file_type == "parquet_split_multi":
                base_dir = config.infile
                runs = [
                    os.path.join(base_dir, d)
                    for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))
                ]
                logger.info(
                    f"Found {len(runs)} runs in split parquet multi directory: {base_dir}",
                )
                logger.info(
                    "For memory efficiency, we will apply weights to each run individually to avoid loading all runs at once.",
                )
                for run in runs:
                    run_config = config.copy()
                    run_config.infile = run
                    if infile != outfile:
                        run_config.outfile = os.path.join(
                            os.path.basename(outfile), run
                        )
                    else:
                        run_config.outfile = run
                    run_config.prefix = os.path.splitext(run)[0]
                    run_config.file_type = "parquet_split"
                    PyProphetWeightApplier(weights_path, run_config).run()
            else:
                PyProphetWeightApplier(weights_path, config).run()
        else:
            logger.info(
                f"Conducting {level} semi-supervised learning.",
            )
            PyProphetLearner(config).run()
    else:
        logger.info(
            f"Applying {level} weights from {apply_weights} to the full data set.",
        )
        config.context = "score_apply"
        if config.file_type == "parquet_split_multi":
            base_dir = config.infile
            runs = [
                os.path.join(base_dir, d)
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]
            logger.info(
                f"Found {len(runs)} runs in split parquet multi directory: {base_dir}",
            )
            logger.info(
                "For memory efficiency, we will apply weights to each run individually to avoid loading all runs at once.",
            )
            for run in runs:
                run_config = config.copy()
                run_config.infile = run
                if infile != outfile:
                    run_config.outfile = os.path.join(os.path.basename(outfile), run)
                else:
                    run_config.outfile = run
                run_config.prefix = os.path.splitext(run)[0]
                run_config.file_type = "parquet_split"
                PyProphetWeightApplier(apply_weights, run_config).run()
        else:
            PyProphetWeightApplier(apply_weights, config).run()
