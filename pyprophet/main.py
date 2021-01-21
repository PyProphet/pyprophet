import pandas as pd
import numpy as np
import click
import sys

from .runner import PyProphetLearner, PyProphetWeightApplier
from .ipf import infer_peptidoforms
from .levels_contexts import infer_peptides, infer_proteins, infer_genes, subsample_osw, reduce_osw, merge_osw, backpropagate_oswr
from .export import export_tsv, export_score_plots
from .export_compound import export_compound_tsv
from .filter import filter_sqmass
from .data_handling import (transform_pi0_lambda, transform_threads, transform_subsample_ratio, check_sqlite_table)
from functools import update_wrapper
import sqlite3
from tabulate import tabulate

from hyperopt import hp

try:
    profile
except NameError:
    def profile(fun):
        return fun


@click.group(chain=True)
@click.version_option()
def cli():
    """
    PyProphet: Semi-supervised learning and scoring of OpenSWATH results.

    Visit http://openswath.org for usage instructions and help.
    """

# PyProphet semi-supervised learning and scoring
@cli.command()
# # File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='PyProphet output file.')
# Semi-supervised learning
@click.option('--classifier', default='LDA', show_default=True, type=click.Choice(['LDA', 'XGBoost']), help='Either a "LDA" or "XGBoost" classifier is used for semi-supervised learning.')
@click.option('--xgb_autotune/--no-xgb_autotune', default=False, show_default=True, help='XGBoost: Autotune hyperparameters.')

@click.option('--apply_weights', type=click.Path(exists=True), help='Apply PyProphet score weights file instead of semi-supervised learning.')
@click.option('--xeval_fraction', default=0.5, show_default=True, type=float, help='Data fraction used for cross-validation of semi-supervised learning step.')
@click.option('--xeval_num_iter', default=10, show_default=True, type=int, help='Number of iterations for cross-validation of semi-supervised learning step.')
@click.option('--ss_initial_fdr', default=0.15, show_default=True, type=float, help='Initial FDR cutoff for best scoring targets.')
@click.option('--ss_iteration_fdr', default=0.05, show_default=True, type=float, help='Iteration FDR cutoff for best scoring targets.')
@click.option('--ss_num_iter', default=10, show_default=True, type=int, help='Number of iterations for semi-supervised learning step.')
@click.option('--ss_main_score', default="var_xcorr_shape", show_default=True, type=str, help='Main score to start semi-supervised-learning.')
@click.option('--ss_score_filter', default='', help='Specify scores which should used for scoring. In addition specific predefined profiles can be used. For example for metabolomis data use "metabolomics".  Please specify any additional input as follows: "var_ms1_xcorr_coelution,var_library_corr,var_xcorr_coelution,etc."')
# Statistics
@click.option('--group_id', default="group_id", show_default=True, type=str, help='Group identifier for calculation of statistics.')
@click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
@click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
@click.option('--pi0_lambda', default=[0.1,0.5,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='bootstrap', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
@click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
@click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
@click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
@click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
@click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
@click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
@click.option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
# OpenSWATH options
@click.option('--level', default='ms2', show_default=True, type=click.Choice(['ms1', 'ms2', 'ms1ms2', 'transition']), help='Either "ms1", "ms2", "ms1ms2" or "transition"; the data level selected for scoring. "ms1ms2 integrates both MS1- and MS2-level scores and can be used instead of "ms2"-level results."')
# IPF options
@click.option('--ipf_max_peakgroup_rank', default=1, show_default=True, type=int, help='Assess transitions only for candidate peak groups until maximum peak group rank.')
@click.option('--ipf_max_peakgroup_pep', default=0.7, show_default=True, type=float, help='Assess transitions only for candidate peak groups until maximum posterior error probability.')
@click.option('--ipf_max_transition_isotope_overlap', default=0.5, show_default=True, type=float, help='Maximum isotope overlap to consider transitions in IPF.')
@click.option('--ipf_min_transition_sn', default=0, show_default=True, type=float, help='Minimum log signal-to-noise level to consider transitions in IPF. Set -1 to disable this filter.')
# TRIC
@click.option('--tric_chromprob/--no-tric_chromprob', default=False, show_default=True, help='Whether chromatogram probabilities for TRIC should be computed.')
# Processing
@click.option('--threads', default=1, show_default=True, type=int, help='Number of threads used for semi-supervised learning. -1 means all available CPUs.', callback=transform_threads)
@click.option('--test/--no-test', default=False, show_default=True, help='Run in test mode with fixed seed.')
def score(infile, outfile, classifier, xgb_autotune, apply_weights, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, tric_chromprob, threads, test, ss_score_filter):
    """
    Conduct semi-supervised learning and error-rate estimation for MS1, MS2 and transition-level data. 
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    # Prepare XGBoost-specific parameters
    xgb_hyperparams = {'autotune': xgb_autotune, 'autotune_num_rounds': 10, 'num_boost_round': 100, 'early_stopping_rounds': 10, 'test_size': 0.33}

    xgb_params = {'eta': 0.3, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'colsample_bynode': 1, 'lambda': 1, 'alpha': 0, 'scale_pos_weight': 1, 'silent': 1, 'objective': 'binary:logitraw', 'nthread': 1, 'eval_metric': 'auc'}

    xgb_params_space = {'eta': hp.uniform('eta', 0.0, 0.3), 'gamma': hp.uniform('gamma', 0.0, 0.5), 'max_depth': hp.quniform('max_depth', 2, 8, 1), 'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1), 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'colsample_bynode': 1, 'lambda': hp.uniform('lambda', 0.0, 1.0), 'alpha': hp.uniform('alpha', 0.0, 1.0), 'scale_pos_weight': 1.0, 'silent': 1, 'objective': 'binary:logitraw', 'nthread': 1, 'eval_metric': 'auc'}

    if not apply_weights:
        PyProphetLearner(infile, outfile, classifier, xgb_hyperparams, xgb_params, xgb_params_space, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, tric_chromprob, threads, test, ss_score_filter).run()
    else:
        PyProphetWeightApplier(infile, outfile, classifier, xgb_hyperparams, xgb_params, xgb_params_space, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, tric_chromprob, threads, test, apply_weights, ss_score_filter).run()


# IPF
@cli.command()
# File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='PyProphet output file.')
# IPF parameters
@click.option('--ipf_ms1_scoring/--no-ipf_ms1_scoring', default=True, show_default=True, help='Use MS1 precursor data for IPF.')
@click.option('--ipf_ms2_scoring/--no-ipf_ms2_scoring', default=True, show_default=True, help='Use MS2 precursor data for IPF.')
@click.option('--ipf_h0/--no-ipf_h0', default=True, show_default=True, help='Include possibility that peak groups are not covered by peptidoform space.')
@click.option('--ipf_grouped_fdr/--no-ipf_grouped_fdr', default=False, show_default=True, help='[Experimental] Compute grouped FDR instead of pooled FDR to better support data where peak groups are evaluated to originate from very heterogeneous numbers of peptidoforms.')
@click.option('--ipf_max_precursor_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to consider scored precursors in IPF.')
@click.option('--ipf_max_peakgroup_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to consider scored peak groups in IPF.')
@click.option('--ipf_max_precursor_peakgroup_pep', default=0.4, show_default=True, type=float, help='Maximum BHM layer 1 integrated precursor peakgroup PEP to consider in IPF.')
@click.option('--ipf_max_transition_pep', default=0.6, show_default=True, type=float, help='Maximum PEP to consider scored transitions in IPF.')
def ipf(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_grouped_fdr, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep):
    """
    Infer peptidoforms after scoring of MS1, MS2 and transition-level data.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_grouped_fdr, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep)


# Peptide-level inference
@cli.command()
# File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='PyProphet output file.')
# Context
@click.option('--context', default='run-specific', show_default=True, type=click.Choice(['run-specific', 'experiment-wide', 'global']), help='Context to estimate protein-level FDR control.')
# Statistics
@click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
@click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
@click.option('--pi0_lambda', default=[0.1,0.5,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='bootstrap', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
@click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
@click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
@click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
@click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
@click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
@click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
@click.option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
def peptide(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):
    """
    Infer peptides and conduct error-rate estimation in different contexts.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_peptides(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)


# Gene-level inference
@cli.command()
# File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='PyProphet output file.')
# Context
@click.option('--context', default='run-specific', show_default=True, type=click.Choice(['run-specific', 'experiment-wide', 'global']), help='Context to estimate gene-level FDR control.')
# Statistics
@click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
@click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
@click.option('--pi0_lambda', default=[0.1,0.5,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='bootstrap', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
@click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
@click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
@click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
@click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
@click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
@click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
@click.option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
def gene(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):
    """
    Infer genes and conduct error-rate estimation in different contexts.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_genes(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

# Protein-level inference
@cli.command()
# File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='PyProphet output file.')
# Context
@click.option('--context', default='run-specific', show_default=True, type=click.Choice(['run-specific', 'experiment-wide', 'global']), help='Context to estimate protein-level FDR control.')
# Statistics
@click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
@click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
@click.option('--pi0_lambda', default=[0.1,0.5,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='bootstrap', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
@click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
@click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
@click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
@click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
@click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
@click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
@click.option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
def protein(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):
    """
    Infer proteins and conduct error-rate estimation in different contexts.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_proteins(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)


# Subsample OpenSWATH file to minimum for integrated scoring
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='OpenSWATH input file.')
@click.option('--out','outfile', type=click.Path(exists=False), help='Subsampled OSWS output file.')
@click.option('--subsample_ratio', default=1, show_default=True, type=float, help='Subsample ratio used per input file.', callback=transform_subsample_ratio)
@click.option('--test/--no-test', default=False, show_default=True, help='Run in test mode with fixed seed.')
def subsample(infile, outfile, subsample_ratio, test):
    """
    Subsample OpenSWATH file to minimum for integrated scoring
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    subsample_osw(infile, outfile, subsample_ratio, test)


# Reduce scored PyProphet file to minimum for global scoring
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Scored PyProphet input file.')
@click.option('--out','outfile', type=click.Path(exists=False), help='Reduced OSWR output file.')
def reduce(infile, outfile):
    """
    Reduce scored PyProphet file to minimum for global scoring
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    reduce_osw(infile, outfile)


# Merging of multiple runs
@cli.command()
@click.argument('infiles', nargs=-1, type=click.Path(exists=True))
@click.option('--out','outfile', required=True, type=click.Path(exists=False), help='Merged OSW output file.')
@click.option('--same_run/--no-same_run', default=False, help='Assume input files are from same run (deletes run information).')
@click.option('--template','templatefile', required=True, type=click.Path(exists=False), help='Template OSW file.')
@click.option('--merged_post_scored_runs', is_flag=True, help='Merge OSW output files that have already been scored.')
def merge(infiles, outfile, same_run, templatefile, merged_post_scored_runs):
    """
    Merge multiple OSW files and (for large experiments, it is recommended to subsample first).
    """

    if len(infiles) < 1:
        raise click.ClickException("At least one PyProphet input file needs to be provided.")

    merge_osw(infiles, outfile, templatefile, same_run, merged_post_scored_runs)


# Backpropagate multi-run peptide and protein scores to single files
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='Single run PyProphet input file.')
@click.option('--out','outfile', type=click.Path(exists=False), help='Single run (with multi-run scores) PyProphet output file.')
@click.option('--apply_scores', required=True, type=click.Path(exists=True), help='PyProphet multi-run scores file to apply.')
def backpropagate(infile, outfile, apply_scores):
    """
    Backpropagate multi-run peptide and protein scores to single files
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    backpropagate_oswr(infile, outfile, apply_scores)


# Export TSV
@cli.command()
# File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='Output TSV/CSV (matrix, legacy_split, legacy_merged) file.')
@click.option('--format', default='legacy_split', show_default=True, type=click.Choice(['matrix', 'legacy_split', 'legacy_merged','score_plots']), help='Export format, either matrix, legacy_split/legacy_merged (mProphet/PyProphet) or score_plots format.')
@click.option('--csv/--no-csv', 'outcsv', default=False, show_default=True, help='Export CSV instead of TSV file.')
# Context
@click.option('--transition_quantification/--no-transition_quantification', default=True, show_default=True, help='[format: legacy] Report aggregated transition-level quantification.')
@click.option('--max_transition_pep', default=0.7, show_default=True, type=float, help='[format: legacy] Maximum PEP to retain scored transitions for quantification (requires transition-level scoring).')
@click.option('--ipf', default='peptidoform', show_default=True, type=click.Choice(['peptidoform','augmented','disable']), help='[format: matrix/legacy] Should IPF results be reported if present? "peptidoform": Report results on peptidoform-level, "augmented": Augment OpenSWATH results with IPF scores, "disable": Ignore IPF results')
@click.option('--ipf_max_peptidoform_pep', default=0.4, show_default=True, type=float, help='[format: matrix/legacy] IPF: Filter results to maximum run-specific peptidoform-level PEP.')
@click.option('--max_rs_peakgroup_qvalue', default=0.05, show_default=True, type=float, help='[format: matrix/legacy] Filter results to maximum run-specific peak group-level q-value.')
@click.option('--peptide/--no-peptide', default=True, show_default=True, help='Append peptide-level error-rate estimates if available.')
@click.option('--max_global_peptide_qvalue', default=0.01, show_default=True, type=float, help='[format: matrix/legacy] Filter results to maximum global peptide-level q-value.')
@click.option('--protein/--no-protein', default=True, show_default=True, help='Append protein-level error-rate estimates if available.')
@click.option('--max_global_protein_qvalue', default=0.01, show_default=True, type=float, help='[format: matrix/legacy] Filter results to maximum global protein-level q-value.')
def export(infile, outfile, format, outcsv, transition_quantification, max_transition_pep, ipf, ipf_max_peptidoform_pep, max_rs_peakgroup_qvalue, peptide, max_global_peptide_qvalue, protein, max_global_protein_qvalue):
    """
    Export TSV/CSV tables
    """
    if format == "score_plots":
        export_score_plots(infile)
    else:
        if outfile is None:
            if outcsv:
                outfile = infile.split(".osw")[0] + ".csv"
            else:
                outfile = infile.split(".osw")[0] + ".tsv"
        else:
            outfile = outfile

        export_tsv(infile, outfile, format, outcsv, transition_quantification, max_transition_pep, ipf, ipf_max_peptidoform_pep, max_rs_peakgroup_qvalue, peptide, max_global_peptide_qvalue, protein, max_global_protein_qvalue)


# Export Compound TSV
@cli.command()
#File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='Output TSV/CSV (matrix, legacy_merged) file.')
@click.option('--format', default='legacy_merged', show_default=True, type=click.Choice(['matrix', 'legacy_merged','score_plots']), help='Export format, either matrix, legacy_merged (PyProphet) or score_plots format.')
@click.option('--csv/--no-csv', 'outcsv', default=False, show_default=True, help='Export CSV instead of TSV file.')
# Context
@click.option('--max_rs_peakgroup_qvalue', default=0.05, show_default=True, type=float, help='[format: matrix/legacy] Filter results to maximum run-specific peak group-level q-value.')
def export_compound(infile, outfile, format, outcsv, max_rs_peakgroup_qvalue):
    """
    Export Compound TSV/CSV tables
    """
    if format == "score_plots":
        export_score_plots(infile)
    else:
        if outfile is None:
            if outcsv:
                outfile = infile.split(".osw")[0] + ".csv"
            else:
                outfile = infile.split(".osw")[0] + ".tsv"
        else:
            outfile = outfile

        export_compound_tsv(infile, outfile, format, outcsv, max_rs_peakgroup_qvalue)

# Filter sqMass files
@cli.command()
# File handling
@click.argument('sqmassfiles', nargs=-1, type=click.Path(exists=True))
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--max_precursor_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to retain scored precursors in sqMass.')
@click.option('--max_peakgroup_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to retain scored peak groups in sqMass.')
@click.option('--max_transition_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to retain scored transitions in sqMass.')
def filter(sqmassfiles, infile, max_precursor_pep, max_peakgroup_pep, max_transition_pep):
    """
    Filter sqMass files
    """

    filter_sqmass(sqmassfiles, infile, max_precursor_pep, max_peakgroup_pep, max_transition_pep)

# Print statistics
@cli.command()
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
def statistics(infile):
    """
    Print PyProphet statistics
    """

    con = sqlite3.connect(infile)

    qts = [0.01, 0.05, 0.10]

    for qt in qts:
        if check_sqlite_table(con, 'SCORE_MS2'):
            peakgroups = pd.read_sql('SELECT * FROM SCORE_MS2 INNER JOIN FEATURE ON SCORE_MS2.feature_id = FEATURE.id INNER JOIN PRECURSOR ON FEATURE.precursor_id = PRECURSOR.id INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID WHERE RANK==1 AND DECOY==0;' , con)

            click.echo("Total peakgroups (q-value<%s): %s" % (qt, len(peakgroups[peakgroups['QVALUE']<qt][['FEATURE_ID']].drop_duplicates())))
            click.echo("Total peakgroups per run (q-value<%s):" % qt)
            click.echo(tabulate(peakgroups[peakgroups['QVALUE']<qt].groupby(['FILENAME'])['FEATURE_ID'].nunique().reset_index(), showindex=False))
            click.echo(10*"=")

        if check_sqlite_table(con, 'SCORE_PEPTIDE'):
            peptides_global = pd.read_sql('SELECT * FROM SCORE_PEPTIDE INNER JOIN PEPTIDE ON SCORE_PEPTIDE.peptide_id = PEPTIDE.id WHERE CONTEXT=="global" AND DECOY==0;' , con)
            peptides = pd.read_sql('SELECT * FROM SCORE_PEPTIDE INNER JOIN PEPTIDE ON SCORE_PEPTIDE.peptide_id = PEPTIDE.id INNER JOIN RUN ON SCORE_PEPTIDE.RUN_ID = RUN.ID WHERE DECOY==0;' , con)

            click.echo("Total peptides (global context) (q-value<%s): %s" % (qt, len(peptides_global[peptides_global['QVALUE']<qt][['PEPTIDE_ID']].drop_duplicates())))
            click.echo(tabulate(peptides[peptides['QVALUE']<qt].groupby(['FILENAME'])['PEPTIDE_ID'].nunique().reset_index(), showindex=False))
            click.echo(10*"=")

        if check_sqlite_table(con, 'SCORE_PROTEIN'):
            proteins_global = pd.read_sql('SELECT * FROM SCORE_PROTEIN INNER JOIN PROTEIN ON SCORE_PROTEIN.protein_id = PROTEIN.id WHERE CONTEXT=="global" AND DECOY==0;' , con)
            proteins = pd.read_sql('SELECT * FROM SCORE_PROTEIN INNER JOIN PROTEIN ON SCORE_PROTEIN.protein_id = PROTEIN.id INNER JOIN RUN ON SCORE_PROTEIN.RUN_ID = RUN.ID WHERE DECOY==0;' , con)

            click.echo("Total proteins (global context) (q-value<%s): %s" % (qt, len(proteins_global[proteins_global['QVALUE']<qt][['PROTEIN_ID']].drop_duplicates())))
            click.echo(tabulate(proteins[proteins['QVALUE']<qt].groupby(['FILENAME'])['PROTEIN_ID'].nunique().reset_index(), showindex=False))
            click.echo(10*"=")
