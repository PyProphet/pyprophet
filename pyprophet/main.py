# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    def profile(fun):
        return fun

import click
from .std_logger import logging
import sys

from .runner import PyProphetLearner, PyProphetWeightApplier
from .ipf import infer_peptidoforms
from .levels_contexts import infer_peptides, infer_proteins, merge_osw # , infer_protein_groups
from .export import export_tsv

from .config import (transform_pi0_lambda, transform_threads, transform_subsample_ratio, set_parameters)

from functools import update_wrapper

import pandas as pd
pd.options.display.width = 220
pd.options.display.precision = 6

import numpy as np

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
@click.option('--apply_weights', type=click.Path(exists=True), help='Apply PyProphet score weights file instead of semi-supervised learning.')
@click.option('--xeval_fraction', default=0.5, show_default=True, type=float, help='Data fraction used for cross-validation of semi-supervised learning step.')
@click.option('--xeval_iterations', default=10, show_default=True, type=int, help='Number of iterations for cross-validation of semi-supervised learning step.')
@click.option('--initial_fdr', default=0.15, show_default=True, type=float, help='Initial FDR cutoff for best scoring targets.')
@click.option('--iteration_fdr', default=0.02, show_default=True, type=float, help='Iteration FDR cutoff for best scoring targets.')
@click.option('--ss_iterations', default=10, show_default=True, type=int, help='Number of iterations for semi-supervised learning step.')
# Statistics
@click.option('--group_id', default="group_id", show_default=True, type=str, help='Group identifier for calculation of statistics.')
@click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
@click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
@click.option('--pi0_lambda', default=[0.4,0,0], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='smoother', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
@click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
@click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
@click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
@click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
@click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
@click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
@click.option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
# OpenSWATH options
@click.option('--level', default='ms2', show_default=True, type=click.Choice(['ms1', 'ms2', 'transition']), help='Either "ms1", "ms2" or "transition"; the data level selected for scoring.')
# IPF options
@click.option('--ipf_max_peakgroup_rank', default=1, show_default=True, type=int, help='Assess transitions only for candidate peak groups until maximum peak group rank.')
@click.option('--ipf_max_peakgroup_pep', default=0.3, show_default=True, type=float, help='Assess transitions only for candidate peak groups until maximum posterior error probability.')
@click.option('--ipf_max_transition_isotope_overlap', default=0.5, show_default=True, type=float, help='Maximum isotope overlap to consider transitions in IPF.')
@click.option('--ipf_min_transition_sn', default=0, show_default=True, type=float, help='Minimum log signal-to-noise level to consider transitions in IPF. Set -1 to disable this filter.')
# TRIC
@click.option('--tric_chromprob/--no-tric_chromprob', default=False, show_default=True, help='Whether chromatogram probabilities for TRIC should be computed.')
# Processing
@click.option('--threads', default=1, show_default=True, type=int, help='Number of threads used for semi-supervised learning. -1 means all available CPUs.', callback=transform_threads)
@click.option('--test/--no-test', default=False, show_default=True, help='Run in test mode with fixed seed.')
def score(infile, outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, ss_iterations, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, tric_chromprob, threads, test):
    """
    Conduct semi-supervised learning and error-rate estimation for MS1, MS2 and transition-level data. 
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    set_parameters(xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, ss_iterations, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, tric_chromprob, threads, test)

    if not apply_weights:
        PyProphetLearner(infile, outfile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn).run()
    else:
        PyProphetWeightApplier(infile, outfile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, apply_weights).run()


# IPF
@cli.command()
# File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', type=click.Path(exists=False), help='PyProphet output file.')
# IPF parameters
@click.option('--ipf_ms1_scoring/--no-ipf_ms1_scoring', default=True, show_default=True, help='Use MS1 scores for IPF.')
@click.option('--ipf_ms2_scoring/--no-ipf_ms2_scoring', default=True, show_default=True, help='Use MS2 scores for IPF.')
@click.option('--ipf_h0/--no-ipf_h0', default=True, show_default=True, help='Include possibility that peak groups are not covered by peptidoform space.')
@click.option('--ipf_max_precursor_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to consider scored precursors in IPF.')
@click.option('--ipf_max_peakgroup_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to consider scored peak groups in IPF.')
@click.option('--ipf_max_precursor_peakgroup_pep', default=0.4, show_default=True, type=float, help='Maximum BHM layer 1 integrated precursor peakgroup PEP to consider in IPF.')
@click.option('--ipf_max_transition_pep', default=0.6, show_default=True, type=float, help='Maximum PEP to consider scored transitions in IPF.')
def ipf(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep):
    """
    Infer peptidoforms after scoring of MS1, MS2 and transition-level data.
    """

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep)


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
@click.option('--pi0_lambda', default=[0.1,1.0,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='smoother', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
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
@click.option('--pi0_lambda', default=[0.1,1.0,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
@click.option('--pi0_method', default='smoother', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
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

# # Protein Group-level inference
# @cli.command()
# # File handling
# @click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
# @click.option('--out', 'outfile', type=click.Path(exists=False), help='PyProphet output file.')
# # Grouping
# @click.option('--peptide_qvalue', default=0.01, show_default=True, type=float, help='Global peptide q-value cutoff to select peptides for protein grouping.')
# # Context
# @click.option('--context', default='global', show_default=True, type=click.Choice(['run-specific', 'experiment-wide', 'global']), help='Context to estimate protein-level FDR control.')
# # Statistics
# @click.option('--parametric/--no-parametric', default=False, show_default=True, help='Do parametric estimation of p-values.')
# @click.option('--pfdr/--no-pfdr', default=False, show_default=True, help='Compute positive false discovery rate (pFDR) instead of FDR.')
# @click.option('--pi0_lambda', default=[0.1,1.0,0.05], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4, 0, 0.', callback=transform_pi0_lambda)
# @click.option('--pi0_method', default='smoother', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')
# @click.option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')
# @click.option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')
# @click.option('--lfdr_truncate/--no-lfdr_truncate', show_default=True, default=True, help='If True, local FDR values >1 are set to 1.')
# @click.option('--lfdr_monotone/--no-lfdr_monotone', show_default=True, default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')
# @click.option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')
# @click.option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')
# @click.option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')
# def protein_group(infile, outfile, peptide_qvalue, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

#     if outfile is None:
#         outfile = infile
#     else:
#         outfile = outfile

#     infer_protein_groups(infile, outfile, peptide_qvalue, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

# Merging & subsampling of multiple runs
@cli.command()
@click.argument('infiles', nargs=-1, type=click.Path(exists=True))
@click.option('--out','outfile', type=click.Path(exists=False), help='Merged OSW output file.')
@click.option('--subsample_ratio', default=1, show_default=True, type=float, help='Subsample ratio used per input file.', callback=transform_subsample_ratio)
@click.option('--test/--no-test', default=False, show_default=True, help='Run in test mode with fixed seed.')
def merge(infiles, outfile, subsample_ratio, test):
    """
    Merge multiple OSW files and optionally subsample the data for faster learning.
    """

    if len(infiles) < 1:
        sys.exit("Error: At least one PyProphet input file needs to be defined.")

    merge_osw(infiles, outfile, subsample_ratio, test)

# Export TSV
@cli.command()
# File handling
@click.option('--in', 'infile', required=True, type=click.Path(exists=True), help='PyProphet input file.')
@click.option('--out', 'outfile', required=True, type=click.Path(exists=False), help='Output TSV/CSV file.')
@click.option('--format', default='legacy', show_default=True, type=click.Choice(['matrix', 'legacy']), help='Export format, either matrix or legacy mProphet/PyProphet format.')
@click.option('--csv/--no-csv', 'outcsv', default=False, show_default=True, help='Export CSV instead of TSV file.')
# Context
@click.option('--ipf/--no-ipf', default=True, show_default=True, help='Use IPF peptidoform-level data if available. Replaces FullPeptideName and m_score columns with IPF estimates. Note: Results do not contain decoys.')
@click.option('--peptide/--no-peptide', default=True, show_default=True, help='Append peptide-level error-rate estimates if available.')
@click.option('--protein/--no-protein', default=True, show_default=True, help='Append protein-level error-rate estimates if available.')
def export(infile, outfile, format, outcsv, ipf, peptide, protein):
    """
    Export TSV/CSV tables
    """

    export_tsv(infile, outfile, format, outcsv, ipf, peptide, protein)
