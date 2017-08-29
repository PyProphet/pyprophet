# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    def profile(fun):
        return fun

import abc
import click
from std_logger import logging
import sys
import random
import time
import warnings

from pyprophet import PyProphet
from config import CONFIG, set_pandas_print_options
from report import save_report
from levels_contexts import infer_peptides, infer_proteins

from .main_helpers import (transform_pi0_lambda, transform_threads, transform_random_seed, set_parameters, filterChromByLabels)
from ipf import infer_peptidoforms

from functools import update_wrapper

import pandas as pd
import numpy as np
import sqlite3
from shutil import copyfile

import collections
PyProphetResult = collections.namedtuple('PyProphetResult', ['table', 'parameters'])


class PyProphetRunner(object):

    __metaclass__ = abc.ABCMeta

    """Base class for workflow of command line tool
    """

    def __init__(self, table, infile, outfile, mode, level):
        self.table = table
        self.infile = infile
        self.outfile = outfile
        self.prefix = os.path.splitext(outfile)[0]
        self.mode = mode
        self.level = level

    @abc.abstractmethod
    def run_algo(self):
        pass

    @abc.abstractmethod
    def extra_writes(self):
        pass

    def run(self):

        extra_writes = dict(self.extra_writes())

        self.check_cols = [CONFIG.get("group_id"), "run_id", "decoy"]

        logging.info("config settings:")
        for k, v in sorted(CONFIG.config.items()):
            logging.info("    %s: %s" % (k, v))

        start_at = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (result, scorer, weights) = self.run_algo()

        needed = time.time() - start_at

        set_pandas_print_options()
        self.print_summary(result)

        if self.mode == 'tsv':
            self.save_tsv_results(result, extra_writes, scorer.pi0)
            self.save_tsv_weights(weights, extra_writes)
        elif self.mode == 'osw':
            self.save_osw_results(result, extra_writes, scorer.pi0)
            self.save_osw_weights(weights)

        seconds = int(needed)
        msecs = int(1000 * (needed - seconds))

        click.echo("TIME: %d seconds and %d msecs wall time" % (seconds, msecs))

    def print_summary(self, result):
        if result.summary_statistics is not None:
            click.echo("=" * 98)
            click.echo(result.summary_statistics)
            click.echo("=" * 98)

    def save_tsv_results(self, result, extra_writes, pi0):
        summ_stat_path = extra_writes.get("summ_stat_path")
        if summ_stat_path is not None:
            result.summary_statistics.to_csv(summ_stat_path, sep="\t", index=False)
            click.echo("WRITTEN: " + summ_stat_path)

        full_stat_path = extra_writes.get("full_stat_path")
        if full_stat_path is not None:
            result.final_statistics.to_csv(full_stat_path, sep="\t", index=False)
            click.echo("WRITTEN: " + full_stat_path)

        output_path = extra_writes.get("output_path")
        if output_path is not None:
            result.scored_tables.to_csv(output_path, sep="\t", index=False)
            click.echo("WRITTEN: " + output_path)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(extra_writes.get("report_path"), output_path, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)
            click.echo("WRITTEN: " + extra_writes.get("report_path"))

    def save_tsv_weights(self, weights, extra_writes):
        weights['level'] = self.level
        trained_weights_path = extra_writes.get("trained_weights_path")
        if trained_weights_path is not None:
            weights.to_csv(trained_weights_path, sep="\t", index=False)
            click.echo("WRITTEN: " + trained_weights_path)

    def save_osw_results(self, result, extra_writes, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        con = sqlite3.connect(self.outfile)

        if self.level == "ms2":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_MS2')
            con.commit()
            c.fetchall()

            df = result.scored_tables[['feature_id','d_score','peak_group_rank','p_value','q_value','pep']]
            df.columns = ['FEATURE_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_MS2"
            df.to_sql(table, con, index=False, dtype={"FEATURE_ID": "TEXT"})
        elif self.level == "ms1":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_MS1')
            con.commit()
            c.fetchall()

            df = result.scored_tables[['feature_id','d_score','peak_group_rank','p_value','q_value','pep']]
            df.columns = ['FEATURE_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_MS1"
            df.to_sql(table, con, index=False, dtype={"FEATURE_ID": "TEXT"})
        elif self.level == "transition":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_TRANSITION')
            con.commit()
            c.fetchall()

            df = result.scored_tables[['feature_id','transition_id','d_score','peak_group_rank','p_value','q_value','pep']]
            df.columns = ['FEATURE_ID','TRANSITION_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_TRANSITION"
            df.to_sql(table, con, index=False, dtype={"FEATURE_ID": "TEXT","TRANSITION_ID": "TEXT"})

        con.close()
        click.echo("WRITTEN: " + self.outfile)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(extra_writes.get("report_path"), self.outfile, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)
            click.echo("WRITTEN: " + extra_writes.get("report_path"))

    def save_osw_weights(self, weights):
        weights['level'] = self.level
        con = sqlite3.connect(self.outfile)

        c = con.cursor()
        c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_WEIGHTS"')
        if c.fetchone()[0] == 1:
            c.execute('DELETE FROM PYPROPHET_WEIGHTS WHERE LEVEL =="' + self.level + '"')
        c.fetchall()

        weights.to_sql("PYPROPHET_WEIGHTS", con, index=False, if_exists='append')

class PyProphetLearner(PyProphetRunner):

    def run_algo(self):
        (result, scorer, weights) = PyProphet().learn_and_apply(self.table)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_" + self.level + "_scored.txt")
        yield "summ_stat_path", os.path.join(self.prefix + "_" + self.level + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_" + self.level + "_full_stat.csv")
        yield "trained_weights_path", os.path.join(self.prefix + "_" + self.level + "_weights.txt")
        yield "report_path", os.path.join(self.prefix + "_" + self.level + "_report.pdf")

class PyProphetWeightApplier(PyProphetRunner):

    def __init__(self, table, infile, outfile, mode, level, apply_weights):
        super(PyProphetWeightApplier, self).__init__(table, infile, outfile, mode, level)
        if not os.path.exists(apply_weights):
            sys.exit("Error: Weights file %s does not exist." % apply_weights)
        if self.mode == "tsv":
            try:
                self.persisted_weights = pd.read_csv(apply_weights, sep="\t")
                if self.level != self.persisted_weights['level'].unique()[0]:
                    sys.exit("Error: Weights file has wrong level.")
            except Exception:
                import traceback
                traceback.print_exc()
                raise
        elif self.mode == "osw":
            try:
                con = sqlite3.connect(infile)

                data = pd.read_sql_query("SELECT * FROM PYPROPHET_WEIGHTS WHERE LEVEL=='" + self.level + "'", con)
                data.columns = [col.lower() for col in data.columns]
                con.close()
                self.persisted_weights = data
                if self.level != self.persisted_weights['level'].unique()[0]:
                    sys.exit("Error: Weights file has wrong level.")
            except Exception:
                import traceback
                traceback.print_exc()
                raise

    def run_algo(self):
        (result, scorer, weights) = PyProphet().apply_weights(self.table, self.persisted_weights)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_" + self.level + "_scored.txt")
        yield "summ_stat_path", os.path.join(self.prefix + "_" + self.level + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_" + self.level + "_full_stat.csv")
        yield "report_path", os.path.join(self.prefix + "_" + self.level + "_report.pdf")

@click.group(chain=True)
@click.version_option()
def cli():
    """
    PyProphet helptext
    """

@cli.resultcallback()
def process_commands(processors):
    """This result callback is invoked with an iterable of all the chained
    subcommands.  As in this example each subcommand returns a function
    we can chain them together to feed one into the other, similar to how
    a pipe on unix works.
    """
    # Start with an empty iterable.
    stream = ()

    # Pipe it through all stream processors.
    for processor in processors:
        stream = processor(stream)

    # Evaluate the stream and throw away the items.
    for _ in stream:
        pass

def processor(f):
    """Helper decorator to rewrite a function so that it returns another
    function from it.
    """
    def new_func(*args, **kwargs):
        def processor(stream):
            return f(stream, *args, **kwargs)
        return processor
    return update_wrapper(new_func, f)

def generator(f):
    """Similar to the :func:`processor` but passes through old values
    unchanged and does not pass through the values as parameter.
    """
    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in f(*args, **kwargs):
            yield item
    return update_wrapper(new_func, f)

# TSV file handling
@cli.command("tsv")
@click.argument('infile', nargs=1, type=click.Path(exists=True))
# Function
@processor
def tsv(ctx, infile):
    table = pd.read_csv(infile, "\t")
    return(PyProphetResult(table,{'infile': infile, 'level': 'ms2', 'mode': 'tsv'}))

# OSW file handling
@cli.command("osw")
@click.argument('infile', nargs=1, type=click.Path(exists=True))
# OSW options
@click.option('--level', default='ms2', show_default=True, type=click.Choice(['ms1', 'ms2', 'transition']), help='Either "ms1", "ms2" or "transition"; the data level selected for scoring.')
# IPF options
@click.option('--ipf_max_pgrank', default=1, show_default=True, type=int, help='Assess transitions only for candidate peak groups until maximum peak group rank.')
@click.option('--ipf_max_pgpep', default=0.3, show_default=True, type=float, help='Assess transitions only for candidate peak groups until maximum posterior error probability.')
# Function
@processor
def osw(ctx, infile, level, ipf_max_pgrank, ipf_max_pgpep):
    con = sqlite3.connect(infile)

    if level == "ms2":
        data = pd.read_sql_query("SELECT *, RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_MS2 INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID INNER JOIN (SELECT ID, DECOY FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID;", con)
    elif level == "ms1":
        data = pd.read_sql_query("SELECT *, RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_MS1 INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID INNER JOIN (SELECT ID, DECOY FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID;", con)
    elif level == "transition":
        # TODO: Remove VAR_LOG_SN_SCORE by changing default values in OSW
        data = pd.read_sql_query("SELECT TRANSITION.DECOY AS DECOY, FEATURE_TRANSITION.*, RUN_ID || '_' || FEATURE_TRANSITION.FEATURE_ID || '_' || PRECURSOR_ID || '_' || TRANSITION_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_TRANSITION INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID FROM FEATURE) AS FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID INNER JOIN (SELECT ID, DECOY FROM TRANSITION) AS TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE VAR_LOG_SN_SCORE > 0 AND RANK <= " + str(ipf_max_pgrank) + " AND PEP <= " + str(ipf_max_pgpep) + " AND PRECURSOR.DECOY == 0 ORDER BY FEATURE_ID, PRECURSOR_ID, TRANSITION_ID;", con)
    else:
        sys.exit("Error: Unspecified data level selected.")


    data.columns = [col.lower() for col in data.columns]
    con.close()

    return(PyProphetResult(data,{'infile': infile, 'level': level, 'mode': 'osw'}))

# PyProphet semi-supervised learning and scoring
@cli.command("score")
# # File handling
@click.option('--outfile', type=click.Path(exists=False), help='PyProphet output file.')
# Semi-supervised learning
@click.option('--apply_weights', type=click.Path(exists=True), help='Apply PyProphet score weights file instead of semi-supervised learning.')
@click.option('--xeval_fraction', default=0.5, show_default=True, type=float, help='Data fraction used for cross-validation of semi-supervised learning step.')
@click.option('--xeval_iterations', default=10, show_default=True, type=int, help='Number of iterations for cross-validation of semi-supervised learning step.')
@click.option('--initial_fdr', default=0.15, show_default=True, type=float, help='Initial FDR cutoff for best scoring targets.')
@click.option('--iteration_fdr', default=0.02, show_default=True, type=float, help='Iteration FDR cutoff for best scoring targets.')
@click.option('--subsample/--no-subsample', default=False, show_default=True, help='Subsample input data to speed up semi-supervised learning.')
@click.option('--subsample_rate', default=0.1, show_default=True, type=float, help='Subsampling rate of input data.')
# Statistics
@click.option('--group_id', default="transition_group_id", show_default=True, type=str, help='Group identifier for calculation of statistics.')
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
# Processing
@click.option('--threads', default=1, show_default=True, type=int, help='Number of threads used for semi-supervised learning. -1 means all available CPUs.', callback=transform_threads)
@click.option('--test/--no-test', default=False, show_default=True, help='Run in test mode with fixed seed.')
@click.option('--random_seed', default=None, show_default=True, type=int, help='Set fixed seed to integer value.', callback=transform_random_seed)
# Function
@processor
def score(result, outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, subsample, subsample_rate, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test, random_seed):

    table = result.table
    infile = result.parameters['infile']
    level = result.parameters['level']
    mode = result.parameters['mode']

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    set_parameters(outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, subsample, subsample_rate, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test, random_seed)

    learning_mode = not apply_weights

    if learning_mode:
        PyProphetLearner(table, infile, outfile, mode, level).run()

    elif apply_weights:
        PyProphetWeightApplier(table, infile, outfile, mode, level, apply_weights).run()

    yield True

# IPF
@cli.command("ipf")
# File handling
@click.argument('infile', nargs=1, type=click.Path(exists=True))
@click.option('--outfile', type=click.Path(exists=False), help='PyProphet output file.')
# IPF parameters
@click.option('--ipf_ms1_scoring/--no-ipf_ms1_scoring', default=True, show_default=True, help='Use MS1 scores for IPF.')
@click.option('--ipf_ms2_scoring/--no-ipf_ms2_scoring', default=True, show_default=True, help='Use MS2 scores for IPF.')
@click.option('--ipf_h0/--no-ipf_h0', default=True, show_default=True, help='Include possibility that peak groups are not covered by peptidoform space.')
@click.option('--ipf_max_precursor_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to consider scored precursors in IPF.')
@click.option('--ipf_max_peakgroup_pep', default=0.7, show_default=True, type=float, help='Maximum PEP to consider scored precursors in IPF.')
@click.option('--ipf_max_precursor_peakgroup_pep', default=0.4, show_default=True, type=float, help='Maximum BHM layer 1 integrated precursor peakgroup PEP to consider in IPF.')
@click.option('--ipf_max_transition_pep', default=0.6, show_default=True, type=float, help='Maximum PEP to consider scored transitions in IPF.')
# Function
@processor
def ipf(ctx, infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep):

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep)

    yield True

# Peptide-level inference
@cli.command("peptide")
# File handling
@click.argument('infile', nargs=1, type=click.Path(exists=True))
@click.option('--outfile', type=click.Path(exists=False), help='PyProphet output file.')
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
# Function
@processor
def peptide(ctx, infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_peptides(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

    yield True

# Protein-level inference
@cli.command("protein")
# # File handling
@click.argument('infile', nargs=1, type=click.Path(exists=True))
@click.option('--outfile', type=click.Path(exists=False), help='PyProphet output file.')
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
# Function
@processor
def protein(ctx, infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps):

    if outfile is None:
        outfile = infile
    else:
        outfile = outfile

    infer_proteins(infile, outfile, context, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps)

    yield True

# Merging & subsampling of multiple runs
@cli.command("merge")
@click.argument('infiles', nargs=-1, type=click.Path(exists=True))
@click.option('--outfile', type=click.Path(exists=False), help='Merged OSW output file.')
@click.option('--subsample_ratio', default=1, show_default=True, type=float, help='Subsample ratio used per input file')
# Function
@processor
def merge(ctx, infiles, outfile, subsample_ratio):
    for infile in infiles:
        if infile == infiles[0]:
            # Copy the first file to have a template
            copyfile(infile, outfile)
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            c.execute('DELETE FROM RUN')
            c.execute('DELETE FROM FEATURE')
            c.execute('DELETE FROM FEATURE_MS1')
            c.execute('DELETE FROM FEATURE_MS2')
            c.execute('DELETE FROM FEATURE_TRANSITION')
            conn.commit()
            c.fetchall()

        c.execute('ATTACH DATABASE "'+ infile + '" AS sdb')
        c.execute('INSERT INTO RUN SELECT * FROM sdb.RUN')
        c.execute('INSERT INTO FEATURE SELECT * FROM sdb.FEATURE WHERE PRECURSOR_ID IN (SELECT PRECURSOR_ID FROM sdb.FEATURE ORDER BY RANDOM() LIMIT (SELECT ROUND(' + str(subsample_ratio) + '*COUNT(DISTINCT PRECURSOR_ID)) FROM sdb.FEATURE))')
        c.execute('INSERT INTO FEATURE_MS1 SELECT * FROM sdb.FEATURE_MS1 WHERE sdb.FEATURE_MS1.FEATURE_ID IN (SELECT ID FROM FEATURE)')
        c.execute('INSERT INTO FEATURE_MS2 SELECT * FROM sdb.FEATURE_MS2 WHERE sdb.FEATURE_MS2.FEATURE_ID IN (SELECT ID FROM FEATURE)')
        c.execute('INSERT INTO FEATURE_TRANSITION SELECT * FROM sdb.FEATURE_TRANSITION WHERE sdb.FEATURE_TRANSITION.FEATURE_ID IN (SELECT ID FROM FEATURE)')
        logging.info("Merging file " + infile + " to " + outfile + ".")

    c.execute('VACUUM')
    conn.commit()
    c.fetchall()
    conn.close()
    logging.info("All OSW files merged.")

    yield True
