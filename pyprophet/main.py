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
import zlib

import numpy as np

from pyprophet import PyProphet
from config import CONFIG, set_pandas_print_options
from report import save_report

from .main_helpers import (transform_pi0_lambda, transform_threads, transform_random_seed, set_parameters, create_pathes, check_if_any_exists, filterChromByLabels)

from functools import update_wrapper

import pandas as pd

class PyProphetRunner(object):

    __metaclass__ = abc.ABCMeta

    """Base class for workflow of command line tool
    """

    def __init__(self, table, prefix):
        self.table = table
        self.prefix = prefix

    @abc.abstractmethod
    def run_algo(self):
        pass

    @abc.abstractmethod
    def extra_writes(self):
        pass

    def run(self):

        extra_writes = dict(self.extra_writes("./"))

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
        pi0 = None if scorer is None else scorer.pi0
        self.save_results(result, extra_writes, pi0)

        self.save_weights(weights, extra_writes)

        seconds = int(needed)
        msecs = int(1000 * (needed - seconds))
        minutes = int(needed / 60.0)

        print "NEEDED",
        if minutes:
            print minutes, "minutes and",

        print "%d seconds and %d msecs wall time" % (seconds, msecs)
        print

    def print_summary(self, result):
        if result.summary_statistics is not None:
            print
            print "=" * 98
            print
            print result.summary_statistics
            print
            print "=" * 98
        print

    def save_results(self, result, extra_writes, pi0):
        summ_stat_path = extra_writes.get("summ_stat_path")
        if summ_stat_path is not None:
            result.summary_statistics.to_csv(summ_stat_path, sep="\t", index=False)
            print "WRITTEN: ", summ_stat_path

        full_stat_path = extra_writes.get("full_stat_path")
        if full_stat_path is not None:
            result.final_statistics.to_csv(full_stat_path, sep="\t", index=False)
            print "WRITTEN: ", full_stat_path

        output_path = extra_writes.get("output_path")
        if output_path is not None:
            result.scored_tables.to_csv(output_path, sep="\t", index=False)
            print "WRITTEN: ", output_path

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values
            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values

            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(extra_writes.get("report_path"), output_path, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)
            print "WRITTEN: ", extra_writes.get("report_path")

    def save_weights(self, weights, extra_writes):
        trained_weights_path = extra_writes.get("trained_weights_path")
        if trained_weights_path is not None:
            np.savetxt(trained_weights_path, weights, delimiter="\t")
            print "WRITTEN: ", trained_weights_path


class PyProphetLearner(PyProphetRunner):

    def run_algo(self):
        (result, scorer, weights) = PyProphet().learn_and_apply(self.table)
        return (result, scorer, weights)

    def extra_writes(self, dirname):
        yield "output_path", os.path.join(dirname, self.prefix + "_scored.txt")
        yield "summ_stat_path", os.path.join(dirname, self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(dirname, self.prefix + "_full_stat.csv")
        yield "trained_weights_path", os.path.join(dirname, self.prefix + "_weights.txt")
        yield "report_path", os.path.join(dirname, self.prefix + "_report.pdf")

class PyProphetWeightApplier(PyProphetRunner):

    def __init__(self, table, prefix, merge_results, apply_weights, delim_in, delim_out):
        super(PyProphetWeightApplier, self).__init__(table, prefix, merge_results, delim_in,
                                                     delim_out)
        if not os.path.exists(apply_weights):
            raise Exception("weights file %s does not exist" % apply_weights)
        try:
            self.persisted_weights = np.loadtxt(apply_weights)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def run_algo(self):
        (result, scorer, weights) = PyProphet().apply_weights(self.table, self.delim_in,
                                                              self.check_cols,
                                                              self.persisted_weights)
        return (result, scorer, weights)

    def extra_writes(self, dirname):
        yield "output_path", os.path.join(dirname, self.prefix + "_scored.txt")
        yield "summ_stat_path", os.path.join(dirname, self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(dirname, self.prefix + "_full_stat.csv")
        yield "report_path", os.path.join(dirname, self.prefix + "_report.pdf")

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


def copy_filename(new, old):
    new.filename = old.filename
    return new


@cli.command("tsv")
@click.argument('infile', nargs=1, type=click.Path(exists=True))
@processor
def tsv(ctx, infile):
    table = pd.read_csv(infile, "\t")
    return(table)

@cli.command("run")
# # File handling
@click.option('--outfile', default="test123.txt", type=click.Path(exists=False), help='PyProphet output file.')

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

@processor
def run(table, outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, subsample, subsample_rate, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test, random_seed):

    set_parameters(outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, subsample, subsample_rate, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test, random_seed)

    apply_scorer = CONFIG.get("apply_scorer")
    apply_weights = CONFIG.get("apply_weights")
    prefix = CONFIG.get("target.prefix")
    merge_results = CONFIG.get("multiple_files.merge_results")
    delim_in = CONFIG.get("delim.in")
    delim_out = CONFIG.get("delim.out")
    out_of_core = CONFIG.get("out_of_core")

    random_seed = CONFIG.get("random_seed")
    num_processes = CONFIG.get("num_processes")

    if random_seed is not None and num_processes != 1:
        sys.exit("Setting random seed does not work if you run pyprophet with multiple "
                        "processes. Using more than one process is rarely faster.")

    if random_seed is not None:
        random.seed(random_seed)

    if apply_scorer and apply_weights:
        sys.exit("can not apply scorer and weights at the same time")

    learning_mode = not apply_weights

    if learning_mode:
        PyProphetLearner(table, prefix).run()

    elif apply_weights:
        PyProphetWeightApplier(
            [table], prefix, merge_results, apply_weights, delim_in, delim_out).run()

    yield True