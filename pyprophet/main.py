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
import cPickle
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

class PyProphetRunner(object):

    __metaclass__ = abc.ABCMeta

    """Base class for workflow of command line tool
    """

    def __init__(self, pathes, prefix, merge_results, delim_in, delim_out):
        self.pathes = pathes
        self.prefix = prefix
        self.merge_results = merge_results
        self.delim_in = delim_in
        self.delim_out = delim_out

    @abc.abstractmethod
    def run_algo(self):
        pass

    @abc.abstractmethod
    def extra_writes(self):
        pass

    def determine_output_dir_name(self):

        # from now on: paramterchecks above only for learning mode

        dirname = CONFIG.get("target.dir")
        if dirname is None:
            dirnames = set(os.path.dirname(path) for path in self.pathes)
            # is always ok for not learning_mode, which includes that pathes has only one entry
            if len(dirnames) > 1:
                raise Exception("could not derive common directory name of input files, please use "
                                "--target.dir option")
            dirname = dirnames.pop()

        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
            logging.info("created folder %s" % dirname)
        return dirname

    def create_out_pathes(self, dirname):
        if self.merge_results:
            assert self.prefix is not None
            out_pathes = [create_pathes(self.prefix, dirname)]

        elif len(self.pathes) == 1:
            assert self.prefix is not None
            out_pathes = [create_pathes(self.prefix, dirname)]
        else:
            out_pathes = []
            for path in self.pathes:
                specific_prefix = os.path.splitext(os.path.basename(path))[0]
                out_pathes.append(create_pathes(specific_prefix, dirname))
        return out_pathes

    def run(self):

        dirname = self.determine_output_dir_name()
        out_pathes = self.create_out_pathes(dirname)

        extra_writes = dict(self.extra_writes(dirname))

        to_check = list(v for p in out_pathes for v in p.values())
        to_check.extend(extra_writes.values())

        if not CONFIG.get("target.overwrite"):
            error = check_if_any_exists(to_check)
            if error:
                return False

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
        pvalues = None if scorer is None else scorer.target_pvalues
        pi0 = None if scorer is None else scorer.pi0
        self.save_results(result, extra_writes, out_pathes, pvalues, pi0)

        self.save_scorer(scorer, extra_writes)
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

    def save_results(self, result, extra_writes, out_pathes, pvalues, pi0):
        summ_stat_path = extra_writes.get("summ_stat_path")
        if summ_stat_path is not None:
            result.summary_statistics.to_csv(summ_stat_path, self.delim_out, index=False)
            print "WRITTEN: ", summ_stat_path

        full_stat_path = extra_writes.get("full_stat_path")
        if full_stat_path is not None:
            result.final_statistics.to_csv(full_stat_path, sep=self.delim_out, index=False)
            print "WRITTEN: ", full_stat_path

        for input_path, scored_table, out_path in zip(self.pathes, result.scored_tables, out_pathes):

            cutoff = CONFIG.get("d_score.cutoff")
            scored_table.to_csv(out_path.scored_table, out_path.filtered_table, cutoff, sep=self.delim_out, index=False)
            print "WRITTEN: ", out_path.scored_table
            print "WRITTEN: ", out_path.filtered_table

            if CONFIG.get("rewrite_sqmass"):

                # get basepath
                basepath = input_path.split(".tsv")[0]
                basepath = basepath.split(".txt")[0]
                basepath = basepath.split(".csv")[0]

                # try to find a matching sqMass file
                sqmass_file = None
                if os.path.exists(basepath + ".chrom.sqMass"):
                    sqmass_file = basepath + ".chrom.sqMass"
                elif os.path.exists(basepath + ".sqMass"):
                    sqmass_file = basepath + ".sqMass"

                # get selected chromatograms on the filtered table
                df = scored_table.df[scored_table.df.d_score > cutoff]
                fragment_anno = df.aggr_Fragment_Annotation.unique()
                prec_anno = df.aggr_prec_Fragment_Annotation.unique()

                labels = []
                for l in fragment_anno:
                    labels.extend( l.split(";") )
                for l in prec_anno:
                    labels.extend( l.split(";") )

                filterChromByLabels(sqmass_file, out_path.filtered_chroms, labels)

            if result.final_statistics is not None:

                cutoffs = result.final_statistics["cutoff"].values
                svalues = result.final_statistics["svalue"].values
                qvalues = result.final_statistics["qvalue"].values
                # pvalues = result.final_statistics["pvalue"].values
                decoys, targets, top_decoys, top_targets = scored_table.scores()
                plot_data = save_report(
                    out_path.report, self.prefix, decoys, targets, top_decoys, top_targets,
                    cutoffs, svalues, qvalues, pvalues, pi0)
                print "WRITTEN: ", out_path.report

                cutoffs, svalues, qvalues, top_targets, top_decoys = plot_data
                # for (name, values) in [("cutoffs", cutoffs), ("svalues", svalues), ("qvalues", qvalues),
                #                        ("d_scores_top_target_peaks", top_targets),
                #                        ("d_scores_top_decoy_peaks", top_decoys)]:
                #     path = out_path[name]
                #     with open(path, "w") as fp:
                #         fp.write(" ".join("%e" % v for v in values))
                #     print "WRITTEN: ", path

    def save_scorer(self, scorer, extra_writes):
        pickled_scorer_path = extra_writes.get("pickled_scorer_path")
        if pickled_scorer_path is not None:
            assert scorer is not None, "invalid setting, should never happend"
            bin_data = zlib.compress(cPickle.dumps(scorer, protocol=2))
            with open(pickled_scorer_path, "wb") as fp:
                fp.write(bin_data)
            print "WRITTEN: ", pickled_scorer_path

    def save_weights(self, weights, extra_writes):
        trained_weights_path = extra_writes.get("trained_weights_path")
        if trained_weights_path is not None:
            np.savetxt(trained_weights_path, weights, delimiter="\t")
            print "WRITTEN: ", trained_weights_path


class PyProphetLearner(PyProphetRunner):

    def run_algo(self):
        (result, scorer, weights) = PyProphet().learn_and_apply(self.pathes, self.delim_in,
                                                                self.check_cols)
        return (result, scorer, weights)

    def extra_writes(self, dirname):
        yield "summ_stat_path", os.path.join(dirname, self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(dirname, self.prefix + "_full_stat.csv")
        yield "pickled_scorer_path", os.path.join(dirname, self.prefix + "_scorer.bin")
        yield "trained_weights_path", os.path.join(dirname, self.prefix + "_weights.txt")


class PyProphetOutOfCoreLearner(PyProphetLearner):

    def run_algo(self):
        (result, scorer, weights) = PyProphet().learn_and_apply_out_of_core(self.pathes,
                                                                            self.delim_in,
                                                                            self.check_cols)
        return (result, scorer, weights)


class PyProphetWeightApplier(PyProphetRunner):

    def __init__(self, pathes, prefix, merge_results, apply_weights, delim_in, delim_out):
        super(PyProphetWeightApplier, self).__init__(pathes, prefix, merge_results, delim_in,
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
        (result, scorer, weights) = PyProphet().apply_weights(self.pathes, self.delim_in,
                                                              self.check_cols,
                                                              self.persisted_weights)
        return (result, scorer, weights)

    def extra_writes(self, dirname):
        yield "summ_stat_path", os.path.join(dirname, self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(dirname, self.prefix + "_full_stat.csv")
        yield "pickled_scorer_path", os.path.join(dirname, self.prefix + "_scorer.bin")


class PyProphetOutOfCoreWeightApplier(PyProphetWeightApplier):

    def run_algo(self):
        (result, scorer, weights) = PyProphet().apply_weights_out_of_core(self.pathes, self.delim_in,
                                                                          self.check_cols,
                                                                          self.persisted_weights)
        return (result, scorer, weights)


class PyProphetOutOfCoreScorerApplier(PyProphetRunner):

    def __init__(self, pathes, prefix, merge_results, apply_scorer, delim_in, delim_out):
        super(PyProphetOutOfCoreScorerApplier, self).__init__(pathes, prefix, merge_results, delim_in,
                                                              delim_out)
        if not os.path.exists(apply_scorer):
            raise Exception("persisted scorer file %s does not exist" % apply_scorer)
        try:
            self.persisted_scorer = cPickle.loads(zlib.decompress(open(apply_scorer, "rb").read()))
            self.persisted_scorer.merge_results = merge_results
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def run_algo(self):
        (result, scorer, weights) = PyProphet().apply_scorer_out_of_core(self.pathes, self.delim_in,
                                                                         self.check_cols,
                                                                         self.persisted_scorer)
        return (result, scorer, weights)

    def extra_writes(self, dirname):
        """empty generator, see
        http://stackoverflow.com/questions/13243766/python-empty-generator-function
        """
        return
        yield


@click.command()

@click.version_option()

# # File handling
@click.argument('infiles', nargs=-1, type=click.Path(exists=True))
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

def main(infiles, outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, subsample, subsample_rate, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test, random_seed):

    pathes = set_parameters(infiles, outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, subsample, subsample_rate, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test, random_seed)

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

    learning_mode = not apply_scorer and not apply_weights

    if learning_mode:
        if out_of_core:
            PyProphetOutOfCoreLearner(pathes, prefix, merge_results, delim_in, delim_out).run()
        else:
            PyProphetLearner(pathes, prefix, merge_results, delim_in, delim_out).run()

    elif apply_weights:
        if out_of_core:
            PyProphetOutOfCoreWeightApplier(
                pathes, prefix, merge_results, apply_weights, delim_in, delim_out).run()
        else:
            PyProphetWeightApplier(
                pathes, prefix, merge_results, apply_weights, delim_in, delim_out).run()

    else:
        if out_of_core:
            logging.info("out_of_core setting ignored: this parameter has no influence for "
                         "applying a persisted scorer")
        PyProphetOutOfCoreScorerApplier(
            pathes, prefix, merge_results, apply_scorer, delim_in, delim_out).run()

