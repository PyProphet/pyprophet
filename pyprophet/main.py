# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    def profile(fun):
        return fun

from pyprophet import PyProphet
from config import CONFIG, set_pandas_print_options
from report import save_report, export_mayu, mayu_cols
import sys
import time
import warnings
import logging
import cPickle
import zlib
import numpy as np
import pandas as pd

format_ = "%(levelname)s -- [pid=%(process)s] : %(asctime)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=format_)


def print_help():
    print
    script = os.path.basename(sys.argv[0])
    print "usage:"
    print "       %s [options] input_file [input_file ...]" % script
    print "   or "
    print "       %s --help" % script
    print "   or "
    print "       %s --version" % script
    dump_config_info(CONFIG.config, CONFIG.info)


def print_version():
    import version
    print "%d.%d.%d" % version.version


def dump_config_info(config, info):
    print
    print "parameters:"
    print
    for k, v in sorted(config.items()):
        comment = info.get(k, "")
        print "    --%-40s   default: %-5r %s" % (k, v, comment)
    print


def dump_config(config):
    print
    print "used parameters:"
    print
    for k, v in sorted(config.items()):
        print "    %-40s   : %r" % (k, v)
    print


def _create_path(prefix, dirname):

    class Pathes(dict):

        def __init__(self, prefix, dirname, **kw):
            for k, postfix in kw.items():
                self[k] = os.path.join(dirname, prefix + postfix)
        __getattr__ = dict.__getitem__

    return Pathes(prefix, dirname,
                  scored_table="_with_dscore.csv",
                  filtered_table="_with_dscore_filtered.csv",
                  report="_report.pdf",
                  cutoffs="_cutoffs.txt",
                  svalues="_svalues.txt",
                  qvalues="_qvalues.txt",
                  d_scores_top_target_peaks="_dscores_top_target_peaks.txt",
                  d_scores_top_decoy_peaks="_dscores_top_decoy_peaks.txt",
                  mayu_cutoff="_mayu.cutoff",
                  mayu_fasta="_mayu.fasta",
                  mayu_csv="_mayu.csv",
                  )


def main():
    _main(sys.argv[1:])


def _main(args):

    options = dict()
    pathes = []

    if "--help" in args:
        print_help()
        return

    if "--version" in args:
        print_version()
        return

    for arg in args:
        if arg.startswith("--"):
            if "=" in arg:
                pre, __, post = arg.partition("=")
                options[pre[2:]] = post
            else:
                options[arg[2:]] = True
        else:
            pathes.append(arg)

    if not pathes:
        print_help()
        raise Exception("no input file given")

    CONFIG.update(options)
    dump_config(CONFIG.config)

    apply_scorer = CONFIG.get("apply_scorer")
    apply_weights = CONFIG.get("apply_weights")

    learning_mode = not apply_scorer and not apply_weights

    """
    valid combinations

    learning? len(pathes)>1? merge_results?  prefix? reaction

    NO        YES            ???             ???     not allowed

              NO             ???             ???     ok

    YES       YES            YES             ???     ok

                             ???             YES     prefix ignored, ok

                                             NO      ok

              NO             ???             ???     ok

    """

    # line 1 from above:
    if len(pathes) > 1 and not learning_mode:
        raise Exception("multiple input files are only allowed for learning a shared model")

    # from now on: paramterchecks above only for learning mode

    dirname = CONFIG.get("target.dir")
    if dirname is None:
        dirnames = set(os.path.dirname(path) for path in pathes)
        # is always ok for not learning_mode, which includes that pathes has only one entry
        if len(dirnames) > 1:
            raise Exception("could not derive common directory name of input files, please use "
                            "--target.dir option")
        dirname = dirnames.pop()

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        logging.info("created folder %s" % dirname)

    prefix = CONFIG.get("target.prefix")

    merge_results = CONFIG.get("multiple_files.merge_results")

    # line 4 from above:
    if learning_mode and len(pathes) > 1 and not merge_results and prefix:
        logging.warn("ignore --target.prefix=%r" % prefix)

    if prefix is None:
        prefixes = [os.path.splitext(os.path.basename(path))[0] for path in pathes]
        common_prefix = os.path.commonprefix(prefixes)
        # is always ok for not learning_mode, which includes that pathes has only one entry
        if not common_prefix:
            raise Exception("could not derive common prefix of input file names, please use "
                            "--target.prefix option")
        prefix = common_prefix

    persisted_scorer = None
    if apply_scorer:
        if not os.path.exists(apply_scorer):
            raise Exception("scorer file %s does not exist" % apply_scorer)
        try:
            persisted_scorer = cPickle.loads(zlib.decompress(open(apply_scorer, "rb").read()))
        except:
            import traceback
            traceback.print_exc()
            raise

    apply_existing_scorer = persisted_scorer is not None

    persisted_weights = None
    if apply_weights:
        if apply_existing_scorer:
            raise Exception("can not apply existing scorer and existing weights at the same time")
        if not os.path.exists(apply_weights):
            raise Exception("weights file %s does not exist" % apply_weights)
        try:
            persisted_weights = np.loadtxt(apply_weights)

        except:
            import traceback
            traceback.print_exc()
            raise

    apply_existing_weights = persisted_weights is not None

    """
    len(pathes) > 1 and merge_results -> 1 output
    else:                                len(pathes) output
    """

    if merge_results:
        assert prefix
        out_pathes = [_create_path(prefix, dirname)]

    else:
        out_pathes = []
        for path in pathes:
            if len(pathes) > 1:
                specific_prefix = os.path.splitext(os.path.basename(path))[0]
            else:
                specific_prefix = prefix
            out_pathes.append(_create_path(specific_prefix, dirname))

    summ_stat_path = full_stat_path = None

    if not apply_existing_scorer:
        # this is the only case where not statics will be written:
        summ_stat_path = os.path.join(dirname, prefix + "_summary_stat.csv")
        full_stat_path = os.path.join(dirname, prefix + "_full_stat.csv")

    if not apply_existing_scorer:
        pickled_scorer_path = os.path.join(dirname, prefix + "_scorer.bin")

    if not apply_existing_weights:
        trained_weights_path = os.path.join(dirname, prefix + "_weights.txt")

    if not CONFIG.get("target.overwrite"):
        found_exsiting_file = False
        to_check = list(k for p in out_pathes for k in p.keys())
        if summ_stat_path:
            to_check.append(summ_stat_path)
        if full_stat_path:
            to_check.append(full_stat_path)
        if not apply_existing_scorer:
            to_check.append(pickled_scorer_path)
        if not apply_existing_weights:
            to_check.append(trained_weights_path)
        for p in to_check:
            if os.path.exists(p):
                found_exsiting_file = True
                print "ERROR: %s already exists" % p
        if found_exsiting_file:
            print
            print "please use --target.overwrite option"
            print
            return

    logging.info("config settings:")
    for k, v in sorted(CONFIG.config.items()):
        logging.info("    %s: %s" % (k, v))
    start_at = time.time()

    check_cols = ["transition_group_id", "run_id", "decoy"]
    if CONFIG.get("export.mayu"):
        check_cols += mayu_cols()

    delim_in = CONFIG.get("delim.in")
    delim_out = CONFIG.get("delim.out")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # was kommt genau zurück ????
        (tables, result, needed_to_persist, trained_weights) = PyProphet().process_csv(pathes,
                                                                                       delim_in,
                                                                                       persisted_scorer,
                                                                                       persisted_weights,
                                                                                       check_cols)
        (summ_stat, full_stat, scored_tables) = result

    if merge_results and len(scored_tables) > 0:
        scored_tables = [pd.concat(scored_tables)]

    needed = time.time() - start_at

    set_pandas_print_options()

    if summ_stat is not None:
        print
        print "=" * 98
        print
        print summ_stat
        print
        print "=" * 98

    print
    if summ_stat is not None:
        summ_stat.to_csv(summ_stat_path, delim_out, index=False)
        print "WRITTEN: ", summ_stat_path

    if full_stat is not None:
        full_stat.to_csv(full_stat_path, sep=delim_out, index=False)
        print "WRITTEN: ", full_stat_path

        for scored_table, out_path in zip(scored_tables, out_pathes):
            plot_data = save_report(out_path.report, prefix, scored_table, full_stat)
            print "WRITTEN: ", out_path.report

            cutoffs, svalues, qvalues, top_target, top_decoys = plot_data
            for (name, values) in [("cutoffs", cutoffs), ("svalues", svalues), ("qvalues", qvalues),
                                ("d_scores_top_target_peaks", top_target),
                                ("d_scores_top_decoy_peaks", top_decoys)]:
                path = out_path[name]
                with open(path, "w") as fp:
                    fp.write(" ".join("%e" % v for v in values))
                print "WRITTEN: ", path

    for scored_table, out_path in zip(scored_tables, out_pathes):

        scored_table.to_csv(out_path.scored_table, sep=delim_out, index=False)
        print "WRITTEN: ", out_path.scored_table

        filtered_table = scored_table[scored_table.d_score > CONFIG.get("d_score.cutoff")]

        filtered_table.to_csv(out_path.filtered_table, sep=delim_out, index=False)
        print "WRITTEN: ", out_path.filtered_table

        if CONFIG.get("export.mayu"):
            if full_stat:
                export_mayu(out_pathes.mayu_cutoff, out_pathes.mayu_fasta,
                            out_pathes.mayu_csv, scored_table, full_stat)
                print "WRITTEN: ", out_pathes.mayu_cutoff
                print "WRITTEN: ", out_pathes.mayu_fasta
                print "WRITTEN: ", out_pathes.mayu_csv
            else:
                logging.warn("can not write mayu table in this case")

    if not apply_existing_scorer:
        bin_data = zlib.compress(cPickle.dumps(needed_to_persist, protocol=2))
        with open(pickled_scorer_path, "wb") as fp:
            fp.write(bin_data)
        print "WRITTEN: ", pickled_scorer_path

    if not apply_existing_weights:
        np.savetxt(trained_weights_path, trained_weights, delimiter="\t")
        print "WRITTEN: ", trained_weights_path

    seconds = int(needed)
    msecs = int(1000 * (needed - seconds))
    minutes = int(needed / 60.0)

    print "NEEDED",
    if minutes:
        print minutes, "minutes and",

    print "%d seconds and %d msecs wall time" % (seconds, msecs)
    print
