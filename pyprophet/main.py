# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    profile = lambda x: x

from pyprophet import PyProphet
from config import standard_config, fix_config_types
from report import save_report, export_mayu
import sys
import time
import warnings
import logging
import cPickle
import zlib
import numpy as np

def print_help():
    print
    script = os.path.basename(sys.argv[0])
    print "usage:"
    print "       %s [options] input_file" % script
    print "   or "
    print "       %s --help" % script
    print "   or "
    print "       %s --version" % script
    CONFIG, info = standard_config()
    dump_config_info(CONFIG, info)


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


def main():
    _main(sys.argv[1:])


def _main(args):

    options = dict()
    path = None

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
            if path is not None:
                print_help()
                raise Exception("duplicate input file argument")
            path = arg

    if path is None:
        print_help()
        raise Exception("no input file given")

    CONFIG, info = standard_config()
    CONFIG.update(options)
    fix_config_types(CONFIG)
    dump_config(CONFIG)

    delim_in = CONFIG.get("delim.in", ",")
    delim_out = CONFIG.get("delim.out", ",")

    dirname = CONFIG.get("target.dir", None)
    if dirname is None:
        dirname = os.path.dirname(path)

    basename = os.path.basename(path)
    prefix, __ = os.path.splitext(basename)

    persisted_scorer = None
    apply_scorer = CONFIG.get("apply_scorer")
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
    apply_weights = CONFIG.get("apply_weights")
    if apply_weights:
        if not os.path.exists(apply_weights):
            raise Exception("weights file %s does not exist" % apply_weights)
        try:
            persisted_weights = np.loadtxt(apply_weights)

        except:
            import traceback
            traceback.print_exc()
            raise

    apply_existing_weights = persisted_weights is not None


    class Pathes(dict):

        def __init__(self, prefix=prefix, dirname=dirname, **kw):
            for k, postfix in kw.items():
                self[k] = os.path.join(dirname, prefix + postfix)
        __getattr__ = dict.__getitem__

    pathes = Pathes(scored_table="_with_dscore.csv",
                    filtered_table="_with_dscore_filtered.csv",
                    final_stat="_full_stat.csv",
                    summ_stat="_summary_stat.csv",
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

    if not apply_existing_scorer:
        pickled_scorer_path = os.path.join(dirname, prefix + "_scorer.bin")

    if not apply_existing_weights:
        trained_weights_path = os.path.join(dirname, prefix + "_weights.txt")

    if not CONFIG.get("target.overwrite", False):
        found_exsiting_file = False
        to_check = list(pathes.keys())
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

    format_ = "%(levelname)s -- [pid=%(process)s] : %(asctime)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=format_)
    logging.info("config settings:")
    for k, v in sorted(CONFIG.items()):
        logging.info("    %s: %s" % (k, v))
    start_at = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, needed_to_persist, trained_weights = PyProphet().process_csv(path, delim_in, persisted_scorer, persisted_weights)
        (summ_stat, final_stat, scored_table) = result
    needed = time.time() - start_at

    print
    print "=" * 78
    print
    print summ_stat
    print
    print "=" * 78

    print
    if summ_stat is not None:
        summ_stat.to_csv(pathes.summ_stat, sep=delim_out, index=False)
        print "WRITTEN: ", pathes.summ_stat
    if final_stat is not None:
        final_stat.to_csv(pathes.final_stat, sep=delim_out, index=False)
        print "WRITTEN: ", pathes.final_stat
        plot_data = save_report(pathes.report, basename, scored_table, final_stat)
        print "WRITTEN: ", pathes.report
        cutoffs, svalues, qvalues, top_target, top_decoys = plot_data
        for (name, values) in [("cutoffs", cutoffs), ("svalues", svalues), ("qvalues", qvalues),
                               ("d_scores_top_target_peaks", top_target),
                               ("d_scores_top_decoy_peaks", top_decoys)]:
            path = pathes[name]
            with open(path, "w") as fp:
                fp.write(" ".join("%e" % v for v in values))
            print "WRITTEN: ", path
    scored_table.to_csv(pathes.scored_table, sep=delim_out, index=False)
    print "WRITTEN: ", pathes.scored_table

    filtered_table = scored_table[scored_table.d_score > CONFIG.get("d_score.cutoff")]

    filtered_table.to_csv(pathes.filtered_table, sep=delim_out, index=False)
    print "WRITTEN: ", pathes.filtered_table

    if not apply_existing_scorer:
        bin_data = zlib.compress(cPickle.dumps(needed_to_persist, protocol=2))
        with open(pickled_scorer_path, "wb") as fp:
            fp.write(bin_data)
        print "WRITTEN: ", pickled_scorer_path

    if not apply_existing_weights:
        np.savetxt(trained_weights_path,trained_weights,delimiter="\t")
        print "WRITTEN: ", trained_weights_path

    if CONFIG.get("export.mayu", True):
        export_mayu(pathes.mayu_cutoff, pathes.mayu_fasta, pathes.mayu_csv, scored_table, final_stat)
        print "WRITTEN: ", pathes.mayu_cutoff
        print "WRITTEN: ", pathes.mayu_fasta
        print "WRITTEN: ", pathes.mayu_csv
    print

    seconds = int(needed)
    msecs = int(1000 * (needed - seconds))
    minutes = int(needed / 60.0)

    print "NEEDED",
    if minutes:
        print minutes, "minutes and",

    print "%d seconds and %d msecs wall time" % (seconds, msecs)
    print
