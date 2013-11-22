# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except:
    profile = lambda x: x

from pyprophet import PyProphet
from config import standard_config, fix_config_types
import sys
import time
import warnings
import logging
import cPickle
import zlib


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
    prefix, __ = os.path.splitext(os.path.basename(path))

    persisted = None
    apply_ = CONFIG.get("apply")
    if apply_:
        if not os.path.exists(apply_):
            raise Exception("scorer file %s does not exist" % apply_)
        try:
            persisted = cPickle.loads(zlib.decompress(open(apply_, "rb").read()))
        except:
            import traceback
            traceback.print_exc()
            raise

    apply_existing_scorer = persisted is not None

    scored_table_path = os.path.join(dirname, prefix + "_with_dscore.csv")
    final_stat_path = os.path.join(dirname, prefix + "_full_stat.csv")
    summ_stat_path = os.path.join(dirname, prefix + "_summary_stat.csv")

    if not apply_existing_scorer:
        pickled_scorer_path = os.path.join(dirname, prefix + "_scorer.bin")

    if not CONFIG.get("target.overwrite", False):
        found_exsiting_file = False
        to_check = (scored_table_path, final_stat_path, summ_stat_path)
        if not apply_existing_scorer:
            to_check = to_check + (pickled_scorer_path,)
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
        result, needed_to_persist = PyProphet().process_csv(path, delim_in, persisted)
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
        summ_stat.to_csv(summ_stat_path, sep=delim_out, index=False)
        print "WRITTEN: ", summ_stat_path
    if final_stat is not None:
        final_stat.to_csv(final_stat_path, sep=delim_out, index=False)
        print "WRITTEN: ", final_stat_path
    scored_table.to_csv(scored_table_path, sep=delim_out, index=False)
    print "WRITTEN: ", scored_table_path

    if not apply_existing_scorer:
        bin_data = zlib.compress(cPickle.dumps(needed_to_persist, protocol=2))
        with open(pickled_scorer_path, "wb") as fp:
            fp.write(bin_data)
        print "WRITTEN: ", pickled_scorer_path
    print

    seconds = int(needed)
    msecs = int(1000 * (needed - seconds))
    minutes = int(needed / 60.0)

    print "NEEDED",
    if minutes:
        print minutes, "minutes and",

    print "%d seconds and %d msecs wall time" % (seconds, msecs)
    print
