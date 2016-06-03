# encoding: utf-8

import os
import sys

from config import CONFIG


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


def parse_cmdline(args):

    options = dict()
    pathes = []

    if "--help" in args:
        print_help()
        sys.exit(0)

    if "--version" in args:
        print_version()
        sys.exit(0)

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

    return pathes


def create_pathes(prefix, dirname):

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


def check_if_any_exists(to_check):
    found_exsiting_file = False
    for p in to_check:
        if os.path.exists(p):
            found_exsiting_file = True
            print "ERROR: %s already exists" % p
    if found_exsiting_file:
        print
        print "please use --target.overwrite option"
        print
    return found_exsiting_file
