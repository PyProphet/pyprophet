# encoding: utf-8

import os
import sys

from config import CONFIG, MS1_CONFIG, MS2_CONFIG, UIS_CONFIG


def print_help():
    print
    script = os.path.basename(sys.argv[0])
    print "usage:"
    print "       %s [options] input_file [input_file ...]" % script
    print "   or "
    print "       %s --help" % script
    print "   or "
    print "       %s --version" % script
    print "   or prepend flag (ms1, ms2, uis) to parameters to specifically make changes for that level:"
    print "       %s --ms1.final_statistics.nonparam_null_dist input_file [input_file ...]" % script
    dump_config_info(CONFIG.config, CONFIG.info)
    sys.exit()


def print_version():
    import version
    print "%d.%d.%d" % version.version
    sys.exit()


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
    ms1_options = dict()
    ms2_options = dict()
    uis_options = dict()

    pathes = []

    if "--help" in args:
        print_help()
        sys.exit(0)

    if "--version" in args:
        print_version()
        sys.exit(0)

    for arg in args:
        if arg.startswith("--ms1."):
            if "=" in arg:
                pre, __, post = arg.partition("=")
                ms1_options[pre[6:]] = post
            else:
                ms1_options[arg[6:]] = True
        elif arg.startswith("--ms2."):
            if "=" in arg:
                pre, __, post = arg.partition("=")
                ms2_options[pre[6:]] = post
            else:
                ms2_options[arg[6:]] = True
        elif arg.startswith("--uis."):
            if "=" in arg:
                pre, __, post = arg.partition("=")
                uis_options[pre[6:]] = post
            else:
                uis_options[arg[6:]] = True
        elif arg.startswith("--"):
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
    print "General configuration:"
    dump_config(CONFIG.config)

    MS1_CONFIG.update(options)
    MS1_CONFIG.update(ms1_options)
    if (MS1_CONFIG.get("ms1_scoring.enable")):
        print "MS1 configuration:"
        dump_config(MS1_CONFIG.config)

    MS2_CONFIG.update(options)
    MS2_CONFIG.update(ms2_options)
    if (MS2_CONFIG.get("ms2_scoring.enable")):
        print "MS2 configuration:"
        dump_config(MS2_CONFIG.config)

    UIS_CONFIG.update(options)
    UIS_CONFIG.update(uis_options)
    if (UIS_CONFIG.get("uis_scoring.enable")):
        print "UIS configuration:"
        dump_config(UIS_CONFIG.config)

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

