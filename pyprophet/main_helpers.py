# encoding: utf-8

import os
import sys

import click
from .version import version

from config import CONFIG


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
                  filtered_chroms="_with_dscore_filtered.sqMass",
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


# Filter a sqMass chromatogram file by given input labels
def filterChromByLabels(infile, outfile, labels):
    import sqlite3
    conn = sqlite3.connect(infile)
    c = conn.cursor()

    labels = [ "'" + l + "'" for l in labels]
    labels_stmt = get_ids_stmt(labels)

    stmt = "SELECT ID FROM CHROMATOGRAM WHERE NATIVE_ID IN %s" % labels_stmt
    keep_ids = [i[0] for i in list(c.execute(stmt))]
    print("Keep %s chromatograms" % len(keep_ids) )

    nr_chrom = list(c.execute("SELECT COUNT(*) FROM CHROMATOGRAM"))[0][0]
    nr_spec = list(c.execute("SELECT COUNT(*) FROM SPECTRUM"))[0][0]

    assert(nr_chrom > 0)
    assert(nr_spec == 0)

    copyDatabase(c, conn, outfile, keep_ids)

def copy_table(c, conn, keep_ids, tbl, id_col):
    stmt = "CREATE TABLE other.%s AS SELECT * FROM %s WHERE %s IN " % (tbl, tbl, id_col)
    stmt += get_ids_stmt(keep_ids) + ";"
    c.execute(stmt)
    conn.commit()

def copyDatabase(c, conn, outfile, keep_ids):
    c.execute("ATTACH DATABASE '%s' AS other;" % outfile)

    # Tables: 
    #  - DATA
    #  - SPECTRUM
    #  - RUN
    #  - RUN_EXTRA
    #  - CHROMATOGRAM
    #  - PRODUCT
    #  - PRECURSOR

    # copy over data that matches the selected ids
    copy_table(c, conn, keep_ids, "PRECURSOR", "CHROMATOGRAM_ID")
    copy_table(c, conn, keep_ids, "PRODUCT", "CHROMATOGRAM_ID")
    copy_table(c, conn, keep_ids, "DATA", "CHROMATOGRAM_ID")
    copy_table(c, conn, keep_ids, "CHROMATOGRAM", "ID")

    # copy over data and create indices
    c.execute("CREATE TABLE other.RUN AS SELECT * FROM RUN");
    c.execute("CREATE TABLE other.SPECTRUM AS SELECT * FROM SPECTRUM");
    c.execute("CREATE TABLE other.RUN_EXTRA AS SELECT * FROM RUN_EXTRA");

    c.execute("CREATE INDEX other.data_chr_idx ON DATA(CHROMATOGRAM_ID);")
    c.execute("CREATE INDEX other.data_sp_idx ON DATA(SPECTRUM_ID);")
    c.execute("CREATE INDEX other.spec_rt_idx ON SPECTRUM(RETENTION_TIME);")
    c.execute("CREATE INDEX other.spec_mslevel ON SPECTRUM(MSLEVEL);")
    c.execute("CREATE INDEX other.spec_run ON SPECTRUM(RUN_ID);")
    c.execute("CREATE INDEX other.chrom_run ON CHROMATOGRAM(RUN_ID);")

    conn.commit()

def get_ids_stmt(keep_ids):
    ids_stmt = "("
    for myid in keep_ids:
        ids_stmt += str(myid) + ","
    ids_stmt = ids_stmt[:-1]
    ids_stmt += ")"
    return ids_stmt 

