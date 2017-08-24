# encoding: utf-8

import os
import sys
import numpy as np

import click
from .version import version

from config import CONFIG

import ntpath


# Parameter transformation functions
def transform_pi0_lambda(ctx, param, value):
  if value[1] == 0 and value[2] == 0:
      pi0_lambda = value[0]
  elif 0 <= value[0] < 1 and value[0] <= value[1] <= 1 and 0 < value[2] < 1:
      pi0_lambda = np.arange(value[0], value[1], value[2])
  else:
      sys.exit('Error: Wrong input values for pi0_lambda. pi0_lambda must be within [0,1).')
  return(pi0_lambda)

def transform_threads(ctx, param, value):
    if value == -1:
        value = multiprocessing.cpu_count()
    return(value)

def transform_random_seed(ctx, param, value):
    if value is None:
        value = np.random.randint(0, sys.maxint)
    return(value)

def dump_config(config):
    print
    print "used parameters:"
    print
    for k, v in sorted(config.items()):
        print "    %-40s   : %r" % (k, v)
    print


def set_parameters(infiles, outfile, apply_weights, xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, subsample, subsample_rate, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, threads, test, random_seed):

    options = dict()

    if len(infiles) < 1:
        sys.exit('Use option --help for further information.')

    if outfile != None:
        options['target.prefix'] = outfile
    elif outfile == None and len(infiles) == 1:
        options['target.prefix'] = os.path.splitext(ntpath.basename(infiles[0]))[0]
    else:
        sys.exit('Error: Specify output filename when analysing multiple input files together.')
    
    options['apply_weights'] = apply_weights
    options['xeval.fraction'] = xeval_fraction
    options['xeval.num_iter'] = xeval_iterations
    options['semi_supervised_learner.initial_fdr'] = initial_fdr
    options['semi_supervised_learner.initial_lambda'] = pi0_lambda
    options['semi_supervised_learner.iteration_fdr'] = iteration_fdr
    options['semi_supervised_learner.iteration_lambda'] = pi0_lambda
    options['semi_supervised_learner.num_iter'] = xeval_iterations
    options['out_of_core'] = subsample
    options['out_of_core.sampling_rate'] = subsample_rate

    options['group_id'] = group_id
    options['final_statistics.emp_p'] = parametric
    options['final_statistics.pfdr'] = pfdr
    options['final_statistics.lambda'] = pi0_lambda
    options['final_statistics.pi0_method'] = pi0_method
    options['final_statistics.pi0_smooth_df'] = pi0_smooth_df
    options['final_statistics.pi0_smooth_log_pi0'] = pi0_smooth_log_pi0
    options['final_statistics.lfdr_trunc'] = lfdr_truncate
    options['final_statistics.lfdr_monotone'] = lfdr_monotone
    options['final_statistics.lfdr_transf'] = lfdr_transformation
    options['final_statistics.lfdr_adj'] = lfdr_adj
    options['final_statistics.lfdr_eps'] = lfdr_eps

    options['num_processes'] = threads
    options['is_test'] = test

    options['random_seed'] = random_seed


    CONFIG.update(options)
    dump_config(CONFIG.config)

    return infiles


def create_pathes(prefix, dirname):

    class Pathes(dict):

        def __init__(self, prefix, dirname, **kw):
            for k, postfix in kw.items():
                self[k] = os.path.join(dirname, prefix + postfix)
        __getattr__ = dict.__getitem__

    return Pathes(prefix, dirname,
                  scored_table="_scored.txt",
                  filtered_table="_with_dscore_filtered.csv",
                  filtered_chroms="_scored.sqMass",
                  report="_report.pdf",
                  cutoffs="_cutoffs.txt",
                  svalues="_svalues.txt",
                  qvalues="_qvalues.txt",
                  d_scores_top_target_peaks="_dscores_top_target_peaks.txt",
                  d_scores_top_decoy_peaks="_dscores_top_decoy_peaks.txt",
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

