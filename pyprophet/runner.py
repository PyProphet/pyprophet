import abc
import click
import sys
import os
import time
import warnings
import pandas as pd
import numpy as np
import sqlite3

from .pyprophet import PyProphet
from .report import save_report
from .data_handling import is_sqlite_file, check_sqlite_table
from shutil import copyfile

try:
    profile
except NameError:
    def profile(fun):
        return fun


class PyProphetRunner(object):

    __metaclass__ = abc.ABCMeta

    """Base class for workflow of command line tool
    """

    def __init__(self, infile, outfile, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, tric_chromprob, threads, test):
        def read_tsv(infile):
            table = pd.read_csv(infile, "\t")
            return(table)

        def read_osw(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn):
            con = sqlite3.connect(infile)

            if level == "ms2" or level == "ms1ms2":
                if not check_sqlite_table(con, "FEATURE_MS2"):
                    sys.exit("Error: MS2-level feature table not present in file.")
                table = pd.read_sql_query('''
SELECT *,
       RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
FROM FEATURE_MS2
INNER JOIN
  (SELECT RUN_ID,
          ID,
          PRECURSOR_ID,
          EXP_RT
   FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID
INNER JOIN
  (SELECT ID,
          DECOY
   FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
ORDER BY RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
''', con)
            elif level == "ms1":
                if not check_sqlite_table(con, "FEATURE_MS1"):
                    sys.exit("Error: MS1-level feature table not present in file.")
                table = pd.read_sql_query('''
SELECT *,
       RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
FROM FEATURE_MS1
INNER JOIN
  (SELECT RUN_ID,
          ID,
          PRECURSOR_ID,
          EXP_RT
   FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID
INNER JOIN
  (SELECT ID,
          DECOY
   FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
ORDER BY RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
''', con)
            elif level == "transition":
                if not check_sqlite_table(con, "FEATURE_TRANSITION"):
                    sys.exit("Error: Transition-level feature table not present in file.")
                table = pd.read_sql_query('''
SELECT TRANSITION.DECOY AS DECOY,
       FEATURE_TRANSITION.*,
       RUN_ID || '_' || FEATURE_TRANSITION.FEATURE_ID || '_' || PRECURSOR_ID || '_' || TRANSITION_ID AS GROUP_ID
FROM FEATURE_TRANSITION
INNER JOIN
  (SELECT RUN_ID,
          ID,
          PRECURSOR_ID,
          EXP_RT
   FROM FEATURE) AS FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN
  (SELECT ID,
          DECOY
   FROM TRANSITION) AS TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
WHERE RANK <= %s
  AND PEP <= %s
  AND VAR_ISOTOPE_OVERLAP_SCORE <= %s
  AND VAR_LOG_SN_SCORE > %s
  AND PRECURSOR.DECOY == 0
ORDER BY RUN_ID,
         PRECURSOR.ID,
         FEATURE.EXP_RT,
         TRANSITION.ID;
''' % (ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn), con)
            else:
                sys.exit("Error: Unspecified data level selected.")

            # Append MS1 scores to MS2 table if selected
            if level == "ms1ms2":
                if not check_sqlite_table(con, "FEATURE_MS1"):
                    sys.exit("Error: MS1-level feature table not present in file.")
                ms1_table = pd.read_sql_query('SELECT * FROM FEATURE_MS1;', con)

                ms1_scores = [c for c in ms1_table.columns if c.startswith("VAR_")]
                ms1_table = ms1_table[['FEATURE_ID'] + ms1_scores]
                ms1_table.columns = ['FEATURE_ID'] + ["VAR_MS1_" + s.split("VAR_")[1] for s in ms1_scores]

                table = pd.merge(table, ms1_table, how='left', on='FEATURE_ID')

            # Format table
            table.columns = [col.lower() for col in table.columns]

            # Mark main score column
            if ss_main_score.lower() in table.columns:
                table = table.rename(index=str, columns={ss_main_score.lower(): "main_"+ss_main_score.lower()})
            else:
                sys.exit("Error: Main score column not present in data.")

            con.close()
            return(table)

        # Main function
        if is_sqlite_file(infile):
            self.mode = 'osw'
            self.table = read_osw(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn)
        else:
            self.mode = 'tsv'
            self.table = read_tsv(infile)

        self.infile = infile
        self.outfile = outfile
        self.xeval_fraction = xeval_fraction
        self.xeval_num_iter = xeval_num_iter
        self.ss_initial_fdr = ss_initial_fdr
        self.ss_iteration_fdr = ss_iteration_fdr
        self.ss_num_iter = ss_num_iter
        self.ss_main_score = ss_main_score
        self.group_id = group_id
        self.parametric = parametric
        self.pfdr = pfdr
        self.pi0_lambda = pi0_lambda
        self.pi0_method = pi0_method
        self.pi0_smooth_df = pi0_smooth_df
        self.pi0_smooth_log_pi0 = pi0_smooth_log_pi0
        self.lfdr_truncate = lfdr_truncate
        self.lfdr_monotone = lfdr_monotone
        self.lfdr_transformation = lfdr_transformation
        self.lfdr_adj = lfdr_adj
        self.lfdr_eps = lfdr_eps
        self.level = level
        self.tric_chromprob = tric_chromprob
        self.threads = threads
        self.test = test

        self.prefix = os.path.splitext(outfile)[0]

    @abc.abstractmethod
    def run_algo(self):
        pass

    @abc.abstractmethod
    def extra_writes(self):
        pass

    def run(self):

        extra_writes = dict(self.extra_writes())

        self.check_cols = [self.group_id, "run_id", "decoy"]

        start_at = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (result, scorer, weights) = self.run_algo()

        needed = time.time() - start_at

        self.print_summary(result)

        if self.mode == 'tsv':
            self.save_tsv_results(result, extra_writes, scorer.pi0)
            self.save_tsv_weights(weights, extra_writes)
        elif self.mode == 'osw':
            self.save_osw_results(result, extra_writes, scorer.pi0)
            self.save_osw_weights(weights)

        seconds = int(needed)
        msecs = int(1000 * (needed - seconds))

        click.echo("Info: Total time: %d seconds and %d msecs wall time" % (seconds, msecs))

    def print_summary(self, result):
        if result.summary_statistics is not None:
            click.echo("=" * 80)
            click.echo(result.summary_statistics)
            click.echo("=" * 80)

    def save_tsv_results(self, result, extra_writes, pi0):
        summ_stat_path = extra_writes.get("summ_stat_path")
        if summ_stat_path is not None:
            result.summary_statistics.to_csv(summ_stat_path, sep=",", index=False)
            click.echo("Info: %s written." % summ_stat_path)

        full_stat_path = extra_writes.get("full_stat_path")
        if full_stat_path is not None:
            result.final_statistics.to_csv(full_stat_path, sep=",", index=False)
            click.echo("Info: %s written." % full_stat_path)

        output_path = extra_writes.get("output_path")
        if output_path is not None:
            result.scored_tables.to_csv(output_path, sep="\t", index=False)
            click.echo("Info: %s written." % output_path)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(extra_writes.get("report_path"), output_path, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)
            click.echo("Info: %s written." % extra_writes.get("report_path"))

    def save_tsv_weights(self, weights, extra_writes):
        weights['level'] = self.level
        trained_weights_path = extra_writes.get("trained_weights_path")
        if trained_weights_path is not None:
            weights.to_csv(trained_weights_path, sep=",", index=False)
            click.echo("Info: %s written." % trained_weights_path)

    def save_osw_results(self, result, extra_writes, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        con = sqlite3.connect(self.outfile)

        if self.level == "ms2" or self.level == "ms1ms2":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_MS2;')
            con.commit()
            c.fetchall()

            df = result.scored_tables
            if 'h_score' in df.columns:
                df = df[['feature_id','d_score','h_score','h0_score','peak_group_rank','p_value','q_value','pep']]
                df.columns = ['FEATURE_ID','SCORE','HSCORE','H0SCORE','RANK','PVALUE','QVALUE','PEP']
            else:
                df = df[['feature_id','d_score','peak_group_rank','p_value','q_value','pep']]
                df.columns = ['FEATURE_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_MS2"
            df.to_sql(table, con, index=False)
        elif self.level == "ms1":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_MS1;')
            con.commit()
            c.fetchall()

            df = result.scored_tables
            if 'h_score' in df.columns:
                df = df[['feature_id','d_score','h_score','h0_score','peak_group_rank','p_value','q_value','pep']]
                df.columns = ['FEATURE_ID','SCORE','HSCORE','H0SCORE','RANK','PVALUE','QVALUE','PEP']
            else:
                df = df[['feature_id','d_score','peak_group_rank','p_value','q_value','pep']]
                df.columns = ['FEATURE_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_MS1"
            df.to_sql(table, con, index=False)
        elif self.level == "transition":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_TRANSITION;')
            con.commit()
            c.fetchall()

            df = result.scored_tables[['feature_id','transition_id','d_score','peak_group_rank','p_value','q_value','pep']]
            df.columns = ['FEATURE_ID','TRANSITION_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_TRANSITION"
            df.to_sql(table, con, index=False)

        con.close()
        click.echo("Info: %s written." % self.outfile)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(os.path.join(self.prefix + "_" + self.level + "_report.pdf"), self.outfile, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)
            click.echo("Info: %s written." %  os.path.join(self.prefix + "_" + self.level + "_report.pdf"))

    def save_osw_weights(self, weights):
        weights['level'] = self.level
        con = sqlite3.connect(self.outfile)

        c = con.cursor()
        c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_WEIGHTS";')
        if c.fetchone()[0] == 1:
            c.execute('DELETE FROM PYPROPHET_WEIGHTS WHERE LEVEL =="%s"' % self.level)
        c.fetchall()

        weights.to_sql("PYPROPHET_WEIGHTS", con, index=False, if_exists='append')


class PyProphetLearner(PyProphetRunner):

    def run_algo(self):
        (result, scorer, weights) = PyProphet(self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, self.tric_chromprob, self.threads, self.test).learn_and_apply(self.table)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_scored.tsv")
        yield "summ_stat_path", os.path.join(self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_full_stat.csv")
        yield "trained_weights_path", os.path.join(self.prefix + "_weights.csv")
        yield "report_path", os.path.join(self.prefix + "_report.pdf")


class PyProphetWeightApplier(PyProphetRunner):

    def __init__(self, infile, outfile, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, tric_chromprob, threads, test, apply_weights):
        super(PyProphetWeightApplier, self).__init__(infile, outfile, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, tric_chromprob, threads, test)
        if not os.path.exists(apply_weights):
            sys.exit("Error: Weights file %s does not exist." % apply_weights)
        if self.mode == "tsv":
            try:
                self.persisted_weights = pd.read_csv(apply_weights, sep=",")
                if self.level != self.persisted_weights['level'].unique()[0]:
                    sys.exit("Error: Weights file has wrong level.")
            except Exception:
                import traceback
                traceback.print_exc()
                raise
        elif self.mode == "osw":
            try:
                con = sqlite3.connect(apply_weights)

                data = pd.read_sql_query("SELECT * FROM PYPROPHET_WEIGHTS WHERE LEVEL=='%s'" % self.level, con)
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
        (result, scorer, weights) = PyProphet(self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, self.tric_chromprob, self.threads, self.test).apply_weights(self.table, self.persisted_weights)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_scored.tsv")
        yield "summ_stat_path", os.path.join(self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_full_stat.csv")
        yield "report_path", os.path.join(self.prefix + "_report.pdf")
