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
from .config import CONFIG
from .report import save_report
from .data_handling import isSQLite3
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

    def __init__(self, infile, outfile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn):
        def read_tsv(infile):
            table = pd.read_csv(infile, "\t")
            return(table)

        def read_osw(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn):
            con = sqlite3.connect(infile)

            if level == "ms2":
                table = pd.read_sql_query("SELECT *, RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_MS2 INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID, EXP_RT FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID INNER JOIN (SELECT ID, DECOY FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID ORDER BY RUN_ID, PRECURSOR.ID ASC, FEATURE.EXP_RT ASC;", con)
            elif level == "ms1":
                table = pd.read_sql_query("SELECT *, RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_MS1 INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID, EXP_RT FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID INNER JOIN (SELECT ID, DECOY FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID ORDER BY  RUN_ID, PRECURSOR.ID ASC, FEATURE.EXP_RT ASC;", con)
            elif level == "transition":
                table = pd.read_sql_query("SELECT TRANSITION.DECOY AS DECOY, FEATURE_TRANSITION.*, RUN_ID || '_' || FEATURE_TRANSITION.FEATURE_ID || '_' || PRECURSOR_ID || '_' || TRANSITION_ID AS GROUP_ID, VAR_XCORR_SHAPE AS MAIN_VAR_XCORR_SHAPE FROM FEATURE_TRANSITION INNER JOIN (SELECT RUN_ID, ID, PRECURSOR_ID, EXP_RT FROM FEATURE) AS FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID INNER JOIN (SELECT ID, DECOY FROM TRANSITION) AS TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID WHERE RANK <= " + str(ipf_max_peakgroup_rank) + " AND PEP <= " + str(ipf_max_peakgroup_pep) + " AND VAR_ISOTOPE_OVERLAP_SCORE <= " + str(ipf_max_transition_isotope_overlap) + " AND VAR_LOG_SN_SCORE > " + str(ipf_min_transition_sn) + " AND PRECURSOR.DECOY == 0 ORDER BY  RUN_ID, PRECURSOR.ID, FEATURE.EXP_RT, TRANSITION.ID;", con)
            else:
                sys.exit("Error: Unspecified data level selected.")


            table.columns = [col.lower() for col in table.columns]
            con.close()

            return(table)

        if isSQLite3(infile):
            self.mode = 'osw'
            self.table = read_osw(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn)
        else:
            self.mode = 'tsv'
            self.table = read_tsv(infile)

        self.infile = infile
        self.outfile = outfile
        self.prefix = os.path.splitext(outfile)[0]
        self.level = level

    @abc.abstractmethod
    def run_algo(self):
        pass

    @abc.abstractmethod
    def extra_writes(self):
        pass

    def run(self):

        extra_writes = dict(self.extra_writes())

        self.check_cols = [CONFIG.get("group_id"), "run_id", "decoy"]

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

        click.echo("TOTAL TIME: %d seconds and %d msecs wall time" % (seconds, msecs))

    def print_summary(self, result):
        if result.summary_statistics is not None:
            click.echo("=" * 98)
            click.echo(result.summary_statistics)
            click.echo("=" * 98)

    def save_tsv_results(self, result, extra_writes, pi0):
        summ_stat_path = extra_writes.get("summ_stat_path")
        if summ_stat_path is not None:
            result.summary_statistics.to_csv(summ_stat_path, sep=",", index=False)
            click.echo("WRITTEN: " + summ_stat_path)

        full_stat_path = extra_writes.get("full_stat_path")
        if full_stat_path is not None:
            result.final_statistics.to_csv(full_stat_path, sep=",", index=False)
            click.echo("WRITTEN: " + full_stat_path)

        output_path = extra_writes.get("output_path")
        if output_path is not None:
            result.scored_tables.to_csv(output_path, sep="\t", index=False)
            click.echo("WRITTEN: " + output_path)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(extra_writes.get("report_path"), output_path, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)
            click.echo("WRITTEN: " + extra_writes.get("report_path"))

    def save_tsv_weights(self, weights, extra_writes):
        weights['level'] = self.level
        trained_weights_path = extra_writes.get("trained_weights_path")
        if trained_weights_path is not None:
            weights.to_csv(trained_weights_path, sep=",", index=False)
            click.echo("WRITTEN: " + trained_weights_path)

    def save_osw_results(self, result, extra_writes, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        con = sqlite3.connect(self.outfile)

        if self.level == "ms2":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_MS2')
            con.commit()
            c.fetchall()

            df = result.scored_tables[['feature_id','d_score','peak_group_rank','p_value','q_value','pep']]
            df.columns = ['FEATURE_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_MS2"
            df.to_sql(table, con, index=False)
        elif self.level == "ms1":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_MS1')
            con.commit()
            c.fetchall()

            df = result.scored_tables[['feature_id','d_score','peak_group_rank','p_value','q_value','pep']]
            df.columns = ['FEATURE_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_MS1"
            df.to_sql(table, con, index=False)
        elif self.level == "transition":
            c = con.cursor()
            c.execute('DROP TABLE IF EXISTS SCORE_TRANSITION')
            con.commit()
            c.fetchall()

            df = result.scored_tables[['feature_id','transition_id','d_score','peak_group_rank','p_value','q_value','pep']]
            df.columns = ['FEATURE_ID','TRANSITION_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
            table = "SCORE_TRANSITION"
            df.to_sql(table, con, index=False)

        con.close()
        click.echo("WRITTEN: " + self.outfile)

        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(os.path.join(self.prefix + "_" + self.level + "_report.pdf"), self.outfile, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0)
            click.echo("WRITTEN: " + os.path.join(self.prefix + "_" + self.level + "_report.pdf"))

    def save_osw_weights(self, weights):
        weights['level'] = self.level
        con = sqlite3.connect(self.outfile)

        c = con.cursor()
        c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_WEIGHTS"')
        if c.fetchone()[0] == 1:
            c.execute('DELETE FROM PYPROPHET_WEIGHTS WHERE LEVEL =="' + self.level + '"')
        c.fetchall()

        weights.to_sql("PYPROPHET_WEIGHTS", con, index=False, if_exists='append')


class PyProphetLearner(PyProphetRunner):

    def run_algo(self):
        (result, scorer, weights) = PyProphet().learn_and_apply(self.table)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_scored.tsv")
        yield "summ_stat_path", os.path.join(self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_full_stat.csv")
        yield "trained_weights_path", os.path.join(self.prefix + "_weights.csv")
        yield "report_path", os.path.join(self.prefix + "_report.pdf")


class PyProphetWeightApplier(PyProphetRunner):

    def __init__(self, infile, outfile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, apply_weights):
        super(PyProphetWeightApplier, self).__init__(infile, outfile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn)
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

                data = pd.read_sql_query("SELECT * FROM PYPROPHET_WEIGHTS WHERE LEVEL=='" + self.level + "'", con)
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
        (result, scorer, weights) = PyProphet().apply_weights(self.table, self.persisted_weights)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_scored.tsv")
        yield "summ_stat_path", os.path.join(self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_full_stat.csv")
        yield "report_path", os.path.join(self.prefix + "_report.pdf")
