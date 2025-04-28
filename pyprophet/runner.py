import abc
import click
import sys
import os
import time
import warnings
import pandas as pd
import numpy as np
import polars as pl
import sqlite3
import pickle

from .pyprophet import PyProphet
from .glyco.scoring import partial_score, combined_score
from .glyco.stats import ErrorStatisticsCalculator
from.glyco.report import save_report as save_report_glyco
from .report import save_report
from .data_handling import is_sqlite_file, check_sqlite_table, is_parquet_file, get_parquet_column_names
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

    def __init__(self, infile, outfile, classifier, xgb_hyperparams, xgb_params, xgb_params_space, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, add_alignment_features, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, glyco, density_estimator, grid_size, tric_chromprob, threads, test, ss_score_filter, color_palette, main_score_selection_report):
        def read_tsv(infile):
            table = pd.read_csv(infile, sep="\t")
            return(table)

        def read_osw(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn):
            con = sqlite3.connect(infile)

            if level == "ms2" or level == "ms1ms2":
                if not check_sqlite_table(con, "FEATURE_MS2"):
                    raise click.ClickException("MS2-level feature table not present in file.")

                con.executescript('''
                                    CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);
                                    ''')

                if not glyco:
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
                                                        CHARGE AS PRECURSOR_CHARGE,
                                                        DECOY
                                                FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                                                INNER JOIN
                                                (SELECT PRECURSOR_ID AS ID,
                                                        COUNT(*) AS TRANSITION_COUNT
                                                FROM TRANSITION_PRECURSOR_MAPPING
                                                INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
                                                WHERE DETECTING==1
                                                GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
                                                ORDER BY RUN_ID,
                                                        PRECURSOR.ID ASC,
                                                        FEATURE.EXP_RT ASC;
                                                ''', con)
                else:
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
                                                        CHARGE AS PRECURSOR_CHARGE,
                                                        DECOY
                                                FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                                                INNER JOIN
                                                (SELECT PRECURSOR_ID AS ID,
                                                        DECOY_PEPTIDE,
                                                        DECOY_GLYCAN
                                                FROM PRECURSOR_GLYCOPEPTIDE_MAPPING
                                                INNER JOIN GLYCOPEPTIDE
                                                ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID == GLYCOPEPTIDE.ID) AS DECOY
                                                ON FEATURE.PRECURSOR_ID = DECOY.ID
                                                INNER JOIN
                                                (SELECT PRECURSOR_ID AS ID,
                                                        COUNT(*) AS TRANSITION_COUNT
                                                FROM TRANSITION_PRECURSOR_MAPPING
                                                INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
                                                WHERE DETECTING==1
                                                GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
                                                ORDER BY RUN_ID,
                                                        PRECURSOR.ID ASC,
                                                        FEATURE.EXP_RT ASC;
                                                ''', con)
            elif level == "ms1":
                if not check_sqlite_table(con, "FEATURE_MS1"):
                    raise click.ClickException("MS1-level feature table not present in file.")

                con.executescript('''
                                    CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);
                                    ''')
                if not glyco:
                    table = pd.read_sql_query(
                        """
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
                                CHARGE AS PRECURSOR_CHARGE,
                                DECOY
                        FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                        ORDER BY RUN_ID,
                                PRECURSOR.ID ASC,
                                FEATURE.EXP_RT ASC;
                        """,
                        con,
                    )
                else:
                    if not check_sqlite_table(con, "SCORE_MS2"):
                        raise click.ClickException("MS1-level scoring for glycoform inference requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first.")
                    if not check_sqlite_table(con, "FEATURE_MS1"):
                        raise click.ClickException("MS1-level feature table not present in file.")

                    table = pd.read_sql_query(
                        '''
                        SELECT DECOY.*,
                            FEATURE_MS1.*, 
                            FEATURE.*,
                            PRECURSOR.*,
                            RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
                        FROM FEATURE_MS1
                        INNER JOIN
                        (SELECT RUN_ID,
                            ID,
                            PRECURSOR_ID,
                            EXP_RT
                        FROM FEATURE) AS FEATURE ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
                        
                        INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID

                        INNER JOIN
                        (SELECT ID,
                            CHARGE AS PRECURSOR_CHARGE,
                            DECOY
                        FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                        
                        INNER JOIN
                            (SELECT PRECURSOR_ID AS ID,
                                    DECOY_PEPTIDE,
                                    DECOY_GLYCAN
                            FROM PRECURSOR_GLYCOPEPTIDE_MAPPING
                            INNER JOIN GLYCOPEPTIDE 
                            ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID == GLYCOPEPTIDE.ID) AS DECOY 
                            ON FEATURE.PRECURSOR_ID = DECOY.ID

                        WHERE RANK <= %s
                        ORDER BY RUN_ID,
                            PRECURSOR.ID ASC,
                            FEATURE.EXP_RT ASC;
                        ''' % (ipf_max_peakgroup_rank),
                            con,
                        )
            elif level == "transition":
                if not check_sqlite_table(con, "SCORE_MS2"):
                    raise click.ClickException("Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first.")
                if not check_sqlite_table(con, "FEATURE_TRANSITION"):
                    raise click.ClickException("Transition-level feature table not present in file.")

                con.executescript('''
                                    CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
                                    CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
                                    CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);
                                    CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);
                                    ''')

                table = pd.read_sql_query('''
                                            SELECT TRANSITION.DECOY AS DECOY,
                                                FEATURE_TRANSITION.*,
                                                PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
                                                TRANSITION.PRODUCT_CHARGE AS PRODUCT_CHARGE,
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
                                                    CHARGE AS PRODUCT_CHARGE,
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
            elif level == "alignment":
                if not check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
                    raise click.ClickException("MS2-level feature alignemnt table not present in file.")

                con.executescript('''
                                CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                                CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                                CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                                CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);
                                ''')

                table = pd.read_sql_query('''
                            SELECT
                                ALIGNED_FEATURE_ID AS FEATURE_ID,
                                XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                                XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE_TO_REFERENCE, 
                                MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE, 
                                XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,  
                                XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE, 
                                MI_TO_ALL AS VAR_MI_TO_ALL, 
                                RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION, 
                                PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO,
                                LABEL AS DECOY,
                                ALIGNED_FILENAME || '_' || FEATURE.PRECURSOR_ID AS GROUP_ID
                            FROM FEATURE_MS2_ALIGNMENT
                            LEFT JOIN
                            (SELECT RUN_ID,
                                    ID,
                                    PRECURSOR_ID,
                                    EXP_RT
                            FROM FEATURE) AS FEATURE ON REFERENCE_FEATURE_ID = FEATURE.ID
                            ''', con)
                # Map DECOY to 1 and -1 to 0 and 1
                table['DECOY'] = table['DECOY'].map({1: 0, -1: 1})
            else:
                raise click.ClickException("Unspecified data level selected.")

            # Append MS1 scores to MS2 table if selected
            if level == "ms1ms2":
                if not check_sqlite_table(con, "FEATURE_MS1"):
                    raise click.ClickException("MS1-level feature table not present in file.")
                ms1_table = pd.read_sql_query('SELECT * FROM FEATURE_MS1;', con)

                ms1_scores = [c for c in ms1_table.columns if c.startswith("VAR_")]
                ms1_table = ms1_table[['FEATURE_ID'] + ms1_scores]
                ms1_table.columns = ['FEATURE_ID'] + ["VAR_MS1_" + s.split("VAR_")[1] for s in ms1_scores]

                table = pd.merge(table, ms1_table, how='left', on='FEATURE_ID')

            if add_alignment_features:
                # Append MS2 alignment scores to MS2 table if selected
                if level == "ms2" or level == "ms1ms2":
                    if not check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
                        raise click.ClickException("MS2-level feature alignment table not present in file.")

                    if not check_sqlite_table(con, "SCORE_ALIGNMENT"):
                        raise click.ClickException("To add MS2-level alignment features, alignment-level first needs to be performed. Please run 'pyprophet score --level=alignment' on this file first.")

                    alignment_table = pd.read_sql_query(
                        """SELECT 
                                ALIGNED_FEATURE_ID AS FEATURE_ID,
                                PRECURSOR_ID,
                                XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                                XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE_TO_REFERENCE, 
                                MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE, 
                                XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,  
                                XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE_TO_ALL, 
                                MI_TO_ALL AS VAR_MI_TO_ALL, 
                                RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION, 
                                PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO 
                            FROM 
                            (SELECT DISTINCT * FROM FEATURE_MS2_ALIGNMENT) AS FEATURE_MS2_ALIGNMENT
                            INNER JOIN 
                            (SELECT DISTINCT *, MIN(QVALUE) FROM SCORE_ALIGNMENT GROUP BY FEATURE_ID) AS SCORE_ALIGNMENT 
                            ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_MS2_ALIGNMENT.ALIGNED_FEATURE_ID
                            WHERE LABEL = 1""",
                        con,
                    )

                    if 'PRECURSOR_ID' in table.columns:
                        table = pd.merge(table, alignment_table, how='left', on=['FEATURE_ID', 'PRECURSOR_ID'])
                    else:
                        table = pd.merge(table, alignment_table, how='left', on='FEATURE_ID')  

                # Append TRANSITION alignment scores to TRANSITION table if selected
                if level == "transition":
                    if not check_sqlite_table(con, "FEATURE_TRANSITION_ALIGNMENT"):
                        raise click.ClickException("Transition-level feature alignment table not present in file.")

                    if not check_sqlite_table(con, "SCORE_ALIGNMENT"):
                        raise click.ClickException("To add Transition-level alignment features, alignment-level first needs to be performed. Please run 'pyprophet score --level=alignment' on this file first.")

                    alignment_table = pd.read_sql_query(
                        """SELECT 
                                FEATURE_TRANSITION_ALIGNMENT.FEATURE_ID,
                                TRANSITION_ID,
                                XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                                XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE_TO_REFERENCE, 
                                MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE, 
                                XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,  
                                XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE_TO_ALL, 
                                MI_TO_ALL AS VAR_MI_TO_ALL, 
                                RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION, 
                                PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO
                            FROM FEATURE_TRANSITION_ALIGNMENT
                            INNER JOIN 
                            (SELECT DISTINCT *, MIN(QVALUE) FROM SCORE_ALIGNMENT GROUP BY FEATURE_ID) AS SCORE_ALIGNMENT 
                            ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_TRANSITION_ALIGNMENT.FEATURE_ID 
                        """,
                        con
                    )
                    table = pd.merge(table, alignment_table, how='left', on=['FEATURE_ID', 'TRANSITION_ID'])

                cols = ['VAR_XCORR_COELUTION_TO_REFERENCE', 'VAR_XCORR_SHAPE_TO_REFERENCE', 'VAR_MI_TO_REFERENCE', 'VAR_XCORR_COELUTION_TO_ALL',  'VAR_XCORR_SHAPE_TO_ALL', 'VAR_MI_TO_ALL', 'VAR_RETENTION_TIME_DEVIATION', 'VAR_PEAK_INTENSITY_RATIO']    
                # Fill in missing values for cols2 with -1
                table[cols] = table[cols].fillna(-1)       

            # Format table
            table.columns = [col.lower() for col in table.columns]

            # Mark main score column
            if ss_main_score.lower() in table.columns:
                table = table.rename(index=str, columns={ss_main_score.lower(): "main_"+ss_main_score.lower()})
            elif ss_main_score.lower() == "swath_pretrained":
                # Add a pretrained main score corresponding to the original implementation in OpenSWATH
                # This is optimized for 32-windows SCIEX TripleTOF 5600 data
                table['main_var_pretrained'] = -( -0.19011762 * table['var_library_corr']
                                                +  2.47298914 * table['var_library_rmsd']
                                                +  5.63906731 * table['var_norm_rt_score']
                                                + -0.62640133 * table['var_isotope_correlation_score']
                                                +  0.36006925 * table['var_isotope_overlap_score']
                                                +  0.08814003 * table['var_massdev_score']
                                                +  0.13978311 * table['var_xcorr_coelution']
                                                + -1.16475032 * table['var_xcorr_shape']
                                                + -0.19267813 * table['var_yseries_score']
                                                + -0.61712054 * table['var_log_sn_score'])
            else:
                raise click.ClickException(f"Main score ({ss_main_score.lower()}) column not present in data. Current columns: {table.columns}")

            # Enable transition count & precursor / product charge scores for XGBoost-based classifier
            if classifier == 'XGBoost' and level!='alignment':
                click.echo("Info: Enable number of transitions & precursor / product charge scores for XGBoost-based classifier")
                table = table.rename(index=str, columns={'precursor_charge': 'var_precursor_charge', 'product_charge': 'var_product_charge', 'transition_count': 'var_transition_count'})

            con.close()
            return(table)

        def read_parquet(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn):
            import pyarrow.parquet as pq

            all_column_names = get_parquet_column_names(infile)

            if level == "ms2" or level == "ms1ms2":
                if not any([col.startswith("FEATURE_MS2_") for col in all_column_names]):
                    raise click.ClickException("MS2-level feature columns are not present in parquet file.")

                cols = [
                    'RUN_ID',
                    'PRECURSOR_ID',
                    'PRECURSOR_CHARGE',
                    'FEATURE_ID',
                    'EXP_RT',
                    'PRECURSOR_DECOY',
                    'TRANSITION_DETECTING'
                ]
                # Filter columes names for FEATURE_MS2_
                feature_ms2_cols = [col for col in all_column_names if col.startswith('FEATURE_MS2_')]
                # Read the parquet file with selected columns
                table = pl.read_parquet(infile, columns=cols + feature_ms2_cols)

                # Drop rows with nulls in key columns
                table = (
                    table
                    .drop_nulls(subset=['RUN_ID', 'FEATURE_ID'] + feature_ms2_cols)
                )

                # Rename columns
                table = table.rename({
                    'PRECURSOR_DECOY': 'DECOY',
                    **{col: col.replace('FEATURE_MS2_', '') 
                    for col in table.columns 
                    if col.startswith('FEATURE_MS2_')}
                })

                # Create transition count table
                table_transition = (
                    table.select(['RUN_ID', 'PRECURSOR_ID', 'FEATURE_ID', 'EXP_RT', 'DECOY', 'TRANSITION_DETECTING'])
                    .explode('TRANSITION_DETECTING')
                    .filter(pl.col('TRANSITION_DETECTING') == 1)
                    .group_by(['RUN_ID', 'PRECURSOR_ID', 'FEATURE_ID', 'EXP_RT', 'DECOY'])
                    .agg(pl.count().alias('TRANSITION_COUNT'))
                )

                # Join with main table
                table = table.join(
                    table_transition,
                    on=['RUN_ID', 'PRECURSOR_ID', 'FEATURE_ID', 'EXP_RT', 'DECOY'],
                    how='left'
                ).drop('TRANSITION_DETECTING')

                # Create GROUP_ID column
                table = table.with_columns(
                    (pl.col('RUN_ID').cast(pl.Utf8) + '_' + pl.col('PRECURSOR_ID').cast(pl.Utf8))
                    .alias('GROUP_ID')
                )
                table = table.to_pandas()
            elif level == "ms1":
                if not any([col.startswith("FEATURE_MS1_") for col in all_column_names]):
                    raise click.ClickException("MS1-level feature columns are not present in parquet file.")

                cols = [
                    'RUN_ID',
                    'PRECURSOR_ID',
                    'PRECURSOR_CHARGE',
                    'FEATURE_ID',
                    'EXP_RT',
                    'PRECURSOR_DECOY'
                ]
                # Filter columes names for FEATURE_MS1_
                feature_ms1_cols = [col for col in all_column_names if col.startswith('FEATURE_MS1_')]
                # Read the parquet file with selected columns
                table = pl.read_parquet(infile, columns=cols + feature_ms1_cols)

                # Drop rows with nulls in key columns
                table = table.drop_nulls(subset=['RUN_ID', 'FEATURE_ID'] + feature_ms1_cols)

                # Rename columns - remove 'FEATURE_MS1_' prefix and rename PRECURSOR_DECOY
                table = table.rename({
                    'PRECURSOR_DECOY': 'DECOY',
                    **{col: col.replace('FEATURE_MS1_', '') 
                    for col in table.columns 
                    if col.startswith('FEATURE_MS1_')}
                })

                # Create GROUP_ID column by concatenating RUN_ID and PRECURSOR_ID
                table = table.with_columns(
                    (pl.col('RUN_ID').cast(pl.Utf8) + "_" + pl.col('PRECURSOR_ID').cast(pl.Utf8))
                    .alias('GROUP_ID')
                )
                table = table.to_pandas()
            elif level == "transition":
                if not any([col.startswith("SCORE_MS2_") for col in all_column_names]):
                    raise click.ClickException("Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first.")
                if not any([col.startswith("FEATURE_TRANSITION_") for col in all_column_names]):
                    raise click.ClickException("Transition-level feature columns are not present in parquet file.")

                cols = [
                    'RUN_ID',
                    'FEATURE_ID',
                    'PRECURSOR_ID',
                    'TRANSITION_ID',
                    'PRECURSOR_DECOY',
                    'TRANSITION_DECOY',
                    'PRECURSOR_CHARGE',
                    'TRANSITION_CHARGE',
                ]

                # Read only needed columns from parquet
                feature_transition_ms2_score_cols = [col for col in all_column_names if col.startswith('SCORE_MS2_') or col.startswith('FEATURE_TRANSITION_')]

                table = pl.read_parquet(
                    infile,
                    columns=cols + feature_transition_ms2_score_cols
                )

                # Explode list columns
                exploding_cols = [
                    col for col in table.columns 
                    if col.startswith('TRANSITION_') or col.startswith('FEATURE_TRANSITION_')
                ]

                table = table.explode(exploding_cols)

                # Apply filters
                table = table.filter(
                    (pl.col('PRECURSOR_DECOY') == 0) &
                    (pl.col('SCORE_MS2_RANK') <= ipf_max_peakgroup_rank) &
                    (pl.col('SCORE_MS2_PEP') <= ipf_max_peakgroup_pep) &
                    (pl.col('FEATURE_TRANSITION_VAR_ISOTOPE_OVERLAP_SCORE') <= ipf_max_transition_isotope_overlap) &
                    (pl.col('FEATURE_TRANSITION_VAR_LOG_SN_SCORE') > ipf_min_transition_sn)
                )

                # Drop SCORE_MS2_ columns
                score_cols = [col for col in table.columns if col.startswith('SCORE_MS2_')]
                table = table.drop(score_cols)

                # Rename columns
                table = table.rename({
                    'TRANSITION_DECOY': 'DECOY',
                    **{col: col.replace('FEATURE_TRANSITION_', '') 
                    for col in table.columns 
                    if col.startswith('FEATURE_TRANSITION_')}
                })

                # Create GROUP_ID column
                table = table.with_columns(
                    (pl.col('RUN_ID').cast(pl.Utf8) + '_' + 
                    pl.col('FEATURE_ID').cast(pl.Utf8) + '_' + 
                    pl.col('PRECURSOR_ID').cast(pl.Utf8) + '_' + 
                    pl.col('TRANSITION_ID').cast(pl.Utf8)).alias('GROUP_ID')
                )
                table = table.to_pandas()
            else:
                raise click.ClickException("Unspecified data level selected.")

            # Append MS1 scores to MS2 table if selected
            if level == "ms1ms2":
                if not any([col.startswith("FEATURE_MS1_") for col in all_column_names]):
                    raise click.ClickException("MS1-level feature columns are not present in parquet file.")

                # Filter columes names for FEATURE_MS2_
                feature_ms1_cols = [col for col in all_column_names if col.startswith('FEATURE_MS1_VAR')]
                # Read the parquet file
                ms1_table = pd.read_parquet(infile, columns=['FEATURE_ID'] + feature_ms1_cols)
                # Rename columns with 'FEATURE_MS1_VAR_' to 'VAR_MS1_'
                ms1_table.columns = [col.replace('FEATURE_MS1_VAR_', 'VAR_MS1_') for col in ms1_table.columns]

                table = pd.merge(table, ms1_table, how='left', on='FEATURE_ID')

            # Format table
            table.columns = [col.lower() for col in table.columns]

            # Mark main score column
            if ss_main_score.lower() in table.columns:
                table = table.rename(index=str, columns={ss_main_score.lower(): "main_"+ss_main_score.lower()})
            elif ss_main_score.lower() == "swath_pretrained":
                # Add a pretrained main score corresponding to the original implementation in OpenSWATH
                # This is optimized for 32-windows SCIEX TripleTOF 5600 data
                table['main_var_pretrained'] = -( -0.19011762 * table['var_library_corr']
                                                +  2.47298914 * table['var_library_rmsd']
                                                +  5.63906731 * table['var_norm_rt_score']
                                                + -0.62640133 * table['var_isotope_correlation_score']
                                                +  0.36006925 * table['var_isotope_overlap_score']
                                                +  0.08814003 * table['var_massdev_score']
                                                +  0.13978311 * table['var_xcorr_coelution']
                                                + -1.16475032 * table['var_xcorr_shape']
                                                + -0.19267813 * table['var_yseries_score']
                                                + -0.61712054 * table['var_log_sn_score'])
            else:
                raise click.ClickException(f"Main score ({ss_main_score.lower()}) column not present in data. Current columns: {table.columns}")

            # Enable transition count & precursor / product charge scores for XGBoost-based classifier
            if classifier == 'XGBoost' and level!='alignment':
                click.echo("Info: Enable number of transitions & precursor / product charge scores for XGBoost-based classifier")
                table = table.rename(index=str, columns={'precursor_charge': 'var_precursor_charge', 'product_charge': 'var_product_charge', 'transition_count': 'var_transition_count'})

            return(table)

        # Check for auto main score selection
        if ss_main_score=="auto":
            # Set starting default main score
            ss_main_score = "var_xcorr_shape"
            use_dynamic_main_score = True
        else:
            use_dynamic_main_score = False

        # Main function
        if is_sqlite_file(infile):
            self.mode = 'osw'
            self.table = read_osw(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn)
        elif is_parquet_file(infile):
            self.mode = 'parquet'
            self.table = read_parquet(infile, level, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn)
        else:
            self.mode = 'tsv'
            self.table = read_tsv(infile)    

        self.infile = infile
        self.outfile = outfile
        self.classifier = classifier
        self.xgb_hyperparams = xgb_hyperparams
        self.xgb_params = xgb_params
        self.xgb_params_space = xgb_params_space
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
        self.glyco = glyco
        self.density_estimator = density_estimator
        self.grid_size = grid_size
        self.tric_chromprob = tric_chromprob
        self.threads = threads
        self.test = test
        self.ss_score_filter = ss_score_filter
        self.color_palette = color_palette
        self.main_score_selection_report = main_score_selection_report
        self.ss_use_dynamic_main_score = use_dynamic_main_score

        self.prefix = os.path.splitext(outfile)[0]

    @abc.abstractmethod
    def run_algo(self, part=None):
        pass

    @abc.abstractmethod
    def extra_writes(self):
        pass

    def run(self):

        extra_writes = dict(self.extra_writes())

        self.check_cols = [self.group_id, "run_id", "decoy"]

        if self.glyco and self.level in ["ms2", "ms1ms2"]:
            start_at = time.time()

            start_at_peptide = time.time()
            click.echo("*" * 30 + "  Glycoform Scoring  " + "*" * 30)
            click.echo("-" * 80)
            click.echo("Info: Scoring peptide part")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result_peptide, scorer_peptide, weights_peptide) = \
                    self.run_algo(part='peptide')
            end_at_peptide = time.time() - start_at_peptide
            seconds = int(end_at_peptide)
            msecs = int(1000 * (end_at_peptide - seconds))
            click.echo("Info: peptide part scored: %d seconds and %d msecs" % (seconds, msecs))

            start_at_glycan = time.time()
            click.echo("-" * 80)
            click.echo("Info: Scoring glycan part")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result_glycan, scorer_glycan, weights_glycan) = \
                    partial_score(self, part='glycan')

            end_at_glycan = time.time() - start_at_glycan
            seconds = int(end_at_glycan)
            msecs = int(1000 * (end_at_glycan - seconds))
            click.echo("Info: glycan part scored: %d seconds and %d msecs" % (seconds, msecs))

            start_at_combined = time.time()
            click.echo("-" * 80)
            click.echo("Info: Calculating combined scores")
            (result_combined, weights_combined) = \
                combined_score(self.group_id, result_peptide, result_glycan)

            if isinstance(weights_combined, pd.DataFrame):
                click.echo(weights_combined)

            end_at_combined= time.time() - start_at_combined
            seconds = int(end_at_combined)
            msecs = int(1000 * (end_at_combined - seconds))
            click.echo("Info: combined scores calculated: %d seconds and %d msecs" % (seconds, msecs))

            start_at_stats = time.time()
            click.echo("-" * 80)
            click.echo("Info: Calculating error statistics")
            error_stat = ErrorStatisticsCalculator(
                result_combined,
                density_estimator=self.density_estimator,
                grid_size=self.grid_size,
                parametric=self.parametric,
                pfdr=self.pfdr,
                pi0_lambda=self.pi0_lambda, pi0_method=self.pi0_method,
                pi0_smooth_df=self.pi0_smooth_df,
                pi0_smooth_log_pi0=self.pi0_smooth_log_pi0,
                lfdr_truncate=self.lfdr_truncate,
                lfdr_monotone=self.lfdr_monotone,
                lfdr_transformation=self.lfdr_transformation,
                lfdr_adj=self.lfdr_adj, lfdr_eps=self.lfdr_eps,
                tric_chromprob=self.tric_chromprob,
            )
            result, pi0 = error_stat.error_statistics()

            end_at_stats = time.time() - start_at_stats
            seconds = int(end_at_stats)
            msecs = int(1000 * (end_at_stats - seconds))
            click.echo("Info: error statistics finished: %d seconds and %d msecs" % (seconds, msecs))

            if all((
            isinstance(w, pd.DataFrame)
            for w in [weights_peptide, weights_glycan, weights_combined]
            )):
                weights = pd.concat((
                    weights_peptide.assign(part='peptide'),
                    weights_glycan.assign(part='glycan'),
                    weights_combined.assign(part='combined')
                ), ignore_index=True)
            else:
                weights = {
                'peptide': weights_peptide,
                'glycan': weights_glycan,
                'combined': weights_combined
                }

            needed = time.time() - start_at
        else:
            start_at = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (result, scorer, weights) = self.run_algo()
            needed = time.time() - start_at

        self.print_summary(result)

        if self.mode == 'tsv':
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                self.save_tsv_results(result, extra_writes, pi0)
            else:
                self.save_tsv_results(result, extra_writes, scorer.pi0) 
            if self.classifier == 'LDA':
                self.save_tsv_weights(weights, extra_writes)
            elif self.classifier == 'XGBoost':
                self.save_bin_weights(weights, extra_writes)

        elif self.mode == 'osw':
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                self.save_osw_results(result, extra_writes, pi0)
            else:
                self.save_osw_results(result, extra_writes, scorer.pi0)
            self.save_osw_weights(weights)

        elif self.mode == 'parquet':
            self.save_parquet_results(result, extra_writes, scorer.pi0)
            if self.classifier == 'LDA':
                self.save_tsv_weights(weights, extra_writes)
            elif self.classifier == 'XGBoost':
                self.save_bin_weights(weights, extra_writes)

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

            save_report(extra_writes.get("report_path"), output_path, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0, self.color_palette)
            click.echo("Info: %s written." % extra_writes.get("report_path"))

    def save_tsv_weights(self, weights, extra_writes):
        weights['level'] = self.level
        trained_weights_path = extra_writes.get("trained_weights_path")
        if trained_weights_path is not None:
            weights.to_csv(trained_weights_path, sep=",", index=False, mode='a')
            click.echo("Info: %s written." % trained_weights_path)

    def save_osw_results(self, result, extra_writes, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        con = sqlite3.connect(self.outfile)

        if self.glyco and self.level in ["ms2", "ms1ms2"]:
            if self.level == "ms2" or self.level == "ms1ms2":
                c = con.cursor()
                c.execute('DROP TABLE IF EXISTS SCORE_MS2;')
                c.execute('DROP TABLE IF EXISTS SCORE_MS2_PART_PEPTIDE;')
                c.execute('DROP TABLE IF EXISTS SCORE_MS2_PART_GLYCAN;')
                con.commit()
                c.fetchall()

                table = "SCORE_MS2"

            elif self.level == "ms1":
                c = con.cursor()
                c.execute('DROP TABLE IF EXISTS SCORE_MS1;')
                c.execute('DROP TABLE IF EXISTS SCORE_MS1_PART_PEPTIDE;')
                c.execute('DROP TABLE IF EXISTS SCORE_MS1_PART_GLYCAN;')
                con.commit()
                c.fetchall()

                table = "SCORE_MS1"

            df = result.scored_tables
            if 'h_score' in df.columns:
                df = df[['feature_id','d_score_combined','h_score','h0_score','peak_group_rank','q_value','pep']]
                df.columns = ['FEATURE_ID','SCORE','HSCORE','H0SCORE','RANK','QVALUE','PEP']
            else:
                df = df[['feature_id','d_score_combined','peak_group_rank','q_value','pep']]
                df.columns = ['FEATURE_ID','SCORE','RANK','QVALUE','PEP']
            df.to_sql(table, con, index=False)

            for part in ['peptide', 'glycan']:
                df = result.scored_tables
                df = df[['feature_id','d_score_' + part,'pep_' + part]]
                df.columns = ['FEATURE_ID','SCORE','PEP']
                df.to_sql(table + "_PART_" + part.upper(), con, index=False)
        else:
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
            elif self.level == "alignment":
                c = con.cursor()
                c.execute('DROP TABLE IF EXISTS SCORE_ALIGNMENT;')
                con.commit()
                c.fetchall()

                df = result.scored_tables[['feature_id','d_score','peak_group_rank','p_value','q_value','pep']]
                df.columns = ['FEATURE_ID','SCORE','RANK','PVALUE','QVALUE','PEP']
                table = "SCORE_ALIGNMENT"
                df.to_sql(table, con, index=False)

        con.close()
        click.echo("Info: %s written." % self.outfile)

        if result.final_statistics is not None:
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                save_report_glyco(
                    os.path.join(self.prefix + "_" + self.level + "_report.pdf"),
                self.outfile + ': ' + self.level + '-level scoring',
                result.scored_tables,
                result.final_statistics,
                pi0
                )
            else:
                cutoffs = result.final_statistics["cutoff"].values
                svalues = result.final_statistics["svalue"].values
                qvalues = result.final_statistics["qvalue"].values

                pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
                top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
                top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

                save_report(os.path.join(self.prefix + "_" + self.level + "_report.pdf"), self.outfile, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0, self.color_palette)
            click.echo("Info: %s written." %  os.path.join(self.prefix + "_" + self.level + "_report.pdf"))

    def save_osw_weights(self, weights):
        if self.classifier == "LDA":
            weights['level'] = self.level
            con = sqlite3.connect(self.outfile)

            c = con.cursor()
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="GLYCOPEPTIDEPROPHET_WEIGHTS";')
                if c.fetchone()[0] == 1:
                    c.execute('DELETE FROM GLYCOPEPTIDEPROPHET_WEIGHTS WHERE LEVEL =="%s"' % self.level)
            else:
                c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_WEIGHTS";')
                if c.fetchone()[0] == 1:
                    c.execute('DELETE FROM PYPROPHET_WEIGHTS WHERE LEVEL =="%s"' % self.level)
            c.close()

            # print(weights)

            weights.to_sql("PYPROPHET_WEIGHTS", con, index=False, if_exists='append')

        elif self.classifier == "XGBoost":
            con = sqlite3.connect(self.outfile)

            c = con.cursor()
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="GLYCOPEPTIDEPROPHET_XGB";')
                if c.fetchone()[0] == 1:
                    c.execute('DELETE FROM GLYCOPEPTIDEPROPHET_XGB WHERE LEVEL =="%s"' % self.level)
                else:
                    c.execute('CREATE TABLE GLYCOPEPTIDEPROPHET_XGB (level TEXT, xgb BLOB)')

                c.execute('INSERT INTO GLYCOPEPTIDEPROPHET_XGB VALUES(?, ?)', [self.level, pickle.dumps(weights)])
            else:
                c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_XGB";')
                if c.fetchone()[0] == 1:
                    c.execute('DELETE FROM PYPROPHET_XGB WHERE LEVEL =="%s"' % self.level)
                else:
                    c.execute('CREATE TABLE PYPROPHET_XGB (level TEXT, xgb BLOB)')

                c.execute('INSERT INTO PYPROPHET_XGB VALUES(?, ?)', [self.level, pickle.dumps(weights)])
            con.commit()
            c.close()

    def save_bin_weights(self, weights, extra_writes):
        trained_weights_path = extra_writes.get("trained_model_path_" + self.level)
        if trained_weights_path is not None:
            with open(trained_weights_path, 'wb') as file:
                self.persisted_weights = pickle.dump(weights, file)
            click.echo("Info: %s written." % trained_weights_path)

    def save_parquet_results(self, result, extra_writes, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        if self.level == "ms2" or self.level == "ms1ms2":
            # Read the parquet file
            init_df = pl.read_parquet(self.outfile)

            # Check and drop SCORE_MS2_ columns if they exist
            score_ms2_cols = [col for col in init_df.columns if col.startswith("SCORE_MS2_")]
            if score_ms2_cols:
                click.echo(click.style("Warn: Dropping existing SCORE_MS2_ columns from the output file.", fg='yellow'))
                init_df = init_df.drop(score_ms2_cols)

            # Process the result scores
            df = pl.from_pandas(result.scored_tables)
            if 'h_score' in df.columns:
                df = df.select([
                    'feature_id', 'd_score', 'h_score', 'h0_score',
                    'peak_group_rank', 'p_value', 'q_value', 'pep'
                ])
                df.columns = ['FEATURE_ID', 'SCORE', 'HSCORE', 'H0SCORE', 
                            'RANK', 'PVALUE', 'QVALUE', 'PEP']
            else:
                df = df.select([
                    'feature_id', 'd_score', 'peak_group_rank',
                    'p_value', 'q_value', 'pep'
                ])
                df.columns = ['FEATURE_ID', 'SCORE', 'RANK', 'PVALUE', 'QVALUE', 'PEP']

            # Add SCORE_MS2_ prefix to columns except FEATURE_ID
            new_columns = ['FEATURE_ID'] + [f'SCORE_MS2_{col}' for col in df.columns[1:]]
            df = df.rename(dict(zip(df.columns, new_columns)))

            # Merge with initial dataframe
            df = init_df.join(df, on='FEATURE_ID', how='left', coalesce=True)

            # Write to parquet
            df.write_parquet(
                self.outfile,
                compression="zstd",
                compression_level=11
            )
        elif self.level == "ms1":
            # Read the parquet file
            init_df = pl.read_parquet(self.outfile)

            # Check and drop SCORE_MS1_ columns if they exist
            score_ms1_cols = [col for col in init_df.columns if col.startswith("SCORE_MS1_")]
            if score_ms1_cols:
                click.echo(click.style("Warn: Dropping existing SCORE_MS1_ columns from the output file.", fg='yellow'))
                init_df = init_df.drop(score_ms1_cols)

            # Process the result scores
            df = pl.from_pandas(result.scored_tables)
            if 'h_score' in df.columns:
                df = df.select([
                    'feature_id', 'd_score', 'h_score', 'h0_score',
                    'peak_group_rank', 'p_value', 'q_value', 'pep'
                ])
                df = df.rename({
                    'feature_id': 'FEATURE_ID',
                    'd_score': 'SCORE',
                    'h_score': 'HSCORE',
                    'h0_score': 'H0SCORE',
                    'peak_group_rank': 'RANK',
                    'p_value': 'PVALUE',
                    'q_value': 'QVALUE',
                    'pep': 'PEP'
                })
            else:
                df = df.select([
                    'feature_id', 'd_score', 'peak_group_rank',
                    'p_value', 'q_value', 'pep'
                ])
                df = df.rename({
                    'feature_id': 'FEATURE_ID',
                    'd_score': 'SCORE',
                    'peak_group_rank': 'RANK',
                    'p_value': 'PVALUE',
                    'q_value': 'QVALUE',
                    'pep': 'PEP'
                })

            # Add SCORE_MS1_ prefix to columns except FEATURE_ID
            new_columns = {
                col: f'SCORE_MS1_{col}' if col != 'FEATURE_ID' else col
                for col in df.columns
            }
            df = df.rename(new_columns)

            # Merge with initial dataframe
            df = init_df.join(df, on='FEATURE_ID', how='left')

            # Write to parquet
            df.write_parquet(
                self.outfile,
                compression="zstd",
                compression_level=11
            )
        elif self.level == "transition":
            # Read the parquet file
            init_df = pl.read_parquet(self.outfile)

            # Check and drop SCORE_TRANSITION_ columns if they exist
            score_transition_cols = [col for col in init_df.columns if col.startswith("SCORE_TRANSITION_")]
            if score_transition_cols:
                click.echo(click.style("Warn: Dropping existing SCORE_TRANSITION_ columns from the output file.", fg='yellow'))
                init_df = init_df.drop(score_transition_cols)

            # Get TRANSITION_ and FEATURE_TRANSITION_ columns and explode them
            transition_cols = [col for col in init_df.columns if col.startswith('TRANSITION_') or col.startswith('FEATURE_TRANSITION_')]
            transition_cols+=['PRODUCT_MZ', 'ANNOTATION', 'PEPTIDE_IPF_ID']
            
            # Create sub_init_df with only the transition columns to explode
            sub_init_df = init_df.select(['PRECURSOR_ID', 'FEATURE_ID']+transition_cols)
            sub_init_df = sub_init_df.explode(transition_cols)
            
            # Drop transition_cols from init_df
            init_df = init_df.drop(transition_cols)

            # Process the scored tables
            df = pl.DataFrame(result.scored_tables).select([
                'feature_id', 'transition_id', 'd_score', 
                'peak_group_rank', 'p_value', 'q_value', 'pep'
            ]).rename({
                'feature_id': 'FEATURE_ID',
                'transition_id': 'TRANSITION_ID',
                'd_score': 'SCORE',
                'peak_group_rank': 'RANK',
                'p_value': 'PVALUE',
                'q_value': 'QVALUE',
                'pep': 'PEP'
            })

            # Add SCORE_TRANSITION_ prefix to columns except FEATURE_ID and TRANSITION_ID
            df = df.rename({
                col: f'SCORE_TRANSITION_{col}' 
                for col in df.columns 
                if col not in ['FEATURE_ID', 'TRANSITION_ID']
            })

            # Merge with initial dataframe
            df = sub_init_df.join(
                df, 
                on=['FEATURE_ID', 'TRANSITION_ID'], 
                how='left',
                coalesce=True
            )
            
            del sub_init_df

            # Identify columns to group by (non-transition columns except PRODUCT_MZ and ANNOTATION)
            collapse_columns = [
                col for col in df.columns 
                if not any(col.startswith(prefix) for prefix in ['TRANSITION_', 'FEATURE_TRANSITION_', 'SCORE_TRANSITION_'])
                and col not in ['PRODUCT_MZ', 'ANNOTATION', 'PEPTIDE_IPF_ID']
            ]

            # Group and aggregate
            df = df.group_by(collapse_columns).agg(pl.all())
            
            # Merge with init_df
            df = init_df.join(
                df, 
                on=['PRECURSOR_ID', 'FEATURE_ID'], 
                how='left',
                coalesce=True
            )
            
            del init_df

            # Write to parquet
            df.write_parquet(
                self.outfile,
                compression="zstd",
                compression_level=11
            )
        
        click.echo("Info: %s written." % self.outfile)
        
        if result.final_statistics is not None:
            cutoffs = result.final_statistics["cutoff"].values
            svalues = result.final_statistics["svalue"].values
            qvalues = result.final_statistics["qvalue"].values

            pvalues = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["p_value"].values
            top_targets = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 0)]["d_score"].values
            top_decoys = result.scored_tables.loc[(result.scored_tables.peak_group_rank == 1) & (result.scored_tables.decoy == 1)]["d_score"].values

            save_report(os.path.join(self.prefix + "_" + self.level + "_report.pdf"), self.outfile, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0, self.color_palette)
            click.echo("Info: %s written." % os.path.join(self.prefix + "_" + self.level + "_report.pdf"))

class PyProphetLearner(PyProphetRunner):

    def run_algo(self, part=None):
        if self.glyco:
            if self.level in ['ms2', 'ms1ms2'] and part != 'peptide' and part != 'glycan':
                raise click.ClickException("For glycopeptide MS2-level scoring, please specify either 'peptide' or 'glycan' as part.")
            
            if 'decoy' in self.table.columns and self.level!='transition':
                self.table = self.table.drop(columns=['decoy'])
            if self.level == 'ms2' or self.level == 'ms1ms2':
                self.table = self.table.rename(columns={'decoy_' + part: 'decoy'})
            elif self.level == 'ms1':
                self.table = self.table.rename(columns={'decoy_glycan': 'decoy'})
            
        (result, scorer, weights) = PyProphet(self.classifier, self.xgb_hyperparams, self.xgb_params, self.xgb_params_space, self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, self.tric_chromprob, self.threads, self.test, self.ss_score_filter, self.color_palette, self.main_score_selection_report, self.outfile, self.level, self.ss_use_dynamic_main_score).learn_and_apply(self.table)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_scored.tsv")
        yield "summ_stat_path", os.path.join(self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_full_stat.csv")
        yield "trained_weights_path", os.path.join(self.prefix + "_weights.csv")
        yield "trained_model_path_ms1", os.path.join(self.prefix + "_ms1_model.bin")
        yield "trained_model_path_ms1ms2", os.path.join(self.prefix + "_ms1ms2_model.bin")
        yield "trained_model_path_ms2", os.path.join(self.prefix + "_ms2_model.bin")
        yield "trained_model_path_transition", os.path.join(self.prefix + "_transition_model.bin")
        yield "report_path", os.path.join(self.prefix + "_report.pdf")


class PyProphetWeightApplier(PyProphetRunner):

    def __init__(self, infile, outfile, classifier, xgb_hyperparams, xgb_params, xgb_params_space, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, add_alignment_features, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, glyco, density_estimator, grid_size, tric_chromprob, threads, test, apply_weights, ss_score_filter, color_palette, main_score_selection_report):
        super(PyProphetWeightApplier, self).__init__(infile, outfile, classifier, xgb_hyperparams, xgb_params, xgb_params_space, xeval_fraction, xeval_num_iter, ss_initial_fdr, ss_iteration_fdr, ss_num_iter, ss_main_score, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, level, add_alignment_features, ipf_max_peakgroup_rank, ipf_max_peakgroup_pep, ipf_max_transition_isotope_overlap, ipf_min_transition_sn, glyco, density_estimator, grid_size, tric_chromprob, threads, test, ss_score_filter, color_palette, main_score_selection_report)
        if not os.path.exists(apply_weights):
            raise click.ClickException("Weights file %s does not exist." % apply_weights)
        if self.mode == "tsv":
            if self.classifier == "LDA":
                try:
                    self.persisted_weights = pd.read_csv(apply_weights, sep=",")
                    if self.level != self.persisted_weights['level'].unique()[0]:
                        raise click.ClickException("Weights file has wrong level.")
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise
            elif self.classifier == "XGBoost":
                with open(apply_weights, 'rb') as file:
                    self.persisted_weights = pickle.load(file)
        elif self.mode == "osw":
            if self.classifier == "LDA":
                try:
                    con = sqlite3.connect(apply_weights)

                    if not check_sqlite_table(con, "PYPROPHET_WEIGHTS"):
                        raise click.ClickException("PYPROPHET_WEIGHTS table is not present in file, cannot apply weights for LDA classifier! Make sure you have run the scoring on a subset of the data first, or that you supplied the right `--classifier` parameter.")
                    data = pd.read_sql_query("SELECT * FROM PYPROPHET_WEIGHTS WHERE LEVEL=='%s'" % self.level, con)
                    data.columns = [col.lower() for col in data.columns]
                    con.close()
                    self.persisted_weights = data
                    if self.level != self.persisted_weights['level'].unique()[0]:
                        raise click.ClickException("Weights file has wrong level.")
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise
            elif self.classifier == "XGBoost":
                try:
                    con = sqlite3.connect(apply_weights)

                    if not check_sqlite_table(con, "PYPROPHET_XGB"):
                        raise click.ClickException("PYPROPHET_XGB table is not present in file, cannot apply weights for XGBoost classifier! Make sure you have run the scoring on a subset of the data first, or that you supplied the right `--classifier` parameter.")
                    data = con.execute("SELECT xgb FROM PYPROPHET_XGB WHERE LEVEL=='%s'" % self.level).fetchone()
                    con.close()
                    self.persisted_weights = pickle.loads(data[0])
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise
                
    def run_algo(self):
        (result, scorer, weights) = PyProphet(self.classifier, self.xgb_hyperparams, self.xgb_params, self.xgb_params_space, self.xeval_fraction, self.xeval_num_iter, self.ss_initial_fdr, self.ss_iteration_fdr, self.ss_num_iter, self.group_id, self.parametric, self.pfdr, self.pi0_lambda, self.pi0_method, self.pi0_smooth_df, self.pi0_smooth_log_pi0, self.lfdr_truncate, self.lfdr_monotone, self.lfdr_transformation, self.lfdr_adj, self.lfdr_eps, self.tric_chromprob, self.threads, self.test, self.ss_score_filter, self.color_palette, self.main_score_selection_report, self.outfile, self.level, self.ss_use_dynamic_main_score).apply_weights(self.table, self.persisted_weights)
        return (result, scorer, weights)

    def extra_writes(self):
        yield "output_path", os.path.join(self.prefix + "_scored.tsv")
        yield "summ_stat_path", os.path.join(self.prefix + "_summary_stat.csv")
        yield "full_stat_path", os.path.join(self.prefix + "_full_stat.csv")
        yield "report_path", os.path.join(self.prefix + "_report.pdf")
