import pandas as pd
import numpy as np
import polars as pl
import scipy as sp
import sqlite3
import duckdb
import sys
import click
import time

from scipy.stats import rankdata
from .data_handling import check_sqlite_table, create_index_if_not_exists, is_parquet_file, get_parquet_column_names
from shutil import copyfile


def compute_model_fdr(data_in):
    data = np.asarray(data_in)

    # compute model based FDR estimates from posterior error probabilities
    order = np.argsort(data)

    ranks = np.zeros(data.shape[0], dtype=int)
    fdr = np.zeros(data.shape[0])

    # rank data with with maximum ranks for ties
    ranks[order] = rankdata(data[order], method='max')

    # compute FDR/q-value by using cumulative sum of maximum rank for ties
    fdr[order] = data[order].cumsum()[ranks[order]-1] / ranks[order]

    return fdr


def read_pyp_peakgroup_precursor(path, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring):
    click.echo("Info: Reading precursor-level data ... ", nl=False)
    # precursors are restricted according to ipf_max_peakgroup_pep to exclude very poor peak groups
    start = time.time()
    
    # Connect to DuckDB database
    con = duckdb.connect(database=path, read_only=False)
    
    # Connectin with sqlite3 for checking tables
    con_sqlite = sqlite3.connect(path)
    
    # only use MS2 precursors
    if not ipf_ms1_scoring and ipf_ms2_scoring:
        if not check_sqlite_table(con_sqlite, "SCORE_MS2") or not check_sqlite_table(con_sqlite, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS2 and transition-level data before running IPF.")

        create_index_if_not_exists(con, 'idx_transition_id', 'TRANSITION', 'ID')
        create_index_if_not_exists(con, 'idx_precursor_precursor_id', 'PRECURSOR', 'ID')
        create_index_if_not_exists(con, 'idx_feature_precursor_id', 'FEATURE', 'PRECURSOR_ID')
        create_index_if_not_exists(con, 'idx_feature_feature_id', 'FEATURE', 'ID')
        create_index_if_not_exists(con, 'idx_score_ms2_feature_id', 'SCORE_MS2', 'FEATURE_ID')
        create_index_if_not_exists(con, 'idx_score_transition_feature_id', 'SCORE_TRANSITION', 'FEATURE_ID')
        create_index_if_not_exists(con, 'idx_score_transition_transition_id', 'SCORE_TRANSITION', 'TRANSITION_ID')

        data = con.execute('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       NULL AS MS1_PRECURSOR_PEP,
       SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN
  (SELECT FEATURE_ID,
          PEP
   FROM SCORE_TRANSITION
   INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
   WHERE TRANSITION.TYPE=''
     AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < ?;
''', [ipf_max_peakgroup_pep]).df()

    # only use MS1 precursors
    elif ipf_ms1_scoring and not ipf_ms2_scoring:
        if not check_sqlite_table(con_sqlite, "SCORE_MS1") or not check_sqlite_table(con_sqlite, "SCORE_MS2") or not check_sqlite_table(con_sqlite, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS1, MS2 and transition-level data before running IPF.")

        create_index_if_not_exists(con, 'idx_precursor_precursor_id', 'PRECURSOR', 'ID')
        create_index_if_not_exists(con, 'idx_feature_precursor_id', 'FEATURE', 'PRECURSOR_ID')
        create_index_if_not_exists(con, 'idx_feature_feature_id', 'FEATURE', 'ID')
        create_index_if_not_exists(con, 'idx_score_ms1_feature_id', 'SCORE_MS1', 'FEATURE_ID')
        create_index_if_not_exists(con, 'idx_score_ms2_feature_id', 'SCORE_MS2', 'FEATURE_ID')

        data = con.execute('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
       NULL AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < ?;
''', [ipf_max_peakgroup_pep]).df()

    # use both MS1 and MS2 precursors
    elif ipf_ms1_scoring and ipf_ms2_scoring:
        if not check_sqlite_table(con_sqlite, "SCORE_MS1") or not check_sqlite_table(con_sqlite, "SCORE_MS2") or not check_sqlite_table(con_sqlite, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS1, MS2 and transition-level data before running IPF.")

        create_index_if_not_exists(con, 'idx_transition_id', 'TRANSITION', 'ID')
        create_index_if_not_exists(con, 'idx_precursor_precursor_id', 'PRECURSOR', 'ID')
        create_index_if_not_exists(con, 'idx_feature_precursor_id', 'FEATURE', 'PRECURSOR_ID')
        create_index_if_not_exists(con, 'idx_feature_feature_id', 'FEATURE', 'ID')
        create_index_if_not_exists(con, 'idx_score_ms1_feature_id', 'SCORE_MS1', 'FEATURE_ID')
        create_index_if_not_exists(con, 'idx_score_ms2_feature_id', 'SCORE_MS2', 'FEATURE_ID')
        create_index_if_not_exists(con, 'idx_score_transition_feature_id', 'SCORE_TRANSITION', 'FEATURE_ID')
        create_index_if_not_exists(con, 'idx_score_transition_transition_id', 'SCORE_TRANSITION', 'TRANSITION_ID')

        data = con.execute('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       SCORE_MS1.PEP AS MS1_PRECURSOR_PEP,
       SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN
  (SELECT FEATURE_ID,
          PEP
   FROM SCORE_TRANSITION
   INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
   WHERE TRANSITION.TYPE=''
     AND TRANSITION.DECOY=0) AS SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < ?;
''', [ipf_max_peakgroup_pep]).df()

    # do not use any precursor information
    else:
        if not check_sqlite_table(con_sqlite, "SCORE_MS2") or not check_sqlite_table(con_sqlite, "SCORE_TRANSITION"):
            raise click.ClickException("Apply scoring to MS2 and transition-level data before running IPF.")

        create_index_if_not_exists(con, 'idx_precursor_precursor_id', 'PRECURSOR', 'ID')
        create_index_if_not_exists(con, 'idx_feature_precursor_id', 'FEATURE', 'PRECURSOR_ID')
        create_index_if_not_exists(con, 'idx_feature_feature_id', 'FEATURE', 'ID')
        create_index_if_not_exists(con, 'idx_score_ms2_feature_id', 'SCORE_MS2', 'FEATURE_ID')

        data = con.execute('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP,
       NULL AS MS1_PRECURSOR_PEP,
       NULL AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
WHERE PRECURSOR.DECOY=0
  AND SCORE_MS2.PEP < ?;
''', [ipf_max_peakgroup_pep]).df()

    data.columns = [col.lower() for col in data.columns]
    con.close()
    con_sqlite.close()
    
    end = time.time()
    click.echo(f"{end-start:.4f} seconds")

    return data


def read_pyp_transition(path, ipf_max_transition_pep, ipf_h0):
    click.echo("Info: Reading peptidoform-level data ... ", nl=False)
     # only the evidence is restricted to ipf_max_transition_pep, the peptidoform-space is complete
    start = time.time()
    
    con = duckdb.connect(database=path, read_only=False)
    
    create_index_if_not_exists(con, 'idx_transition_peptide_mapping_transition_id', 'TRANSITION_PEPTIDE_MAPPING', 'TRANSITION_ID')
    create_index_if_not_exists(con, 'idx_transition_id', 'TRANSITION', 'ID')
    create_index_if_not_exists(con, 'idx_score_transition_feature_id', 'SCORE_TRANSITION', 'FEATURE_ID')
    create_index_if_not_exists(con, 'idx_score_transition_transition_id', 'SCORE_TRANSITION', 'TRANSITION_ID')
    
    # transition-level evidence
    evidence = con.execute('''
SELECT FEATURE_ID,
       TRANSITION_ID,
       PEP
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
WHERE TRANSITION.TYPE != ''
  AND TRANSITION.DECOY = 0
  AND PEP < ?;
''', [ipf_max_transition_pep]).df()
    evidence.columns = [col.lower() for col in evidence.columns]
    
    # transition-level bitmask
    bitmask = con.execute('''
SELECT DISTINCT TRANSITION.ID AS TRANSITION_ID,
                PEPTIDE_ID,
                1 AS BMASK
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.TYPE != ''
  AND TRANSITION.DECOY = 0;
''').df()
    bitmask.columns = [col.lower() for col in bitmask.columns]
    
    # potential peptidoforms per feature
    num_peptidoforms = con.execute('''
SELECT FEATURE_ID,
       COUNT(DISTINCT PEPTIDE_ID) AS NUM_PEPTIDOFORMS
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.TYPE != ''
  AND TRANSITION.DECOY = 0
GROUP BY FEATURE_ID
ORDER BY FEATURE_ID;
''').df()
    num_peptidoforms.columns = [col.lower() for col in num_peptidoforms.columns]
    
    # peptidoform space per feature
    peptidoforms = con.execute('''
SELECT DISTINCT FEATURE_ID,
                PEPTIDE_ID
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.TYPE != ''
  AND TRANSITION.DECOY = 0
ORDER BY FEATURE_ID;
''').df()
    peptidoforms.columns = [col.lower() for col in peptidoforms.columns]
    
    con.close()
    
    # add h0 (peptide_id: -1) to peptidoform-space if necessary
    if ipf_h0:
        peptidoforms = pd.concat([peptidoforms, pd.DataFrame({'feature_id': peptidoforms['feature_id'].unique(), 'peptide_id': -1})])
    
    # generate transition-peptidoform table
    trans_pf = pd.merge(evidence, peptidoforms, how='outer', on='feature_id')
    
    # apply bitmask
    trans_pf_bm = pd.merge(trans_pf, bitmask, how='left', on=['transition_id', 'peptide_id']).fillna(0)
    
    # append number of peptidoforms
    data = pd.merge(trans_pf_bm, num_peptidoforms, how='inner', on='feature_id')
    
    end = time.time()
    click.echo(f"{end-start:.4f} seconds")
    
    return data


def read_pyp_parquet_peakgroup_precursor(path, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring):
    import pyarrow.parquet as pq

    all_column_names = get_parquet_column_names(path)
    
    click.echo("Info: Reading precursor-level data ... ", nl=False)
    # precursors are restricted according to ipf_max_peakgroup_pep to exclude very poor peak groups
    start = time.time()
    
    # only use MS2 precursors
    if not ipf_ms1_scoring and ipf_ms2_scoring:
        if not any([col.startswith("SCORE_MS2_") for col in all_column_names]) or not any([col.startswith("SCORE_TRANSITION_") for col in all_column_names]):
            raise click.ClickException("Apply scoring to MS2 and transition-level data before running IPF.")

        cols = ['FEATURE_ID', 'SCORE_MS2_PEP', 'PRECURSOR_DECOY', 'TRANSITION_TYPE', 'TRANSITION_DECOY', 'SCORE_TRANSITION_PEP']
        data = pl.read_parquet(path, columns=cols)

        data = (
            data.explode([
                'TRANSITION_TYPE', 
                'TRANSITION_DECOY', 
                'SCORE_TRANSITION_PEP'
            ])
            .filter(
                (pl.col('TRANSITION_TYPE') == '') &
                (pl.col('TRANSITION_DECOY') == 0) &
                (pl.col('PRECURSOR_DECOY') == 0) &
                (pl.col('SCORE_MS2_PEP') < ipf_max_peakgroup_pep) 
            )
            .rename({
                'SCORE_MS2_PEP': 'MS2_PEAKGROUP_PEP',
                'SCORE_TRANSITION_PEP': 'MS2_PRECURSOR_PEP'
            })
            .with_columns(
                pl.lit(None).cast(pl.Float64).alias('MS1_PRECURSOR_PEP')
            )
            .drop(['TRANSITION_TYPE', 'TRANSITION_DECOY', 'PRECURSOR_DECOY'])
            .select([
                'FEATURE_ID',
                'MS2_PEAKGROUP_PEP',
                'MS1_PRECURSOR_PEP',
                'MS2_PRECURSOR_PEP'
            ])
            .to_pandas()
        ) 

    # only use MS1 precursors
    elif ipf_ms1_scoring and not ipf_ms2_scoring:
        if not any([col.startswith("SCORE_MS1_") for col in all_column_names]) or not any([col.startswith("SCORE_MS2_") for col in all_column_names]) or not any([col.startswith("SCORE_TRANSITION_") for col in all_column_names]):
            raise click.ClickException("Apply scoring to MS1, MS2 and transition-level data before running IPF.")

        cols = ['FEATURE_ID', 'SCORE_MS1_PEP', 'PRECURSOR_DECOY', 'SCORE_MS2_PEP']
        data = pl.read_parquet(path, columns=cols)
        data = (
            data.filter(
                (pl.col('PRECURSOR_DECOY') == 0) &
                (pl.col('SCORE_MS2_PEP') < ipf_max_peakgroup_pep)
            )
            .rename({
                'SCORE_MS1_PEP': 'MS1_PRECURSOR_PEP',
                'SCORE_MS2_PEP': 'MS2_PEAKGROUP_PEP'
            })
            .with_columns(
                pl.lit(None).cast(pl.Float64).alias('MS2_PRECURSOR_PEP')
            )
            .drop(['PRECURSOR_DECOY'])
            .select([
                'FEATURE_ID',
                'MS2_PEAKGROUP_PEP',
                'MS1_PRECURSOR_PEP',
                'MS2_PRECURSOR_PEP'
            ])
            .to_pandas()
        )

    # use both MS1 and MS2 precursors
    elif ipf_ms1_scoring and ipf_ms2_scoring:
        if not any([col.startswith("SCORE_MS1_") for col in all_column_names]) or not any([col.startswith("SCORE_MS2_") for col in all_column_names]) or not any([col.startswith("SCORE_TRANSITION_") for col in all_column_names]):
            raise click.ClickException("Apply scoring to MS1, MS2 and transition-level data before running IPF.")

        cols = ['FEATURE_ID', 'SCORE_MS1_PEP', 'PRECURSOR_DECOY', 'SCORE_MS2_PEP', 'TRANSITION_TYPE', 'TRANSITION_DECOY', 'SCORE_TRANSITION_PEP']
        data = pl.read_parquet(path, columns=cols)
        data = (
            data.explode([
                'TRANSITION_TYPE', 
                'TRANSITION_DECOY', 
                'SCORE_TRANSITION_PEP'
            ])
            .filter(
                (pl.col('TRANSITION_TYPE') == '') &
                (pl.col('TRANSITION_DECOY') == 0) &
                (pl.col('PRECURSOR_DECOY') == 0) &
                (pl.col('SCORE_MS2_PEP') < ipf_max_peakgroup_pep) 
            )
            .rename({
                'SCORE_MS1_PEP': 'MS1_PRECURSOR_PEP',
                'SCORE_MS2_PEP': 'MS2_PEAKGROUP_PEP',
                'SCORE_TRANSITION_PEP': 'MS2_PRECURSOR_PEP'
            })
            .drop(['TRANSITION_TYPE', 'TRANSITION_DECOY', 'PRECURSOR_DECOY'])
            .select([
                'FEATURE_ID',
                'MS2_PEAKGROUP_PEP',
                'MS1_PRECURSOR_PEP',
                'MS2_PRECURSOR_PEP'
            ])
            .to_pandas()
        )

    # do not use any precursor information
    else:
        if not any([col.startswith("SCORE_MS2_") for col in all_column_names]) or not any([col.startswith("SCORE_TRANSITION_") for col in all_column_names]):
            raise click.ClickException("Apply scoring to MS2 and transition-level data before running IPF.")

        cols = ['FEATURE_ID', 'SCORE_MS2_PEP', 'PRECURSOR_DECOY']
        data = pl.read_parquet(path, columns=cols)
        data = (
            data.filter(
                (pl.col('PRECURSOR_DECOY') == 0) &
                (pl.col('SCORE_MS2_PEP') < ipf_max_peakgroup_pep)
            )
            .rename({
                'SCORE_MS2_PEP': 'MS2_PEAKGROUP_PEP'
            })
            .with_columns(
                pl.lit(None).cast(pl.Float64).alias('MS1_PRECURSOR_PEP'),
                pl.lit(None).cast(pl.Float64).alias('MS2_PRECURSOR_PEP')
            )
            .drop(['PRECURSOR_DECOY'])
            .select([
                'FEATURE_ID',
                'MS2_PEAKGROUP_PEP',
                'MS1_PRECURSOR_PEP',
                'MS2_PRECURSOR_PEP'
            ])
            .to_pandas()
        )

    data.columns = [col.lower() for col in data.columns]
    
    end = time.time()
    click.echo(f"{end-start:.4f} seconds")

    return data


def read_pyp_parquet_transition(path, ipf_max_transition_pep, ipf_h0):
    click.echo("Info: Reading peptidoform-level data ... ", nl=False)
     # only the evidence is restricted to ipf_max_transition_pep, the peptidoform-space is complete
    start = time.time()
    
    
    # transition-level evidence
    cols = ['FEATURE_ID', 'TRANSITION_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY', 'SCORE_TRANSITION_PEP']
    evidence = pl.read_parquet(path, columns=cols)
    evidence = (
        evidence.explode(['TRANSITION_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY', 'SCORE_TRANSITION_PEP'])
    .filter(
        (pl.col('TRANSITION_TYPE') != '') &
        (pl.col('TRANSITION_DECOY') == 0) &
        (pl.col('SCORE_TRANSITION_PEP') < ipf_max_transition_pep)
    )
    .rename({'SCORE_TRANSITION_PEP': 'PEP'})
    .drop(['TRANSITION_TYPE', 'TRANSITION_DECOY'])
    .select([
        'FEATURE_ID',
        'TRANSITION_ID',
        'PEP'
    ])
    .unique()
    .to_pandas()
    )
    evidence.columns = [col.lower() for col in evidence.columns]
    
    # transition-level bitmask
    cols = ['TRANSITION_ID', 'PEPTIDE_IPF_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY']
    bitmask = pl.read_parquet(path, columns=cols)
    bitmask = (
        bitmask.explode(['TRANSITION_ID', 'PEPTIDE_IPF_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY'])
        .filter(
            (pl.col('TRANSITION_TYPE') != '') &
            (pl.col('TRANSITION_DECOY') == 0) &
            (pl.col('PEPTIDE_IPF_ID').is_not_null())
        )
        .with_columns(
            pl.col('PEPTIDE_IPF_ID').cast(pl.Int64).alias('PEPTIDE_ID'),
            pl.lit(1).cast(pl.Int64).alias('BMASK')
        )
        .drop(['TRANSITION_TYPE', 'TRANSITION_DECOY'])
        .select([
            'TRANSITION_ID',
            'PEPTIDE_ID',
            'BMASK'
        ])
        .unique()
        .to_pandas()
    )
    bitmask.columns = [col.lower() for col in bitmask.columns]
    
    # potential peptidoforms per feature
    cols = ['FEATURE_ID', 'PEPTIDE_IPF_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY']
    num_peptidoforms = pl.read_parquet(path, columns=cols)
    num_peptidoforms = (
        num_peptidoforms.explode(['PEPTIDE_IPF_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY'])
        .filter(
            (pl.col('TRANSITION_TYPE') != '') &
            (pl.col('TRANSITION_DECOY') == 0) &
            (pl.col('PEPTIDE_IPF_ID').is_not_null())
        )
        .with_columns(
            pl.col('PEPTIDE_IPF_ID').cast(pl.Int64).alias('PEPTIDE_ID'),
            pl.lit(1).cast(pl.Int64).alias('BMASK')
        )
        .drop(['TRANSITION_TYPE', 'TRANSITION_DECOY'])
        .unique()
        .group_by(['FEATURE_ID'])
        .agg(pl.col('PEPTIDE_ID').n_unique().alias('NUM_PEPTIDOFORMS'))
        .sort('FEATURE_ID')
        .select([
            'FEATURE_ID',
            'NUM_PEPTIDOFORMS'
        ])
        .to_pandas()
    )
    num_peptidoforms.columns = [col.lower() for col in num_peptidoforms.columns]
    
    # peptidoform space per feature
    cols = ['FEATURE_ID', 'PEPTIDE_IPF_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY']
    peptidoforms = pl.read_parquet(path, columns=cols)
    peptidoforms = (
        peptidoforms.explode(['PEPTIDE_IPF_ID', 'TRANSITION_TYPE', 'TRANSITION_DECOY'])
        .filter(
            (pl.col('TRANSITION_TYPE') != '') &
            (pl.col('TRANSITION_DECOY') == 0) &
            (pl.col('PEPTIDE_IPF_ID').is_not_null())
        )
        .with_columns(
            pl.col('PEPTIDE_IPF_ID').cast(pl.Int64).alias('PEPTIDE_ID'),
            pl.lit(1).cast(pl.Int64).alias('BMASK')
        )
        .drop(['TRANSITION_TYPE', 'TRANSITION_DECOY'])
        .unique()
        .sort(['FEATURE_ID'])
        .group_by(['FEATURE_ID', 'PEPTIDE_ID'])
        .agg(pl.all())
        .select([
            'FEATURE_ID',
            'PEPTIDE_ID'
        ])
        .to_pandas()
    )
    peptidoforms.columns = [col.lower() for col in peptidoforms.columns]
    
    # add h0 (peptide_id: -1) to peptidoform-space if necessary
    if ipf_h0:
        peptidoforms = pd.concat([peptidoforms, pd.DataFrame({'feature_id': peptidoforms['feature_id'].unique(), 'peptide_id': -1})])
    
    # generate transition-peptidoform table
    trans_pf = pd.merge(evidence, peptidoforms, how='outer', on='feature_id')
    
    # apply bitmask
    trans_pf_bm = pd.merge(trans_pf, bitmask, how='left', on=['transition_id', 'peptide_id']).fillna(0)
    
    # append number of peptidoforms
    data = pd.merge(trans_pf_bm, num_peptidoforms, how='inner', on='feature_id')
    
    end = time.time()
    click.echo(f"{end-start:.4f} seconds")
    
    return data


def prepare_precursor_bm(data):
    # MS1-level precursors
    ms1_precursor_data = data[['feature_id','ms2_peakgroup_pep','ms1_precursor_pep']].dropna(axis=0, how='any')
    ms1_bm_data = pd.concat([pd.DataFrame({'feature_id': ms1_precursor_data['feature_id'], 'prior': 1-ms1_precursor_data['ms2_peakgroup_pep'], 'evidence': 1-ms1_precursor_data['ms1_precursor_pep'], 'hypothesis': True}), pd.DataFrame({'feature_id': ms1_precursor_data['feature_id'], 'prior': ms1_precursor_data['ms2_peakgroup_pep'], 'evidence': ms1_precursor_data['ms1_precursor_pep'], 'hypothesis': False})])

    # MS2-level precursors
    ms2_precursor_data = data[['feature_id','ms2_peakgroup_pep','ms2_precursor_pep']].dropna(axis=0, how='any')
    ms2_bm_data = pd.concat([pd.DataFrame({'feature_id': ms2_precursor_data['feature_id'], 'prior': 1-ms2_precursor_data['ms2_peakgroup_pep'], 'evidence': 1-ms2_precursor_data['ms2_precursor_pep'], 'hypothesis': True}), pd.DataFrame({'feature_id': ms2_precursor_data['feature_id'], 'prior': ms2_precursor_data['ms2_peakgroup_pep'], 'evidence': ms2_precursor_data['ms2_precursor_pep'], 'hypothesis': False})])

    # missing precursor data
    missing_precursor_data = data[['feature_id','ms2_peakgroup_pep']].dropna(axis=0, how='any').drop_duplicates()
    missing_bm_data = pd.concat([pd.DataFrame({'feature_id': missing_precursor_data['feature_id'], 'prior': 1-missing_precursor_data['ms2_peakgroup_pep'], 'evidence': 0, 'hypothesis': True}), pd.DataFrame({'feature_id': missing_precursor_data['feature_id'], 'prior': missing_precursor_data['ms2_peakgroup_pep'], 'evidence': 1, 'hypothesis': False})])

    # combine precursor data
    precursor_bm_data = pd.concat([ms1_bm_data, ms2_bm_data])
    # append missing precursors if no MS1/MS2 evidence is available
    precursor_bm_data = pd.concat([precursor_bm_data, missing_bm_data.loc[~missing_bm_data['feature_id'].isin(precursor_bm_data['feature_id'])]])

    return(precursor_bm_data)


def transfer_confident_evidence_across_runs(df1, across_run_confidence_threshold, group_cols=['feature_id', 'transition_id', 'peptide_id', 'bmask', 'num_peptidoforms', 'alignment_group_id'], value_cols=['pep', 'precursor_peakgroup_pep']):
    feature_ids = np.unique(df1['feature_id'])
    df_list = []
    for feature_id in feature_ids:
        tmp_df = df1[(df1['feature_id'] == feature_id) | ((df1['feature_id'] != feature_id) & (df1['pep'] <= across_run_confidence_threshold))]
        tmp_df['feature_id'] = feature_id
        df_list.append(tmp_df)
    df_filtered = pd.concat(df_list)
    
    # Group by relevant columns and apply min reduction
    df_result = df_filtered.groupby(
        group_cols,
        as_index=False
    )[value_cols].min()
    
    return df_result


def prepare_transition_bm(data, propagate_signal_across_runs, across_run_confidence_threshold):
    # Propagate peps <= threshold for aligned feature groups across runs
    if propagate_signal_across_runs: 
        ## Separate out features that need propagation and those that don't to avoid calling apply on the features that don't need propagated peps
        non_prop_data = data.loc[ data['feature_id']==data['alignment_group_id']]
        prop_data = data.loc[ data['feature_id']!=data['alignment_group_id']]
        
        start = time.time()
        # Group by alignment_group_id and apply function in parallel
        data_with_confidence = (
            prop_data.groupby("alignment_group_id", group_keys=False)
            .apply(lambda df: transfer_confident_evidence_across_runs(df, across_run_confidence_threshold))
            .reset_index(drop=True)
        )
        end = time.time()
        click.echo(f"\nInfo: Propagating signal for {len(prop_data['feature_id'].unique())} aligned features of total {len(data['feature_id'].unique())} features across runs ... {end-start:.4f} seconds")
        
        ## Concat non prop data with prop data
        data = pd.concat([non_prop_data, data_with_confidence], ignore_index=True)
  
    # peptide_id = -1 indicates h0, i.e. the peak group is wrong!
    # initialize priors
    data.loc[data.peptide_id != -1, 'prior'] = (1-data.loc[data.peptide_id != -1, 'precursor_peakgroup_pep']) / data.loc[data.peptide_id != -1, 'num_peptidoforms'] # potential peptidoforms
    data.loc[data.peptide_id == -1, 'prior'] = data.loc[data.peptide_id == -1, 'precursor_peakgroup_pep'] # h0
    
    # set evidence
    data.loc[data.bmask == 1, 'evidence'] = (1-data.loc[data.bmask == 1, 'pep']) # we have evidence FOR this peptidoform or h0
    data.loc[data.bmask == 0, 'evidence'] = data.loc[data.bmask == 0, 'pep'] # we have evidence AGAINST this peptidoform or h0
    
    if propagate_signal_across_runs:
        cols = ['feature_id', 'alignment_group_id', 'num_peptidoforms','prior','evidence','peptide_id']
    else:
        cols = ['feature_id', 'num_peptidoforms','prior','evidence','peptide_id']
    data = data[cols]
    data = data.rename(columns=lambda x: x.replace('peptide_id','hypothesis'))
    
    return data


def apply_bm(data):
    # compute likelihood * prior per feature & hypothesis
    # all priors are identical but pandas DF multiplication requires aggregation, so we use min()
    pp_data = (data.groupby(['feature_id',"hypothesis"])["evidence"].prod() * data.groupby(['feature_id',"hypothesis"])["prior"].min()).reset_index()
    pp_data.columns = ['feature_id','hypothesis','likelihood_prior']

    # compute likelihood sum per feature
    pp_data['likelihood_sum'] = pp_data.groupby('feature_id')['likelihood_prior'].transform("sum")

    # compute posterior hypothesis probability
    pp_data['posterior'] = pp_data['likelihood_prior'] / pp_data['likelihood_sum']

    return pp_data.fillna(value = 0)


def precursor_inference(data, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep, ipf_max_precursor_peakgroup_pep):
    # prepare MS1-level precursor data
    if ipf_ms1_scoring:
        ms1_precursor_data = data[data['ms1_precursor_pep'] < ipf_max_precursor_pep][['feature_id','ms1_precursor_pep']].drop_duplicates()
    else:
        ms1_precursor_data = data[['feature_id']].drop_duplicates()
        ms1_precursor_data['ms1_precursor_pep'] = np.nan

    # prepare MS2-level precursor data
    if ipf_ms2_scoring:
        ms2_precursor_data = data[data['ms2_precursor_pep'] < ipf_max_precursor_pep][['feature_id','ms2_precursor_pep']].drop_duplicates()
    else:
        ms2_precursor_data = data[['feature_id']].drop_duplicates()
        ms2_precursor_data['ms2_precursor_pep'] = np.nan

    # prepare MS2-level peak group data
    ms2_pg_data = data[['feature_id','ms2_peakgroup_pep']].drop_duplicates()

    if ipf_ms1_scoring or ipf_ms2_scoring:
        # merge MS1- & MS2-level precursor and peak group data
        precursor_data = ms2_precursor_data.merge(ms1_precursor_data, on=['feature_id'], how='outer').merge(ms2_pg_data, on=['feature_id'], how='outer')

        # prepare precursor-level Bayesian model
        click.echo("Info: Preparing precursor-level data ... ", nl=False)
        start = time.time()
        precursor_data_bm = prepare_precursor_bm(precursor_data)
        end = time.time()
        click.echo(f"{end-start:.4f} seconds")

        # compute posterior precursor probability
        click.echo("Info: Conducting precursor-level inference ... ", nl=False)
        start = time.time()
        prec_pp_data = apply_bm(precursor_data_bm)
        prec_pp_data['precursor_peakgroup_pep'] = 1 - prec_pp_data['posterior']

        inferred_precursors = prec_pp_data[prec_pp_data['hypothesis']][['feature_id','precursor_peakgroup_pep']]
        end = time.time()
        click.echo(f"{end-start:.4f} seconds")
    else:
        # no precursor-level data on MS1 and/or MS2 should be used; use peak group-level data
        click.echo("Info: Skipping precursor-level inference.")
        inferred_precursors = ms2_pg_data.rename(columns=lambda x: x.replace('ms2_peakgroup_pep', 'precursor_peakgroup_pep'))

    inferred_precursors = inferred_precursors[(inferred_precursors['precursor_peakgroup_pep'] < ipf_max_precursor_peakgroup_pep)]

    return inferred_precursors


def peptidoform_inference(transition_table, precursor_data, ipf_grouped_fdr, propagate_signal_across_runs, across_run_confidence_threshold):
    transition_table = pd.merge(transition_table, precursor_data, on='feature_id')

    # compute transition posterior probabilities
    click.echo("Info: Preparing peptidoform-level data ... ", nl=False)
    start = time.time()
    transition_data_bm = prepare_transition_bm(transition_table, propagate_signal_across_runs, across_run_confidence_threshold)
    end = time.time()
    click.echo(f"{end-start:.4f} seconds")

    # compute posterior peptidoform probability
    click.echo("Info: Conducting peptidoform-level inference ... ", nl=False)
    start = time.time()
    pf_pp_data = apply_bm(transition_data_bm)
    pf_pp_data['pep'] = 1 - pf_pp_data['posterior']
    
    # compute model-based FDR
    if ipf_grouped_fdr:
        pf_pp_data['qvalue'] = pd.merge(pf_pp_data, transition_data_bm[['feature_id', 'num_peptidoforms']].drop_duplicates(), on=['feature_id'], how='inner').groupby('num_peptidoforms')['pep'].transform(compute_model_fdr)
    else:
        pf_pp_data['qvalue'] = compute_model_fdr(pf_pp_data['pep'])

    # merge precursor-level data with UIS data
    result = pf_pp_data.merge(precursor_data[['feature_id','precursor_peakgroup_pep']].drop_duplicates(), on=['feature_id'], how='inner')
    
    end = time.time()
    click.echo(f"{end-start:.4f} seconds")

    return result


def get_feature_mapping_across_runs(infile, ipf_max_alignment_pep=1):
    click.echo("Info: Reading Across Run Feature Alignment Mapping ... ", nl=False)
    start = time.time()

    con = sqlite3.connect(infile)

    data = pd.read_sql_query(
        f"""SELECT  
                DENSE_RANK() OVER (ORDER BY PRECURSOR_ID, ALIGNMENT_ID) AS ALIGNMENT_GROUP_ID,
                ALIGNED_FEATURE_ID AS FEATURE_ID 
                FROM (SELECT DISTINCT * FROM FEATURE_MS2_ALIGNMENT) AS FEATURE_MS2_ALIGNMENT
                INNER JOIN 
                (SELECT DISTINCT *, MIN(QVALUE) FROM SCORE_ALIGNMENT GROUP BY FEATURE_ID) AS SCORE_ALIGNMENT 
                ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_MS2_ALIGNMENT.ALIGNED_FEATURE_ID
                WHERE LABEL = 1
                AND SCORE_ALIGNMENT.PEP < {ipf_max_alignment_pep}
                ORDER BY ALIGNMENT_GROUP_ID""",
        con,
    )

    data.columns = [col.lower() for col in data.columns]
    con.close()
    
    end = time.time()
    click.echo(f"{end-start:.4f} seconds")

    return data


def infer_peptidoforms(infile, outfile, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0, ipf_grouped_fdr, ipf_max_precursor_pep, ipf_max_peakgroup_pep, ipf_max_precursor_peakgroup_pep, ipf_max_transition_pep, propagate_signal_across_runs, ipf_max_alignment_pep=1, across_run_confidence_threshold=0.5):
    click.echo("Info: Starting IPF (Inference of PeptidoForms).")

    # precursor level
    if is_parquet_file(infile):
        precursor_table = read_pyp_parquet_peakgroup_precursor(infile, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring)
    else:
        precursor_table = read_pyp_peakgroup_precursor(infile, ipf_max_peakgroup_pep, ipf_ms1_scoring, ipf_ms2_scoring)
    precursor_data = precursor_inference(precursor_table, ipf_ms1_scoring, ipf_ms2_scoring, ipf_max_precursor_pep, ipf_max_precursor_peakgroup_pep)

    # peptidoform level
    if is_parquet_file(infile):
        peptidoform_table = read_pyp_parquet_transition(infile, ipf_max_transition_pep, ipf_h0)
    else:
        peptidoform_table = read_pyp_transition(infile, ipf_max_transition_pep, ipf_h0)
    ## prepare for propagating signal across runs for aligned features
    if propagate_signal_across_runs:
        across_run_feature_map = get_feature_mapping_across_runs(infile, ipf_max_alignment_pep)
        peptidoform_table = peptidoform_table.merge(across_run_feature_map, how='left', on='feature_id')
        ## Fill missing alignment_group_id with feature_id for those that are not aligned
        peptidoform_table["alignment_group_id"] = peptidoform_table["alignment_group_id"].astype(object)
        mask = peptidoform_table["alignment_group_id"].isna()
        peptidoform_table.loc[mask, "alignment_group_id"] = peptidoform_table.loc[mask, "feature_id"].astype(str)

        peptidoform_table = peptidoform_table.astype({'alignment_group_id':'int64'})
    
    peptidoform_data = peptidoform_inference(peptidoform_table, precursor_data, ipf_grouped_fdr, propagate_signal_across_runs, across_run_confidence_threshold)

    # finalize results and write to table
    click.echo("Info: Storing results.")
    peptidoform_data = peptidoform_data[peptidoform_data['hypothesis']!=-1][['feature_id','hypothesis','precursor_peakgroup_pep','qvalue','pep']]
    peptidoform_data.columns = ['FEATURE_ID','PEPTIDE_ID','PRECURSOR_PEAKGROUP_PEP','QVALUE','PEP']
    
    # Convert feature_id to int64
    peptidoform_data = peptidoform_data.astype({'FEATURE_ID':'int64'})

    if is_parquet_file(infile):
        if infile != outfile:
            copyfile(infile, outfile)
        else:
            outfile = infile
        
        # Read the parquet file
        init_df = pl.read_parquet(outfile)

        # Select and process columns
        sub_init_df = (
            init_df.select(['PEPTIDE_ID', 'PRECURSOR_ID', 'PEPTIDE_IPF_ID', 'FEATURE_ID'])
            .explode('PEPTIDE_IPF_ID')
            .drop_nulls(subset=['PEPTIDE_IPF_ID'])
            .with_columns(pl.col('PEPTIDE_IPF_ID').cast(pl.Int64))
        )

        peptidoform_data = pl.from_pandas(peptidoform_data)

        # Join with peptidoform_data
        peptidoform_data_m = (
            sub_init_df.join(
                peptidoform_data,
                left_on=['FEATURE_ID', 'PEPTIDE_IPF_ID'],
                right_on=['FEATURE_ID', 'PEPTIDE_ID'],
                how='inner',
                coalesce=True
            )
            .unique()  # Equivalent to drop_duplicates()
            .drop(['PEPTIDE_IPF_ID'])  # Drop columns
            .rename({
                'PRECURSOR_PEAKGROUP_PEP': 'SCORE_IPF_PRECURSOR_PEAKGROUP_PEP',
                'QVALUE': 'SCORE_IPF_QVALUE',
                'PEP': 'SCORE_IPF_PEP'
            })
        )

        # Merge back with original data
        peptidoform_data_m = (
            init_df.join(
                peptidoform_data_m,
                on=['PEPTIDE_ID', 'PRECURSOR_ID', 'FEATURE_ID'],
                how='left'
            )
        )

        # Write to parquet
        peptidoform_data_m.write_parquet(
            outfile,
            compression="zstd",
            compression_level=11
        )
    else:
        if infile != outfile:
            copyfile(infile, outfile)

        con = sqlite3.connect(outfile)

        peptidoform_data.to_sql("SCORE_IPF", con, index=False, if_exists='replace')
        con.close()
