import time
import click
import sqlite3
import pandas as pd
import numpy as np
from shutil import copyfile

from pyprophet.data_handling import check_sqlite_table
from pyprophet.ipf import apply_bm, compute_model_fdr, get_feature_mapping_across_runs, transfer_confident_evidence_across_runs

from .pepmass import GlycoPeptideMassCalculator



def read_glycoform_space(path, use_glycan_composition):
    click.echo("Info: Reading glycoform spaces.")
    
    con = sqlite3.connect(path)
    
    if not check_sqlite_table(con, "SCORE_TRANSITION"):
        raise click.ClickException("Apply scoring to transition-level data before running glycoform inference.")
    
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_transition_glycopeptide_mapping_transition_id ON TRANSITION_GLYCOPEPTIDE_MAPPING (TRANSITION_ID);
CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);
''')
    
    glycoforms = pd.read_sql_query('''
SELECT DISTINCT SCORE_TRANSITION.FEATURE_ID AS FEATURE_ID,
                GLYCOPEPTIDE.ID AS GLYCOPEPTIDE_ID,
                GLYCAN_STRUCT,
                GLYCAN_COMPOSITION
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_GLYCOPEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_GLYCOPEPTIDE_MAPPING.TRANSITION_ID
INNER JOIN GLYCOPEPTIDE ON GLYCOPEPTIDE.ID = TRANSITION_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID
INNER JOIN GLYCOPEPTIDE_GLYCAN_MAPPING ON GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCOPEPTIDE_ID = GLYCOPEPTIDE.ID
INNER JOIN GLYCAN ON GLYCAN.ID = GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCAN_ID
WHERE TRANSITION.DECOY = 0
ORDER BY FEATURE_ID;
''', con)
    glycoforms.columns = [col.lower() for col in glycoforms.columns]
    
    num_glycoforms = pd.read_sql_query('''
SELECT FEATURE_ID,
       COUNT(DISTINCT GLYCOPEPTIDE_ID) AS NUM_GLYCOFORMS
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_GLYCOPEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_GLYCOPEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.DECOY = 0
GROUP BY FEATURE_ID
ORDER BY FEATURE_ID;
''', con)
    num_glycoforms.columns = [col.lower() for col in num_glycoforms.columns]
    
    con.close()
        
    h0 = glycoforms[['feature_id']].drop_duplicates()
    h0['glycopeptide_id'] = -1
    h0['glycan_struct'] = ''
    h0['glycan_composition'] = ''
    
    glycoforms = pd.concat([
        glycoforms, 
        h0
    ], sort=True)
    
    data = pd.merge(glycoforms, num_glycoforms, how='left', on='feature_id')
    
    if use_glycan_composition:
        data['num_glycan_composition'] = data \
            .groupby(['feature_id']) \
            ['glycan_composition'] \
            .transform(lambda x: len(x[x != ''].unique()))
        data['num_glycan_struct'] = data \
            .groupby(['feature_id', 'glycan_composition']) \
            ['glycan_struct'] \
            .transform(lambda x: len(x))
    
    return data


def read_ms2_peakgroup(path, max_peakgroup_pep):
    click.echo("Info: Reading MS2 peak group data.")
    
    con = sqlite3.connect(path)
    
    if not check_sqlite_table(con, "SCORE_MS2"):
        raise click.ClickException("Apply scoring to MS2-level data before running glycoform inference.")
    
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS1 (FEATURE_ID);
''')
    
    data = pd.read_sql_query('''
SELECT FEATURE.ID AS FEATURE_ID,
       SCORE_MS2.PEP AS MS2_PEAKGROUP_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
WHERE PRECURSOR.DECOY = 0
  AND SCORE_MS2.PEP < %s;
''' % max_peakgroup_pep, con)
    data.columns = [col.lower() for col in data.columns]
    
    con.close()    
    
    return data


def read_ms1_precursor(path, max_precursor_pep):
    click.echo("Info: Reading MS1 precursor-level data.")
    
    con = sqlite3.connect(path)
    
    if not check_sqlite_table(con, "SCORE_MS1"):
        raise click.ClickException("Apply scoring to MS1-level data before running glycoform inference.")

    
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_ms1_feature_id ON SCORE_MS1 (FEATURE_ID);
''')
    
    query = '''
SELECT FEATURE.ID AS FEATURE_ID,
       GLYCOPEPTIDE_ID,
       GLYCAN_STRUCT,
       PRECURSOR_MZ,
       PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
       SCORE_MS1.PEP AS MS1_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
INNER JOIN FEATURE_MS1 ON FEATURE.ID = FEATURE_MS1.FEATURE_ID

INNER JOIN
  (SELECT PRECURSOR_ID,
          GLYCOPEPTIDE.ID AS GLYCOPEPTIDE_ID,
          GLYCAN_STRUCT
   FROM GLYCOPEPTIDE
   INNER JOIN PRECURSOR_GLYCOPEPTIDE_MAPPING ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID = GLYCOPEPTIDE.ID
   INNER JOIN GLYCOPEPTIDE_GLYCAN_MAPPING ON GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCOPEPTIDE_ID = GLYCOPEPTIDE.ID
   INNER JOIN GLYCAN ON GLYCAN.ID = GLYCOPEPTIDE_GLYCAN_MAPPING.GLYCAN_ID) AS GLYCOPEPTIDE 
  ON GLYCOPEPTIDE.PRECURSOR_ID = PRECURSOR.ID

WHERE PRECURSOR.DECOY = 0
  AND SCORE_MS1.PEP < %s 
  AND FEATURE_MS1.AREA_INTENSITY > 0;
''' % max_precursor_pep

    data = pd.read_sql_query(query, con)    
    data.columns = [col.lower() for col in data.columns]
    con.close()

    return data


def read_ms2_precursor(path, max_precursor_pep):
    click.echo("Info: Reading MS2 precursor-level data.")
    
    con = sqlite3.connect(path)
    
    if not check_sqlite_table(con, "SCORE_TRANSITION"):
        raise click.ClickException("Apply scoring to transition-level data before running glycoform inference.")
        
    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);
''')
    
    query = '''
SELECT FEATURE.ID AS FEATURE_ID,
       GLYCOPEPTIDE_TRANSITION.GLYCOPEPTIDE_ID AS GLYCOPEPTIDE_ID,
       GLYCOPEPTIDE_TRANSITION.TRANSITION_ID AS TRANSITION_ID,      
       SCORE_TRANSITION.PEP AS MS2_PRECURSOR_PEP
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN
  (SELECT PRECURSOR_ID,
          GLYCOPEPTIDE.ID AS GLYCOPEPTIDE_ID,          
          TRANSITION_GLYCOPEPTIDE_MAPPING.TRANSITION_ID AS TRANSITION_ID
   FROM GLYCOPEPTIDE
   INNER JOIN PRECURSOR_GLYCOPEPTIDE_MAPPING ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID = GLYCOPEPTIDE.ID
   INNER JOIN TRANSITION_GLYCOPEPTIDE_MAPPING ON TRANSITION_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID = GLYCOPEPTIDE.ID
   INNER JOIN TRANSITION ON TRANSITION_GLYCOPEPTIDE_MAPPING.TRANSITION_ID = TRANSITION.ID
   WHERE TRANSITION.TYPE=''
     AND TRANSITION.DECOY=0
  ) AS GLYCOPEPTIDE_TRANSITION
  ON GLYCOPEPTIDE_TRANSITION.PRECURSOR_ID = PRECURSOR.ID

LEFT JOIN SCORE_TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = GLYCOPEPTIDE_TRANSITION.TRANSITION_ID 
                          AND SCORE_TRANSITION.FEATURE_ID = FEATURE.ID

WHERE PRECURSOR.DECOY = 0
  AND SCORE_TRANSITION.PEP < %s;
''' % max_precursor_pep

    data = pd.read_sql_query(query, con)    
    data.columns = [col.lower() for col in data.columns]
    con.close()
    
    return data


def read_transition(path, max_transition_pep):    
    click.echo("Info: Reading transition-level data.")
    
    con = sqlite3.connect(path)
    
    if not check_sqlite_table(con, "SCORE_TRANSITION"):
        raise click.ClickException("Apply scoring to transition-level data before running glycoform inference.")

    con.executescript('''
CREATE INDEX IF NOT EXISTS idx_transition_glycopeptide_mapping_transition_id ON TRANSITION_GLYCOPEPTIDE_MAPPING (TRANSITION_ID);
CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);
CREATE INDEX IF NOT EXISTS idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);
''')
    
    data = pd.read_sql_query('''
SELECT FEATURE_ID,
       TRANSITION.ID AS TRANSITION_ID,
       GLYCOPEPTIDE_ID,
       PEP
FROM SCORE_TRANSITION
INNER JOIN TRANSITION ON SCORE_TRANSITION.TRANSITION_ID = TRANSITION.ID
INNER JOIN TRANSITION_GLYCOPEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_GLYCOPEPTIDE_MAPPING.TRANSITION_ID
WHERE TRANSITION.TYPE!=''
  AND TRANSITION.DECOY=0
  AND PEP < %s
ORDER BY FEATURE_ID;
 ''' % max_transition_pep, con)
    data.columns = [col.lower() for col in data.columns]
    
    return data


def calculate_ms2_peakgroup_prior(data, use_glycan_composition):    
    if use_glycan_composition:
        return data.apply(
            lambda x: (1 - x['ms2_peakgroup_pep']) / x['num_glycan_composition'] / x['num_glycan_struct'] \
                if x['glycopeptide_id'] != -1 else x['ms2_peakgroup_pep'], 
            axis=1   
        )
    else:
        return data.apply(
            lambda x: (1 - x['ms2_peakgroup_pep']) / x['num_glycoforms'] \
                if x['glycopeptide_id'] != -1 else x['ms2_peakgroup_pep'], 
            axis=1   
        )

def prepare_ms1_precursor_bm(glycoforms, 
                             ms1_precursor_table, 
                             ms2_peakgroup_table, 
                             use_glycan_composition,
                             ms1_mz_window=10,
                             ms1_mz_window_unit='ppm'):
    mass_calculator = GlycoPeptideMassCalculator()    
    
    if ms1_mz_window_unit == 'ppm':
        def match_precursor_mz(precursor_mz, precursor_charge, 
                               glycan_mw_1, glycan_mw_2): 
            return abs(glycan_mw_2 - glycan_mw_1) / \
                (precursor_charge * precursor_mz) \
                <= ms1_mz_window * 1e-6
    elif ms1_mz_window_unit in {'Da', 'Th'}:
        def match_precursor_mz(precursor_mz, precursor_charge, 
                               glycan_mw_1, glycan_mw_2): 
            return abs(glycan_mw_2 - glycan_mw_1) / precursor_charge \
                <= ms1_mz_window
    else:
        raise ValueError('invalid m/z extraction window unit: ' + \
                         str(ms1_mz_window_unit))
        
    data = pd.merge(
        glycoforms, ms1_precursor_table,
        on=['feature_id'],
        how='left',
        suffixes=['', '_ms1']
    ).drop_duplicates()
    
    data = pd.merge(data, ms2_peakgroup_table, how='inner', on='feature_id')
    data['prior'] = calculate_ms2_peakgroup_prior(
        data, 
        use_glycan_composition=use_glycan_composition
    )
    
    data['match_precursor'] = data \
        .apply(
            lambda x: \
                np.isnan(x['ms1_precursor_pep']) or \
                (x['glycopeptide_id'] != -1) and \
                match_precursor_mz(
                    x['precursor_mz'],
                    x['precursor_charge'],
                    mass_calculator.glycan_mw(x['glycan_struct']),
                    mass_calculator.glycan_mw(x['glycan_struct_ms1']),
                ),
            axis=1
        )
                
    data['evidence'] = \
        1 - data['ms1_precursor_pep'].fillna(1)
    data['evidence'] = data \
        .apply(
            lambda x: \
                x['evidence'] \
                if x['glycopeptide_id'] != -1 and x['match_precursor'] \
                else 1 - x['evidence'],
            axis=1
        )
            
    return data \
        [[
            'feature_id', 'num_glycoforms', 
            'prior', 'evidence', 'glycopeptide_id'
        ]] \
        .rename(columns={'glycopeptide_id': 'hypothesis'})


def prepare_ms2_precursor_bm(glycoforms, 
                             ms2_precursor_table, ms2_peakgroup_table,
                             use_glycan_composition):
    evidence = ms2_precursor_table \
        [['feature_id', 'transition_id', 'ms2_precursor_pep']] \
        .drop_duplicates()        
    data = pd.merge(glycoforms, evidence, how='left', on='feature_id')
        
    bitmask = ms2_precursor_table \
        [['transition_id', 'glycopeptide_id']] \
        .drop_duplicates()
    bitmask['bmask'] = 1
    data = pd.merge(data, bitmask, how='left', on=['transition_id','glycopeptide_id']) 

    data.dropna(subset=['transition_id'], inplace=True)
    data.fillna(value={'bmask': 0}, inplace=True)
    data.dropna(subset=['ms2_precursor_pep'], inplace=True) 
    data['evidence'] = data.apply(
        lambda x: 1 - x['ms2_precursor_pep'] \
            if x['bmask'] == 1 else x['ms2_precursor_pep'], 
        axis=1
    )
    
    data = pd.merge(data, ms2_peakgroup_table, how='inner', on='feature_id')
    data['prior'] = calculate_ms2_peakgroup_prior(
        data,
        use_glycan_composition=use_glycan_composition
    )
    
    return data \
        [[
            'feature_id', 'num_glycoforms', 
            'prior', 'evidence', 'glycopeptide_id'
        ]] \
        .rename(columns={'glycopeptide_id': 'hypothesis'})
        
        
def prepare_transition_bm(glycoforms, transition_table, 
                          precursor_peakgroup_data, propagate_signal_across_runs, across_run_confidence_threshold):       
    # Propagate peps <= threshold for aligned feature groups across runs
    if propagate_signal_across_runs: 
        evidence = transition_table \
        [['feature_id', 'alignment_group_id', 'transition_id', 'pep']] \
        .drop_duplicates()
        ## Separate out features that need propagation and those that don't to avoid calling apply on the features that don't need propagated peps
        non_prop_data = evidence.loc[ evidence['feature_id']==evidence['alignment_group_id']]
        prop_data = evidence.loc[ evidence['feature_id']!=evidence['alignment_group_id']]
        
        start = time.time()
        # Group by alignment_group_id and apply function in parallel
        data_with_confidence = (
            prop_data.groupby("alignment_group_id", group_keys=False)
            .apply(lambda df: transfer_confident_evidence_across_runs(df, across_run_confidence_threshold, ['feature_id', 'transition_id'], ['pep']))
            .reset_index(drop=True)
        )
        end = time.time()
        click.echo(f"\nInfo: Propagating signal for {len(prop_data['feature_id'].unique())} aligned features of total {len(evidence['feature_id'].unique())} features across runs ... {end-start:.4f} seconds")
        
        ## Concat non prop data with prop data
        evidence = pd.concat([non_prop_data, data_with_confidence], ignore_index=True)
        
        ## Drop alignment_group_id column
        evidence.drop(columns=['alignment_group_id'], inplace=True)
    else:
        evidence = transition_table \
        [['feature_id', 'transition_id', 'pep']] \
        .drop_duplicates()
          
    data = pd.merge(glycoforms, evidence, how='outer', on='feature_id')
    
    bitmask = transition_table \
        [['transition_id', 'glycopeptide_id']] \
        .drop_duplicates()
    bitmask['bmask'] = 1
    data = pd.merge(data, bitmask, how='left', on=['transition_id','glycopeptide_id']) 

    data.dropna(subset=['transition_id'], inplace=True)
    data.fillna(value={'bmask': 0}, inplace=True)
    
    data['evidence'] = data.apply(
        lambda x: 1 - x['pep'] if x['bmask'] == 1 else x['pep'], 
        axis=1
    )
    
    data = pd.merge(
        data, 
        precursor_peakgroup_data[[
            'feature_id', 'glycopeptide_id', 
            'precursor_peakgroup_pep'
        ]], 
        how='inner', 
        on=['feature_id', 'glycopeptide_id']
    )
    data['prior'] = 1 - data['precursor_peakgroup_pep']     
    
    return data \
        [[
            'feature_id', 'num_glycoforms', 
            'prior', 'evidence', 'glycopeptide_id'
        ]] \
        .rename(columns={'glycopeptide_id': 'hypothesis'})
        
        
def infer_glycoforms(infile, outfile, 
                     ms1_precursor_scoring, 
                     ms2_precursor_scoring, 
                     grouped_fdr, 
                     max_precursor_pep, 
                     max_peakgroup_pep, 
                     max_precursor_peakgroup_pep,
                     max_transition_pep,
                     use_glycan_composition,
                     ms1_mz_window=10,
                     ms1_mz_window_unit='ppm',
                     propagate_signal_across_runs=False,
                     max_alignment_pep=0.4,
                     across_run_confidence_threshold=0.5,
                     ):
    click.echo("Info: Starting inference of glycoforms.")
    
    glycoforms = read_glycoform_space(
        infile, 
        use_glycan_composition=use_glycan_composition
    )
    ms2_peakgroup_table = read_ms2_peakgroup(infile, max_peakgroup_pep)
    
    if ms1_precursor_scoring:
        ms1_precursor_table = read_ms1_precursor(
            infile,
            max_precursor_pep=max_precursor_pep
        )
        ms1_precursor_bm_data = prepare_ms1_precursor_bm(
            glycoforms=glycoforms,
            ms1_precursor_table=ms1_precursor_table,
            ms2_peakgroup_table=ms2_peakgroup_table,
            use_glycan_composition=use_glycan_composition,
            ms1_mz_window=ms1_mz_window,
            ms1_mz_window_unit=ms1_mz_window_unit
        )
    
    if ms2_precursor_scoring:        
        ms2_precursor_table = read_ms2_precursor(
            infile,
            max_precursor_pep
        )        
        ms2_precursor_bm_data = prepare_ms2_precursor_bm(
            glycoforms=glycoforms,
            ms2_precursor_table=ms2_precursor_table,
            ms2_peakgroup_table=ms2_peakgroup_table,
            use_glycan_composition=use_glycan_composition
        )
        
    if ms1_precursor_scoring and ms2_precursor_scoring:
        precursor_bm_data = pd.concat([
            ms1_precursor_bm_data, 
            ms2_precursor_bm_data
        ])
    elif ms1_precursor_scoring:
        precursor_bm_data = ms1_precursor_bm_data
    elif ms2_precursor_scoring:
        precursor_bm_data = ms2_precursor_bm_data
    else:
        precursor_bm_data = None
    
    
    if precursor_bm_data is not None:
        prec_pp_data = apply_bm(precursor_bm_data)    
        prec_pp_data['precursor_peakgroup_pep'] = 1 - prec_pp_data['posterior']    
        precursor_peakgroup_data = prec_pp_data \
            .rename(columns={'hypothesis': 'glycopeptide_id'}) \
            [['feature_id', 'glycopeptide_id', 'precursor_peakgroup_pep']]            
    else:
        precursor_peakgroup_data = pd.merge(
            glycoforms, 
            ms2_peakgroup_table,
            how='inner', 
            on='feature_id'
        )
        precursor_peakgroup_data['precursor_peakgroup_pep'] = \
            calculate_ms2_peakgroup_prior(
                precursor_peakgroup_data,
                use_glycan_composition=use_glycan_composition
            )
            
    precursor_peakgroup_data = precursor_peakgroup_data.loc[
        (precursor_peakgroup_data['glycopeptide_id'] == -1) | \
        (precursor_peakgroup_data['precursor_peakgroup_pep'] < \
         max_precursor_peakgroup_pep)
    ]
        
    transition_table = read_transition(
        infile, 
        max_transition_pep
    )
    
    click.echo("Conducting glycoform-level inference ... ")
    ## prepare for propagating signal across runs for aligned features
    if propagate_signal_across_runs:
        across_run_feature_map = get_feature_mapping_across_runs(infile, max_alignment_pep)
        transition_table = pd.merge(transition_table, across_run_feature_map, on='feature_id', how='left')
        ## Fill missing alignment_group_id with feature_id for those that are not aligned
        transition_table["alignment_group_id"] = transition_table["alignment_group_id"].astype(object)
        mask = transition_table["alignment_group_id"].isna()
        transition_table.loc[mask, "alignment_group_id"] = transition_table.loc[mask, "feature_id"].astype(str)
        transition_table = transition_table.astype({'alignment_group_id':'int64'})
    
    transition_bm_data = prepare_transition_bm(
        glycoforms=glycoforms,
        transition_table=transition_table,
        precursor_peakgroup_data=precursor_peakgroup_data, propagate_signal_across_runs=propagate_signal_across_runs, 
        across_run_confidence_threshold=across_run_confidence_threshold
    )
    
    glycoform_pp_data = apply_bm(transition_bm_data)
        
    if use_glycan_composition:
        glycoform_pp_data = pd.merge(
            glycoform_pp_data,
            glycoforms \
                [[
                    'feature_id',  'glycopeptide_id', 
                    'glycan_composition'
                ]] \
                .drop_duplicates(['feature_id',  'glycopeptide_id']) \
                .rename(columns={'glycopeptide_id': 'hypothesis'}),
            how='left'
        )
                
        index = glycoform_pp_data \
            .groupby(['feature_id', 'glycan_composition']) \
            .apply(lambda x: x['posterior'].idxmax())
        
        glycoform_pp_data['posterior'] = glycoform_pp_data \
            .groupby(['feature_id', 'glycan_composition']) \
            ['posterior'] \
            .transform(lambda x: np.sum(x))
        
        glycoform_pp_data = glycoform_pp_data.iloc[index]
        
    glycoform_pp_data['pep'] = 1 - glycoform_pp_data['posterior']

    if grouped_fdr:
        glycoform_pp_data['qvalue'] = pd \
            .merge(
                glycoform_pp_data, 
                transition_bm_data[['feature_id', 'num_glycoforms']] \
                    .drop_duplicates(), 
                on=['feature_id'], 
                how='inner'
            ) \
            .groupby('num_glycoforms')['pep'] \
            .transform(compute_model_fdr)
    else:
        glycoform_pp_data['qvalue'] = \
            compute_model_fdr(glycoform_pp_data['pep'])
    
    click.echo("Info: Storing results.")
    glycoform_data = glycoform_pp_data \
        .rename(columns={'hypothesis': 'glycopeptide_id'}) \
        .merge(
            precursor_peakgroup_data[[
                'feature_id', 
                'glycopeptide_id', 
                'precursor_peakgroup_pep'
            ]] \
            .drop_duplicates(), 
            on=['feature_id', 'glycopeptide_id'], 
            how='inner'
        ) \
        [[
            'feature_id', 
            'glycopeptide_id', 
            'precursor_peakgroup_pep',
            'qvalue', 
            'pep'
        ]]        
    glycoform_data.columns = [x.upper() for x in glycoform_data.columns]
    
    if infile != outfile:
        copyfile(infile, outfile)

    con = sqlite3.connect(outfile)
    glycoform_data.to_sql(
        "SCORE_GLYCOFORM", con, 
        index=False, if_exists='replace'
    )
    con.close()
    
    