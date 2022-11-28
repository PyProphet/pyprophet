import pandas as pd
import numpy as np
import sqlite3
import click
from datetime import datetime

from .data_handling import check_sqlite_table


# Filter a sqMass chromatogram file by given input labels
def filter_chrom_by_labels(infile, outfile, labels):
    conn = sqlite3.connect(infile)
    c = conn.cursor()

    labels = [ "'" + str(l) + "'" for l in labels]
    labels_stmt = get_ids_stmt(labels)

    stmt = "SELECT ID FROM CHROMATOGRAM WHERE NATIVE_ID IN %s" % labels_stmt
    keep_ids = [i[0] for i in list(c.execute(stmt))]
    click.echo("Keep %s chromatograms" % len(keep_ids) )

    nr_chrom = list(c.execute("SELECT COUNT(*) FROM CHROMATOGRAM"))[0][0]
    nr_spec = list(c.execute("SELECT COUNT(*) FROM SPECTRUM"))[0][0]

    assert(nr_chrom > 0)
    assert(nr_spec == 0)

    copy_database(c, conn, outfile, keep_ids)


def copy_table(c, conn, keep_ids, tbl, id_col, omit_tables=[]):
    if tbl not in omit_tables:
        stmt = "CREATE TABLE other.%s AS SELECT * FROM %s WHERE %s IN " % (tbl, tbl, id_col)
        stmt += get_ids_stmt(keep_ids) + ";"
        c.execute(stmt)
        conn.commit()

def create_index_if(c, conn, stmt, tbl, omit_tables=[]):
    if tbl not in omit_tables:
        c.execute(stmt)
        conn.commit()

def copy_database(c, conn, outfile, keep_ids):
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


def filter_sqmass(sqmassfiles, infile, max_precursor_pep, max_peakgroup_pep, max_transition_pep):
    con = sqlite3.connect(infile)

    # process each sqmassfile independently
    for sqm_in in sqmassfiles:
        sqm_out = sqm_in.split(".sqMass")[0] + "_filtered.sqMass"

        if check_sqlite_table(con, 'SCORE_MS1') and check_sqlite_table(con, 'SCORE_MS2') and check_sqlite_table(con, 'SCORE_TRANSITION'):
            transitions = pd.read_sql_query('''
SELECT TRANSITION_ID AS transition_id
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN SCORE_TRANSITION ON FEATURE.ID = SCORE_TRANSITION.FEATURE_ID
INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID
WHERE SCORE_MS1.PEP <= {0}
  AND SCORE_MS2.PEP <= {1}
  AND SCORE_TRANSITION.PEP <= {2}
  AND FILENAME LIKE '%{3}%';
'''.format(max_precursor_pep, max_peakgroup_pep, max_transition_pep, sqm_in.split(".sqMass")[0]), con)['transition_id'].values

        elif check_sqlite_table(con, 'SCORE_MS1') and check_sqlite_table(con, 'SCORE_MS2') and not check_sqlite_table(con, 'SCORE_TRANSITION'):
            transitions = pd.read_sql_query('''
SELECT TRANSITION_ID AS transition_id
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS1 ON FEATURE.ID = SCORE_MS1.FEATURE_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN FEATURE_TRANSITION ON FEATURE.ID = FEATURE_TRANSITION.FEATURE_ID
INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID
WHERE SCORE_MS1.PEP <= {0}
  AND SCORE_MS2.PEP <= {1}
  AND FILENAME LIKE '%{2}%';
'''.format(max_precursor_pep, max_peakgroup_pep, sqm_in.split(".sqMass")[0]), con)['transition_id'].values

        elif not check_sqlite_table(con, 'SCORE_MS1') and check_sqlite_table(con, 'SCORE_MS2') and not check_sqlite_table(con, 'SCORE_TRANSITION'):
            transitions = pd.read_sql_query('''
SELECT TRANSITION_ID AS transition_id
FROM PRECURSOR
INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
INNER JOIN FEATURE_TRANSITION ON FEATURE.ID = FEATURE_TRANSITION.FEATURE_ID
INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID
WHERE SCORE_MS2.PEP <= {0}
  AND FILENAME LIKE '%{1}%';
}
'''.format(max_peakgroup_pep, sqm_in.split(".sqMass")[0]), con)['transition_id'].values
            
        else:
            raise click.ClickException("Conduct scoring on MS1, MS2 and/or transition-level before filtering.")

        filter_chrom_by_labels(sqm_in, sqm_out, transitions)

def filter_osw(oswfiles, remove_decoys=True, omit_tables=[], max_gene_fdr=None, max_protein_fdr=None, max_peptide_fdr=None, max_ms2_fdr=None):

    # process each oswfile independently
    for osw_in in oswfiles:
        osw_out = osw_in.split(".osw")[0] + "_filtered.osw"

        click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Begin filtering {osw_in} to {osw_out}...")

        conn = sqlite3.connect(osw_in)
        c = conn.cursor()

        c.execute("ATTACH DATABASE '%s' AS other;" % osw_out)

        if remove_decoys:
            decoy_query = " AND DECOY=0"
        else:
            decoy_query = ""

        # Table(s) - GENE and SCORE_GENE
        if max_gene_fdr is not None and check_sqlite_table(conn, 'SCORE_GENE'):
            gene_ids = np.unique(list(c.execute(f"SELECT GENE_ID FROM SCORE_GENE INNER JOIN GENE ON GENE.ID = SCORE_GENE.GENE_ID WHERE QVALUE <= {max_gene_fdr} {decoy_query}")))
            click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for {len(gene_ids)} gene ids with gene score q-value <= {max_gene_fdr} with decoy removal = {remove_decoys}...")
            # Copy filtered tables
            copy_table(c, conn, gene_ids, "GENE", "ID", omit_tables)
            copy_table(c, conn, gene_ids, "SCORE_GENE", "GENE_ID", omit_tables)
        else:
            # Copy original full tables
            gene_ids = np.unique(list(c.execute(f"SELECT ID FROM GENE WHERE ID IS NOT NULL {decoy_query}")))            
            if len(gene_ids)!=0:
                copy_table(c, conn, gene_ids, "GENE", "ID", omit_tables)
            elif check_sqlite_table(conn, 'GENE'):
                c.execute('CREATE TABLE other.GENE as SELECT * FROM GENE')
                conn.commit()
            if check_sqlite_table(conn, 'SCORE_GENE'):
                copy_table(c, conn, gene_ids, "SCORE_GENE", "GENE_ID", omit_tables)

        # Table(s) - PROTEIN and SCORE_PROTEIN
        if max_protein_fdr is not None and check_sqlite_table(conn, 'SCORE_PROTEIN'):
            protein_ids = np.unique(list(c.execute(f"SELECT PROTEIN_ID FROM SCORE_PROTEIN INNER JOIN PROTEIN ON PROTEIN.ID = SCORE_PROTEIN.PROTEIN_ID WHERE QVALUE <= {max_protein_fdr} {decoy_query}")))
            click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for {len(protein_ids)} protein ids with protein score q-value <= {max_protein_fdr} with decoy removal = {remove_decoys}...")
            # Copy filtered tables
            copy_table(c, conn, protein_ids, "PROTEIN", "ID", omit_tables)
            copy_table(c, conn, protein_ids, "SCORE_PROTEIN", "PROTEIN_ID", omit_tables)
        else:
            # Copy original full tables
            protein_ids = np.unique(list(c.execute(f"SELECT ID FROM PROTEIN WHERE ID IS NOT NULL {decoy_query}")))            
            copy_table(c, conn, protein_ids, "PROTEIN", "ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_PROTEIN'):
                copy_table(c, conn, protein_ids, "SCORE_PROTEIN", "PROTEIN_ID", omit_tables)

        # Table(s) - PEPTIDE, SCORE_PEPTIDE and PEPTIDE_XXXX_MAPPING
        if max_peptide_fdr is not None and check_sqlite_table(conn, 'SCORE_PEPTIDE'):
            peptide_ids = np.unique(list(c.execute(f"SELECT PEPTIDE_ID FROM SCORE_PEPTIDE INNER JOIN PEPTIDE ON PEPTIDE.ID = SCORE_PEPTIDE.PEPTIDE_ID WHERE QVALUE <= {max_peptide_fdr} {decoy_query}")))
            click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for {len(peptide_ids)} peptide ids with peptide score q-value <= {max_peptide_fdr} with decoy removal = {remove_decoys}...")
            # Copy filtered tables
            copy_table(c, conn, peptide_ids, "PEPTIDE", "ID", omit_tables)
            copy_table(c, conn, peptide_ids, "SCORE_PEPTIDE", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PRECURSOR_PEPTIDE_MAPPING", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PEPTIDE_PROTEIN_MAPPING", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PEPTIDE_GENE_MAPPING", "PEPTIDE_ID", omit_tables)
        else:
            # Copy original full tables
            peptide_ids = np.unique(list(c.execute(f"SELECT ID FROM PEPTIDE WHERE ID IS NOT NULL {decoy_query}")))
            copy_table(c, conn, peptide_ids, "PEPTIDE", "ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PRECURSOR_PEPTIDE_MAPPING", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PEPTIDE_PROTEIN_MAPPING", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PEPTIDE_GENE_MAPPING", "PEPTIDE_ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_PEPTIDE'):
                copy_table(c, conn, peptide_ids, "SCORE_PEPTIDE", "PEPTIDE_ID", omit_tables)
        
        # Table(s) - SCORE_MS2, FEATURE, FEATURE_MS1,  FEATURE_MS2, FEATURE_TRANSITION, PRECURSOR
        if max_ms2_fdr is not None and check_sqlite_table(conn, 'SCORE_MS2'):
            feature_precursor_ids = np.array(list(c.execute(f"SELECT FEATURE_ID, PRECURSOR_ID FROM SCORE_MS2 INNER JOIN (SELECT FEATURE.ID, PRECURSOR_ID FROM FEATURE INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID WHERE PRECURSOR.ID IS NOT NULL {decoy_query}) AS FEATURE ON FEATURE.ID = SCORE_MS2.FEATURE_ID WHERE QVALUE <= {max_peptide_fdr}")))
            feature_ids = np.unique(feature_precursor_ids[:,0])
            precursor_ids = np.unique(feature_precursor_ids[:,1])
            click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for {len(feature_ids)} feature ids across {len(precursor_ids)} unique precursor ids with ms2 score q-value <= {max_ms2_fdr} with decoy removal = {remove_decoys}...")
            # Copy filtered tables
            copy_table(c, conn, feature_ids, "FEATURE", "ID", omit_tables)
            if check_sqlite_table(conn, 'FEATURE_MS1'):
                copy_table(c, conn, feature_ids, "FEATURE_MS1", "FEATURE_ID", omit_tables)
            copy_table(c, conn, feature_ids, "FEATURE_MS2", "FEATURE_ID", omit_tables)
            if check_sqlite_table(conn, 'FEATURE_TRANSITION'):
                copy_table(c, conn, feature_ids, "FEATURE_TRANSITION", "FEATURE_ID", omit_tables)
            copy_table(c, conn, feature_ids, "SCORE_MS2", "FEATURE_ID", omit_tables)
            copy_table(c, conn, precursor_ids, "PRECURSOR", "ID", omit_tables)
        else:
            # Copy original full tables
            feature_precursor_ids = np.array(list(c.execute(f"SELECT ID, PRECURSOR_ID FROM (SELECT FEATURE.ID, PRECURSOR_ID FROM FEATURE INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID WHERE PRECURSOR.ID IS NOT NULL {decoy_query}) AS FEATURE")))
            feature_ids = np.unique(feature_precursor_ids[:,0])
            precursor_ids = np.unique(feature_precursor_ids[:,1])
            # Copy original full tables
            copy_table(c, conn, feature_ids, "FEATURE", "ID", omit_tables)
            if check_sqlite_table(conn, 'FEATURE_MS1'):
                copy_table(c, conn, feature_ids, "FEATURE_MS1", "ID", omit_tables)
            copy_table(c, conn, feature_ids, "FEATURE_MS2", "ID", omit_tables)
            if check_sqlite_table(conn, 'FEATURE_TRANSITION'):
                copy_table(c, conn, feature_ids, "FEATURE_TRANSITION", "FEATURE_ID", omit_tables)
            copy_table(c, conn, precursor_ids, "PRECURSOR", "ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_MS2'):
                copy_table(c, conn, feature_ids, "SCORE_MS2", "FEATURE_ID", omit_tables)
        
        # Table(s) - TRANSITION, TRANSITION_PRECURSOR_MAPPING, TRANSITION_PEPTIDE_MAPPING
        transition_ids = np.unique(list(c.execute(f"SELECT ID FROM TRANSITION LEFT JOIN (SELECT * FROM TRANSITION_PRECURSOR_MAPPING WHERE PRECURSOR_ID IN {tuple(precursor_ids)}) AS TRANSITION_PRECURSOR_MAPPING ON TRANSITION.ID = TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID LEFT JOIN (SELECT * FROM TRANSITION_PEPTIDE_MAPPING WHERE PEPTIDE_ID IN {tuple(peptide_ids)}) AS TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID")))
        # Ensure there are ids to filter for
        assert (len(transition_ids) >0), "There seems to be no transition ids to retain after filtering..."
        click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for  {len(transition_ids)} transition ids for {len(peptide_ids)} peptides ids and {len(precursor_ids)} precursor ids...")
        # Copy filtered tables
        copy_table(c, conn, transition_ids, "TRANSITION", "ID", omit_tables)
        copy_table(c, conn, transition_ids, "TRANSITION_PRECURSOR_MAPPING", "TRANSITION_ID", omit_tables)
        copy_table(c, conn, transition_ids, "TRANSITION_PEPTIDE_MAPPING", "TRANSITION_ID", omit_tables)
        if check_sqlite_table(conn, 'SCORE_TRANSITION'):
            copy_table(c, conn, transition_ids, "SCORE_TRANSITION", "TRANSITION_ID", omit_tables)

        # Table(s) - RUN, VERSION
        c.execute('CREATE TABLE other.RUN as SELECT * FROM RUN')
        c.execute('CREATE TABLE other.VERSION as SELECT * FROM VERSION')
        conn.commit()

        # Create Indexes
        create_index_if(c, conn, "CREATE INDEX other.idx_feature_feature_id ON FEATURE (ID);", "FEATURE", omit_tables) 
        create_index_if(c, conn, "CREATE INDEX other.idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);", "FEATURE_MS1", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);", "FEATURE_MS2", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);", "FEATURE", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_feature_run_id ON FEATURE (RUN_ID);", "FEATURE", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);", "FEATURE_TRANSITION", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);", "FEATURE_TRANSITION", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_peptide_peptide_id ON PEPTIDE (ID);", "PEPTIDE", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_peptide_protein_mapping_peptide_id ON PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID);", "PEPTIDE_PROTEIN_MAPPING", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_precursor_peptide_mapping_peptide_id ON PRECURSOR_PEPTIDE_MAPPING (PEPTIDE_ID);", "PRECURSOR_PEPTIDE_MAPPING", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_precursor_peptide_mapping_precursor_id ON PRECURSOR_PEPTIDE_MAPPING (PRECURSOR_ID);", "PRECURSOR_PEPTIDE_MAPPING", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_precursor_precursor_id ON PRECURSOR (ID);", "PRECURSOR", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_protein_protein_id ON PROTEIN (ID);", "PROTEIN", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_run_run_id ON RUN (ID)", "RUN", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);", "SCORE_MS2", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_score_protein_protein_id ON SCORE_PROTEIN (PROTEIN_ID);", "SCORE_PROTEIN", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_score_transition_feature_id ON SCORE_TRANSITION (FEATURE_ID);", "SCORE_TRANSITION", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_score_transition_transition_id ON SCORE_TRANSITION (TRANSITION_ID);", "SCORE_TRANSITION", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_transition_id ON TRANSITION (ID);", "TRANSITION", omit_tables)  
        create_index_if(c, conn, "CREATE INDEX other.idx_transition_peptide_mapping_transition_id ON TRANSITION_PEPTIDE_MAPPING (TRANSITION_ID);", "TRANSITION_PEPTIDE_MAPPING", omit_tables)  

        # Detach and close
        c.execute("DETACH DATABASE 'other';")
        conn.close()

        click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Finished filtering {osw_in} to {osw_out}...")