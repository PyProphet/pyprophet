import pandas as pd
import numpy as np
import sqlite3
import click
from datetime import datetime

from .data_handling import check_sqlite_table


# Filter a sqMass chromatogram file by given input labels
def filter_chrom_by_labels(infile, outfile, labels):
    if len(labels) == 0:
        raise click.ClickException("No transition ids to filter chromatograms, try adjust the filtering criteria.")
    
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


def copy_table(c, conn, keep_ids, tbl, id_col, omit_tables=[], extra_id_col=None, extra_keep_ids=None):
    if tbl not in omit_tables:
        stmt = "CREATE TABLE other.%s AS SELECT * FROM %s WHERE %s IN " % (tbl, tbl, id_col)
        stmt += get_ids_stmt(keep_ids) 
        if extra_id_col is not None and extra_keep_ids is not None:
            stmt += " AND %s IN " % extra_id_col
            stmt += get_ids_stmt(extra_keep_ids)
            if extra_id_col == "RUN_ID":
                stmt += " OR %s IS NULL" % extra_id_col
        stmt += ";"
        
        c.execute(stmt)
        conn.commit()

def create_index_if(c, conn, stmt, tbl, omit_tables=[]):
    if tbl not in omit_tables and check_sqlite_table(conn, tbl):
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


def filter_sqmass(sqmassfiles, infile=None, max_precursor_pep=0.7, max_peakgroup_pep=0.7, max_transition_pep=0.7, keep_naked_peptides=[], remove_decoys=True):
    if infile is not None:
        con = sqlite3.connect(infile)
        
    if remove_decoys:
        decoy_query = " AND DECOY=0"
    else:
        decoy_query = ""

    # process each sqmassfile independently
    for sqm_in in sqmassfiles:
        click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Begin filtering {sqm_in}...")
        sqm_out = sqm_in.split(".sqMass")[0] + "_filtered.sqMass"

        if infile is not None:
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
    AND FILENAME LIKE '%{3}%'
    {4};
    '''.format(max_precursor_pep, max_peakgroup_pep, max_transition_pep, sqm_in.split(".sqMass")[0], decoy_query), con)['transition_id'].values

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
    AND FILENAME LIKE '%{2}%'
    {3};
    '''.format(max_precursor_pep, max_peakgroup_pep, sqm_in.split(".sqMass")[0], decoy_query), con)['transition_id'].values

            elif not check_sqlite_table(con, 'SCORE_MS1') and check_sqlite_table(con, 'SCORE_MS2') and not check_sqlite_table(con, 'SCORE_TRANSITION'):
                transitions = pd.read_sql_query('''
    SELECT TRANSITION_ID AS transition_id
    FROM PRECURSOR
    INNER JOIN FEATURE ON PRECURSOR.ID = FEATURE.PRECURSOR_ID
    INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
    INNER JOIN FEATURE_TRANSITION ON FEATURE.ID = FEATURE_TRANSITION.FEATURE_ID
    INNER JOIN RUN ON FEATURE.RUN_ID = RUN.ID
    WHERE SCORE_MS2.PEP <= {0}
    AND FILENAME LIKE '%{1}%'
    {2};
    '''.format(max_peakgroup_pep, sqm_in.split(".sqMass")[0], decoy_query), con)['transition_id'].values
            else:
                raise click.ClickException("Conduct scoring on MS1, MS2 and/or transition-level before filtering.")
            
        elif len(keep_naked_peptides) != 0:
            con = sqlite3.connect(sqm_in)
            transitions = pd.read_sql_query(f'''
    SELECT NATIVE_ID
    FROM CHROMATOGRAM
    INNER JOIN PRECURSOR ON PRECURSOR.CHROMATOGRAM_ID = CHROMATOGRAM.ID
    WHERE PRECURSOR.PEPTIDE_SEQUENCE IN ('{"','".join(keep_naked_peptides)}') ''', con)['NATIVE_ID'].values
            con.close()
        else:
            raise click.ClickException("Please provide either an associated OSW file to filter based on scoring or a list of peptides to keep.")

        filter_chrom_by_labels(sqm_in, sqm_out, transitions)

def filter_osw(oswfiles, remove_decoys=True, omit_tables=[], max_gene_fdr=None, max_protein_fdr=None, max_peptide_fdr=None, max_ms2_fdr=None, keep_naked_peptides=[], run_ids=[]):

    print("Filtering OSW files...")
    print("Parameters:")
    print("  - remove_decoys: %s" % remove_decoys)
    print("  - omit_tables: %s" % omit_tables)
    print("  - max_gene_fdr: %s" % max_gene_fdr)
    print("  - max_protein_fdr: %s" % max_protein_fdr)
    print("  - max_peptide_fdr: %s" % max_peptide_fdr)
    print("  - max_ms2_fdr: %s" % max_ms2_fdr)
    print("  - keep_naked_peptides: %s" % keep_naked_peptides)
    print("  - run_ids: %s" % run_ids)
        
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

        # Generate gene, protein, peptide, precursor and transition id table for specific filters
        # keep_peptides is a list of strings of peptide sequences to keep
        if len(keep_naked_peptides) != 0:

            keep_peptides_ids = np.unique(list(c.execute(f"""SELECT ID FROM PEPTIDE WHERE UNMODIFIED_SEQUENCE IN ('{"','".join(keep_naked_peptides)}')""")))
            assert (len(keep_peptides_ids) >0), "There seems to be no peptides in the UNMODIFIED_SEQUENCE column in the PEPTIDE table matching the peptides in the keep_naked_peptides list... {keep_naked_peptides}"
            keep_peptide_ids_stmt = get_ids_stmt(keep_peptides_ids)

            keep_precursor_ids = np.unique(list(c.execute(f"SELECT PRECURSOR_ID FROM PRECURSOR_PEPTIDE_MAPPING WHERE PEPTIDE_ID IN {keep_peptide_ids_stmt}")))
            keep_precursor_ids_stmt = get_ids_stmt(keep_precursor_ids)

            keep_transition_ids = np.unique(list(c.execute(f"SELECT TRANSITION_ID FROM TRANSITION_PRECURSOR_MAPPING WHERE PRECURSOR_ID IN {keep_precursor_ids_stmt}")))

            if check_sqlite_table(conn, 'TRANSITION_PEPTIDE_MAPPING'):
                keep_transition_pep_ids = np.unique(list(c.execute(f"SELECT TRANSITION_ID FROM TRANSITION_PEPTIDE_MAPPING WHERE PEPTIDE_ID IN {keep_peptide_ids_stmt}")))
                keep_transition_ids = np.hstack((keep_transition_ids, keep_transition_pep_ids))

            keep_protein_ids = np.unique(list(c.execute(f"SELECT PROTEIN_ID FROM PEPTIDE_PROTEIN_MAPPING WHERE PEPTIDE_ID IN {keep_peptide_ids_stmt}")))

            if check_sqlite_table(conn, 'GENE'):
                keep_gene_ids = np.unique(list(c.execute(f"SELECT GENE_ID FROM PEPTIDE_GENE_MAPPING WHERE PEPTIDE_ID IN {keep_peptide_ids_stmt}")))
        else:
            keep_peptides_ids = None
            keep_precursor_ids = None
            keep_transition_ids = None
            keep_protein_ids = None
            if check_sqlite_table(conn, 'GENE'):
                keep_gene_ids = None
            
        if len(run_ids) != 0:
            keep_feature_ids = np.unique(list(c.execute(f"""SELECT ID FROM FEATURE WHERE RUN_ID IN ('{"','".join(run_ids)}')""")))
            extra_run_id_col = "RUN_ID"
            keep_run_ids = run_ids
        else:
            keep_feature_ids = None
            extra_run_id_col = None
            keep_run_ids = None
        
        # Table(s) - GENE and SCORE_GENE
        if max_gene_fdr is not None and check_sqlite_table(conn, 'SCORE_GENE'):
            gene_ids = np.unique(list(c.execute(f"SELECT GENE_ID FROM SCORE_GENE INNER JOIN GENE ON GENE.ID = SCORE_GENE.GENE_ID WHERE QVALUE <= {max_gene_fdr} {decoy_query}")))
            # Further reduce gene_ids only if gene_ids is also in keep_gene_ids
            if keep_gene_ids is not None:
                gene_ids = np.intersect1d(gene_ids, keep_gene_ids)
            click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for {len(gene_ids)} gene ids with gene score q-value <= {max_gene_fdr} with decoy removal = {remove_decoys}...")
            # Copy filtered tables
            copy_table(c, conn, gene_ids, "GENE", "ID", omit_tables)
            copy_table(c, conn, gene_ids, "SCORE_GENE", "GENE_ID", omit_tables, extra_run_id_col, keep_run_ids)
        else:
            if check_sqlite_table(conn, 'GENE'):
                # Copy original full tables
                gene_ids = np.unique(list(c.execute(f"SELECT ID FROM GENE WHERE ID IS NOT NULL {decoy_query}")))  
                # Further reduce gene_ids only if gene_ids is also in keep_gene_ids
                if keep_gene_ids is not None:
                    gene_ids = np.intersect1d(gene_ids, keep_gene_ids)          
                if len(gene_ids)!=0:
                    copy_table(c, conn, gene_ids, "GENE", "ID", omit_tables)
                elif check_sqlite_table(conn, 'GENE'):
                    c.execute('CREATE TABLE other.GENE as SELECT * FROM GENE')
                    conn.commit()
            if check_sqlite_table(conn, 'SCORE_GENE'):
                copy_table(c, conn, gene_ids, "SCORE_GENE", "GENE_ID", omit_tables, extra_run_id_col, keep_run_ids)

        # Table(s) - PROTEIN and SCORE_PROTEIN
        if max_protein_fdr is not None and check_sqlite_table(conn, 'SCORE_PROTEIN'):
            protein_ids = np.unique(list(c.execute(f"SELECT PROTEIN_ID FROM SCORE_PROTEIN INNER JOIN PROTEIN ON PROTEIN.ID = SCORE_PROTEIN.PROTEIN_ID WHERE QVALUE <= {max_protein_fdr} {decoy_query}")))
            # Further reduce protein_ids only if protein_ids is also in keep_protein_ids
            if keep_protein_ids is not None:
                protein_ids = np.intersect1d(protein_ids, keep_protein_ids)
            click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for {len(protein_ids)} protein ids with protein score q-value <= {max_protein_fdr} with decoy removal = {remove_decoys}...")
            # Copy filtered tables
            copy_table(c, conn, protein_ids, "PROTEIN", "ID", omit_tables)
            copy_table(c, conn, protein_ids, "SCORE_PROTEIN", "PROTEIN_ID", omit_tables, extra_run_id_col, keep_run_ids)
        else:
            # Copy original full tables
            protein_ids = np.unique(list(c.execute(f"SELECT ID FROM PROTEIN WHERE ID IS NOT NULL {decoy_query}")))          
            # Further reduce protein_ids only if protein_ids is also in keep_protein_ids
            if keep_protein_ids is not None:
                protein_ids = np.intersect1d(protein_ids, keep_protein_ids)
            copy_table(c, conn, protein_ids, "PROTEIN", "ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_PROTEIN'):
                copy_table(c, conn, protein_ids, "SCORE_PROTEIN", "PROTEIN_ID", omit_tables, extra_run_id_col, keep_run_ids)

        # Table(s) - PEPTIDE, SCORE_PEPTIDE and PEPTIDE_XXXX_MAPPING
        if max_peptide_fdr is not None and check_sqlite_table(conn, 'SCORE_PEPTIDE'):
            peptide_ids = np.unique(list(c.execute(f"SELECT PEPTIDE_ID FROM SCORE_PEPTIDE INNER JOIN PEPTIDE ON PEPTIDE.ID = SCORE_PEPTIDE.PEPTIDE_ID WHERE QVALUE <= {max_peptide_fdr} {decoy_query}")))
            # Further reduce peptide_ids only if peptide_ids is also in keep_peptides_ids
            if keep_peptides_ids is not None:
                peptide_ids = np.intersect1d(peptide_ids, keep_peptides_ids)
            click.echo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Filtering for {len(peptide_ids)} peptide ids with peptide score q-value <= {max_peptide_fdr} with decoy removal = {remove_decoys}...")
            # Copy filtered tables
            copy_table(c, conn, peptide_ids, "PEPTIDE", "ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_IPF'):
                copy_table(c, conn, peptide_ids, "SCORE_IPF", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "SCORE_PEPTIDE", "PEPTIDE_ID", omit_tables, extra_run_id_col, keep_run_ids)
            copy_table(c, conn, peptide_ids, "PRECURSOR_PEPTIDE_MAPPING", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PEPTIDE_PROTEIN_MAPPING", "PEPTIDE_ID", omit_tables)
            if check_sqlite_table(conn, 'PEPTIDE_GENE_MAPPING'):
                copy_table(c, conn, peptide_ids, "PEPTIDE_GENE_MAPPING", "PEPTIDE_ID", omit_tables)
        else:
            # Copy original full tables
            peptide_ids = np.unique(list(c.execute(f"SELECT ID FROM PEPTIDE WHERE ID IS NOT NULL {decoy_query}")))
            # Further reduce peptide_ids only if peptide_ids is also in keep_peptides_ids
            if keep_peptides_ids is not None:
                peptide_ids = np.intersect1d(peptide_ids, keep_peptides_ids)
            copy_table(c, conn, peptide_ids, "PEPTIDE", "ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PRECURSOR_PEPTIDE_MAPPING", "PEPTIDE_ID", omit_tables)
            copy_table(c, conn, peptide_ids, "PEPTIDE_PROTEIN_MAPPING", "PEPTIDE_ID", omit_tables)
            if check_sqlite_table(conn, 'PEPTIDE_GENE_MAPPING'):
                copy_table(c, conn, peptide_ids, "PEPTIDE_GENE_MAPPING", "PEPTIDE_ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_IPF'):
                copy_table(c, conn, peptide_ids, "SCORE_IPF", "PEPTIDE_ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_PEPTIDE'):
                copy_table(c, conn, peptide_ids, "SCORE_PEPTIDE", "PEPTIDE_ID", omit_tables, extra_run_id_col, keep_run_ids)
        
        # Table(s) - SCORE_MS2, FEATURE, FEATURE_MS1,  FEATURE_MS2, FEATURE_TRANSITION, PRECURSOR
        if max_ms2_fdr is not None and check_sqlite_table(conn, 'SCORE_MS2'):
            feature_precursor_ids = np.array(list(c.execute(f"SELECT FEATURE_ID, PRECURSOR_ID FROM SCORE_MS2 INNER JOIN (SELECT FEATURE.ID, PRECURSOR_ID FROM FEATURE INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID WHERE PRECURSOR.ID IS NOT NULL {decoy_query}) AS FEATURE ON FEATURE.ID = SCORE_MS2.FEATURE_ID WHERE QVALUE <= {max_ms2_fdr}")))
            feature_ids = np.unique(feature_precursor_ids[:,0])
            precursor_ids = np.unique(feature_precursor_ids[:,1])
            # Further reduce precursor_ids only if precursor_ids is also in keep_precursor_ids
            if keep_precursor_ids is not None:
                precursor_ids = np.intersect1d(precursor_ids, keep_precursor_ids)
                # further reduce feature_ids
                feature_ids = np.unique(list(c.execute(f"SELECT ID FROM FEATURE WHERE PRECURSOR_ID IN {get_ids_stmt(precursor_ids)}")))
            # Further reduce feature_ids only if feature_ids is also in keep_feature_ids
            if keep_feature_ids is not None:
                feature_ids = np.intersect1d(feature_ids, keep_feature_ids)
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
            # Further reduce precursor_ids only if precursor_ids is also in keep_precursor_ids
            if keep_precursor_ids is not None:
                precursor_ids = np.intersect1d(precursor_ids, keep_precursor_ids)
                # further reduce feature_ids
                feature_ids = np.unique(list(c.execute(f"SELECT ID FROM FEATURE WHERE PRECURSOR_ID IN {get_ids_stmt(precursor_ids)}")))
            # Further reduce feature_ids only if feature_ids is also in keep_feature_ids
            if keep_feature_ids is not None:
                feature_ids = np.intersect1d(feature_ids, keep_feature_ids)
            # Copy original full tables
            copy_table(c, conn, feature_ids, "FEATURE", "ID", omit_tables)
            if check_sqlite_table(conn, 'FEATURE_MS1'):
                copy_table(c, conn, feature_ids, "FEATURE_MS1", "FEATURE_ID", omit_tables)
            copy_table(c, conn, feature_ids, "FEATURE_MS2", "FEATURE_ID", omit_tables)
            if check_sqlite_table(conn, 'FEATURE_TRANSITION'):
                copy_table(c, conn, feature_ids, "FEATURE_TRANSITION", "FEATURE_ID", omit_tables)
            copy_table(c, conn, precursor_ids, "PRECURSOR", "ID", omit_tables)
            if check_sqlite_table(conn, 'SCORE_MS2'):
                copy_table(c, conn, feature_ids, "SCORE_MS2", "FEATURE_ID", omit_tables)
        
        # Table(s) - TRANSITION, TRANSITION_PRECURSOR_MAPPING, TRANSITION_PEPTIDE_MAPPING
        transition_ids = np.unique(list(c.execute(f"SELECT ID FROM TRANSITION LEFT JOIN (SELECT * FROM TRANSITION_PRECURSOR_MAPPING WHERE PRECURSOR_ID IN {tuple(precursor_ids)}) AS TRANSITION_PRECURSOR_MAPPING ON TRANSITION.ID = TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID LEFT JOIN (SELECT * FROM TRANSITION_PEPTIDE_MAPPING WHERE PEPTIDE_ID IN {tuple(peptide_ids)}) AS TRANSITION_PEPTIDE_MAPPING ON TRANSITION.ID = TRANSITION_PEPTIDE_MAPPING.TRANSITION_ID")))
        # Further reduce transition_ids only if transition_ids is also in keep_transition_ids
        if keep_transition_ids is not None:
            transition_ids = np.intersect1d(transition_ids, keep_transition_ids)
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
        if len(run_ids) != 0:
            run_query = f""" WHERE ID IN ('{"','".join(run_ids)}')"""
        else:
            run_query = ""
        c.execute(f'CREATE TABLE other.RUN as SELECT * FROM RUN {run_query}')
        if check_sqlite_table(conn, 'VERSION'):
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