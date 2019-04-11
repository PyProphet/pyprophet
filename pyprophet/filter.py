import pandas as pd
import sqlite3
import click

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


def copy_table(c, conn, keep_ids, tbl, id_col):
    stmt = "CREATE TABLE other.%s AS SELECT * FROM %s WHERE %s IN " % (tbl, tbl, id_col)
    stmt += get_ids_stmt(keep_ids) + ";"
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

