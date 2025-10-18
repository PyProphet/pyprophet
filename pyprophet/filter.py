import duckdb
import numpy as np
import click
from datetime import datetime

from .io.util import check_sqlite_table


def check_sqlite_table_duckdb(con, table_name):
    """Check if a table exists in a DuckDB-attached SQLite database."""
    try:
        result = con.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchone()
        return result is not None
    except:
        return False


def filter_osw(
    oswfiles,
    remove_decoys=True,
    omit_tables=[],
    max_gene_fdr=None,
    max_protein_fdr=None,
    max_peptide_fdr=None,
    max_ms2_fdr=None,
    keep_naked_peptides=[],
    run_ids=[],
):
    """Filter OSW files using DuckDB for better performance."""

    print("Filtering OSW files with DuckDB...")
    print("Parameters:")
    print("  - remove_decoys: %s" % remove_decoys)
    print("  - max_gene_fdr: %s" % max_gene_fdr)
    print("  - max_protein_fdr: %s" % max_protein_fdr)
    print("  - max_peptide_fdr: %s" % max_peptide_fdr)
    print("  - max_ms2_fdr: %s" % max_ms2_fdr)

    # Process each OSW file
    for osw_in in oswfiles:
        osw_out = osw_in.split(".osw")[0] + "_filtered.osw"

        click.echo(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Begin filtering {osw_in} to {osw_out}..."
        )

        # Create DuckDB connection
        con = duckdb.connect()
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")

        # Attach input and output databases
        con.execute(f"ATTACH '{osw_in}' AS src (TYPE SQLITE)")
        con.execute(f"ATTACH '{osw_out}' AS dst (TYPE SQLITE)")

        # Build filters - use proper WHERE clause format
        decoy_where = "AND DECOY=0" if remove_decoys else ""
        run_where = (
            f"AND RUN_ID IN ({','.join([str(r) for r in run_ids])})" if run_ids else ""
        )

        # Peptide sequence filter
        if keep_naked_peptides:
            peptide_seq_where = (
                f"AND UNMODIFIED_SEQUENCE IN ('"
                + "','".join(keep_naked_peptides)
                + "')"
            )
        else:
            peptide_seq_where = ""

        # === GENE LEVEL ===
        if max_gene_fdr is not None and check_sqlite_table_duckdb(
            con, "src.SCORE_GENE"
        ):
            click.echo("  Filtering GENE level...")

            con.execute(f"""
                CREATE TABLE dst.GENE AS
                SELECT DISTINCT g.*
                FROM src.GENE g
                INNER JOIN src.SCORE_GENE sg ON g.ID = sg.GENE_ID
                WHERE sg.QVALUE <= {max_gene_fdr}
                {peptide_seq_where}
            """)

            con.execute(f"""
                CREATE TABLE dst.SCORE_GENE AS
                SELECT sg.*
                FROM src.SCORE_GENE sg
                INNER JOIN dst.GENE g ON sg.GENE_ID = g.ID
                WHERE sg.QVALUE <= {max_gene_fdr} 
                {run_where.replace("RUN_ID", "sg.RUN_ID") if run_ids else ""}
            """)
        elif check_sqlite_table_duckdb(con, "src.GENE"):
            con.execute("CREATE TABLE dst.GENE AS SELECT * FROM src.GENE")
            if check_sqlite_table_duckdb(con, "src.SCORE_GENE"):
                con.execute(
                    "CREATE TABLE dst.SCORE_GENE AS SELECT * FROM src.SCORE_GENE"
                )

        # === PROTEIN LEVEL ===
        if max_protein_fdr is not None:
            click.echo("  Filtering PROTEIN level...")

            con.execute(f"""
                CREATE TABLE dst.PROTEIN AS
                SELECT DISTINCT p.*
                FROM src.PROTEIN p
                INNER JOIN src.SCORE_PROTEIN sp ON p.ID = sp.PROTEIN_ID
                WHERE sp.QVALUE <= {max_protein_fdr}
                {decoy_where}
            """)

            if check_sqlite_table_duckdb(con, "src.SCORE_PROTEIN"):
                con.execute(f"""
                    CREATE TABLE dst.SCORE_PROTEIN AS
                    SELECT sp.*
                    FROM src.SCORE_PROTEIN sp
                    INNER JOIN dst.PROTEIN p ON sp.PROTEIN_ID = p.ID
                    WHERE sp.QVALUE <= {max_protein_fdr}
                    {run_where.replace("RUN_ID", "sp.RUN_ID") if run_ids else ""}
                """)
        else:
            con.execute(f"""
                CREATE TABLE dst.PROTEIN AS
                SELECT * FROM src.PROTEIN
                WHERE 1=1 {decoy_where}
            """)
            if check_sqlite_table_duckdb(con, "src.SCORE_PROTEIN"):
                con.execute(
                    "CREATE TABLE dst.SCORE_PROTEIN AS SELECT * FROM src.SCORE_PROTEIN"
                )

        # === PEPTIDE LEVEL ===
        if max_peptide_fdr is not None:
            click.echo("  Filtering PEPTIDE level...")

            con.execute(f"""
                CREATE TABLE dst.PEPTIDE AS
                SELECT DISTINCT p.*
                FROM src.PEPTIDE p
                INNER JOIN src.SCORE_PEPTIDE sp ON p.ID = sp.PEPTIDE_ID
                WHERE sp.QVALUE <= {max_peptide_fdr}
                {peptide_seq_where}
                {decoy_where}
            """)

            if check_sqlite_table_duckdb(con, "src.SCORE_PEPTIDE"):
                con.execute(f"""
                    CREATE TABLE dst.SCORE_PEPTIDE AS
                    SELECT sp.*
                    FROM src.SCORE_PEPTIDE sp
                    INNER JOIN dst.PEPTIDE p ON sp.PEPTIDE_ID = p.ID
                    WHERE sp.QVALUE <= {max_peptide_fdr}
                    {run_where.replace("RUN_ID", "sp.RUN_ID") if run_ids else ""}
                """)
        else:
            con.execute(f"""
                CREATE TABLE dst.PEPTIDE AS
                SELECT * FROM src.PEPTIDE
                WHERE 1=1 {peptide_seq_where} {decoy_where}
            """)
            if check_sqlite_table_duckdb(con, "src.SCORE_PEPTIDE"):
                con.execute(
                    "CREATE TABLE dst.SCORE_PEPTIDE AS SELECT * FROM src.SCORE_PEPTIDE"
                )

        # === PRECURSOR/FEATURE LEVEL ===
        if max_ms2_fdr is not None:
            click.echo("  Filtering PRECURSOR/FEATURE level...")

            con.execute(f"""
                CREATE TABLE dst.FEATURE AS
                SELECT DISTINCT f.*
                FROM src.FEATURE f
                INNER JOIN src.SCORE_MS2 sm ON f.ID = sm.FEATURE_ID
                WHERE sm.QVALUE <= {max_ms2_fdr}
                {run_where.replace("RUN_ID", "f.RUN_ID")}
            """)

            con.execute(f"""
                CREATE TABLE dst.PRECURSOR AS
                SELECT DISTINCT pr.*
                FROM src.PRECURSOR pr
                INNER JOIN dst.FEATURE f ON pr.ID = f.PRECURSOR_ID
                WHERE 1=1 {decoy_where}
            """)

            if check_sqlite_table_duckdb(con, "src.SCORE_MS2"):
                con.execute(f"""
                    CREATE TABLE dst.SCORE_MS2 AS
                    SELECT sm.*
                    FROM src.SCORE_MS2 sm
                    INNER JOIN dst.FEATURE f ON sm.FEATURE_ID = f.ID
                    WHERE sm.QVALUE <= {max_ms2_fdr}
                """)

            if check_sqlite_table_duckdb(con, "src.SCORE_MS1"):
                con.execute(f"""
                    CREATE TABLE dst.SCORE_MS1 AS
                    SELECT sm.*
                    FROM src.SCORE_MS1 sm
                    INNER JOIN dst.FEATURE f ON sm.FEATURE_ID = f.ID
                """)
        else:
            con.execute(f"""
                CREATE TABLE dst.FEATURE AS
                SELECT * FROM src.FEATURE
                WHERE 1=1 {run_where.replace("RUN_ID", "RUN_ID")}
            """)
            con.execute(f"""
                CREATE TABLE dst.PRECURSOR AS
                SELECT pr.*
                FROM src.PRECURSOR pr
                WHERE EXISTS (
                    SELECT 1 FROM dst.FEATURE f WHERE f.PRECURSOR_ID = pr.ID
                ) {decoy_where}
            """)
            if check_sqlite_table_duckdb(con, "src.SCORE_MS2"):
                con.execute(
                    "CREATE TABLE dst.SCORE_MS2 AS SELECT sm.* FROM src.SCORE_MS2 sm INNER JOIN dst.FEATURE f ON sm.FEATURE_ID = f.ID"
                )
            if check_sqlite_table_duckdb(con, "src.SCORE_MS1"):
                con.execute(
                    "CREATE TABLE dst.SCORE_MS1 AS SELECT sm.* FROM src.SCORE_MS1 sm INNER JOIN dst.FEATURE f ON sm.FEATURE_ID = f.ID"
                )

        # === TRANSITION LEVEL ===
        click.echo("  Filtering TRANSITION level...")
        con.execute(f"""
            CREATE TABLE dst.TRANSITION AS
            SELECT DISTINCT t.*
            FROM src.TRANSITION t
            INNER JOIN src.TRANSITION_PRECURSOR_MAPPING tpm ON t.ID = tpm.TRANSITION_ID
            INNER JOIN dst.PRECURSOR p ON tpm.PRECURSOR_ID = p.ID
        """)

        if check_sqlite_table_duckdb(con, "src.FEATURE_TRANSITION"):
            con.execute(f"""
                CREATE TABLE dst.FEATURE_TRANSITION AS
                SELECT ft.*
                FROM src.FEATURE_TRANSITION ft
                INNER JOIN dst.FEATURE f ON ft.FEATURE_ID = f.ID
                INNER JOIN dst.TRANSITION t ON ft.TRANSITION_ID = t.ID
            """)

        if check_sqlite_table_duckdb(con, "src.SCORE_TRANSITION"):
            con.execute(f"""
                CREATE TABLE dst.SCORE_TRANSITION AS
                SELECT st.*
                FROM src.SCORE_TRANSITION st
                INNER JOIN dst.FEATURE f ON st.FEATURE_ID = f.ID
            """)

        # === MAPPING TABLES ===
        click.echo("  Creating mapping tables...")

        con.execute(f"""
            CREATE TABLE dst.PRECURSOR_PEPTIDE_MAPPING AS
            SELECT DISTINCT ppm.*
            FROM src.PRECURSOR_PEPTIDE_MAPPING ppm
            INNER JOIN dst.PEPTIDE p ON ppm.PEPTIDE_ID = p.ID
            INNER JOIN dst.PRECURSOR pr ON ppm.PRECURSOR_ID = pr.ID
        """)

        con.execute(f"""
            CREATE TABLE dst.PEPTIDE_PROTEIN_MAPPING AS
            SELECT DISTINCT ppm.*
            FROM src.PEPTIDE_PROTEIN_MAPPING ppm
            INNER JOIN dst.PEPTIDE p ON ppm.PEPTIDE_ID = p.ID
            INNER JOIN dst.PROTEIN pr ON ppm.PROTEIN_ID = pr.ID
        """)

        con.execute(f"""
            CREATE TABLE dst.TRANSITION_PRECURSOR_MAPPING AS
            SELECT DISTINCT tpm.*
            FROM src.TRANSITION_PRECURSOR_MAPPING tpm
            INNER JOIN dst.TRANSITION t ON tpm.TRANSITION_ID = t.ID
            INNER JOIN dst.PRECURSOR p ON tpm.PRECURSOR_ID = p.ID
        """)

        # === RUN AND VERSION ===
        con.execute(f"""
            CREATE TABLE dst.RUN AS
            SELECT * FROM src.RUN
            WHERE 1=1 {run_where.replace("RUN_ID", "ID") if run_ids else ""}
        """)

        if check_sqlite_table_duckdb(con, "src.VERSION"):
            con.execute("CREATE TABLE dst.VERSION AS SELECT * FROM src.VERSION")

        # === CREATE INDEXES (after detaching, directly on output file) ===
        click.echo("  Creating indexes...")
        con.execute("DETACH src")
        con.execute("DETACH dst")
        con.close()

        # Re-attach only the output database to create indexes
        con = duckdb.connect()
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")
        con.execute(f"ATTACH '{osw_out}' AS db (TYPE SQLITE)")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON db.FEATURE (ID)",
            "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON db.FEATURE (PRECURSOR_ID)",
            "CREATE INDEX IF NOT EXISTS idx_feature_run_id ON db.FEATURE (RUN_ID)",
            "CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON db.PEPTIDE (ID)",
            "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON db.PRECURSOR (ID)",
            "CREATE INDEX IF NOT EXISTS idx_protein_protein_id ON db.PROTEIN (ID)",
            "CREATE INDEX IF NOT EXISTS idx_run_run_id ON db.RUN (ID)",
            "CREATE INDEX IF NOT EXISTS idx_transition_id ON db.TRANSITION (ID)",
        ]

        for idx_sql in indexes:
            try:
                con.execute(idx_sql)
            except Exception as e:
                # Still fails with DuckDB SQLite attachment - indexes need to be created via raw SQLite
                pass

        con.close()

        # Use native sqlite3 for index creation (DuckDB's sqlite attachment doesn't support CREATE INDEX)
        import sqlite3

        sqlite_con = sqlite3.connect(osw_out)
        cursor = sqlite_con.cursor()

        sqlite_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID)",
            "CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID)",
            "CREATE INDEX IF NOT EXISTS idx_feature_run_id ON FEATURE (RUN_ID)",
            "CREATE INDEX IF NOT EXISTS idx_peptide_peptide_id ON PEPTIDE (ID)",
            "CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID)",
            "CREATE INDEX IF NOT EXISTS idx_protein_protein_id ON PROTEIN (ID)",
            "CREATE INDEX IF NOT EXISTS idx_run_run_id ON RUN (ID)",
            "CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID)",
        ]

        for idx_sql in sqlite_indexes:
            try:
                cursor.execute(idx_sql)
            except Exception as e:
                click.echo(f"  Warning: Could not create index: {e}")

        sqlite_con.commit()
        sqlite_con.close()

        click.echo(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Finished filtering {osw_in}"
        )


def filter_sqmass(
    sqmassfiles,
    infile=None,
    max_precursor_pep=0.7,
    max_peakgroup_pep=0.7,
    max_transition_pep=0.7,
    keep_naked_peptides=[],
    remove_decoys=True,
):
    """Filter sqMass files using DuckDB for better performance."""

    decoy_query = " AND p.DECOY=0" if remove_decoys else ""

    for sqm_in in sqmassfiles:
        click.echo(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Begin filtering {sqm_in}..."
        )
        sqm_out = sqm_in.split(".sqMass")[0] + "_filtered.sqMass"

        con = duckdb.connect()
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")

        if infile is not None:
            con.execute(f"ATTACH '{infile}' AS osw (TYPE SQLITE)")

            # Check which score tables exist
            has_ms1 = check_sqlite_table_duckdb(con, "osw.SCORE_MS1")
            has_ms2 = check_sqlite_table_duckdb(con, "osw.SCORE_MS2")
            has_transition = check_sqlite_table_duckdb(con, "osw.SCORE_TRANSITION")

            click.echo(
                f"  Score tables: MS1={has_ms1}, MS2={has_ms2}, TRANSITION={has_transition}"
            )

            # Check if FEATURE_TRANSITION exists
            has_feature_transition = check_sqlite_table_duckdb(
                con, "osw.FEATURE_TRANSITION"
            )
            click.echo(f"  FEATURE_TRANSITION table exists: {has_feature_transition}")

            con.execute(f"ATTACH '{sqm_in}' AS sqm (TYPE SQLITE)")

            # Get filename from sqMass for matching
            sqm_basename = sqm_in.split("/")[-1].split(".sqMass")[0].split(".chrom")[0]
            click.echo(f"  Looking for RUN with filename containing: {sqm_basename}")

            # Check what RUNs exist
            runs = con.execute("SELECT ID, FILENAME FROM osw.RUN").fetchall()
            click.echo(f"  Available RUNs: {runs}")

            # Build query based on available tables
            if has_ms1 and has_ms2 and has_transition and has_feature_transition:
                click.echo("  Using MS1+MS2+TRANSITION query path")
                query = f"""
                SELECT DISTINCT ft.TRANSITION_ID as transition_id
                FROM osw.PRECURSOR p
                INNER JOIN osw.FEATURE f ON p.ID = f.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS1 s1 ON f.ID = s1.FEATURE_ID
                INNER JOIN osw.SCORE_MS2 s2 ON f.ID = s2.FEATURE_ID
                INNER JOIN osw.SCORE_TRANSITION st ON f.ID = st.FEATURE_ID
                INNER JOIN osw.FEATURE_TRANSITION ft ON f.ID = ft.FEATURE_ID
                INNER JOIN osw.RUN r ON f.RUN_ID = r.ID
                WHERE s1.PEP <= {max_precursor_pep}
                AND s2.PEP <= {max_peakgroup_pep}
                AND st.PEP <= {max_transition_pep}
                AND r.FILENAME LIKE '%{sqm_basename}%'
                {decoy_query}
                """
            elif has_ms1 and has_ms2 and has_feature_transition:
                click.echo("  Using MS1+MS2 query path")
                query = f"""
                SELECT DISTINCT ft.TRANSITION_ID as transition_id
                FROM osw.PRECURSOR p
                INNER JOIN osw.FEATURE f ON p.ID = f.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS1 s1 ON f.ID = s1.FEATURE_ID
                INNER JOIN osw.SCORE_MS2 s2 ON f.ID = s2.FEATURE_ID
                INNER JOIN osw.FEATURE_TRANSITION ft ON f.ID = ft.FEATURE_ID
                INNER JOIN osw.RUN r ON f.RUN_ID = r.ID
                WHERE s1.PEP <= {max_precursor_pep}
                AND s2.PEP <= {max_peakgroup_pep}
                AND r.FILENAME LIKE '%{sqm_basename}%'
                {decoy_query}
                """
            elif has_ms2 and has_feature_transition:
                click.echo("  Using MS2-only query path")
                query = f"""
                SELECT DISTINCT ft.TRANSITION_ID as transition_id
                FROM osw.PRECURSOR p
                INNER JOIN osw.FEATURE f ON p.ID = f.PRECURSOR_ID
                INNER JOIN osw.SCORE_MS2 s2 ON f.ID = s2.FEATURE_ID
                INNER JOIN osw.FEATURE_TRANSITION ft ON f.ID = ft.FEATURE_ID
                INNER JOIN osw.RUN r ON f.RUN_ID = r.ID
                WHERE s2.PEP <= {max_peakgroup_pep}
                AND r.FILENAME LIKE '%{sqm_basename}%'
                {decoy_query}
                """
            else:
                raise click.ClickException(
                    "Conduct scoring on MS2-level and ensure FEATURE_TRANSITION table exists before filtering."
                )

            click.echo(f"  Executing query...")
            click.echo(f"  Query: {query[:200]}...")

            try:
                result = con.execute(query).fetchnumpy()
                if "transition_id" in result and len(result["transition_id"]) > 0:
                    transitions = result["transition_id"]
                    click.echo(f"  Found {len(transitions)} transition IDs")
                else:
                    click.echo("  Query returned no results")
                    # Try without filename filter
                    click.echo("  Retrying without filename filter...")
                    query_no_filename = query.replace(
                        f"AND r.FILENAME LIKE '%{sqm_basename}%'", ""
                    )
                    result = con.execute(query_no_filename).fetchnumpy()
                    if "transition_id" in result and len(result["transition_id"]) > 0:
                        transitions = result["transition_id"]
                        click.echo(
                            f"  Found {len(transitions)} transition IDs without filename filter"
                        )
                    else:
                        transitions = np.array([])
                        click.echo(
                            "  Still no results - check your PEP thresholds and data"
                        )
            except Exception as e:
                click.echo(f"  Query failed: {e}")
                raise

        elif len(keep_naked_peptides) != 0:
            con.execute(f"ATTACH '{sqm_in}' AS sqm (TYPE SQLITE)")
            peptide_list = "','".join(keep_naked_peptides)
            query = f"""
            SELECT c.NATIVE_ID as transition_id
            FROM sqm.CHROMATOGRAM c
            INNER JOIN sqm.PRECURSOR p ON p.CHROMATOGRAM_ID = c.ID
            WHERE p.PEPTIDE_SEQUENCE IN ('{peptide_list}')
            """
            transitions = con.execute(query).fetchnumpy()["transition_id"]
        else:
            raise click.ClickException(
                "Please provide either an associated OSW file or a list of peptides to keep."
            )

        con.close()

        if len(transitions) > 0:
            filter_chrom_by_labels(sqm_in, sqm_out, transitions)
        else:
            raise click.ClickException("No transition ids to filter chromatograms.")


def filter_chrom_by_labels(infile, outfile, labels):
    """Filter chromatogram file by transition labels."""
    if len(labels) == 0:
        raise click.ClickException("No transition ids to filter chromatograms.")

    con = duckdb.connect()
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")

    con.execute(f"ATTACH '{infile}' AS src (TYPE SQLITE)")
    con.execute(f"ATTACH '{outfile}' AS dst (TYPE SQLITE)")

    labels_str = "','".join([str(l) for l in labels])

    keep_ids = con.execute(f"""
        SELECT ID FROM src.CHROMATOGRAM
        WHERE NATIVE_ID IN ('{labels_str}')
    """).fetchnumpy()["ID"]

    click.echo(f"Keep {len(keep_ids)} chromatograms")

    ids_str = ",".join([str(i) for i in keep_ids])

    con.execute(
        f"CREATE TABLE dst.CHROMATOGRAM AS SELECT * FROM src.CHROMATOGRAM WHERE ID IN ({ids_str})"
    )
    con.execute(
        f"CREATE TABLE dst.PRECURSOR AS SELECT * FROM src.PRECURSOR WHERE CHROMATOGRAM_ID IN ({ids_str})"
    )
    con.execute(
        f"CREATE TABLE dst.PRODUCT AS SELECT * FROM src.PRODUCT WHERE CHROMATOGRAM_ID IN ({ids_str})"
    )
    con.execute(
        f"CREATE TABLE dst.DATA AS SELECT * FROM src.DATA WHERE CHROMATOGRAM_ID IN ({ids_str})"
    )
    con.execute("CREATE TABLE dst.RUN AS SELECT * FROM src.RUN")
    con.execute("CREATE TABLE dst.SPECTRUM AS SELECT * FROM src.SPECTRUM")
    con.execute("CREATE TABLE dst.RUN_EXTRA AS SELECT * FROM src.RUN_EXTRA")

    con.close()

    # Create indexes using native sqlite3
    import sqlite3

    sqlite_con = sqlite3.connect(outfile)
    cursor = sqlite_con.cursor()

    indexes = [
        "CREATE INDEX IF NOT EXISTS data_chr_idx ON DATA(CHROMATOGRAM_ID)",
        "CREATE INDEX IF NOT EXISTS data_sp_idx ON DATA(SPECTRUM_ID)",
        "CREATE INDEX IF NOT EXISTS spec_rt_idx ON SPECTRUM(RETENTION_TIME)",
        "CREATE INDEX IF NOT EXISTS spec_mslevel ON SPECTRUM(MSLEVEL)",
        "CREATE INDEX IF NOT EXISTS spec_run ON SPECTRUM(RUN_ID)",
        "CREATE INDEX IF NOT EXISTS chrom_run ON CHROMATOGRAM(RUN_ID)",
    ]

    for idx_sql in indexes:
        try:
            cursor.execute(idx_sql)
        except Exception:
            pass

    sqlite_con.commit()
    sqlite_con.close()
