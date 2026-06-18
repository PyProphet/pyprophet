import os
import sqlite3
from shutil import copyfile

import click

from .io.util import check_sqlite_table


def subsample_osw(infile, outfile, subsample_ratio, test):
    conn = sqlite3.connect(infile)
    ms1_present = check_sqlite_table(conn, "FEATURE_MS1")
    ms2_present = check_sqlite_table(conn, "FEATURE_MS2")
    transition_present = check_sqlite_table(conn, "FEATURE_TRANSITION")
    ## Check if infile contains multiple entries for run table, if only 1 entry, then infile is a single run, else infile contains multiples run
    n_runs = (
        conn.cursor()
        .execute("SELECT COUNT(*) AS NUMBER_OF_RUNS FROM RUN")
        .fetchall()[0][0]
    )
    multiple_runs = True if n_runs > 1 else False
    if multiple_runs:
        click.echo(f"Warn: There are {n_runs} runs in {infile}")
    conn.close()

    conn = sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript(
        f"""
PRAGMA synchronous = OFF;

ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE RUN AS SELECT * FROM sdb.RUN;

DETACH DATABASE sdb;
"""
    )
    click.echo(f"Info: Propagated runs of file {infile} to {outfile}.")

    if subsample_ratio >= 1.0:
        c.executescript(
            f"""
    ATTACH DATABASE "{infile}" AS sdb;

    CREATE TABLE FEATURE AS SELECT * FROM sdb.FEATURE;

    DETACH DATABASE sdb;
    """
        )
    else:
        if test:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE AS 
SELECT *
FROM sdb.FEATURE
WHERE PRECURSOR_ID IN
    (SELECT ID
     FROM sdb.PRECURSOR
     LIMIT
       (SELECT ROUND({subsample_ratio}*COUNT(DISTINCT ID))
        FROM sdb.PRECURSOR));

DETACH DATABASE sdb;
"""
            )
        else:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE AS 
SELECT *
FROM sdb.FEATURE
WHERE PRECURSOR_ID IN
    (SELECT ID
     FROM sdb.PRECURSOR
     ORDER BY RANDOM()
     LIMIT
       (SELECT ROUND({subsample_ratio}*COUNT(DISTINCT ID))
        FROM sdb.PRECURSOR));

DETACH DATABASE sdb;
"""
            )
    click.echo(f"Info: Subsampled generic features of file {infile} to {outfile}.")

    if ms1_present:
        if subsample_ratio >= 1.0:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE_MS1 AS 
SELECT *
FROM sdb.FEATURE_MS1;

DETACH DATABASE sdb;
"""
            )
        else:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE_MS1 AS 
SELECT *
FROM sdb.FEATURE_MS1
WHERE sdb.FEATURE_MS1.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
"""
            )
        click.echo(f"Info: Subsampled MS1 features of file {infile} to {outfile}.")

    if ms2_present:
        if subsample_ratio >= 1.0:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE_MS2 AS 
SELECT *
FROM sdb.FEATURE_MS2;

DETACH DATABASE sdb;
"""
            )
        else:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE_MS2 AS 
SELECT *
FROM sdb.FEATURE_MS2
WHERE sdb.FEATURE_MS2.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
"""
            )
        click.echo(f"Info: Subsampled MS2 features of file {infile} to {outfile}.")

    if transition_present:
        if subsample_ratio >= 1.0:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE_TRANSITION AS 
SELECT *
FROM sdb.FEATURE_TRANSITION;

DETACH DATABASE sdb;
"""
            )
        else:
            c.executescript(
                f"""
ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE FEATURE_TRANSITION AS 
SELECT *
FROM sdb.FEATURE_TRANSITION
WHERE sdb.FEATURE_TRANSITION.FEATURE_ID IN
    (SELECT ID
     FROM FEATURE);

DETACH DATABASE sdb;
"""
            )
        click.echo(
            f"Info: Subsampled transition features of file {infile} to {outfile}."
        )

    if multiple_runs:
        c.executescript(
            f"""
PRAGMA synchronous = OFF;

ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE PRECURSOR AS 
SELECT * 
FROM sdb.PRECURSOR
WHERE sdb.PRECURSOR.ID IN
    (SELECT PRECURSOR_ID
     FROM FEATURE);

DETACH DATABASE sdb;
"""
        )
        click.echo(
            f"Info: Subsampled precursor table of file {infile} to {outfile}. For scoring merged subsampled file."
        )

        c.executescript(
            f"""
    PRAGMA synchronous = OFF;

    ATTACH DATABASE "{infile}" AS sdb;

    CREATE TABLE TRANSITION_PRECURSOR_MAPPING AS 
    SELECT * 
    FROM sdb.TRANSITION_PRECURSOR_MAPPING
    WHERE sdb.TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID IN
        (SELECT ID
         FROM PRECURSOR);

    DETACH DATABASE sdb;
    """
        )
        click.echo(
            f"Info: Subsampled transition_precursor_mapping table of file {infile} to {outfile}. For scoring merged subsampled file."
        )

        c.executescript(
            f"""
    PRAGMA synchronous = OFF;

    ATTACH DATABASE "{infile}" AS sdb;

    CREATE TABLE TRANSITION AS 
    SELECT * 
    FROM sdb.TRANSITION
    WHERE sdb.TRANSITION.ID IN
        (SELECT TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID
         FROM TRANSITION_PRECURSOR_MAPPING);

    DETACH DATABASE sdb;
    """
        )
        click.echo(
            f"Info: Subsampled transition table of file {infile} to {outfile}. For scoring merged subsampled file."
        )

    conn.commit()
    conn.close()

    click.echo("Info: OSW file was subsampled.")


def reduce_osw(infile, outfile):
    conn = sqlite3.connect(infile)
    if not check_sqlite_table(conn, "SCORE_MS2"):
        raise click.ClickException(
            "Apply scoring to MS2 data before reducing file for multi-run scoring."
        )
    conn.close()

    try:
        os.remove(outfile)
    except OSError:
        pass

    conn = sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript(
        f"""
PRAGMA synchronous = OFF;

ATTACH DATABASE "{infile}" AS sdb;

CREATE TABLE RUN(ID INT PRIMARY KEY NOT NULL,
                 FILENAME TEXT NOT NULL);

INSERT INTO RUN
SELECT *
FROM sdb.RUN;

CREATE TABLE SCORE_MS2(FEATURE_ID INTEGER, SCORE REAL);

INSERT INTO SCORE_MS2 (FEATURE_ID, SCORE)
SELECT FEATURE_ID,
       SCORE
FROM sdb.SCORE_MS2
WHERE RANK == 1;

CREATE TABLE FEATURE(ID INT PRIMARY KEY NOT NULL,
                     RUN_ID INT NOT NULL,
                     PRECURSOR_ID INT NOT NULL);

INSERT INTO FEATURE (ID, RUN_ID, PRECURSOR_ID)
SELECT ID,
       RUN_ID,
       PRECURSOR_ID
FROM sdb.FEATURE
WHERE ID IN
    (SELECT FEATURE_ID
     FROM SCORE_MS2);
"""
    )

    conn.commit()
    conn.close()

    click.echo("Info: OSW file was reduced for multi-run scoring.")


def merge_osw(infiles, outfile, templatefile, same_run, merge_post_scored_runs, fresh=False):
    conn = sqlite3.connect(infiles[0])
    reduced = check_sqlite_table(conn, "SCORE_MS2")
    conn.close()
    if reduced and not merge_post_scored_runs:
        click.echo("Calling reduced osws merge function")
        merge_oswr(infiles, outfile, templatefile, same_run, fresh=fresh)
    elif merge_post_scored_runs:
        click.echo("Calling post scored osws merge function")
        merge_oswps(infiles, outfile, templatefile, same_run, fresh=fresh)
    else:
        click.echo("Calling pre scored osws merge function")
        merge_osws(infiles, outfile, templatefile, same_run, fresh=fresh)


def _table_has_data(outfile, table_name):
    """Check if a table already has data (for resume capability)."""
    try:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        if not check_sqlite_table(conn, table_name):
            conn.close()
            return False
        result = c.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        conn.close()
        return result[0] > 0 if result else False
    except Exception:
        return False


def _get_merge_progress(outfile, tables_to_check, total_files):
    """Get detailed merge progress per table: how many files have been merged for each table."""
    progress = {}
    try:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        
        # Create metadata table if needed (for old partial merges that don't have it)
        if not check_sqlite_table(conn, "MERGE_PROGRESS"):
            click.echo("Info: Creating MERGE_PROGRESS tracking table (old partial merge detected)...")
            c.executescript("""
                CREATE TABLE MERGE_PROGRESS (
                    table_name TEXT PRIMARY KEY,
                    files_completed INTEGER DEFAULT 0
                );
            """)
            
            # For old partial merges: mark RUN as complete (already merged), reset features to 0
            init_values = {}
            for table in tables_to_check:
                if table == "RUN":
                    init_values[table] = total_files  # RUN already has data, mark as complete
                else:
                    init_values[table] = 0  # Feature tables need to be remerged
            
            for table in tables_to_check:
                c.execute(
                    "INSERT INTO MERGE_PROGRESS (table_name, files_completed) VALUES (?, ?)",
                    (table, init_values[table])
                )
            conn.commit()
            click.echo("Info: MERGE_PROGRESS initialized. Feature tables will restart from file 0.")
        
        # Query progress for each table
        for table in tables_to_check:
            try:
                result = c.execute(
                    "SELECT files_completed FROM MERGE_PROGRESS WHERE table_name = ?",
                    (table,)
                ).fetchone()
                progress[table] = result[0] if result else 0
            except:
                progress[table] = 0
        
        conn.close()
        return progress
    except Exception as e:
        click.echo(f"Warning: Could not read merge progress: {e}. Starting fresh.")
        return {table: 0 for table in tables_to_check}



def merge_osws(infiles, outfile, templatefile, same_run, fresh=False):
    import time
    from datetime import timedelta
    
    tables_to_merge = ["RUN", "FEATURE", "FEATURE_MS1", "FEATURE_MS2", "FEATURE_TRANSITION"]
    total_files = len(infiles)  # Calculate early for progress tracking
    
    # Check if this is a resume operation (unless fresh flag is set)
    is_resume = (not fresh) and os.path.exists(outfile) and _table_has_data(outfile, "RUN")
    if is_resume:
        click.echo(f"Info: Resuming merge for {outfile}. Checking which tables still need to be merged...")
        progress = _get_merge_progress(outfile, tables_to_merge, total_files)
        click.echo(f"Merge progress: RUN={progress['RUN']}, FEATURE={progress['FEATURE']}, MS1={progress['FEATURE_MS1']}, MS2={progress['FEATURE_MS2']}, TRANS={progress['FEATURE_TRANSITION']} files merged")
    else:
        # Fresh merge: copy template and create empty tables
        if fresh and os.path.exists(outfile):
            click.echo(f"Info: --fresh flag set. Removing existing {outfile} and starting from scratch...")
            os.remove(outfile)
        elif os.path.exists(outfile):
            os.remove(outfile)
        copyfile(templatefile, outfile)
        progress = {table: 0 for table in tables_to_merge}
    
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    if same_run:
        c.execute("SELECT ID, FILENAME FROM RUN")
        result = c.fetchall()
        if len(result) != 1:
            raise click.ClickException(
                "Input for same-run merge contains more than one run."
            )
        runid, rname = result[0]

    # Only create empty tables if not resuming
    # OR if resuming but MERGE_PROGRESS was just created (old partial merge) - need to clear feature tables
    need_to_recreate_feature_tables = is_resume and all(v == 0 for v in progress.values())
    
    if not is_resume or need_to_recreate_feature_tables:
        if need_to_recreate_feature_tables:
            click.echo("Info: Detected old partial merge without progress tracking. Clearing feature tables to avoid duplicates...")
        
        drop_statements = """
DROP TABLE IF EXISTS FEATURE;
DROP TABLE IF EXISTS FEATURE_MS1;
DROP TABLE IF EXISTS FEATURE_MS2;
DROP TABLE IF EXISTS FEATURE_TRANSITION;
DROP TABLE IF EXISTS SCORE_MS1;
DROP TABLE IF EXISTS SCORE_MS2;
DROP TABLE IF EXISTS SCORE_TRANSITION;
DROP TABLE IF EXISTS SCORE_PEPTIDE;
DROP TABLE IF EXISTS SCORE_PROTEIN;
DROP TABLE IF EXISTS SCORE_IPF;
"""
        # Only drop RUN table if fresh merge (not resuming old partial)
        if not is_resume:
            drop_statements = "DROP TABLE IF EXISTS RUN;\n" + drop_statements
        
        c.executescript(
            f"""
PRAGMA synchronous = OFF;
PRAGMA cache_size = 50000;
PRAGMA temp_store = MEMORY;

{drop_statements}

ATTACH DATABASE "{infiles[0]}" AS sdb;

CREATE TABLE FEATURE AS SELECT * FROM sdb.FEATURE LIMIT 0;
CREATE TABLE FEATURE_MS1 AS SELECT * FROM sdb.FEATURE_MS1 LIMIT 0;
CREATE TABLE FEATURE_MS2 AS SELECT * FROM sdb.FEATURE_MS2 LIMIT 0;
CREATE TABLE FEATURE_TRANSITION AS SELECT * FROM sdb.FEATURE_TRANSITION LIMIT 0;

DETACH DATABASE sdb;
"""
        )
        
        # Only initialize MERGE_PROGRESS for fresh merges (not old partial merges)
        if not is_resume:
            c.executescript("""
DROP TABLE IF EXISTS MERGE_PROGRESS;

CREATE TABLE MERGE_PROGRESS (
    table_name TEXT PRIMARY KEY,
    files_completed INTEGER DEFAULT 0
);

INSERT INTO MERGE_PROGRESS (table_name, files_completed) VALUES ('RUN', 0);
INSERT INTO MERGE_PROGRESS (table_name, files_completed) VALUES ('FEATURE', 0);
INSERT INTO MERGE_PROGRESS (table_name, files_completed) VALUES ('FEATURE_MS1', 0);
INSERT INTO MERGE_PROGRESS (table_name, files_completed) VALUES ('FEATURE_MS2', 0);
INSERT INTO MERGE_PROGRESS (table_name, files_completed) VALUES ('FEATURE_TRANSITION', 0);
""")
        
        if not is_resume:
            # For fresh merges, also create RUN table from first input file
            c.executescript(f"""
ATTACH DATABASE "{infiles[0]}" AS sdb;
CREATE TABLE RUN AS SELECT * FROM sdb.RUN LIMIT 0;
DETACH DATABASE sdb;
""")
    else:
        c.executescript("PRAGMA synchronous = OFF; PRAGMA cache_size = 50000; PRAGMA temp_store = MEMORY;")

    conn.commit()
    conn.close()

    # ===== BATCH PROCESSING OPTIMIZATION =====
    batch_size = 50
    start_time = time.time()

    def estimate_time_remaining(files_done, total_files, elapsed_seconds):
        if files_done == 0:
            return None
        rate = elapsed_seconds / files_done
        remaining_files = total_files - files_done
        remaining_seconds = rate * remaining_files
        return timedelta(seconds=int(remaining_seconds))

    # ===== MERGE RUN DATA (batch processed, resume from last completed file) =====
    run_start_file = progress["RUN"]
    if run_start_file < total_files:
        if not same_run:
            click.echo(f"Merging RUN data (resuming from file {run_start_file}/{total_files})...")
            for batch_num in range(run_start_file, total_files, batch_size):
                batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
                conn = sqlite3.connect(outfile)
                c = conn.cursor()
                
                for idx, infile in enumerate(batch_files):
                    db_alias = f"run_{batch_num}_{idx}"
                    c.executescript(
                        f"""
    ATTACH DATABASE "{infile}" AS {db_alias};
    INSERT INTO RUN SELECT * FROM {db_alias}.RUN;
    DETACH DATABASE {db_alias};
    """
                    )
                
                conn.commit()
                conn.close()
                
                files_done = min(batch_num + batch_size, total_files)
                
                # Update progress tracking
                conn = sqlite3.connect(outfile)
                conn.execute("UPDATE MERGE_PROGRESS SET files_completed = ? WHERE table_name = 'RUN'", (files_done,))
                conn.commit()
                conn.close()
                
                elapsed = time.time() - start_time
                eta = estimate_time_remaining(files_done, total_files, elapsed)
                click.echo(f"Info: Merged RUN data for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))
        else:
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            c.executescript(f"""INSERT INTO RUN (ID, FILENAME) VALUES ({runid}, '{rname}')""")
            conn.execute("UPDATE MERGE_PROGRESS SET files_completed = ? WHERE table_name = 'RUN'", (total_files,))
            conn.commit()
            conn.close()
    else:
        click.echo("Skipping RUN data (already merged)")

    # ===== MERGE FEATURE DATA (batch processed, resume from last completed file) =====
    feature_start_file = progress["FEATURE"]
    if feature_start_file < total_files:
        click.echo(f"\nMerging FEATURE data (resuming from file {feature_start_file}/{total_files})...")
        for batch_num in range(feature_start_file, total_files, batch_size):
            batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            
            for idx, infile in enumerate(batch_files):
                db_alias = f"feat_{batch_num}_{idx}"
                c.executescript(
                    f"""
    ATTACH DATABASE "{infile}" AS {db_alias};
    INSERT INTO FEATURE SELECT * FROM {db_alias}.FEATURE;
    DETACH DATABASE {db_alias};
    """
                )
            
            conn.commit()
            conn.close()
            
            files_done = min(batch_num + batch_size, total_files)
            
            # Update progress tracking
            conn = sqlite3.connect(outfile)
            conn.execute("UPDATE MERGE_PROGRESS SET files_completed = ? WHERE table_name = 'FEATURE'", (files_done,))
            conn.commit()
            conn.close()
            
            elapsed = time.time() - start_time
            eta = estimate_time_remaining(files_done, total_files, elapsed)
            click.echo(f"Info: Merged FEATURE data for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

        if same_run:
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            c.executescript(f"""UPDATE FEATURE SET RUN_ID = {runid}""")
            conn.commit()
            conn.close()
    else:
        click.echo("Skipping FEATURE data (already merged)")

    # ===== MERGE MS1 FEATURES (batch processed, resume from last completed file) =====
    ms1_start_file = progress["FEATURE_MS1"]
    if ms1_start_file < total_files:
        click.echo(f"\nMerging MS1 features (resuming from file {ms1_start_file}/{total_files})...")
        for batch_num in range(ms1_start_file, total_files, batch_size):
            batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            
            for idx, infile in enumerate(batch_files):
                db_alias = f"ms1_{batch_num}_{idx}"
                c.executescript(
                    f"""
    ATTACH DATABASE "{infile}" AS {db_alias};
    INSERT INTO FEATURE_MS1 SELECT * FROM {db_alias}.FEATURE_MS1;
    DETACH DATABASE {db_alias};
    """
                )
            
            conn.commit()
            conn.close()
            
            files_done = min(batch_num + batch_size, total_files)
            
            # Update progress tracking
            conn = sqlite3.connect(outfile)
            conn.execute("UPDATE MERGE_PROGRESS SET files_completed = ? WHERE table_name = 'FEATURE_MS1'", (files_done,))
            conn.commit()
            conn.close()
            
            elapsed = time.time() - start_time
            eta = estimate_time_remaining(files_done, total_files, elapsed)
            click.echo(f"Info: Merged MS1 features for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))
    else:
        click.echo("Skipping MS1 features (already merged)")

    # ===== MERGE MS2 FEATURES (batch processed, resume from last completed file) =====
    ms2_start_file = progress["FEATURE_MS2"]
    if ms2_start_file < total_files:
        click.echo(f"\nMerging MS2 features (resuming from file {ms2_start_file}/{total_files})...")
        for batch_num in range(ms2_start_file, total_files, batch_size):
            batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            
            for idx, infile in enumerate(batch_files):
                db_alias = f"ms2_{batch_num}_{idx}"
                c.executescript(
                    f"""
    ATTACH DATABASE "{infile}" AS {db_alias};
    INSERT INTO FEATURE_MS2 SELECT * FROM {db_alias}.FEATURE_MS2;
    DETACH DATABASE {db_alias};
    """
                )
            
            conn.commit()
            conn.close()
            
            files_done = min(batch_num + batch_size, total_files)
            
            # Update progress tracking
            conn = sqlite3.connect(outfile)
            conn.execute("UPDATE MERGE_PROGRESS SET files_completed = ? WHERE table_name = 'FEATURE_MS2'", (files_done,))
            conn.commit()
            conn.close()
            
            elapsed = time.time() - start_time
            eta = estimate_time_remaining(files_done, total_files, elapsed)
            click.echo(f"Info: Merged MS2 features for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))
    else:
        click.echo("Skipping MS2 features (already merged)")

    # ===== MERGE TRANSITION FEATURES (batch processed, resume from last completed file) =====
    trans_start_file = progress["FEATURE_TRANSITION"]
    if trans_start_file < total_files:
        click.echo(f"\nMerging transition features (resuming from file {trans_start_file}/{total_files})...")
        for batch_num in range(trans_start_file, total_files, batch_size):
            batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            
            for idx, infile in enumerate(batch_files):
                db_alias = f"trans_{batch_num}_{idx}"
                c.executescript(
                    f"""
    ATTACH DATABASE "{infile}" AS {db_alias};
    INSERT INTO FEATURE_TRANSITION SELECT * FROM {db_alias}.FEATURE_TRANSITION;
    DETACH DATABASE {db_alias};
    """
                )
            
            conn.commit()
            conn.close()
            
            files_done = min(batch_num + batch_size, total_files)
            
            # Update progress tracking
            conn = sqlite3.connect(outfile)
            conn.execute("UPDATE MERGE_PROGRESS SET files_completed = ? WHERE table_name = 'FEATURE_TRANSITION'", (files_done,))
            conn.commit()
            conn.close()
            
            elapsed = time.time() - start_time
            eta = estimate_time_remaining(files_done, total_files, elapsed)
            click.echo(f"Info: Merged transition features for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))
    else:
        click.echo("Skipping transition features (already merged)")

    click.echo("\nInfo: All pre-scored OSWS files were merged successfully.")
    total_time = time.time() - start_time
    click.echo(f"Total merge time: {timedelta(seconds=int(total_time))}")



def merge_oswr(infiles, outfile, templatefile, same_run, fresh=False):
    import time
    from datetime import timedelta
    
    # Copy the template to the output file
    if fresh and os.path.exists(outfile):
        click.echo(f"Info: --fresh flag set. Removing existing {outfile} and starting from scratch...")
        os.remove(outfile)
    copyfile(templatefile, outfile)
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    if same_run:
        c.execute("SELECT ID, FILENAME FROM RUN")
        result = c.fetchall()
        if len(result) != 1:
            raise click.ClickException(
                "Input for same-run merge contains more than one run."
            )
        runid, rname = result[0]

    c.executescript(
        """
PRAGMA synchronous = OFF;
PRAGMA cache_size = 50000;
PRAGMA temp_store = MEMORY;

DROP TABLE IF EXISTS RUN;
DROP TABLE IF EXISTS FEATURE;
DROP TABLE IF EXISTS FEATURE_MS1;
DROP TABLE IF EXISTS FEATURE_MS2;
DROP TABLE IF EXISTS FEATURE_TRANSITION;
DROP TABLE IF EXISTS SCORE_MS1;
DROP TABLE IF EXISTS SCORE_MS2;
DROP TABLE IF EXISTS SCORE_TRANSITION;
DROP TABLE IF EXISTS SCORE_PEPTIDE;
DROP TABLE IF EXISTS SCORE_PROTEIN;
DROP TABLE IF EXISTS SCORE_IPF;

CREATE TABLE RUN(ID INT PRIMARY KEY NOT NULL,
                 FILENAME TEXT NOT NULL);
CREATE TABLE SCORE_MS2(FEATURE_ID INTEGER, SCORE REAL);
CREATE TABLE FEATURE(ID INT PRIMARY KEY NOT NULL,
                     RUN_ID INT NOT NULL,
                     PRECURSOR_ID INT NOT NULL);
"""
    )

    conn.commit()
    conn.close()

    # ===== BATCH PROCESSING OPTIMIZATION =====
    batch_size = 50
    total_files = len(infiles)
    start_time = time.time()

    def estimate_time_remaining(files_done, total_files, elapsed_seconds):
        if files_done == 0:
            return None
        rate = elapsed_seconds / files_done
        remaining_files = total_files - files_done
        remaining_seconds = rate * remaining_files
        return timedelta(seconds=int(remaining_seconds))

    # ===== MERGE RUN DATA (batch processed) =====
    if not same_run:
        click.echo("Merging RUN data...")
        for batch_num in range(0, total_files, batch_size):
            batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            
            for idx, infile in enumerate(batch_files):
                db_alias = f"run_{batch_num}_{idx}"
                c.executescript(f'ATTACH DATABASE "{infile}" AS {db_alias}; INSERT INTO RUN SELECT * FROM {db_alias}.RUN; DETACH DATABASE {db_alias};')
            
            conn.commit()
            conn.close()
            
            files_done = min(batch_num + batch_size, total_files)
            elapsed = time.time() - start_time
            eta = estimate_time_remaining(files_done, total_files, elapsed)
            click.echo(f"Info: Merged RUN data for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))
    else:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        c.executescript(f"""INSERT INTO RUN (ID, FILENAME) VALUES ({runid}, '{rname}')""")
        conn.commit()
        conn.close()

    # ===== MERGE FEATURE DATA (batch processed) =====
    click.echo("\nMerging FEATURE data...")
    for batch_num in range(0, total_files, batch_size):
        batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        
        for idx, infile in enumerate(batch_files):
            db_alias = f"feat_{batch_num}_{idx}"
            c.executescript(f'ATTACH DATABASE "{infile}" AS {db_alias}; INSERT INTO FEATURE SELECT * FROM {db_alias}.FEATURE; DETACH DATABASE {db_alias};')
        
        conn.commit()
        conn.close()
        
        files_done = min(batch_num + batch_size, total_files)
        elapsed = time.time() - start_time
        eta = estimate_time_remaining(files_done, total_files, elapsed)
        click.echo(f"Info: Merged FEATURE data for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        c.executescript(f"""UPDATE FEATURE SET RUN_ID = {runid}""")
        conn.commit()
        conn.close()

    # ===== MERGE SCORE_MS2 DATA (batch processed) =====
    click.echo("\nMerging SCORE_MS2 data...")
    for batch_num in range(0, total_files, batch_size):
        batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        
        for idx, infile in enumerate(batch_files):
            db_alias = f"score_{batch_num}_{idx}"
            c.executescript(f'ATTACH DATABASE "{infile}" AS {db_alias}; INSERT INTO SCORE_MS2 SELECT * FROM {db_alias}.SCORE_MS2; DETACH DATABASE {db_alias};')
        
        conn.commit()
        conn.close()
        
        files_done = min(batch_num + batch_size, total_files)
        elapsed = time.time() - start_time
        eta = estimate_time_remaining(files_done, total_files, elapsed)
        click.echo(f"Info: Merged SCORE_MS2 data for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    click.echo("\nInfo: All reduced OSWR files were merged successfully.")
    total_time = time.time() - start_time
    click.echo(f"Total merge time: {timedelta(seconds=int(total_time))}")



def merge_oswps(infiles, outfile, templatefile, same_run, fresh=False):
    import time
    from datetime import timedelta
    
    click.echo("Info: Merging all Scored Runs.")
    click.echo(f"Info: Processing {len(infiles)} OSW files (total ~{len(infiles) * 14}GB)")
    
    # Copy the first file to have a template
    if fresh and os.path.exists(outfile):
        click.echo(f"Info: --fresh flag set. Removing existing {outfile} and starting from scratch...")
        os.remove(outfile)
    copyfile(templatefile, outfile)
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    if same_run:
        c.execute("SELECT ID, FILENAME FROM RUN")
        result = c.fetchall()
        if len(result) != 1:
            raise click.ClickException(
                "Input for same-run merge contains more than one run."
            )
        runid, rname = result[0]

    original_tables = c.execute(
        """ SELECT name FROM sqlite_master WHERE type='table'; """
    )
    original_tables = [name[0] for name in original_tables]
    ## Get Score tables table_present
    score_tables = [name for name in original_tables if "SCORE" in name]
    if len(score_tables) > 0:
        create_scores_query = "\n".join(
            [
                "CREATE TABLE "
                + score_tbl
                + " AS SELECT * FROM sdb."
                + score_tbl
                + " LIMIT 0;"
                for score_tbl in score_tables
            ]
        )
    else:
        create_scores_query = ""

    ## Get Feature Alignment tables table_present
    feature_alignment_tables = [name for name in original_tables if "ALIGNMENT" in name]
    feature_alignment_tables_present = False
    if len(feature_alignment_tables) > 0:
        create_feature_alignment_query = "\n".join(
            [
                "CREATE TABLE "
                + feature_alignment_tbl
                + " AS SELECT * FROM sdb."
                + feature_alignment_tbl
                + " LIMIT 0;"
                for feature_alignment_tbl in feature_alignment_tables
            ]
        )
        feature_alignment_tables_present = True
    else:
        create_feature_alignment_query = ""

    click.echo(f"First File input: {infiles[0]}")

    c.executescript(
        f"""
    PRAGMA synchronous = OFF;
    PRAGMA cache_size = 50000;
    PRAGMA temp_store = MEMORY;
    DROP TABLE IF EXISTS RUN;
    DROP TABLE IF EXISTS FEATURE;
    DROP TABLE IF EXISTS FEATURE_MS1;
    DROP TABLE IF EXISTS FEATURE_MS2;
    DROP TABLE IF EXISTS FEATURE_TRANSITION;
    DROP TABLE IF EXISTS FEATURE_ALIGNMENT;
    DROP TABLE IF EXISTS FEATURE_MS2_ALIGNMENT;
    DROP TABLE IF EXISTS FEATURE_TRANSITION_ALIGNMENT;
    DROP TABLE IF EXISTS SCORE_MS1;
    DROP TABLE IF EXISTS SCORE_MS2;
    DROP TABLE IF EXISTS SCORE_TRANSITION;
    DROP TABLE IF EXISTS SCORE_PEPTIDE;
    DROP TABLE IF EXISTS SCORE_PROTEIN;
    DROP TABLE IF EXISTS SCORE_IPF;
    ATTACH DATABASE "{infiles[0]}" AS sdb;
    CREATE TABLE RUN AS SELECT * FROM sdb.RUN LIMIT 0;
    CREATE TABLE FEATURE AS SELECT * FROM sdb.FEATURE LIMIT 0;
    CREATE TABLE FEATURE_MS1 AS SELECT * FROM sdb.FEATURE_MS1 LIMIT 0;
    CREATE TABLE FEATURE_MS2 AS SELECT * FROM sdb.FEATURE_MS2 LIMIT 0;
    CREATE TABLE FEATURE_TRANSITION AS SELECT * FROM sdb.FEATURE_TRANSITION LIMIT 0;
    {create_feature_alignment_query}
    {create_scores_query}
    DETACH DATABASE sdb;
    """
    )

    conn.commit()
    conn.close()

    # ===== BATCH PROCESSING OPTIMIZATION =====
    # Instead of opening/closing connection 248 times per table,
    # batch process files in groups to reduce I/O overhead
    batch_size = 50  # Process 50 files per batch
    total_files = len(infiles)
    total_batches = (total_files + batch_size - 1) // batch_size
    start_time = time.time()

    def estimate_time_remaining(files_done, total_files, elapsed_seconds):
        """Calculate estimated time remaining"""
        if files_done == 0:
            return None
        rate = elapsed_seconds / files_done
        remaining_files = total_files - files_done
        remaining_seconds = rate * remaining_files
        return timedelta(seconds=int(remaining_seconds))

    # ===== MERGE RUN DATA (batch processed) =====
    if not same_run:
        click.echo("Merging RUN data...")
        for batch_num in range(0, total_files, batch_size):
            batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            
            for infile in batch_files:
                c.executescript(
                    f"""
        ATTACH DATABASE "{infile}" AS sdb_{batch_num};
        INSERT INTO RUN SELECT * FROM sdb_{batch_num}.RUN;
        DETACH DATABASE sdb_{batch_num};
        """
                )
            
            conn.commit()
            conn.close()
            
            files_done = min(batch_num + batch_size, total_files)
            elapsed = time.time() - start_time
            eta = estimate_time_remaining(files_done, total_files, elapsed)
            click.echo(f"Info: Merged RUN data for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))
    else:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        c.executescript(
            f"""INSERT INTO RUN (ID, FILENAME) VALUES ({runid}, '{rname}')"""
        )
        conn.commit()
        conn.close()

    # ===== MERGE FEATURE DATA (batch processed) =====
    click.echo("\nMerging FEATURE data...")
    for batch_num in range(0, total_files, batch_size):
        batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        
        for idx, infile in enumerate(batch_files):
            db_alias = f"feat_{batch_num}_{idx}"
            c.executescript(
                f"""
        ATTACH DATABASE "{infile}" AS {db_alias};
        INSERT INTO FEATURE SELECT * FROM {db_alias}.FEATURE;
        DETACH DATABASE {db_alias};
        """
            )
        
        conn.commit()
        conn.close()
        
        files_done = min(batch_num + batch_size, total_files)
        elapsed = time.time() - start_time
        eta = estimate_time_remaining(files_done, total_files, elapsed)
        click.echo(f"Info: Merged FEATURE data for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        c.executescript(f"UPDATE FEATURE SET RUN_ID = {runid}")
        conn.commit()
        conn.close()

    # ===== MERGE MS1 FEATURES (batch processed) - CRITICAL BOTTLENECK =====
    click.echo("\nMerging MS1 features (this may take a while)...")
    for batch_num in range(0, total_files, batch_size):
        batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        
        for idx, infile in enumerate(batch_files):
            db_alias = f"ms1_{batch_num}_{idx}"
            c.executescript(
                f"""
    ATTACH DATABASE "{infile}" AS {db_alias};
    INSERT INTO FEATURE_MS1 SELECT * FROM {db_alias}.FEATURE_MS1;
    DETACH DATABASE {db_alias};
    """
            )
        
        conn.commit()
        conn.close()
        
        files_done = min(batch_num + batch_size, total_files)
        elapsed = time.time() - start_time
        eta = estimate_time_remaining(files_done, total_files, elapsed)
        click.echo(f"Info: Merged MS1 features for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    # ===== MERGE MS2 FEATURES (batch processed) =====
    click.echo("\nMerging MS2 features...")
    for batch_num in range(0, total_files, batch_size):
        batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        
        for idx, infile in enumerate(batch_files):
            db_alias = f"ms2_{batch_num}_{idx}"
            c.executescript(
                f"""
        ATTACH DATABASE "{infile}" AS {db_alias};
        INSERT INTO FEATURE_MS2 SELECT * FROM {db_alias}.FEATURE_MS2;
        DETACH DATABASE {db_alias};
        """
            )
        
        conn.commit()
        conn.close()
        
        files_done = min(batch_num + batch_size, total_files)
        elapsed = time.time() - start_time
        eta = estimate_time_remaining(files_done, total_files, elapsed)
        click.echo(f"Info: Merged MS2 features for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    # ===== MERGE TRANSITION FEATURES (batch processed) =====
    click.echo("\nMerging transition features...")
    for batch_num in range(0, total_files, batch_size):
        batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        
        for idx, infile in enumerate(batch_files):
            db_alias = f"trans_{batch_num}_{idx}"
            c.executescript(
                f"""
        ATTACH DATABASE "{infile}" AS {db_alias};
        INSERT INTO FEATURE_TRANSITION SELECT * FROM {db_alias}.FEATURE_TRANSITION;
        DETACH DATABASE {db_alias};
        """
            )
        
        conn.commit()
        conn.close()
        
        files_done = min(batch_num + batch_size, total_files)
        elapsed = time.time() - start_time
        eta = estimate_time_remaining(files_done, total_files, elapsed)
        click.echo(f"Info: Merged transition features for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    # ===== MERGE ALIGNMENT FEATURES (batch processed) =====
    if feature_alignment_tables_present:
        for alignment_table in feature_alignment_tables:
            click.echo(f"\nMerging {alignment_table} data...")
            for batch_num in range(0, total_files, batch_size):
                batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
                conn = sqlite3.connect(outfile)
                
                # Pre-check which files have this table
                files_with_table = []
                for infile in batch_files:
                    check_conn = sqlite3.connect(infile)
                    if check_sqlite_table(check_conn, alignment_table):
                        files_with_table.append(infile)
                    check_conn.close()
                
                if files_with_table:
                    c = conn.cursor()
                    for idx, infile in enumerate(files_with_table):
                        db_alias = f"align_{batch_num}_{idx}"
                        c.executescript(
                            f"""
            ATTACH DATABASE "{infile}" AS {db_alias};
            INSERT INTO {alignment_table} SELECT * FROM {db_alias}.{alignment_table};
            DETACH DATABASE {db_alias};
            """
                        )
                    
                    conn.commit()
                
                conn.close()
                
                files_done = min(batch_num + batch_size, total_files)
                elapsed = time.time() - start_time
                eta = estimate_time_remaining(files_done, total_files, elapsed)
                if files_with_table:
                    click.echo(f"Info: Merged {alignment_table} for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    # ===== MERGE SCORE TABLES (batch processed) =====
    for score_tbl in score_tables:
        click.echo(f"\nMerging {score_tbl} data...")
        for batch_num in range(0, total_files, batch_size):
            batch_files = infiles[batch_num:min(batch_num + batch_size, total_files)]
            conn = sqlite3.connect(outfile)
            c = conn.cursor()
            
            for idx, infile in enumerate(batch_files):
                db_alias = f"score_{batch_num}_{idx}"
                c.executescript(
                    f"""
    ATTACH DATABASE "{infile}" AS {db_alias};
    INSERT INTO {score_tbl} SELECT * FROM {db_alias}.{score_tbl};
    DETACH DATABASE {db_alias};
    """
                )
            
            conn.commit()
            conn.close()
            
            files_done = min(batch_num + batch_size, total_files)
            elapsed = time.time() - start_time
            eta = estimate_time_remaining(files_done, total_files, elapsed)
            click.echo(f"Info: Merged {score_tbl} for {files_done}/{total_files} files" + (f" (ETA: {eta})" if eta else ""))

    ## Skip VACUUM for now (it's slow) - SQLite will auto-optimize on next use
    click.echo("\nInfo: All Post-Scored OSWS files were merged successfully.")
    total_time = time.time() - start_time
    click.echo(f"Total merge time: {timedelta(seconds=int(total_time))}")



def backpropagate_oswr(infile, outfile, apply_scores):
    # store data in table
    if infile != outfile:
        copyfile(infile, outfile)

    # find out what tables exist in the scores
    score_con = sqlite3.connect(apply_scores)
    peptide_present = check_sqlite_table(score_con, "SCORE_PEPTIDE")
    protein_present = check_sqlite_table(score_con, "SCORE_PROTEIN")
    score_con.close()
    if not (peptide_present or protein_present):
        raise click.ClickException(
            "Backpropagation requires peptide or protein-level contexts."
        )

    # build up the list
    script = list()
    script.append("PRAGMA synchronous = OFF;")
    script.append("DROP TABLE IF EXISTS SCORE_PEPTIDE;")
    script.append("DROP TABLE IF EXISTS SCORE_PROTEIN;")

    # create the tables
    if peptide_present:
        script.append(
            "CREATE TABLE SCORE_PEPTIDE (CONTEXT TEXT, RUN_ID INTEGER, PEPTIDE_ID INTEGER, SCORE REAL, PVALUE REAL, QVALUE REAL, PEP REAL);"
        )
    if protein_present:
        script.append(
            "CREATE TABLE SCORE_PROTEIN (CONTEXT TEXT, RUN_ID INTEGER, PROTEIN_ID INTEGER, SCORE REAL, PVALUE REAL, QVALUE REAL, PEP REAL);"
        )

    # copy across the tables
    script.append(f'ATTACH DATABASE "{apply_scores}" AS sdb;')
    insert_table_fmt = "INSERT INTO {0}\nSELECT *\nFROM sdb.{0};"
    if peptide_present:
        script.append(insert_table_fmt.format("SCORE_PEPTIDE"))
    if protein_present:
        script.append(insert_table_fmt.format("SCORE_PROTEIN"))

    # execute the script
    conn = sqlite3.connect(outfile)
    c = conn.cursor()
    c.executescript("\n".join(script))
    conn.commit()
    conn.close()

    click.echo("Info: All multi-run data was backpropagated.")
