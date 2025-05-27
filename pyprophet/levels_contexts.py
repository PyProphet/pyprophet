import os
import sqlite3
from shutil import copyfile

import click
from loguru import logger

from ._config import LevelContextIOConfig
from .glyco.stats import statistics_report as glyco_statistics_report
from .io.dispatcher import ReaderDispatcher, WriterDispatcher
from .io.util import check_sqlite_table
from .stats import (
    error_statistics,
    final_err_table,
    lookup_values_from_error_table,
    summary_err_table,
)


def statistics_report(
    data,
    parametric,
    pfdr,
    pi0_lambda,
    pi0_method,
    pi0_smooth_df,
    pi0_smooth_log_pi0,
    lfdr_truncate,
    lfdr_monotone,
    lfdr_transformation,
    lfdr_adj,
    lfdr_eps,
    writer,
):
    error_stat, pi0 = error_statistics(
        data[data.decoy == 0]["score"],
        data[data.decoy == 1]["score"],
        parametric,
        pfdr,
        pi0_lambda,
        pi0_method,
        pi0_smooth_df,
        pi0_smooth_log_pi0,
        True,
        lfdr_truncate,
        lfdr_monotone,
        lfdr_transformation,
        lfdr_adj,
        lfdr_eps,
    )

    stat_table = final_err_table(error_stat)
    summary_table = summary_err_table(error_stat)

    # print summary table
    logger.opt(raw=True).info("=" * 80)
    logger.opt(raw=True).info("\n")
    logger.opt(raw=True).info(summary_table)
    logger.opt(raw=True).info("\n")
    logger.opt(raw=True).info("=" * 80)
    logger.opt(raw=True).info("\n")

    p_values, s_values, peps, q_values = lookup_values_from_error_table(
        data["score"].values, error_stat
    )
    data["p_value"] = p_values
    data["s_value"] = s_values
    data["q_value"] = q_values
    data["pep"] = peps

    writer._write_levels_context_pdf_report(data, stat_table, pi0)

    return data


def infer_genes(
    config: LevelContextIOConfig,
):
    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]

    if context == "run-specific":
        data = data.groupby("run_id").apply(
            statistics_report,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )
    elif context in ["global", "experiment-wide"]:
        data = statistics_report(
            data,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    # Store results
    writer.save_results(data)


def infer_proteins(
    config: LevelContextIOConfig,
):
    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    data.columns = [col.lower() for col in data.columns]

    if context == "run-specific":
        data = data.groupby("run_id").apply(
            statistics_report,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )
    elif context in ["global", "experiment-wide"]:
        data = statistics_report(
            data,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    # Store results
    writer.save_results(data)


def infer_peptides(config: LevelContextIOConfig):
    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)
    writer = WriterDispatcher.get_writer(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    if context == "run-specific":
        data = data.groupby("run_id").apply(
            statistics_report,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    elif context in ["global", "experiment-wide"]:
        data = statistics_report(
            data,
            config.error_estimation_config.parametric,
            config.error_estimation_config.pfdr,
            config.error_estimation_config.pi0_lambda,
            config.error_estimation_config.pi0_method,
            config.error_estimation_config.pi0_smooth_df,
            config.error_estimation_config.pi0_smooth_log_pi0,
            config.error_estimation_config.lfdr_truncate,
            config.error_estimation_config.lfdr_monotone,
            config.error_estimation_config.lfdr_transformation,
            config.error_estimation_config.lfdr_adj,
            config.error_estimation_config.lfdr_eps,
            writer,
        )

    # store data in table
    writer.save_results(data)


def infer_glycopeptides(
    config: LevelContextIOConfig,
):
    """
    Infer glycopeptides
    Adapted from: https://github.com/lmsac/GproDIA/blob/main/src/glycoprophet/level_contexts.py
    """
    context = config.context_fdr
    reader = ReaderDispatcher.get_reader(config)

    if context in ["global", "experiment-wide", "run-specific"]:
        data = reader.read()
    else:
        raise click.ClickException("Unspecified context selected.")

    if context == "run-specific":
        data = (
            data.groupby("run_id")
            .apply(
                glyco_statistics_report,
                config.outfile,
                context,
                "glycopeptide",
                density_estimator=config.density_estimator,
                grid_size=config.grid_size,
                parametric=config.error_estimation_config.parametric,
                pfdr=config.error_estimation_config.pfdr,
                pi0_lambda=config.error_estimation_config.pi0_lambda,
                pi0_method=config.error_estimation_config.pi0_method,
                pi0_smooth_df=config.error_estimation_config.pi0_smooth_df,
                pi0_smooth_log_pi0=config.error_estimation_config.pi0_smooth_log_pi0,
                lfdr_truncate=config.error_estimation_config.lfdr_truncate,
                lfdr_monotone=config.error_estimation_config.lfdr_monotone,
                # lfdr_transformation=lfdr_transformation,
                # lfdr_adj=lfdr_adj, lfdr_eps=lfdr_eps
            )
            .reset_index()
        )

    elif context in ["global", "experiment-wide"]:
        data = glyco_statistics_report(
            data,
            config.outfile,
            context,
            "glycopeptide",
            density_estimator=config.density_estimator,
            grid_size=config.grid_size,
            parametric=config.parametric,
            pfdr=config.pfdr,
            pi0_lambda=config.pi0_lambda,
            pi0_method=config.pi0_method,
            pi0_smooth_df=config.pi0_smooth_df,
            pi0_smooth_log_pi0=config.pi0_smooth_log_pi0,
            lfdr_truncate=config.lfdr_truncate,
            lfdr_monotone=config.lfdr_monotone,
            # lfdr_transformation=lfdr_transformation,
            # lfdr_adj=lfdr_adj, lfdr_eps=lfdr_eps
        )

    # store data in table
    writer = WriterDispatcher.get_writer(config)
    writer.save_results(data)


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


def merge_osw(infiles, outfile, templatefile, same_run, merge_post_scored_runs):
    conn = sqlite3.connect(infiles[0])
    reduced = check_sqlite_table(conn, "SCORE_MS2")
    conn.close()
    if reduced and not merge_post_scored_runs:
        click.echo("Calling reduced osws merge function")
        merge_oswr(infiles, outfile, templatefile, same_run)
    elif merge_post_scored_runs:
        click.echo("Calling post scored osws merge function")
        merge_oswps(infiles, outfile, templatefile, same_run)
    else:
        click.echo("Calling pre scored osws merge function")
        merge_osws(infiles, outfile, templatefile, same_run)


def merge_osws(infiles, outfile, templatefile, same_run):
    # Copy the first file to have a template
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
        f"""
PRAGMA synchronous = OFF;

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

ATTACH DATABASE "{infiles[0]}" AS sdb;

CREATE TABLE RUN AS SELECT * FROM sdb.RUN LIMIT 0;

CREATE TABLE FEATURE AS SELECT * FROM sdb.FEATURE LIMIT 0;

CREATE TABLE FEATURE_MS1 AS SELECT * FROM sdb.FEATURE_MS1 LIMIT 0;

CREATE TABLE FEATURE_MS2 AS SELECT * FROM sdb.FEATURE_MS2 LIMIT 0;

CREATE TABLE FEATURE_TRANSITION AS SELECT * FROM sdb.FEATURE_TRANSITION LIMIT 0;

DETACH DATABASE sdb;
"""
    )

    conn.commit()
    conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Only create a single run entry (all files are presumably from the same run)
        if same_run:
            c.executescript(
                f"""INSERT INTO RUN (ID, FILENAME) VALUES ({runid}, '{rname}')"""
            )
            break
        else:
            c.executescript(
                f"""
    ATTACH DATABASE "{infile}" AS sdb;

    INSERT INTO RUN SELECT * FROM sdb.RUN;

    DETACH DATABASE sdb;
    """
            )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged runs of file {infile} to {outfile}.")

    # Now merge the run-specific data into the output file:
    #   Note: only tables FEATURE, FEATURE_MS1, FEATURE_MS2 and FEATURE_TRANSITION are run-specific
    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f"""
    ATTACH DATABASE "{infile}" AS sdb; 

    INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; 

    DETACH DATABASE sdb;
    """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged generic features of file {infile} to {outfile}.")

    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Fix run id assuming we only have a single run
        c.executescript(f"""UPDATE FEATURE SET RUN_ID = {runid}""")

        conn.commit()
        conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()
        c.executescript(
            f"""
    ATTACH DATABASE "{infile}" AS sdb;

    INSERT INTO FEATURE_MS1
    SELECT *
    FROM sdb.FEATURE_MS1;

    DETACH DATABASE sdb;
    """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged MS1 features of file {infile} to {outfile}.")

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f"""
    ATTACH DATABASE "{infile}" AS sdb;

    INSERT INTO FEATURE_MS2
    SELECT *
    FROM sdb.FEATURE_MS2;

    DETACH DATABASE sdb;
    """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged MS2 features of file {infile} to {outfile}.")

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f"""
    ATTACH DATABASE "{infile}" AS sdb;

    INSERT INTO FEATURE_TRANSITION
    SELECT *
    FROM sdb.FEATURE_TRANSITION;

    DETACH DATABASE sdb;
    """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged transition features of file {infile} to {outfile}.")

    click.echo("Info: All OSWS files were merged.")


def merge_oswr(infiles, outfile, templatefile, same_run):
    # Copy the template to the output file
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

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Only create a single run entry (all files are presumably from the same run)
        if same_run:
            c.executescript(
                f"""INSERT INTO RUN (ID, FILENAME) VALUES ({runid}, '{rname}')"""
            )
            break
        else:
            c.executescript(
                f'ATTACH DATABASE "{infile}" AS sdb; INSERT INTO RUN SELECT * FROM sdb.RUN; DETACH DATABASE sdb;'
            )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged runs of file {infile} to {outfile}.")

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f'ATTACH DATABASE "{infile}" AS sdb; INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; DETACH DATABASE sdb;'
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged generic features of file {infile} to {outfile}.")

    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Fix run id assuming we only have a single run
        c.executescript(f"""UPDATE FEATURE SET RUN_ID = {runid}""")

        conn.commit()
        conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f'ATTACH DATABASE "{infile}" AS sdb; INSERT INTO SCORE_MS2 SELECT * FROM sdb.SCORE_MS2; DETACH DATABASE sdb;'
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged MS2 scores of file {infile} to {outfile}.")

    click.echo("Info: All reduced OSWR files were merged.")


def merge_oswps(infiles, outfile, templatefile, same_run):
    click.echo("Info: Merging all Scored Runs.")
    # Copy the first file to have a template
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

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Only create a single run entry (all files are presumably from the same run)
        if same_run:
            c.executescript(
                f"""INSERT INTO RUN (ID, FILENAME) VALUES ({runid}, '{rname}')"""
            )
            break
        else:
            c.executescript(
                f"""
        ATTACH DATABASE "{infile}" AS sdb;
        INSERT INTO RUN SELECT * FROM sdb.RUN;
        DETACH DATABASE sdb;
        """
            )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged runs of file {infile} to {outfile}.")

    # Now merge the run-specific data into the output file:
    #   Note: only tables FEATURE, FEATURE_MS1, FEATURE_MS2 and FEATURE_TRANSITION are run-specific
    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f"""
        ATTACH DATABASE "{infile}" AS sdb; 
        INSERT INTO FEATURE SELECT * FROM sdb.FEATURE; 
        DETACH DATABASE sdb;
        """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged generic features of file {infile} to {outfile}.")

    if same_run:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        # Fix run id assuming we only have a single run
        c.executescript(f"UPDATE FEATURE SET RUN_ID = {runid}")

        conn.commit()
        conn.close()

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f"""
    ATTACH DATABASE "{infile}" AS sdb;
    INSERT INTO FEATURE_MS1
    SELECT *
    FROM sdb.FEATURE_MS1;
    DETACH DATABASE sdb;
    """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged MS1 features of file {infile} to {outfile}.")

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f"""
        ATTACH DATABASE "{infile}" AS sdb;
        INSERT INTO FEATURE_MS2
        SELECT *
        FROM sdb.FEATURE_MS2;
        DETACH DATABASE sdb;
        """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged MS2 features of file {infile} to {outfile}.")

    for infile in infiles:
        conn = sqlite3.connect(outfile)
        c = conn.cursor()

        c.executescript(
            f"""
        ATTACH DATABASE "{infile}" AS sdb;
        INSERT INTO FEATURE_TRANSITION
        SELECT *
        FROM sdb.FEATURE_TRANSITION;
        DETACH DATABASE sdb;
        """
        )

        conn.commit()
        conn.close()

        click.echo(f"Info: Merged transition features of file {infile} to {outfile}.")

    if feature_alignment_tables_present:
        for infile in infiles:
            # Check if the infile contains the feature_alignment table
            conn = sqlite3.connect(infile)
            feature_alignment_present = check_sqlite_table(conn, "FEATURE_ALIGNMENT")
            conn.close()

            if feature_alignment_present:
                conn = sqlite3.connect(outfile)
                c = conn.cursor()

                c.executescript(
                    f"""
            ATTACH DATABASE "{infile}" AS sdb;
            INSERT INTO FEATURE_ALIGNMENT
            SELECT *
            FROM sdb.FEATURE_ALIGNMENT;
            DETACH DATABASE sdb;
            """
                )

                conn.commit()
                conn.close()

                click.echo(
                    f"Info: Merged feature alignment tables of file {infile} to {outfile}."
                )
            else:
                click.echo(f"Warn: No feature alignment table found in file {infile}.")

        # Merge FEATURE_MS2_ALIGNMENT
        for infile in infiles:
            conn = sqlite3.connect(infile)
            feature_ms2_alignment_present = check_sqlite_table(
                conn, "FEATURE_MS2_ALIGNMENT"
            )
            conn.close()

            if feature_ms2_alignment_present:
                conn = sqlite3.connect(outfile)
                c = conn.cursor()

                c.executescript(
                    f"""
            ATTACH DATABASE "{infile}" AS sdb;
            INSERT INTO FEATURE_MS2_ALIGNMENT
            SELECT *
            FROM sdb.FEATURE_MS2_ALIGNMENT;
            DETACH DATABASE sdb;
            """
                )

                conn.commit()
                conn.close()

                click.echo(
                    f"Info: Merged feature MS2 alignment tables of file {infile} to {outfile}."
                )
            else:
                click.echo(
                    f"Warn: No feature MS2 alignment table found in file {infile}."
                )

        # Merge FEATURE_TRANSITION_ALIGNMENT
        for infile in infiles:
            conn = sqlite3.connect(infile)
            feature_transition_alignment_present = check_sqlite_table(
                conn, "FEATURE_TRANSITION_ALIGNMENT"
            )
            conn.close()

            if feature_transition_alignment_present:
                conn = sqlite3.connect(outfile)
                c = conn.cursor()

                c.executescript(
                    f"""
                ATTACH DATABASE "{infile}" AS sdb;
                INSERT INTO FEATURE_TRANSITION_ALIGNMENT
                SELECT *
                FROM sdb.FEATURE_TRANSITION_ALIGNMENT;
                DETACH DATABASE sdb;
                """
                )

                conn.commit()
                conn.close()

                click.echo(
                    f"Info: Merged feature transition alignment tables of file {infile} to {outfile}."
                )
            else:
                click.echo(
                    f"Warn: No feature transition alignment table found in file {infile}."
                )

    for infile in infiles:
        for score_tbl in score_tables:
            conn = sqlite3.connect(outfile)
            c = conn.cursor()

            c.executescript(
                f"""
    ATTACH DATABASE "{infile}" AS sdb;
    INSERT INTO {score_tbl}
    SELECT *
    FROM sdb.{score_tbl};
    DETACH DATABASE sdb;
    """
            )

            conn.commit()
            conn.close()

            click.echo(f"Info: Merged {score_tbl} table of file {infile} to {outfile}.")

    ## Vacuum to clean and re-write rootpage indexes
    conn = sqlite3.connect(outfile)
    c = conn.cursor()

    c.executescript("VACUUM")

    conn.commit()
    conn.close()

    click.echo(f"Info: Cleaned and re-wrote indexing meta-data for {outfile}.")

    click.echo("Info: All Post-Scored OSWS files were merged.")


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
