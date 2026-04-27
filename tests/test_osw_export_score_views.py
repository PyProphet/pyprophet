import sqlite3

import duckdb

from pyprophet._config import ExportIOConfig
from pyprophet.io.export.osw import OSWWriter
from pyprophet.io.util import load_sqlite_scanner


def _create_score_test_osw(path, include_experiment_wide: bool) -> None:
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE PEPTIDE (ID INTEGER PRIMARY KEY);
            CREATE TABLE PROTEIN (ID INTEGER PRIMARY KEY);
            CREATE TABLE PEPTIDE_PROTEIN_MAPPING (PEPTIDE_ID INTEGER, PROTEIN_ID INTEGER);
            CREATE TABLE FEATURE (ID INTEGER PRIMARY KEY, RUN_ID INTEGER);
            CREATE TABLE SCORE_PEPTIDE (
                CONTEXT TEXT,
                RUN_ID INTEGER,
                PEPTIDE_ID INTEGER,
                SCORE REAL,
                PVALUE REAL,
                QVALUE REAL,
                PEP REAL
            );
            CREATE TABLE SCORE_PROTEIN (
                CONTEXT TEXT,
                RUN_ID INTEGER,
                PROTEIN_ID INTEGER,
                SCORE REAL,
                PVALUE REAL,
                QVALUE REAL,
                PEP REAL
            );
            """
        )
        con.execute("INSERT INTO PEPTIDE VALUES (1)")
        con.execute("INSERT INTO PROTEIN VALUES (10)")
        con.execute("INSERT INTO PEPTIDE_PROTEIN_MAPPING VALUES (1, 10)")
        con.executemany(
            "INSERT INTO FEATURE VALUES (?, ?)",
            [(100, 1), (101, 2)],
        )

        peptide_rows = [("global", None, 1, 5.0, 0.005, 0.0005, 0.0002)]
        protein_rows = [("global", None, 10, 7.0, 0.007, 0.0007, 0.0003)]
        if include_experiment_wide:
            peptide_rows.extend(
                [
                    ("experiment-wide", 1, 1, 3.0, 0.03, 0.003, 0.001),
                    ("experiment-wide", 2, 1, 4.0, 0.04, 0.004, 0.002),
                ]
            )
            protein_rows.extend(
                [
                    ("experiment-wide", 1, 10, 6.0, 0.06, 0.006, 0.004),
                    ("experiment-wide", 2, 10, 8.0, 0.08, 0.008, 0.005),
                ]
            )

        con.executemany(
            "INSERT INTO SCORE_PEPTIDE VALUES (?, ?, ?, ?, ?, ?, ?)",
            peptide_rows,
        )
        con.executemany(
            "INSERT INTO SCORE_PROTEIN VALUES (?, ?, ?, ?, ?, ?, ?)",
            protein_rows,
        )
        con.commit()


def _read_joined_scores(path, peptide_contexts, protein_contexts):
    config = ExportIOConfig(
        infile=str(path),
        outfile=str(path.with_suffix(".parquet")),
        subsample_ratio=1.0,
        level="osw",
        context="export",
        export_format="parquet",
    )
    writer = OSWWriter(config)
    column_info = {
        "score_ms1_exists": False,
        "score_ms2_exists": False,
        "score_ipf_exists": False,
        "score_peptide_exists": True,
        "score_protein_exists": True,
        "score_peptide_contexts": peptide_contexts,
        "score_protein_contexts": protein_contexts,
    }

    con = duckdb.connect(":memory:")
    load_sqlite_scanner(con)
    try:
        score_cols_select, score_table_joins, score_column_views = (
            writer._build_score_column_selection_and_joins(column_info)
        )
        query = f"""
            {score_column_views}
            SELECT
                FEATURE.RUN_ID,
                {score_cols_select}
            FROM sqlite_scan('{path}', 'FEATURE') AS FEATURE
            CROSS JOIN sqlite_scan('{path}', 'PEPTIDE') AS PEPTIDE
            INNER JOIN sqlite_scan('{path}', 'PEPTIDE_PROTEIN_MAPPING') AS PEPTIDE_PROTEIN_MAPPING
                ON PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
            {score_table_joins}
            ORDER BY FEATURE.RUN_ID
        """
        return con.execute(query).fetchdf()
    finally:
        con.close()


def test_export_score_views_keep_global_scores_when_run_id_is_null(tmp_path):
    osw_path = tmp_path / "mixed_contexts.osw"
    _create_score_test_osw(osw_path, include_experiment_wide=True)

    df = _read_joined_scores(
        osw_path,
        peptide_contexts=["experiment-wide", "global"],
        protein_contexts=["experiment-wide", "global"],
    )

    assert df["SCORE_PEPTIDE_GLOBAL_Q_VALUE"].tolist() == [0.0005, 0.0005]
    assert df["SCORE_PROTEIN_GLOBAL_Q_VALUE"].tolist() == [0.0007, 0.0007]
    assert df["SCORE_PEPTIDE_EXPERIMENT_WIDE_Q_VALUE"].tolist() == [0.003, 0.004]
    assert df["SCORE_PROTEIN_EXPERIMENT_WIDE_Q_VALUE"].tolist() == [0.006, 0.008]


def test_export_score_views_match_global_only_scores_when_run_id_is_null(tmp_path):
    osw_path = tmp_path / "global_only_contexts.osw"
    _create_score_test_osw(osw_path, include_experiment_wide=False)

    df = _read_joined_scores(
        osw_path,
        peptide_contexts=["global"],
        protein_contexts=["global"],
    )

    assert df["SCORE_PEPTIDE_GLOBAL_Q_VALUE"].tolist() == [0.0005, 0.0005]
    assert df["SCORE_PROTEIN_GLOBAL_Q_VALUE"].tolist() == [0.0007, 0.0007]
