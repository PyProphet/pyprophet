from __future__ import print_function

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pyprophet._config import LevelContextIOConfig
from pyprophet.io.dispatcher import ReaderDispatcher

pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None

DATA_FOLDER = Path(__file__).parent / "data"


# ================== SHARED FIXTURES ==================
@pytest.fixture
def temp_folder(tmpdir):
    """Fixture providing temporary folder path"""
    return Path(tmpdir.strpath)


@pytest.fixture
def test_data_osw(temp_folder):
    """Fixture providing OSW test file path"""
    src = DATA_FOLDER / "test_data.osw"
    dst = temp_folder / "test_data.osw"
    shutil.copy(src, dst)
    return dst


@pytest.fixture
def test_data_parquet(temp_folder):
    """Fixture providing Parquet test file path"""
    src = DATA_FOLDER / "test_data.parquet"
    dst = temp_folder / "test_data.parquet"
    shutil.copy(src, dst)
    return dst


@pytest.fixture
def test_data_split_parquet(temp_folder):
    """Fixture providing SplitParquet test folder"""
    src = DATA_FOLDER / "test_data.oswpq"
    dst = temp_folder / "test_data.oswpq"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture(params=["osw", "parquet", "split_parquet"])
def input_strategy(request, test_data_osw, test_data_parquet, test_data_split_parquet):
    """Parametrized fixture for different input strategies"""
    strategies = {
        "osw": {
            "path": test_data_osw,
            "reader": "osw",
            "cmd_prefix": f"--in={test_data_osw}",
        },
        "parquet": {
            "path": test_data_parquet,
            "reader": "parquet",
            "cmd_prefix": f"--in={test_data_parquet}",
        },
        "split_parquet": {
            "path": test_data_split_parquet,
            "reader": "parquet_split",
            "cmd_prefix": f"--in={test_data_split_parquet}",
        },
    }
    return strategies[request.param]


# ================== TEST HELPERS ==================
def run_pyprophet_command(cmd, temp_folder):
    """Helper to run pyprophet commands"""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, cwd=temp_folder
        ).decode()
    except subprocess.CalledProcessError as error:
        print(f"Command failed: {cmd}\n{error.output.decode()}")
        raise


def validate_peptide_results(regtest, input_path, input_type, context):
    """Validate peptide-level results"""
    # config = LevelContextIOConfig(
    #     infile=input_path,
    #     outfile=input_path,
    #     subsample_ratio=1.0,
    #     level="peptide",
    #     context="levels_context",
    # )
    # config.file_type = input_type
    # config.context_fdr = context
    # reader = ReaderDispatcher.get_reader(config)

    if input_type == "osw":
        sort_cols = [
            "context",
            "run_id",
            "peptide_id",
            "score",
            "pvalue",
            "qvalue",
            "pep",
        ]
        with sqlite3.connect(input_path) as con:
            peptide_data = pd.read_sql_query(
                "SELECT * FROM SCORE_PEPTIDE ORDER BY CONTEXT, RUN_ID, PEPTIDE_ID, SCORE, PVALUE, QVALUE, PEP LIMIT 100",
                con,
            )
            peptide_data.columns = [col.lower() for col in peptide_data.columns]
            print("Peptide Data:", file=regtest)
            print(peptide_data.sort_values(sort_cols).head(100), file=regtest)
    else:
        # For Parquet/SplitParquet, use appropriate reader
        cols = [
            "RUN_ID",
            "FEATURE_ID",
            "PEPTIDE_ID",
            f"SCORE_PEPTIDE_{context.upper().replace('-', '_')}_SCORE",
            f"SCORE_PEPTIDE_{context.upper().replace('-', '_')}_P_VALUE",
            f"SCORE_PEPTIDE_{context.upper().replace('-', '_')}_Q_VALUE",
            f"SCORE_PEPTIDE_{context.upper().replace('-', '_')}_PEP",
        ]

        peptide_data = pd.read_parquet(input_path, columns=cols)

        # Convert IDs to nullable integers (Int64)
        for id_col in ["RUN_ID", "FEATURE_ID"]:
            peptide_data[id_col] = peptide_data[id_col].astype("Int64")

        # Filter NULLs and sort consistently
        peptide_data = peptide_data[peptide_data["RUN_ID"].notnull()]
        peptide_data = peptide_data.sort_values(cols).reset_index(drop=True)

        print("Peptide Data:", file=regtest)
        print(peptide_data.head(100), file=regtest)


def validate_protein_results(regtest, input_path, input_type, context):
    """Validate protein-level results"""
    if input_type == "osw":
        sort_cols = [
            "context",
            "run_id",
            "protein_id",
            "score",
            "pvalue",
            "qvalue",
            "pep",
        ]
        with sqlite3.connect(input_path) as con:
            protein_data = pd.read_sql_query(
                "SELECT * FROM SCORE_PROTEIN ORDER BY CONTEXT, RUN_ID, PROTEIN_ID, SCORE, PVALUE, QVALUE, PEP LIMIT 100",
                con,
            )
            protein_data.columns = [col.lower() for col in protein_data.columns]
            print("Protein Data:", file=regtest)
            print(protein_data.sort_values(sort_cols).head(100), file=regtest)
    else:
        # For Parquet/SplitParquet, use appropriate reader
        cols = [
            "RUN_ID",
            "FEATURE_ID",
            "PROTEIN_ID",
            f"SCORE_PROTEIN_{context.upper().replace('-', '_')}_SCORE",
            f"SCORE_PROTEIN_{context.upper().replace('-', '_')}_P_VALUE",
            f"SCORE_PROTEIN_{context.upper().replace('-', '_')}_Q_VALUE",
            f"SCORE_PROTEIN_{context.upper().replace('-', '_')}_PEP",
        ]

        protein_data = pd.read_parquet(input_path, columns=cols)

        # Convert IDs to nullable integers
        for id_col in ["RUN_ID", "FEATURE_ID"]:
            protein_data[id_col] = protein_data[id_col].astype("Int64")

        protein_data = protein_data[protein_data["RUN_ID"].notnull()]
        protein_data = protein_data.sort_values(cols).reset_index(drop=True)

        print("Protein Data:", file=regtest)
        print(protein_data.head(100), file=regtest)


# ================== TEST CASES ==================
@pytest.mark.parametrize("context", ["run-specific", "experiment-wide", "global"])
def test_peptide_levels(input_strategy, temp_folder, regtest, context):
    """Test peptide-level analysis with different contexts"""
    # Build base command
    cmd = f"pyprophet score {input_strategy['cmd_prefix']} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # Add peptide command
    cmd += f" && pyprophet infer peptide --pi0_lambda=0.001 0 0 {input_strategy['cmd_prefix']} --context={context}"

    # Execute commands
    run_pyprophet_command(cmd, temp_folder)

    # Validate output
    validate_peptide_results(
        regtest, input_strategy["path"], input_strategy["reader"], context
    )


@pytest.mark.parametrize("context", ["run-specific", "experiment-wide", "global"])
def test_protein_levels(input_strategy, temp_folder, regtest, context):
    """Test protein-level analysis with different contexts"""
    # Build base command
    cmd = f"pyprophet score {input_strategy['cmd_prefix']} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # Add protein command
    cmd += f" && pyprophet infer protein --pi0_lambda=0 0 0 {input_strategy['cmd_prefix']} --context={context}"

    # Execute commands
    run_pyprophet_command(cmd, temp_folder)

    # Validate output
    validate_protein_results(
        regtest,
        input_strategy["path"],
        input_strategy["reader"],
        context,
    )
