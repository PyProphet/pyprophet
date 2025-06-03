from __future__ import print_function

import os
import subprocess
import shutil
import sys
from pathlib import Path

import pandas as pd
import sqlite3

import pytest

# from pyprophet.ipf import read_pyp_peakgroup_precursor
from pyprophet.io.dispatcher import ReaderDispatcher
from pyprophet._config import IPFIOConfig

pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None

DATA_FOLDER = Path(__file__).parent / "data"


# ================== FIXTURES ==================
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


def validate_output(
    regtest, input_path, input_type, ipf_ms1_scoring=True, ipf_ms2_scoring=True
):
    """Validate IPF output data"""

    config = IPFIOConfig(
        infile=input_path,
        outfile=input_path,
        subsample_ratio=1.0,
        level="peakgroup_precursor",
        context="ipf",
    )
    config.file_type = input_type
    config.ipf_ms1_scoring = ipf_ms1_scoring
    config.ipf_ms2_scoring = ipf_ms2_scoring
    reader = ReaderDispatcher.get_reader(config)

    # Validate precursor-level data
    precursor_data = reader.read(level="peakgroup_precursor")
    print("Precursor Data:", file=regtest)
    print(precursor_data.sort_index(axis=1).head(100), file=regtest)

    # Validate transition-level data
    transition_data = reader.read(level="transition")
    print("Transition Data:", file=regtest)
    print(transition_data.sort_index(axis=1).head(100), file=regtest)

    # For OSW files, also validate transition and IPF data
    if input_type == "osw":
        with sqlite3.connect(input_path) as con:
            # IPF data
            ipf_data = pd.read_sql_query("SELECT * FROM SCORE_IPF LIMIT 100", con)
            ipf_data.columns = [col.lower() for col in ipf_data.columns]
            print("IPF Data:", file=regtest)
            print(ipf_data.sort_index(axis=1).head(100), file=regtest)
    elif input_type == "parquet":
        cols = [
            "FEATURE_ID",
            "PEPTIDE_ID",
            "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP",
            "SCORE_IPF_QVALUE",
            "SCORE_IPF_PEP",
        ]
        ipf_data = pd.read_parquet(input_path, columns=cols)
        ipf_data = ipf_data.rename(
            columns={
                "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP": "PRECURSOR_PEAKGROUP_PEP",
                "SCORE_IPF_QVALUE": "QVALUE",
                "SCORE_IPF_PEP": "PEP",
            }
        )
        ipf_data.columns = [col.lower() for col in ipf_data.columns]
        print("IPF Data:", file=regtest)
        print(ipf_data.sort_index(axis=1).head(100), file=regtest)
    elif input_type == "parquet_split":
        cols = [
            "FEATURE_ID",
            "PEPTIDE_ID",
            "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP",
            "SCORE_IPF_QVALUE",
            "SCORE_IPF_PEP",
        ]
        ipf_data = pd.read_parquet(
            os.path.join(input_path, "precursors_features.parquet"), columns=cols
        )
        ipf_data = ipf_data.rename(
            columns={
                "SCORE_IPF_PRECURSOR_PEAKGROUP_PEP": "PRECURSOR_PEAKGROUP_PEP",
                "SCORE_IPF_QVALUE": "QVALUE",
                "SCORE_IPF_PEP": "PEP",
            }
        )
        ipf_data.columns = [col.lower() for col in ipf_data.columns]
        print("IPF Data:", file=regtest)
        print(ipf_data.sort_index(axis=1).head(100), file=regtest)


# ================== TEST CASES ==================
@pytest.mark.parametrize("ipf_ms1_scoring", [True, False], ids=["ms1_on", "ms1_off"])
@pytest.mark.parametrize("ipf_ms2_scoring", [True, False], ids=["ms2_on", "ms2_off"])
@pytest.mark.parametrize("ipf_h0", [True, False], ids=["h0_on", "h0_off"])
def test_ipf_scoring(
    input_strategy, temp_folder, regtest, ipf_ms1_scoring, ipf_ms2_scoring, ipf_h0
):
    """Test IPF scoring with different configurations and input strategies"""
    # Build base command
    cmd = f"pyprophet score {input_strategy['cmd_prefix']} --level=ms1 --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02"

    # Add MS2 scoring
    cmd += f" && pyprophet score {input_strategy['cmd_prefix']} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"

    # Add transition scoring
    cmd += f" && pyprophet score {input_strategy['cmd_prefix']} --level=transition --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02"

    # Add IPF command with parameters
    cmd += f" && pyprophet ipf {input_strategy['cmd_prefix']}"
    cmd += " --ipf_ms1_scoring" if ipf_ms1_scoring else " --no-ipf_ms1_scoring"
    cmd += " --ipf_ms2_scoring" if ipf_ms2_scoring else " --no-ipf_ms2_scoring"
    cmd += " --ipf_h0" if ipf_h0 else " --no-ipf_h0"

    # Execute commands
    run_pyprophet_command(cmd, temp_folder)

    # Validate output
    validate_output(
        regtest,
        str(input_strategy["path"]),
        input_strategy["reader"],
        ipf_ms1_scoring,
        ipf_ms2_scoring,
    )
