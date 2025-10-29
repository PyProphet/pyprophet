from __future__ import print_function

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

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


@pytest.fixture
def test_data_compound_osw(temp_folder):
    """Fixture providing compound OSW test file path"""
    src = DATA_FOLDER / "test_data_compound.osw"
    dst = temp_folder / "test_data_compound.osw"
    shutil.copy(src, dst)
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
        print(f"Command failed: {cmd}\n{error.output.decode()}", file=sys.stderr)
        if "NotImplementedError" in error.output.decode(): # attempt to catch the specific error rather than the CalledProcessError
            raise NotImplementedError
        else:
            raise 


def validate_export_results(
    regtest, input_path, input_type, output_file="test_data.tsv"
):
    """Validate exported results"""
    df = pd.read_csv(output_file, sep="\t", nrows=100)
    print(df.sort_index(axis=1), file=regtest)


# ================== TEST CASES ==================
@pytest.mark.parametrize(
    "transition_quantification,peptide,protein",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_osw_analysis(
    input_strategy, temp_folder, regtest, transition_quantification, peptide, protein
):
    """Test OSW analysis with different combinations of options"""
    # MS1-level
    cmd = f"pyprophet score {input_strategy['cmd_prefix']} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02 && "

    # peptide-level
    cmd += f"pyprophet infer peptide --pi0_lambda=0.001 0 0 {input_strategy['cmd_prefix']} --context=run-specific && "
    cmd += f"pyprophet infer peptide --pi0_lambda=0.001 0 0 {input_strategy['cmd_prefix']} --context=experiment-wide && "
    cmd += f"pyprophet infer peptide --pi0_lambda=0.001 0 0 {input_strategy['cmd_prefix']} --context=global && "

    # protein-level
    cmd += f"pyprophet infer protein --pi0_lambda=0 0 0 {input_strategy['cmd_prefix']} --context=run-specific && "
    cmd += f"pyprophet infer protein --pi0_lambda=0 0 0 {input_strategy['cmd_prefix']} --context=experiment-wide && "
    cmd += f"pyprophet infer protein --pi0_lambda=0 0 0 {input_strategy['cmd_prefix']} --context=global && "

    # export
    cmd += f"pyprophet export tsv {input_strategy['cmd_prefix']} --out={temp_folder}/test_data.tsv --max_rs_peakgroup_qvalue=1 --format=legacy_merged"

    if not transition_quantification:
        cmd += " --no-transition_quantification"
    if not peptide:
        cmd += " --no-peptide"
    if not protein:
        cmd += " --no-protein"

    run_pyprophet_command(cmd, temp_folder)
    validate_export_results(
        regtest,
        input_strategy["path"],
        input_strategy["reader"],
        f"{temp_folder}/test_data.tsv",
    )

@pytest.mark.parametrize(
    "calib, rt_unit",
    [ (True, 'iRT'), (False, 'iRT'), (True, 'RT'), (False, 'RT')]
)
def test_osw_analysis_libExport(input_strategy, temp_folder, regtest, calib, rt_unit
):
    cmd = f"pyprophet score {input_strategy['cmd_prefix']} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02 && "

    # peptide-level
    cmd += f"pyprophet infer peptide --pi0_lambda=0.001 0 0 {input_strategy['cmd_prefix']} --context=global && "

    # protein-level
    cmd += f"pyprophet infer protein --pi0_lambda=0 0 0 {input_strategy['cmd_prefix']} --context=global && "


    # export
    if calib:
        cmd += f"pyprophet export library {input_strategy['cmd_prefix']} --out={temp_folder}/test_lib.tsv --test --max_peakgroup_qvalue=1 --max_global_peptide_qvalue=1 --max_global_protein_qvalue=1 --rt_unit={rt_unit}"
    else:
        cmd += f"pyprophet export library {input_strategy['cmd_prefix']} --out={temp_folder}/test_lib.tsv --test --max_peakgroup_qvalue=1 --max_global_peptide_qvalue=1 --max_global_protein_qvalue=1 --no-rt_calibration --no-im_calibration --no-intensity_calibration --rt_unit={rt_unit}"

    if not input_strategy["reader"] == "parquet_split":
        with pytest.raises(NotImplementedError):
            run_pyprophet_command(cmd, temp_folder)
    else:
        run_pyprophet_command(cmd, temp_folder)
        validate_export_results(
            regtest,
            input_strategy["path"],
            input_strategy["reader"],
            f"{temp_folder}/test_lib.tsv",
        )

def test_osw_unscored(input_strategy, temp_folder, regtest):
    """Test export of unscored OSW data"""
    cmd = f"pyprophet export tsv {input_strategy['cmd_prefix']} --out={temp_folder}/test_data.tsv --format=legacy_merged"
    run_pyprophet_command(cmd, temp_folder)
    validate_export_results(
        regtest,
        input_strategy["path"],
        input_strategy["reader"],
        f"{temp_folder}/test_data.tsv",
    )


@pytest.mark.parametrize(
    "transition_quantification,ipf",
    [
        (False, "disable"),
        (True, "disable"),
        (False, "peptidoform"),
        (False, "augmented"),
    ],
)
def test_ipf_analysis(
    test_data_osw, temp_folder, regtest, transition_quantification, ipf
):
    """Test IPF analysis with different options"""
    # MS1-level
    cmd = f"pyprophet score --in={test_data_osw} --level=ms1 --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02 && "

    # MS2-level
    cmd += f"pyprophet score --in={test_data_osw} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02 && "

    # transition-level
    cmd += f"pyprophet score --in={test_data_osw} --level=transition --test --pi0_lambda=0.1 0 0 --ss_iteration_fdr=0.02 && "

    # IPF
    cmd += f"pyprophet infer peptidoform --in={test_data_osw} && "

    # export
    cmd += f"pyprophet export tsv --in={test_data_osw} --out={temp_folder}/test_data.tsv --no-peptide --no-protein --ipf_max_peptidoform_pep=1 --max_rs_peakgroup_qvalue=1 --format=legacy_merged"

    if not transition_quantification:
        cmd += " --no-transition_quantification"

    cmd += f" --ipf={ipf}"

    run_pyprophet_command(cmd, temp_folder)
    validate_export_results(
        regtest, test_data_osw, "osw", f"{temp_folder}/test_data.tsv"
    )


# Compound tests (only support OSW)
def test_compound_unscored(test_data_compound_osw, temp_folder, regtest):
    """Test export of unscored compound data"""
    cmd = f"pyprophet export compound --in={test_data_compound_osw} --out={temp_folder}/test_data_compound_unscored.tsv --format=legacy_merged"
    run_pyprophet_command(cmd, temp_folder)

    df = pd.read_csv(
        f"{temp_folder}/test_data_compound_unscored.tsv",
        sep="\t",
        nrows=100,
    )
    print(df.sort_index(axis=1), file=regtest)


def test_compound_ms1(test_data_compound_osw, temp_folder, regtest):
    """Test compound analysis with MS1-level scoring"""
    cmd = f"pyprophet score --in={test_data_compound_osw} --level=ms1 --test &&"
    cmd += f"pyprophet export compound --in={test_data_compound_osw} --out={temp_folder}/test_data_compound_ms1.tsv --max_rs_peakgroup_qvalue=0.05 --format=legacy_merged"

    run_pyprophet_command(cmd, temp_folder)

    df = pd.read_csv(f"{temp_folder}/test_data_compound_ms1.tsv", sep="\t", nrows=100)
    print(df.sort_index(axis=1), file=regtest)


def test_compound_ms2(test_data_compound_osw, temp_folder, regtest):
    """Test compound analysis with MS2-level scoring"""
    cmd = f"pyprophet score --in={test_data_compound_osw} --level=ms2 --test &&"
    cmd += f"pyprophet export compound --in={test_data_compound_osw} --out={temp_folder}/test_data_compound_ms2.tsv --max_rs_peakgroup_qvalue=0.05 --format=legacy_merged"

    run_pyprophet_command(cmd, temp_folder)

    df = pd.read_csv(f"{temp_folder}/test_data_compound_ms2.tsv", sep="\t", nrows=100)
    print(df.sort_index(axis=1), file=regtest)


# ================== PARQUET EXPORT TESTS ==================
def test_parquet_export_scored_osw(test_data_osw, temp_folder, regtest):
    """Test exporting scored OSW with SCORE_ tables to parquet format"""
    # Score at MS2 level
    cmd = f"pyprophet score --in={test_data_osw} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02 && "
    
    # Infer peptide level with global context
    cmd += f"pyprophet infer peptide --pi0_lambda=0.001 0 0 --in={test_data_osw} --context=global && "
    
    # Infer protein level with global context
    cmd += f"pyprophet infer protein --pi0_lambda=0 0 0 --in={test_data_osw} --context=global && "
    
    # Export to parquet (should include SCORE_ tables)
    cmd += f"pyprophet export parquet --in={test_data_osw} --out={temp_folder}/test_data_scored.parquet"
    
    run_pyprophet_command(cmd, temp_folder)
    
    # Verify the parquet file exists and has data
    import pyarrow.parquet as pq
    table = pq.read_table(f"{temp_folder}/test_data_scored.parquet")
    df = table.to_pandas()
    
    # Check that we have data
    assert len(df) > 0, "Exported parquet file should not be empty"
    
    # Check that score columns are present
    score_columns = [col for col in df.columns if col.startswith('SCORE_')]
    assert len(score_columns) > 0, "Exported parquet should contain SCORE_ columns"
    
    print(f"Exported {len(df)} rows with {len(df.columns)} columns", file=regtest)
    print(f"Score columns found: {sorted(score_columns)}", file=regtest)
    print(df.head(10).sort_index(axis=1), file=regtest)


def test_parquet_export_no_transition_data(test_data_osw, temp_folder, regtest):
    """Test exporting parquet without transition data using --no-include_transition_data flag"""
    # Score at MS2 level
    cmd = f"pyprophet score --in={test_data_osw} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02 && "
    
    # Infer peptide level with global context
    cmd += f"pyprophet infer peptide --pi0_lambda=0.001 0 0 --in={test_data_osw} --context=global && "
    
    # Infer protein level with global context
    cmd += f"pyprophet infer protein --pi0_lambda=0 0 0 --in={test_data_osw} --context=global && "
    
    # Export to parquet without transition data
    cmd += f"pyprophet export parquet --in={test_data_osw} --out={temp_folder}/test_data_no_transition.parquet --no-include_transition_data"
    
    run_pyprophet_command(cmd, temp_folder)
    
    # Verify the parquet file exists and has data
    import pyarrow.parquet as pq
    table = pq.read_table(f"{temp_folder}/test_data_no_transition.parquet")
    df = table.to_pandas()
    
    # Check that we have data
    assert len(df) > 0, "Exported parquet file should not be empty"
    
    # Check that transition-specific columns are NOT present
    # transition_columns = [col for col in df.columns if 'TRANSITION' in col.upper()]
    # assert len(transition_columns) == 0, "Exported parquet should not contain TRANSITION columns when --no-include_transition_data is used"
    assert df['TRANSITION_ID'].isnull().all(), "TRANSITION_ID column should be empty when --no-include_transition_data is used"
    
    # Check that score columns are present
    score_columns = [col for col in df.columns if col.startswith('SCORE_')]
    assert len(score_columns) > 0, "Exported parquet should contain SCORE_ columns"
    
    print(f"Exported {len(df)} rows with {len(df.columns)} columns (no transition data)", file=regtest)
    print(f"Score columns found: {sorted(score_columns)}", file=regtest)
    print(df.head(10).sort_index(axis=1), file=regtest)


def test_parquet_export_split_format(test_data_osw, temp_folder, regtest):
    """Test exporting to split parquet format with score data"""
    # Score at MS2 level
    cmd = f"pyprophet score --in={test_data_osw} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02 && "
    
    # Infer peptide level with global context
    cmd += f"pyprophet infer peptide --pi0_lambda=0.001 0 0 --in={test_data_osw} --context=global && "
    
    # Infer protein level with global context
    cmd += f"pyprophet infer protein --pi0_lambda=0 0 0 --in={test_data_osw} --context=global && "
    
    # Export to split parquet format
    cmd += f"pyprophet export parquet --in={test_data_osw} --out={temp_folder}/test_data_split --split_transition_data"
    
    run_pyprophet_command(cmd, temp_folder)
    
    # Verify the directory exists and contains parquet files
    import pyarrow.parquet as pq
    split_dir = Path(temp_folder) / "test_data_split"
    assert split_dir.exists(), "Split parquet directory should exist"
    
    precursor_file = split_dir / "precursors_features.parquet"
    transition_file = split_dir / "transition_features.parquet"
    
    assert precursor_file.exists(), "precursors_features.parquet should exist"
    assert transition_file.exists(), "transition_features.parquet should exist"
    
    # Read precursor data
    precursor_table = pq.read_table(str(precursor_file))
    precursor_df = precursor_table.to_pandas()
    
    # Read transition data
    transition_table = pq.read_table(str(transition_file))
    transition_df = transition_table.to_pandas()
    
    # Check that we have data in both files
    assert len(precursor_df) > 0, "Precursor parquet file should not be empty"
    assert len(transition_df) > 0, "Transition parquet file should not be empty"
    
    # Check that score columns are present in precursor file
    precursor_score_columns = [col for col in precursor_df.columns if col.startswith('SCORE_')]
    assert len(precursor_score_columns) > 0, "Precursor parquet should contain SCORE_ columns"
    
    print(f"Precursor data: {len(precursor_df)} rows with {len(precursor_df.columns)} columns", file=regtest)
    print(f"Transition data: {len(transition_df)} rows with {len(transition_df.columns)} columns", file=regtest)
    print(f"Precursor score columns: {sorted(precursor_score_columns)}", file=regtest)
    print("Precursor data sample:", file=regtest)
    print(precursor_df.head(5).sort_index(axis=1), file=regtest)
