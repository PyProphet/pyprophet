import pytest
import os
import shutil
from pathlib import Path
import subprocess
import pandas as pd


from pyprophet.io.levels_context.osw import OSWReader
from pyprophet.io.levels_context.parquet import ParquetReader
from pyprophet.io.levels_context.split_parquet import SplitParquetReader
from pyprophet._config import LevelContextIOConfig, ErrorEstimationConfig


pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None

DATA_FOLDER = Path(__file__).parent / "data"


# ================== TEST UTILITIES ==================


def create_reader_config(level, context, infile, outfile):
    """
    Common config generator to avoid repetition
    """
    return LevelContextIOConfig(
        infile=infile,
        outfile=outfile,
        subsample_ratio=1,
        context="levels_context",
        level=level,
        context_fdr=context,
        error_estimation_config=ErrorEstimationConfig(),
        color_palette="normal",
        density_estimator="gmm",
        grid_size=256,
    )


def compare_dataframes(df1, df2, level_key_id, cols):
    """
    Compare two pandas DataFrames based on the selected columns.

    Parameters:
    - df1 (pandas.DataFrame): The first DataFrame to compare.
    - df2 (pandas.DataFrame): The second DataFrame to compare.
    - cols (list): List of columns to use for comparison.

    Raises:
    - AssertionError: If either DataFrame is empty or if the selected columns do not match.

    Returns:
    - None
    """
    # Sort both dataframes by the same columns
    df1_sorted = df1.sort_values(by=["run_id", level_key_id, "score"]).reset_index(
        drop=True
    )
    df2_sorted = df2.sort_values(by=["run_id", level_key_id, "score"]).reset_index(
        drop=True
    )

    # Ensure that both dataframes are non-empty
    assert not df1_sorted.empty, "First dataframe is empty"
    assert not df2_sorted.empty, "Second dataframe is empty"

    # Perform the comparison on the selected columns
    # We allow for slight numerical differences using relative tolerance (rtol)
    pd.testing.assert_frame_equal(
        df1_sorted[cols], df2_sorted[cols], check_dtype=False, rtol=1e-1
    )


def get_comparison_columns(reader_type, level):
    """Return appropriate columns based on reader type."""

    level_key_id = "peptide_id" if level == "peptide" else "protein_id"
    base_cols = ["run_id", "group_id", level_key_id, "decoy", "score", "context"]

    return base_cols


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
            "reader": OSWReader,
            "file_type": "osw",
            "cmd_prefix": f"--in={test_data_osw}",
        },
        "parquet": {
            "path": test_data_parquet,
            "reader": ParquetReader,
            "file_type": "parquet",
            "cmd_prefix": f"--in={test_data_parquet}",
        },
        "split_parquet": {
            "path": test_data_split_parquet,
            "reader": SplitParquetReader,
            "file_type": "parquet_split",
            "cmd_prefix": f"--in={test_data_split_parquet}",
        },
    }
    return strategies[request.param]


# ================== TEST UTILITIES ==================
def run_pyprophet_command(cmd, temp_folder):
    """Helper to run pyprophet commands"""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, cwd=temp_folder
        ).decode()
    except subprocess.CalledProcessError as error:
        print(f"Command failed: {cmd}\n{error.output.decode()}")
        raise


def run_scoring(temp_folder, input_strategy):
    """Run initial scoring for the test data"""
    cmd = f"pyprophet score {input_strategy['cmd_prefix']} --level=ms2 --test --pi0_lambda=0.001 0 0 --ss_iteration_fdr=0.02"
    run_pyprophet_command(cmd, temp_folder)


def run_level_context(temp_folder, input_strategy, level, context):
    """Run level context inference"""
    cmd = f"pyprophet levels-context {level} {input_strategy['cmd_prefix']} --context={context} --pi0_lambda=0.001 0 0"
    run_pyprophet_command(cmd, temp_folder)


# ================== UPDATED TEST ==================


@pytest.mark.parametrize("level", ["peptide", "protein"])
@pytest.mark.parametrize("context", ["global", "experiment-wide", "run-specific"])
def test_reader_level_context(input_strategy, temp_folder, level, context, request):
    """
    Test function to validate the output of a reader for a given level and context.
    """
    # 1. Run initial scoring
    run_scoring(temp_folder, input_strategy)

    # 2. Run level context inference - we will test the actual CLI command in another test.
    # run_level_context(temp_folder, input_strategy, level, context)

    # 3. Test reading the results
    config = create_reader_config(
        level=level,
        context=context,
        infile=input_strategy["path"],
        outfile=temp_folder / f"output_{level}_{context}",
    )

    # Initialize appropriate reader
    reader_class = input_strategy["reader"]
    reader = reader_class(config)

    # Read and validate data
    df = reader.read()
    level_key_id = "peptide_id" if level == "peptide" else "protein_id"

    # Basic checks
    assert isinstance(df, pd.DataFrame), (
        f"Invalid return type from {reader_class.__name__}"
    )
    assert not df.empty, f"Empty DataFrame from {reader_class.__name__}"
    assert "run_id" in df.columns, "Missing run_id column"
    assert level_key_id in df.columns, f"Missing {level_key_id} column"


@pytest.mark.parametrize("level", ["peptide", "protein"])
@pytest.mark.parametrize("context", ["global", "experiment-wide", "run-specific"])
def test_compare_readers(
    input_strategy,
    temp_folder,
    level,
    context,
    test_data_osw,
    test_data_parquet,
    test_data_split_parquet,
):
    """
    Test that all readers produce consistent output for a given level and context,
    using different input files for each reader type.
    """
    # 1. Run initial scoring for primary reader
    run_scoring(temp_folder, input_strategy)

    # Create config for primary reader
    primary_config = create_reader_config(
        level=level,
        context=context,
        infile=input_strategy["path"],
        outfile=temp_folder / f"output_{level}_{context}",
    )
    primary_config.file_type = input_strategy["file_type"]

    print(f"\nPrimary reader: {input_strategy['reader'].__name__}")
    print(f"Input file: {input_strategy['path']}")

    # Initialize and test primary reader
    primary_reader = input_strategy["reader"](primary_config)
    df_primary = primary_reader.read()
    level_key_id = "peptide_id" if level == "peptide" else "protein_id"
    cols = get_comparison_columns(input_strategy["file_type"], level)

    # Basic dataframe checks
    assert isinstance(df_primary, pd.DataFrame), "Invalid return type"
    assert not df_primary.empty, "Empty dataframe returned"
    assert "run_id" in df_primary.columns, "Missing run_id"
    assert level_key_id in df_primary.columns, f"Missing {level_key_id}"

    # Sort primary dataframe for comparison
    df_primary_sorted = df_primary.sort_values(
        by=["run_id", level_key_id, "score"]
    ).reset_index(drop=True)[cols]

    # Define comparison strategies with their own input files
    comparison_strategies = [
        {
            "reader": OSWReader,
            "path": test_data_osw,
            "file_type": "osw",
            "cmd_prefix": f"--in={test_data_osw}",
        },
        {
            "reader": ParquetReader,
            "path": test_data_parquet,
            "file_type": "parquet",
            "cmd_prefix": f"--in={test_data_parquet}",
        },
        {
            "reader": SplitParquetReader,
            "path": test_data_split_parquet,
            "file_type": "parquet_split",
            "cmd_prefix": f"--in={test_data_split_parquet}",
        },
    ]

    # Exclude current reader type from comparison
    comparison_strategies = [
        s for s in comparison_strategies if s["reader"] != input_strategy["reader"]
    ]

    for strategy in comparison_strategies:
        print(f"\nComparing with: {strategy['reader'].__name__}")
        print(f"Using input file: {strategy['path']}")

        # Run scoring for this comparison strategy
        run_scoring(temp_folder, strategy)

        # Create config for comparison reader
        comp_config = create_reader_config(
            level=level,
            context=context,
            infile=strategy["path"],
            outfile=temp_folder / f"output_{level}_{context}_comp",
        )
        comp_config.file_type = strategy["file_type"]

        # Initialize and test comparison reader
        comp_reader = strategy["reader"](comp_config)
        df_comp = comp_reader.read()
        df_comp_sorted = df_comp.sort_values(
            by=["run_id", level_key_id, "score"]
        ).reset_index(drop=True)[cols]

        compare_dataframes(df_primary_sorted, df_comp_sorted, level_key_id, cols)
