import pytest
import pandas as pd
from hyperopt import hp

from pyprophet.io.scoring.osw import OSWReader
from pyprophet.io.scoring.parquet import ParquetReader
from pyprophet.io.scoring.split_parquet import SplitParquetReader
from pyprophet.io.scoring.tsv import TSVReader  # legacy, limited support
from pyprophet._config import RunnerIOConfig, RunnerConfig


xgb_params = {
    "eta": 0.3,
    "gamma": 0,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "lambda": 1,
    "alpha": 0,
    "scale_pos_weight": 1,
    "verbosity": 0,
    "objective": "binary:logitraw",
    "nthread": 1,
    "eval_metric": "auc",
}


# ================== TEST UTILITIES ==================


def create_reader_config(level, infile, outfile):
    """
    Common config generator to avoid repetition
    """
    return RunnerIOConfig(
        infile=infile,
        outfile=outfile,
        subsample_ratio=1,
        context="score_learn",
        level=level,
        runner=RunnerConfig(
            xgb_params=xgb_params,
        ),
    )


def compare_dataframes(df1, df2, cols):
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
    df1_sorted = df1.sort_values(by=["run_id", "feature_id"]).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=["run_id", "feature_id"]).reset_index(drop=True)

    # Ensure that both dataframes are non-empty
    assert not df1_sorted.empty, "First dataframe is empty"
    assert not df2_sorted.empty, "Second dataframe is empty"

    # Perform the comparison on the selected columns
    pd.testing.assert_frame_equal(df1_sorted[cols], df2_sorted[cols], check_dtype=False)


def get_comparison_columns(reader_type, level):
    """Return appropriate columns based on reader type."""
    if reader_type == "tsv_reader":
        return [
            "run_id",
            "group_id",
            "decoy",
            "main_var_xx_swath_prelim_score",
            "var_xcorr_shape",
        ]
    else:
        # "run_id",
        # "feature_id",
        # "precursor_id",
        # "exp_rt",
        # "var_bseries_score",
        # "var_dotprod_score",
        # "var_intensity_score",
        # "var_isotope_correlation_score",
        # "var_isotope_overlap_score",
        # "var_library_corr",
        # "var_library_dotprod",
        # "var_library_manhattan",
        # "var_library_rmsd",
        # "var_library_rootmeansquare",
        # "var_library_sangle",
        # "var_log_sn_score",
        # "var_manhattan_score",
        # "var_massdev_score",
        # "var_massdev_score_weighted",
        # "var_norm_rt_score",
        # "var_xcorr_coelution",
        # "var_xcorr_coelution_weighted",
        # "main_var_xcorr_shape",
        # "var_xcorr_shape_weighted",
        # "var_yseries_score",
        # "var_elution_model_fit_score",
        # "var_sonar_lag",
        # "var_sonar_shape",
        # "var_sonar_log_sn",
        # "var_sonar_log_diff",
        # "var_sonar_log_trend",
        # "var_sonar_rsq",
        # "transition_count",
        # "group_id",
        base_cols = [
            "run_id",
            "feature_id",
            "precursor_id",
            "exp_rt",
            "main_var_xcorr_shape",
        ]
        if level in ("ms1ms2", "ms2"):
            base_cols += [
                "transition_count",
            ]
        return base_cols


# ================== FIXTURES ==================
# Helper functions for reader creation
def _create_osw_reader(level):
    config = create_reader_config(
        level, "./data/test_data.osw", "./data/tmp_test_data.osw"
    )
    return OSWReader(config)


def _create_parquet_reader(level):
    config = create_reader_config(
        level, "./data/test_data.parquet", "./data/tmp_test_data.parquet"
    )
    return ParquetReader(config)


def _create_split_parquet_reader(level):
    config = create_reader_config(
        level, "./data/test_data.oswpq/", "./data/tmp_test_data.oswpq/"
    )
    return SplitParquetReader(config)


def _create_split_parquet_multi_reader(level):
    config = create_reader_config(
        level, "./data/test_data.oswpqd/", "./data/tmp_test_data.oswpqd/"
    )
    return SplitParquetReader(config)


def _create_tsv_reader(level):
    config = create_reader_config(
        level, "./data/test_data.txt", "./data/tmp_test_data.txt"
    )
    return TSVReader(config)


# Fixtures that depend on the 'level' parameter
@pytest.fixture
def osw_reader(level):
    return _create_osw_reader(level)


@pytest.fixture
def parquet_reader(level):
    return _create_parquet_reader(level)


@pytest.fixture
def split_parquet_reader(level):
    return _create_split_parquet_reader(level)


@pytest.fixture
def split_parquet_multi_reader(level):
    return _create_split_parquet_multi_reader(level)


@pytest.fixture
def tsv_reader(level):
    return _create_tsv_reader(level)


# ================== TESTS ==================
@pytest.mark.parametrize(
    "level", ["ms1", "ms2", "ms1ms2"]
)  # We leaveout "transition", because it requires scoring of MS2 first
@pytest.mark.parametrize(
    "reader_fixture",
    [
        "osw_reader",
        "parquet_reader",
        "split_parquet_reader",
        "split_parquet_multi_reader",
        "tsv_reader",
    ],
)
def test_reader_level(request, level, reader_fixture):
    """
    Test function to validate the output of a reader for a given level.

    Parameters:
    - request: pytest request object
    - level: level to test
    - reader_fixture: fixture name for the reader

    Returns:
    - None
    """
    reader = request.getfixturevalue(reader_fixture)
    df = reader.read()

    # Basic checks for all readers
    assert isinstance(df, pd.DataFrame), (
        f"{reader.__class__.__name__} returned an invalid type"
    )
    assert not df.empty, f"{reader.__class__.__name__} returned an empty DataFrame"

    # Legacy handling for TSVReader
    if isinstance(reader, TSVReader):
        # Specific checks for TSVReader
        assert "run_id" in df.columns, "Missing run_id column in TSVReader"
        assert "group_id" in df.columns, "Missing group_id column in TSVReader"

        # Check there are columns with prefix "var_"
        var_columns = [col for col in df.columns if col.startswith("var_")]
        assert var_columns, "No columns with prefix 'var_' found in TSVReader"
    else:
        # For other readers (OSW, Parquet, SplitParquet)
        assert "feature_id" in df.columns, "Missing feature_id column in the reader"
        assert "precursor_id" in df.columns, "Missing precursor_id column in the reader"


@pytest.mark.parametrize("level", ["ms1", "ms2", "ms1ms2"])
@pytest.mark.parametrize(
    "reader_fixture",
    [
        "osw_reader",
        "parquet_reader",
        "split_parquet_reader",
        "split_parquet_multi_reader",
        # "tsv_reader",  # TSVReader is not included in this test
    ],
)
def test_compare_readers(request, level, reader_fixture):
    """
    Test that all readers produce consistent output for a given level.

    Parameters:
    - request: pytest request object
    - level: level to test
    - reader_fixture: fixture name for the primary reader
    """
    # Get the primary reader
    primary_reader = request.getfixturevalue(reader_fixture)
    df_primary = primary_reader.read()

    # Get columns for comparison based on reader type
    cols = get_comparison_columns(reader_fixture, level)

    # Basic dataframe checks
    assert isinstance(df_primary, pd.DataFrame), "Invalid return type"
    assert not df_primary.empty, "Empty dataframe returned"

    # Reader-specific checks
    assert "feature_id" in df_primary.columns, "Missing feature_id"
    assert "precursor_id" in df_primary.columns, "Missing precursor_id"

    # Sort primary dataframe for comparison
    df_primary_sorted = df_primary.sort_values(
        by=["run_id", "precursor_id", "exp_rt"]
    ).reset_index(drop=True)[cols]

    # Compare with other readers
    comparison_readers = [
        request.getfixturevalue(fixture)
        for fixture in [
            "osw_reader",
            "parquet_reader",
            "split_parquet_reader",
            "split_parquet_multi_reader",
        ]
        if fixture != reader_fixture  # Don't compare with self
    ]

    for comp_reader in comparison_readers:
        df_comp = comp_reader.read()
        df_comp_sorted = df_comp.sort_values(
            by=["run_id", "precursor_id", "exp_rt"]
        ).reset_index(drop=True)[cols]

        compare_dataframes(df_primary_sorted, df_comp_sorted, cols)
