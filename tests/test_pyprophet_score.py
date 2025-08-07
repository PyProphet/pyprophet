from __future__ import print_function

import os
import subprocess
import shutil
import sys
import sqlite3

import pandas as pd
import pytest
from pyprophet.io.dispatcher import ReaderDispatcher
from pyprophet._config import IPFIOConfig

pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ================== CORE TEST UTILITIES ==================
class TestRunner:
    """Handles execution of pyprophet commands and test setup"""

    def __init__(self, temp_folder):
        self.temp_folder = temp_folder
        self._setup_test_environment()

    def _setup_test_environment(self):
        """Prepare test environment"""
        os.chdir(self.temp_folder)

    def copy_test_file(self, filename):
        """Copy test file to temp directory"""
        src_path = os.path.join(DATA_FOLDER, filename)
        shutil.copy(src_path, self.temp_folder)
        return os.path.join(self.temp_folder, filename)

    def copy_test_dir(self, dirname):
        """Copy test directory to temp directory"""
        src_path = os.path.join(DATA_FOLDER, dirname)
        dst_path = os.path.join(self.temp_folder, dirname)
        shutil.copytree(src_path, dst_path)
        return dst_path

    def run_command(self, cmdline):
        """Execute a shell command and return output"""
        try:
            output = subprocess.check_output(
                cmdline, shell=True, stderr=subprocess.STDOUT
            )
            return cmdline + "\n" + str(output)
        except subprocess.CalledProcessError as error:
            print(error, end="", file=sys.stderr)
            raise


# ================== TEST CONFIGURATIONS ==================
class TestConfig:
    """Base configuration for pyprophet test commands"""

    def __init__(self):
        self.base_cmd = "pyprophet score"
        self.params = {
            "pi0_method": "smoother",
            "ss_iteration_fdr": "0.02",
            "test": True,
        }
        self.score_filters = {
            "ms1": "var_isotope_overlap_score,var_massdev_score,var_xcorr_coelution,var_isotope_correlation_score",
            "ms2": "var_isotope_overlap_score,var_isotope_correlation_score,var_intensity_score,var_massdev_score,var_library_corr,var_norm_rt_score",
            "ms1ms2": "var_ms1_isotope_overlap_score,var_ms1_massdev_score,var_ms1_xcorr_coelution,var_ms1_isotope_correlation_score,var_isotope_overlap_score,var_isotope_correlation_score,var_intensity_score,var_massdev_score,var_library_corr,var_norm_rt_score",
            "metabolomics": "metabolomics",
        }

    def build_command(self, input_file, level=None):
        """Construct the pyprophet command"""
        cmd = f"{self.base_cmd} --in={input_file}"

        if level:
            cmd += f" --level={level}"

        for param, value in self.params.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd += f" --{param}"
                else:
                    cmd += f" --{param}={value}"

        return cmd

    def add_score_filter(self, cmd, level):
        """Add score filter if appropriate for level"""
        if level in self.score_filters:
            cmd += f" --ss_score_filter {self.score_filters[level]}"
        return cmd


# ================== TEST STRATEGIES ==================
class TestStrategy:
    """Base class for different test strategies"""

    def __init__(self, runner, config):
        self.runner = runner
        self.config = config

    def prepare(self):
        """Prepare test files"""
        raise NotImplementedError

    def execute(self):
        """Execute the test"""
        raise NotImplementedError

    def verify(self, regtest):
        """Verify results"""
        raise NotImplementedError

    def apply_weights(self):
        """Apply weights"""
        raise NotImplementedError

    def verify_weights(self, regtest):
        """Verify weights"""
        raise NotImplementedError


class TSVTestStrategy(TestStrategy):
    """Test strategy for TSV files"""

    def prepare(self):
        self.input_file = self.runner.copy_test_file("test_data.txt")

    def execute(self, **kwargs):
        cmd = self.config.build_command("test_data.txt")
        if kwargs.get("parametric"):
            cmd += " --parametric"
        if kwargs.get("pfdr"):
            cmd += " --pfdr"
        if kwargs.get("pi0_lambda"):
            cmd += f" --pi0_lambda={kwargs['pi0_lambda']}"

        # Execute single command
        return self.runner.run_command(cmd)

    def verify(self, regtest):
        for suffix in [
            "summary_stat.csv",
            "full_stat.csv",
            "scored.tsv",
            "weights.csv",
        ]:
            df = pd.read_csv(
                f"test_data_{suffix}", sep="\t" if "tsv" in suffix else ","
            )
            print(df.head(100).sort_index(axis=1), file=regtest)


class OSWTestStrategy(TestStrategy):
    """Test strategy for OSW files"""

    def __init__(self, runner, config, is_metabo=False):
        super().__init__(runner, config)
        self.is_metabo = is_metabo

    def prepare(self):
        filename = "test_data_compound.osw" if self.is_metabo else "test_data.osw"
        self.input_file = self.runner.copy_test_file(filename)

    def execute(self, levels=None, **kwargs):
        base_file = "test_data_compound.osw" if self.is_metabo else "test_data.osw"

        if not levels:
            levels = (
                ["ms1", "ms2", "transition"] if not self.is_metabo else ["ms1", "ms2"]
            )

        # Execute each level command separately
        for level in levels:
            level_cmd = self.config.build_command(self.input_file, level)

            if kwargs.get("parametric"):
                level_cmd += " --parametric"
            if kwargs.get("pfdr"):
                level_cmd += " --pfdr"
            if kwargs.get("pi0_lambda"):
                level_cmd += f" --pi0_lambda={kwargs['pi0_lambda']}"
            if kwargs.get("xgboost"):
                level_cmd += " --classifier=XGBoost"
            if kwargs.get("xgboost_tune"):
                level_cmd += " --autotune"
            if kwargs.get("lda_xgboost"):
                level_cmd += " --classifier=LDA_XGBoost"
            if kwargs.get("score_filter"):
                level_cmd = self.config.add_score_filter(level_cmd, level)

            self.runner.run_command(level_cmd)

    def verify(self, regtest):
        if self.is_metabo:
            with sqlite3.connect("test_data_compound.osw") as conn:
                table = pd.read_sql_query(
                    "SELECT * FROM PYPROPHET_WEIGHTS ORDER BY score", conn
                )
        else:
            config = IPFIOConfig(
                infile=self.input_file,
                outfile=self.input_file,
                subsample_ratio=1.0,
                level="peakgroup_precursor",
                context="ipf",
            )
            config.file_type = "osw"
            config.ipf_max_peakgroup_pep = 1.0
            config.ipf_ms1_scoring = True
            config.ipf_ms2_scoring = True
            reader = ReaderDispatcher.get_reader(config)
            table = reader.read(level="peakgroup_precursor")
            sort_cols = [
                "feature_id",
                "ms2_peakgroup_pep",
                "ms1_precursor_pep",
                "ms2_precursor_pep",
            ]
            table = table.sort_values(sort_cols).reset_index(drop=True)

        print(table.head(100).sort_index(axis=1), file=regtest)


class ParquetTestStrategy(TestStrategy):
    """Test strategy for Parquet files"""

    def prepare(self):
        self.input_file = self.runner.copy_test_file("test_data.parquet")
        self.weights_file = "test_data_weights.csv"

    def execute(self, levels=None, **kwargs):
        if not levels:
            levels = ["ms1", "ms2", "transition"]

        # Execute each level command separately
        for level in levels:
            level_cmd = self.config.build_command(self.input_file, level)

            if kwargs.get("subsample_ratio"):
                level_cmd += f" --subsample_ratio={kwargs['subsample_ratio']}"
            if kwargs.get("parametric"):
                level_cmd += " --parametric"
            if kwargs.get("pfdr"):
                level_cmd += " --pfdr"
            if kwargs.get("pi0_lambda"):
                level_cmd += f" --pi0_lambda={kwargs['pi0_lambda']}"
            if kwargs.get("xgboost"):
                level_cmd += " --classifier=XGBoost"
            if kwargs.get("xgboost_tune"):
                level_cmd += " --autotune"
            if kwargs.get("score_filter"):
                level_cmd = self.config.add_score_filter(level_cmd, level)

            self.runner.run_command(level_cmd)

    def verify(self, regtest):
        # Read input file and print the shape of the infilee_dir/transition_featues.parquet file
        data = pd.read_parquet(self.input_file)
        print(data.shape[0], file=regtest)

        config = IPFIOConfig(
            infile=self.input_file,
            outfile=self.input_file,
            subsample_ratio=1.0,
            level="peakgroup_precursor",
            context="ipf",
        )
        config.file_type = "parquet"
        config.ipf_max_peakgroup_pep = 1.0
        config.ipf_ms1_scoring = True
        config.ipf_ms2_scoring = True
        reader = ReaderDispatcher.get_reader(config)
        table = reader.read(level="peakgroup_precursor")
        sort_cols = [
            "feature_id",
            "ms2_peakgroup_pep",
            "ms1_precursor_pep",
            "ms2_precursor_pep",
        ]
        table = table.sort_values(sort_cols).reset_index(drop=True)
        print(table.head(100).sort_index(axis=1), file=regtest)

    def apply_weights(self):
        # Run scoring without weights to generate initial scores
        cmd = f"pyprophet score --level ms2 --pi0_method=smoother --pi0_lambda 0.001 0 0 --in={self.input_file} --test --ss_iteration_fdr=0.02 --subsample_ratio=1.0"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level ms2 --pi0_method=smoother --pi0_lambda 0.4 0 0 --in={self.input_file} --apply_weights={self.weights_file} --test --ss_iteration_fdr=0.02"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level transition --pi0_method=smoother --pi0_lambda 0.001 0 0 --in={self.input_file} --test --ss_iteration_fdr=0.02 --subsample_ratio=1.0"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level transition --pi0_method=smoother --pi0_lambda 0.4 0 0 --in={self.input_file} --apply_weights={self.weights_file} --test --ss_iteration_fdr=0.02"
        self.runner.run_command(cmd)

    def verify_weights(self, regtest):
        data = pd.read_parquet(self.input_file)
        # (96,259, 23)
        print(data.shape[0], file=regtest)

        cols = [col for col in data.columns if col.startswith("SCORE_MS2_")]
        assert len(cols) > 0, "No SCORE_MS2_ columns found in the data"

        cols = [col for col in data.columns if col.startswith("SCORE_TRANSITION_")]
        assert len(cols) > 0, "No SCORE_TRANSITION_ columns found in the data"

        config = IPFIOConfig(
            infile=self.input_file,
            outfile=self.input_file,
            subsample_ratio=1.0,
            level="peakgroup_precursor",
            context="ipf",
        )
        config.file_type = "parquet"
        config.ipf_max_peakgroup_pep = 1.0
        config.ipf_ms1_scoring = False
        config.ipf_ms2_scoring = False
        reader = ReaderDispatcher.get_reader(config)
        table = reader.read(level="peakgroup_precursor")
        sort_cols = [
            "feature_id",
            "ms2_peakgroup_pep",
            "ms1_precursor_pep",
            "ms2_precursor_pep",
        ]
        table = table.sort_values(sort_cols).reset_index(drop=True)
        print(table.head(100).sort_index(axis=1), file=regtest)


class SplitParquetTestStrategy(TestStrategy):
    """Test strategy for Parquet files"""

    def prepare(self):
        self.input_file = self.runner.copy_test_dir("test_data.oswpq")
        self.weights_file = "test_data_weights.csv"

    def execute(self, levels=None, **kwargs):
        if not levels:
            levels = ["ms1", "ms2", "transition"]

        # Execute each level command separately
        for level in levels:
            level_cmd = self.config.build_command(self.input_file, level)

            if kwargs.get("subsample_ratio"):
                level_cmd += f" --subsample_ratio={kwargs['subsample_ratio']}"
            if kwargs.get("parametric"):
                level_cmd += " --parametric"
            if kwargs.get("pfdr"):
                level_cmd += " --pfdr"
            if kwargs.get("pi0_lambda"):
                level_cmd += f" --pi0_lambda={kwargs['pi0_lambda']}"
            if kwargs.get("xgboost"):
                level_cmd += " --classifier=XGBoost"
            if kwargs.get("xgboost_tune"):
                level_cmd += " --autotune"
            if kwargs.get("score_filter"):
                level_cmd = self.config.add_score_filter(level_cmd, level)

            self.runner.run_command(level_cmd)

    def verify(self, regtest):
        # Read input file and print the shape of the infilee_dir/transition_featues.parquet file
        data = pd.read_parquet(
            os.path.join(self.input_file, "transition_features.parquet")
        )
        # (96,259, 23)
        print(data.shape[0], file=regtest)

        config = IPFIOConfig(
            infile=self.input_file,
            outfile=self.input_file,
            subsample_ratio=1.0,
            level="peakgroup_precursor",
            context="ipf",
        )
        config.file_type = "parquet_split"
        config.ipf_max_peakgroup_pep = 1.0
        config.ipf_ms1_scoring = True
        config.ipf_ms2_scoring = True
        reader = ReaderDispatcher.get_reader(config)
        table = reader.read(level="peakgroup_precursor")
        sort_cols = [
            "feature_id",
            "ms2_peakgroup_pep",
            "ms1_precursor_pep",
            "ms2_precursor_pep",
        ]
        table = table.sort_values(sort_cols).reset_index(drop=True)
        print(table.head(100).sort_index(axis=1), file=regtest)

    def apply_weights(self):
        # Run scoring without weights to generate initial scores
        cmd = f"pyprophet score --level ms2 --pi0_method=smoother --pi0_lambda 0.001 0 0 --in={self.input_file} --test --ss_iteration_fdr=0.02 --subsample_ratio=1.0"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level ms2 --pi0_method=smoother --pi0_lambda 0.4 0 0 --in={self.input_file} --apply_weights={self.weights_file} --test --ss_iteration_fdr=0.02"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level transition --pi0_method=smoother --pi0_lambda 0.001 0 0 --in={self.input_file} --test --ss_iteration_fdr=0.02 --subsample_ratio=1.0"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level transition --pi0_method=smoother --pi0_lambda 0.4 0 0 --in={self.input_file} --apply_weights={self.weights_file} --test --ss_iteration_fdr=0.02"
        self.runner.run_command(cmd)

    def verify_weights(self, regtest):
        data = pd.read_parquet(
            os.path.join(
                self.input_file,
                "precursors_features.parquet",
            )
        )
        # (96,259, 23)
        print(data.shape[0], file=regtest)

        # Check for "SCORE_MS2_" columns
        cols = [col for col in data.columns if col.startswith("SCORE_MS2_")]
        assert len(cols) > 0, "No SCORE_MS2_ columns found in the data"

        data = pd.read_parquet(
            os.path.join(
                self.input_file,
                "transition_features.parquet",
            )
        )
        # (96,259, 23)
        print(data.shape[0], file=regtest)
        cols = [col for col in data.columns if col.startswith("SCORE_TRANSITION_")]
        assert len(cols) > 0, "No SCORE_TRANSITION_ columns found in the data"

        config = IPFIOConfig(
            infile=self.input_file,
            outfile=self.input_file,
            subsample_ratio=1.0,
            level="peakgroup_precursor",
            context="ipf",
        )
        config.file_type = "parquet_split"
        config.ipf_max_peakgroup_pep = 1.0
        config.ipf_ms1_scoring = False
        config.ipf_ms2_scoring = False
        reader = ReaderDispatcher.get_reader(config)
        table = reader.read(level="peakgroup_precursor")
        sort_cols = [
            "feature_id",
            "ms2_peakgroup_pep",
            "ms1_precursor_pep",
            "ms2_precursor_pep",
        ]
        table = table.sort_values(sort_cols).reset_index(drop=True)
        print(table.head(100).sort_index(axis=1), file=regtest)


class MultiSplitParquetTestStrategy(TestStrategy):
    """Test strategy for Parquet files"""

    def prepare(self):
        self.input_file = self.runner.copy_test_dir("test_data.oswpqd")
        self.weights_file = "test_data_weights.csv"

    def execute(self, levels=None, **kwargs):
        if not levels:
            levels = ["ms1", "ms2", "transition"]

        # Execute each level command separately
        for level in levels:
            level_cmd = self.config.build_command(self.input_file, level)

            if kwargs.get("subsample_ratio"):
                level_cmd += f" --subsample_ratio={kwargs['subsample_ratio']}"
            if kwargs.get("parametric"):
                level_cmd += " --parametric"
            if kwargs.get("pfdr"):
                level_cmd += " --pfdr"
            if kwargs.get("pi0_lambda"):
                level_cmd += f" --pi0_lambda={kwargs['pi0_lambda']}"
            if kwargs.get("xgboost"):
                level_cmd += " --classifier=XGBoost"
            if kwargs.get("xgboost_tune"):
                level_cmd += " --autotune"
            if kwargs.get("score_filter"):
                level_cmd = self.config.add_score_filter(level_cmd, level)

            self.runner.run_command(level_cmd)

    def verify(self, regtest):
        # Read input file and print the shape of the infilee_dir/transition_featues.parquet file
        data = pd.read_parquet(
            os.path.join(
                self.input_file,
                "napedro_L120420_010_SW.oswpq/transition_features.parquet",
            )
        )
        # (96,259, 23)
        print(data.shape[0], file=regtest)

        config = IPFIOConfig(
            infile=self.input_file,
            outfile=self.input_file,
            subsample_ratio=1.0,
            level="peakgroup_precursor",
            context="ipf",
        )
        config.file_type = "parquet_split_multi"
        config.ipf_max_peakgroup_pep = 1.0
        config.ipf_ms1_scoring = True
        config.ipf_ms2_scoring = True
        reader = ReaderDispatcher.get_reader(config)
        table = reader.read(level="peakgroup_precursor")
        sort_cols = [
            "feature_id",
            "ms2_peakgroup_pep",
            "ms1_precursor_pep",
            "ms2_precursor_pep",
        ]
        table = table.sort_values(sort_cols).reset_index(drop=True)
        print(table.head(100).sort_index(axis=1), file=regtest)

    def apply_weights(self):
        # Run scoring without weights to generate initial scores
        cmd = f"pyprophet score --level ms2 --pi0_method=smoother --pi0_lambda 0.001 0 0 --in={self.input_file} --test --ss_iteration_fdr=0.02 --subsample_ratio=1.0"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level ms2 --pi0_method=smoother --pi0_lambda 0.4 0 0 --in={self.input_file} --apply_weights={self.weights_file} --test --ss_iteration_fdr=0.02"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level transition --pi0_method=smoother --pi0_lambda 0.001 0 0 --in={self.input_file} --test --ss_iteration_fdr=0.02 --subsample_ratio=1.0"
        self.runner.run_command(cmd)

        cmd = f"pyprophet score --level transition --pi0_method=smoother --pi0_lambda 0.4 0 0 --in={self.input_file} --apply_weights={self.weights_file} --test --ss_iteration_fdr=0.02"
        self.runner.run_command(cmd)

    def verify_weights(self, regtest):
        data = pd.read_parquet(
            os.path.join(
                self.input_file,
                "napedro_L120420_010_SW.oswpq/precursors_features.parquet",
            )
        )
        # (96,259, 23)
        print(data.shape[0], file=regtest)

        # Check for "SCORE_MS2_" columns
        cols = [col for col in data.columns if col.startswith("SCORE_MS2_")]
        assert len(cols) > 0, "No SCORE_MS2_ columns found in the data"

        data = pd.read_parquet(
            os.path.join(
                self.input_file,
                "napedro_L120420_010_SW.oswpq/transition_features.parquet",
            )
        )
        # (96,259, 23)
        print(data.shape[0], file=regtest)
        cols = [col for col in data.columns if col.startswith("SCORE_TRANSITION_")]
        assert len(cols) > 0, "No SCORE_TRANSITION_ columns found in the data"

        config = IPFIOConfig(
            infile=self.input_file,
            outfile=self.input_file,
            subsample_ratio=1.0,
            level="peakgroup_precursor",
            context="ipf",
        )
        config.file_type = "parquet_split_multi"
        config.ipf_max_peakgroup_pep = 1.0
        config.ipf_ms1_scoring = False
        config.ipf_ms2_scoring = False
        reader = ReaderDispatcher.get_reader(config)
        table = reader.read(level="peakgroup_precursor")
        sort_cols = [
            "feature_id",
            "ms2_peakgroup_pep",
            "ms1_precursor_pep",
            "ms2_precursor_pep",
        ]
        table = table.sort_values(sort_cols).reset_index(drop=True)
        print(table.head(100).sort_index(axis=1), file=regtest)


# ================== TEST FIXTURES ==================
@pytest.fixture
def test_runner(tmpdir):
    return TestRunner(tmpdir.strpath)


@pytest.fixture
def test_config():
    return TestConfig()


# ================== TEST CASES ==================
def run_generic_test(test_runner, test_config, strategy_class, regtest, **kwargs):
    strategy = strategy_class(test_runner, test_config)
    strategy.prepare()
    strategy.execute(**kwargs)
    strategy.verify(regtest)


def run_generic_test_overwrite(
    test_runner, test_config, strategy_class, regtest, **kwargs
):
    strategy = strategy_class(test_runner, test_config)
    strategy.prepare()
    strategy.execute(**kwargs)
    strategy.verify(regtest)
    # Second run to overwrite the previous results, the rows should be the same'
    # This is important for the parquet tests, to ensure they data is the same and not just adding additional rows
    strategy.execute(**kwargs)
    strategy.verify(regtest)


def run_generic_test_apply_weights(
    test_runner, test_config, strategy_class, regtest, **kwargs
):
    strategy = strategy_class(test_runner, test_config)
    strategy.prepare()
    strategy.apply_weights(**kwargs)
    strategy.verify_weights(regtest)


def run_metabo_test(
    test_runner, test_config, regtest, ms1ms2=False, score_filter=False
):
    strategy = OSWTestStrategy(test_runner, test_config, is_metabo=True)
    strategy.prepare()

    levels = ["ms1", "ms1ms2"] if ms1ms2 else ["ms1", "ms2"]
    strategy.execute(levels=levels, score_filter=score_filter)
    strategy.verify(regtest)


# TSV Tests
def test_tsv_0(test_runner, test_config, regtest):
    run_generic_test(test_runner, test_config, TSVTestStrategy, regtest)


def test_tsv_1(test_runner, test_config, regtest):
    run_generic_test(
        test_runner, test_config, TSVTestStrategy, regtest, parametric=True
    )


def test_tsv_2(test_runner, test_config, regtest):
    run_generic_test(test_runner, test_config, TSVTestStrategy, regtest, pfdr=True)


def test_tsv_3(test_runner, test_config, regtest):
    run_generic_test(
        test_runner, test_config, TSVTestStrategy, regtest, pi0_lambda="0.3 0.55 0.05"
    )


def test_tsv_apply_weights(test_runner, test_config, regtest):
    # Initial scoring
    run_generic_test(test_runner, test_config, TSVTestStrategy, regtest)

    # Apply weights
    cmd = "pyprophet score --pi0_method=smoother --pi0_lambda 0.4 0 0 --in=test_data.txt --apply_weights=test_data_weights.csv --test --ss_iteration_fdr=0.02"
    test_runner.run_command(cmd)


# OSW Tests
def test_osw_0(test_runner, test_config, regtest):
    run_generic_test(
        test_runner, test_config, OSWTestStrategy, regtest, pi0_lambda="0 0 0"
    )


def test_osw_1(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        parametric=True,
        pi0_lambda="0 0 0",
    )


def test_osw_2(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
    )


def test_osw_3(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
    )


def test_osw_4(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
        xgboost=True,
    )


def test_osw_5(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
        xgboost=True,
        xgboost_tune=True,
    )


def test_osw_6(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        score_filter=True,
    )


def test_osw_7(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
        score_filter=True,
    )


def test_osw_8(test_runner, test_config, regtest):
    run_metabo_test(test_runner, test_config, regtest, ms1ms2=False, score_filter=False)


def test_osw_9(test_runner, test_config, regtest):
    run_metabo_test(test_runner, test_config, regtest, ms1ms2=False, score_filter=True)


def test_osw_10(test_runner, test_config, regtest):
    run_metabo_test(test_runner, test_config, regtest, ms1ms2=True, score_filter=True)

# Tests LDA then XGBoost
def test_osw_11(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        OSWTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
        lda_xgboost=True,
    )


# Parquet Tests
def test_parquet_0(test_runner, test_config, regtest):
    run_generic_test(
        test_runner, test_config, ParquetTestStrategy, regtest, pi0_lambda="0 0 0"
    )


def test_parquet_1(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        ParquetTestStrategy,
        regtest,
        parametric=True,
        pi0_lambda="0 0 0",
    )


def test_parquet_2(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        ParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
    )


def test_parquet_3(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        ParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
    )


def test_parquet_6(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        ParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        score_filter=True,
    )


def test_parquet_7(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        ParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
        score_filter=True,
    )


def test_parquet_8(test_runner, test_config, regtest):
    run_generic_test_overwrite(
        test_runner, test_config, ParquetTestStrategy, regtest, pi0_lambda="0 0 0"
    )


def test_parquet_9(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        ParquetTestStrategy,
        regtest,
        pi0_lambda="0 0 0",
        subample_ratio=0.5,
    )


def test_parquet_apply_weights(test_runner, test_config, regtest):
    # Apply weights
    run_generic_test_apply_weights(
        test_runner, test_config, ParquetTestStrategy, regtest
    )


# Split Parquet Tests
def test_split_parquet_0(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        pi0_lambda="0 0 0",
    )


def test_split_parquet_1(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        parametric=True,
        pi0_lambda="0 0 0",
    )


def test_split_parquet_2(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
    )


def test_split_parquet_3(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
    )


def test_split_parquet_6(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        score_filter=True,
    )


def test_split_parquet_7(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
        score_filter=True,
    )


def test_split_parquet_8(test_runner, test_config, regtest):
    run_generic_test_overwrite(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        pi0_lambda="0 0 0",
    )


def test_split_parquet_9(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        SplitParquetTestStrategy,
        regtest,
        pi0_lambda="0 0 0",
        subample_ratio=0.5,
    )


def test_split_parquet_apply_weights(test_runner, test_config, regtest):
    # Apply weights
    run_generic_test_apply_weights(
        test_runner, test_config, SplitParquetTestStrategy, regtest
    )


# Multi-Split Parquet Tests
def test_multi_split_parquet_0(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        pi0_lambda="0 0 0",
    )


def test_multi_split_parquet_1(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        parametric=True,
        pi0_lambda="0 0 0",
    )


def test_multi_split_parquet_2(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
    )


def test_multi_split_parquet_3(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
    )


def test_multi_split_parquet_6(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        score_filter=True,
    )


def test_multi_split_parquet_7(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        pfdr=True,
        pi0_lambda="0 0 0",
        ms1ms2=True,
        score_filter=True,
    )


def test_multi_split_parquet_8(test_runner, test_config, regtest):
    run_generic_test_overwrite(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        pi0_lambda="0 0 0",
    )


def test_multi_split_parquet_9(test_runner, test_config, regtest):
    run_generic_test(
        test_runner,
        test_config,
        MultiSplitParquetTestStrategy,
        regtest,
        pi0_lambda="0 0 0",
        subample_ratio=0.5,
    )


def test_multi_split_parquet_apply_weights(test_runner, test_config, regtest):
    # Apply weights
    run_generic_test_apply_weights(
        test_runner, test_config, MultiSplitParquetTestStrategy, regtest
    )


# Error Case Test
def test_not_unique_tg_id_blocks(test_runner):
    test_runner.copy_test_file("test_invalid_data.txt")
    cmd = "pyprophet score --pi0_method=smoother --pi0_lambda 0.4 0 0 --in=test_invalid_data.txt --test --ss_iteration_fdr=0.02"

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        test_runner.run_command(cmd)

    assert "Error: group_id values do not form unique blocks in input file(s)." in str(
        exc_info.value.output
    )
