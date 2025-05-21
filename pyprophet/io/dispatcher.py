from loguru import logger

from .util import setup_logger
from .scoring.osw import OSWReader as ScoringOSWReader
from .scoring.osw import OSWWriter as ScoringOSWWriter
from .scoring.parquet import ParquetReader as ParquetScoringReader
from .scoring.parquet import ParquetWriter as ParquetScoringWriter
from .scoring.split_parquet import SplitParquetReader as SplitParquetScoringReader
from .scoring.split_parquet import SplitParquetWriter as SplitParquetScoringWriter
from .scoring.tsv import TSVReader as ScoringTSVReader
from .scoring.tsv import TSVWriter as ScoringTSVWriter

from .._config import RunnerIOConfig, IPFIOConfig, LevelContextIOConfig

setup_logger()


class ReaderDispatcher:
    @staticmethod
    def get_reader(config):
        if config.file_type == "osw":
            return ReaderDispatcher._get_osw_reader(config)
        elif config.file_type == "parquet":
            logger.warning("Parquet input is experimental. Proceed with caution.")
            return ReaderDispatcher._get_parquet_reader(config)
        elif config.file_type in ("parquet_split", "parquet_split_multi"):
            logger.warning("Split parquet input is experimental. Proceed with caution.")
            return ReaderDispatcher._get_split_parquet_reader(config)
        elif config.file_type == "tsv":
            return ReaderDispatcher._get_tsv_reader(config)
        else:
            raise ValueError(f"Unsupported file type: {config.file_type}")

    @staticmethod
    def _get_osw_reader(config):
        if isinstance(config, RunnerIOConfig):
            return ScoringOSWReader(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("OSW IPFReader not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("OSW ContextReader not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_parquet_reader(config):
        if isinstance(config, RunnerIOConfig):
            return ParquetScoringReader(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("Parquet IPFReader not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("Parquet ContextReader not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_split_parquet_reader(config):
        if isinstance(config, RunnerIOConfig):
            return SplitParquetScoringReader(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("SplitParquet IPFReader not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("SplitParquet ContextReader not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_tsv_reader(config):
        if isinstance(config, RunnerIOConfig):
            return ScoringTSVReader(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("TSV IPFReader not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("TSV ContextReader not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")


class WriterDispatcher:
    @staticmethod
    def get_writer(config):
        if config.file_type == "osw":
            return WriterDispatcher._get_osw_writer(config)
        elif config.file_type == "parquet":
            return WriterDispatcher._get_parquet_writer(config)
        elif config.file_type in ("parquet_split", "parquet_split_multi"):
            return WriterDispatcher._get_split_parquet_writer(config)
        elif config.file_type == "tsv":
            return WriterDispatcher._get_tsv_writer(config)
        else:
            raise ValueError(f"Unsupported file type: {config.file_type}")

    @staticmethod
    def _get_osw_writer(config):
        if isinstance(config, RunnerIOConfig):
            return ScoringOSWWriter(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("OSW IPFWriter not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("OSW ContextWriter not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_parquet_writer(config):
        if isinstance(config, RunnerIOConfig):
            return ParquetScoringWriter(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("Parquet IPFWriter not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("Parquet ContextWriter not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_split_parquet_writer(config):
        if isinstance(config, RunnerIOConfig):
            return SplitParquetScoringWriter(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("SplitParquet IPFWriter not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("SplitParquet ContextWriter not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_tsv_writer(config):
        if isinstance(config, RunnerIOConfig):
            return ScoringTSVWriter(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("TSV IPFWriter not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("TSV ContextWriter not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")
