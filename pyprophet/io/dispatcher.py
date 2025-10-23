"""
This module provides dispatcher classes for routing I/O configurations to the appropriate
reader and writer implementations.

The dispatchers determine the correct implementation based on the file type and context
(e.g., scoring, IPF, or levels context). Supported file types include:
- OSW
- Parquet
- Split Parquet
- TSV (for scoring only)

Classes:
- ReaderDispatcher: Routes configurations to the appropriate reader implementation.
- WriterDispatcher: Routes configurations to the appropriate writer implementation.
"""

from loguru import logger

from .util import (
    _get_parquet_reader_class_for_config,
    _get_parquet_writer_class_for_config,
)
from .._config import ExportIOConfig, IPFIOConfig, LevelContextIOConfig, RunnerIOConfig

# Export I/O
from .export.osw import OSWReader as ExportOSWReader
from .export.osw import OSWWriter as ExportOSWWriter
from .export.sqmass import SqMassWriter as ExportSqMassWriter

# IPF I/O
from .ipf.osw import OSWReader as IPFOSWReader
from .ipf.osw import OSWWriter as IPFOSWWriter

# Levels Context I/O
from .levels_context.osw import OSWReader as LevelContextOSWReader
from .levels_context.osw import OSWWriter as LevelContextOSWWriter

# Scoring I/O
from .scoring.osw import OSWReader as ScoringOSWReader
from .scoring.osw import OSWWriter as ScoringOSWWriter
from .scoring.tsv import TSVReader as ScoringTSVReader
from .scoring.tsv import TSVWriter as ScoringTSVWriter


class ReaderDispatcher:
    """
    Dispatcher class to route I/O configuration to the appropriate reader implementation.

    Based on the `file_type` and optionally the config context, this class instantiates and returns
    the correct reader (e.g., OSWReader, ParquetReader, SplitParquetReader).

    Supported file types:
    - "osw"
    - "parquet"
    - "parquet_split"
    - "parquet_split_multi"
    - "tsv" (Note: only supported for scoring)
    """

    @staticmethod
    def get_reader(config):
        """
        Return the appropriate reader instance based on the config's file_type and context.

        Args:
            config (BaseIOConfig): Configuration object with file_type, level, and context.

        Returns:
            BaseReader: An instance of a subclass of BaseReader suitable for the given input type.

        Raises:
            ValueError: If an unsupported file type is provided.
        """
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
            return IPFOSWReader(config)
        elif isinstance(config, LevelContextIOConfig):
            return LevelContextOSWReader(config)
        elif isinstance(config, ExportIOConfig):
            return ExportOSWReader(config)
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_parquet_reader(config):
        cls = _get_parquet_reader_class_for_config(config, split=False)
        return cls(config)

    @staticmethod
    def _get_split_parquet_reader(config):
        cls = _get_parquet_reader_class_for_config(config, split=True)
        return cls(config)

    @staticmethod
    def _get_tsv_reader(config):
        if isinstance(config, RunnerIOConfig):
            return ScoringTSVReader(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("TSV IPFReader not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("TSV LevelsContextReader not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")


class WriterDispatcher:
    """
    Dispatcher class to route I/O configuration to the appropriate writer implementation.

    Based on the `file_type` and optionally the config context, this class instantiates and returns
    the correct writer (e.g., OSWWriter, ParquetWriter, SplitParquetWriter).

    Supported file types:
    - "osw"
    - "parquet"
    - "parquet_split"
    - "parquet_split_multi"
    - "tsv"
    """

    @staticmethod
    def get_writer(config):
        """
        Return the appropriate writer instance based on the config's file_type and context.

        Args:
            config (BaseIOConfig): Configuration object with file_type, level, and context.

        Returns:
            BaseWriter: An instance of a subclass of BaseWriter suitable for the given output type.

        Raises:
            ValueError: If an unsupported file type is provided.
        """
        if config.file_type == "osw":
            return WriterDispatcher._get_osw_writer(config)
        if config.file_type == "sqmass":
            return WriterDispatcher._get_sqmass_writer(config)
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
            return IPFOSWWriter(config)
        elif isinstance(config, LevelContextIOConfig):
            return LevelContextOSWWriter(config)
        elif isinstance(config, ExportIOConfig):
            return ExportOSWWriter(config)
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_sqmass_writer(config):
        if isinstance(config, ExportIOConfig):
            return ExportSqMassWriter(config)
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")

    @staticmethod
    def _get_parquet_writer(config):
        cls = _get_parquet_writer_class_for_config(config, split=False)
        return cls(config)

    @staticmethod
    def _get_split_parquet_writer(config):
        cls = _get_parquet_writer_class_for_config(config, split=True)
        return cls(config)

    @staticmethod
    def _get_tsv_writer(config):
        if isinstance(config, RunnerIOConfig):
            return ScoringTSVWriter(config)
        elif isinstance(config, IPFIOConfig):
            raise NotImplementedError("TSV IPFWriter not implemented.")
        elif isinstance(config, LevelContextIOConfig):
            raise NotImplementedError("TSV LevelContextWriter not implemented.")
        else:
            raise ValueError(f"Unsupported config context: {type(config).__name__}")
