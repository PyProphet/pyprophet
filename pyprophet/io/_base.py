from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os


@dataclass
class BaseIOConfig:
    """
    Base configuration class for I/O-related metadata.

    Attributes:
        infile (str): Path to the input file (e.g., .osw, .parquet).
        outfile (str): Path to the output file to be written.
        level (str): Scoring level (e.g., 'ms1', 'ms2', 'transition', 'alignment').
        context (str): Context or mode in which the reader/writer operates
                       (e.g., 'semi_supervised', 'ipf', 'level_context').
        prefix (str): Automatically derived from outfile (e.g., 'results/output' for 'results/output.osw').
    """

    infile: str
    outfile: str
    level: str
    context: str
    prefix: str = field(init=False)

    def __post_init__(self):
        self.prefix = os.path.splitext(self.outfile)[0]


class BaseReader(ABC):
    """
    Abstract base class for implementing readers that load data from different sources (OSW, Parquet, etc.).
    """

    def __init__(self, config: BaseIOConfig):
        """
        Initialize the reader with a given configuration.

        Args:
            config (BaseIOConfig): Configuration object containing input details.
        """
        self.config = config

    @abstractmethod
    def read(self):
        """
        Abstract method to be implemented by subclasses to read data from a specific format.
        """
        raise NotImplementedError("Subclasses must implement 'read'.")


class BaseWriter(ABC):
    """
    Abstract base class for implementing writers that save results to various output formats.
    """

    def __init__(self, config: BaseIOConfig):
        """
        Initialize the writer with a given configuration.

        Args:
            config (BaseIOConfig): Configuration object containing output details.
        """
        self.config = config

    @abstractmethod
    def save_results(self, result, pi0):
        """
        Abstract method to save scoring results and statistical outputs.

        Args:
            result: The result object containing scoring tables.
            pi0: Estimated pi0 value from FDR statistics.
        """
        raise NotImplementedError("Subclasses must implement 'save_results'.")

    @abstractmethod
    def save_weights(self, weights):
        """
        Abstract method to save model weights (e.g., LDA coefficients, XGBoost model).

        Args:
            weights: Model weights or trained object.
        """
        raise NotImplementedError("Subclasses must implement 'save_weights'.")
