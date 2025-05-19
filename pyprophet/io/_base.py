from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import click
import pickle


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

    @property
    def infile(self):
        return self.config.infile

    @property
    def outfile(self):
        return self.config.outfile

    @property
    def classifier(self):
        return self.config.runner.classifier

    @property
    def level(self):
        return self.config.level

    @property
    def glyco(self):
        return self.config.glyco

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

    @property
    def infile(self):
        return self.config.infile

    @property
    def outfile(self):
        return self.config.outfile

    @property
    def classifier(self):
        return self.config.runner.classifier

    @property
    def level(self):
        return self.config.level

    @property
    def glyco(self):
        return self.config.glyco

    @abstractmethod
    def save_results(self, result, pi0):
        """
        Abstract method to save scoring results and statistical outputs.

        Args:
            result: The result object containing scoring tables.
            pi0: Estimated pi0 value from FDR statistics.
        """
        raise NotImplementedError("Subclasses must implement 'save_results'.")

    def save_weights(self, weights):
        """
        Abstract method to save model weights (e.g., LDA coefficients, XGBoost model).

        Args:
            weights: Model weights or trained object.
        """
        if self.classifier == "LDA":
            self._save_tsv_weights(weights)
        elif self.classifier == "XGBoost":
            self._save_bin_weights(weights)
        else:
            raise NotImplementedError(
                f"Classifier {self.classifier} not supported for saving weights."
            )

    def _save_tsv_weights(self, weights):
        """
        Save the model weights to a TSV file.

        Args:
            weights: Model weights or trained object.
        """
        weights["level"] = self.level
        trained_weights_path = self.config.extra_writes.get("trained_weights_path")
        if trained_weights_path is not None:
            weights.to_csv(trained_weights_path, sep=",", index=False, mode="a")
            click.echo("Info: %s written." % trained_weights_path)

    def _save_bin_weights(self, weights):
        """
        Save the model weights to a binary file.

        Args:
            weights: Model weights or trained object.
        """
        trained_weights_path = self.config.extra_writes.get("trained_model_path")
        if trained_weights_path is not None:
            with open(trained_weights_path, "wb") as file:
                self.persisted_weights = pickle.dump(weights, file)
            click.echo("Info: %s written." % trained_weights_path)
