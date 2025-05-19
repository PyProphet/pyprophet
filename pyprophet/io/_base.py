from abc import ABC, abstractmethod
import os
import pickle
import pandas as pd
import click
from .._base import BaseIOConfig
from ..report import save_report


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

    def _prepare_score_dataframe(
        self, df: pd.DataFrame, level: str, prefix: str
    ) -> pd.DataFrame:
        """
        Prepare the score DataFrame
        """
        score_cols = [
            "feature_id",
            "d_score",
            "peak_group_rank",
            "p_value",
            "q_value",
            "pep",
        ]
        if "h_score" in df.columns:
            score_cols.insert(2, "h_score")
            score_cols.insert(3, "h0_score")

        if level == "transition":
            score_cols.insert(1, "transition_id")

        df = df[score_cols].rename(columns=str.upper)

        if level == "transition":
            key_cols = {"FEATURE_ID", "TRANSITION_ID"}
        else:
            key_cols = {"FEATURE_ID"}

        rename_map = {
            col: f"{prefix}{col}" for col in df.columns if col not in key_cols
        }
        return df.rename(columns=rename_map)

    def _get_columns_to_keep(
        self, existing_cols: list[str], score_prefix: str
    ) -> list[str]:
        """
        Get the columns to keep in the DataFrame by removing existing score columns. Mainly for Parquet files.
        """
        drop_cols = [col for col in existing_cols if col.startswith(score_prefix)]
        if drop_cols:
            click.echo(
                click.style(
                    f"Warn: Dropping existing {score_prefix} columns.", fg="yellow"
                )
            )
        return [col for col in existing_cols if not col.startswith(score_prefix)]

    def _write_pdf_report_if_present(self, result, pi0):
        """
        Write a PDF report if the scoring results contain final statistics.
        """

        if result.final_statistics is None:
            return

        df = result.scored_tables
        prefix = self.config.prefix
        level = self.level

        cutoffs = result.final_statistics["cutoff"].values
        svalues = result.final_statistics["svalue"].values
        qvalues = result.final_statistics["qvalue"].values

        pvalues = df[(df.peak_group_rank == 1) & (df.decoy == 0)]["p_value"].values
        top_targets = df[(df.peak_group_rank == 1) & (df.decoy == 0)]["d_score"].values
        top_decoys = df[(df.peak_group_rank == 1) & (df.decoy == 1)]["d_score"].values

        pdf_path = os.path.join(prefix + f"_{level}_report.pdf")
        save_report(
            pdf_path,
            self.outfile,
            top_decoys,
            top_targets,
            cutoffs,
            svalues,
            qvalues,
            pvalues,
            pi0,
            self.config.runner.color_palette,
        )
        click.echo(f"Info: {pdf_path} written.")

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
