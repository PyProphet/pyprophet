from abc import ABC, abstractmethod
import os
import pickle
import pandas as pd
import click
from loguru import logger
from .util import setup_logger
from .._base import BaseIOConfig
from ..report import save_report

setup_logger()


class BaseReader(ABC):
    """
    Abstract base class for implementing readers that load data from different sources (OSW, Parquet, etc.).
    """

    def __init__(self, config: BaseIOConfig):
        """
        Initialize the reader with a given configuration.

        Args:
            config (BaseIOConfig): Configuration object containing input details, and module specific config for params for reading.
        """
        self.config = config

    @property
    def context(self):
        return self.config.context

    @property
    def infile(self):
        return self.config.infile

    @property
    def outfile(self):
        return self.config.outfile

    @property
    def file_type(self):
        return self.config.file_type

    @property
    def subsample_ratio(self):
        return self.config.subsample_ratio

    @property
    def classifier(self):
        return self.config.runner.classifier

    @property
    def level(self):
        return self.config.level

    @property
    def glyco(self):
        return self.config.runner.glyco

    @abstractmethod
    def read(self):
        """
        Abstract method to be implemented by subclasses to read data from a specific format.
        """
        raise NotImplementedError("Subclasses must implement 'read'.")

    def _finalize_feature_table(self, df, ss_main_score):
        df.columns = [c.lower() for c in df.columns]
        main_score = ss_main_score.lower()
        if main_score in df.columns:
            df = df.rename(columns={main_score: "main_" + main_score})
        elif (
            main_score == "swath_pretrained"
        ):  # TODO: Should we deprecate this? This is probably not optimal for newer data from different instruments
            # Add a pretrained main score corresponding to the original implementation in OpenSWATH
            # This is optimized for 32-windows SCIEX TripleTOF 5600 data
            df["main_var_pretrained"] = -(
                -0.19011762 * df["var_library_corr"]
                + 2.47298914 * df["var_library_rmsd"]
                + 5.63906731 * df["var_norm_rt_score"]
                + -0.62640133 * df["var_isotope_correlation_score"]
                + 0.36006925 * df["var_isotope_overlap_score"]
                + 0.08814003 * df["var_massdev_score"]
                + 0.13978311 * df["var_xcorr_coelution"]
                + -1.16475032 * df["var_xcorr_shape"]
                + -0.19267813 * df["var_yseries_score"]
                + -0.61712054 * df["var_log_sn_score"]
            )
        else:
            raise click.ClickException(
                f"Main score ({main_score}) not found in input columns: {df.columns}"
            )

        if self.classifier == "XGBoost" and self.level != "alignment":
            logger.info(
                "Enable number of transitions & precursor / product charge scores for XGBoost-based classifier"
            )
            df = df.rename(
                columns={
                    "precursor_charge": "var_precursor_charge",
                    "product_charge": "var_product_charge",
                    "transition_count": "var_transition_count",
                }
            )
        return df


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
    def context(self):
        return self.config.context

    @property
    def infile(self):
        return self.config.infile

    @property
    def outfile(self):
        return self.config.outfile

    @property
    def file_type(self):
        return self.config.file_type

    @property
    def classifier(self):
        return self.config.runner.classifier

    @property
    def level(self):
        return self.config.level

    @property
    def glyco(self):
        return self.config.runner.glyco

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

        if level == "alignment":
            score_cols.insert(2, "alignment_id")
            score_cols.insert(2, "decoy")
            # Reverse map DECOY 0 to 1 and 1 to -1
            # arycal saves a label column to indicate 1 as target aligned peaks and -1 as the random/shuffled decoy aligned peak
            df["decoy"] = df["decoy"].map({0: 1, 1: -1})

        df = df[score_cols].rename(columns=str.upper)
        df = df.rename(columns={"D_SCORE": "SCORE"})

        if level not in ("ms2", "ms1ms2") and self.config.file_type == "osw":
            df = df.rename(columns={"PEAK_GROUP_RANK": "RANK"})

        if level == "transition":
            key_cols = {"FEATURE_ID", "TRANSITION_ID"}
        elif level == "alignment":
            key_cols = {"ALIGNMENT_ID", "FEATURE_ID", "DECOY"}
        else:
            key_cols = {"FEATURE_ID"}

        if self.config.file_type == "osw":
            return df

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
            logger.warning(
                f"Warn: Dropping existing {score_prefix} columns.",
            )

        return [col for col in existing_cols if not col.startswith(score_prefix)]

    def _write_pdf_report_if_present(self, result, pi0):
        """
        Write a PDF report if the scoring results contain final statistics.
        """

        if result.final_statistics is None:
            return

        df = result.scored_tables
        if self.level == "alignment":
            # Map Decoy 1 to 0 and -1 to 1
            df["decoy"] = df["decoy"].map({1: 0, -1: 1})

        prefix = self.config.prefix
        level = self.level

        cutoffs = result.final_statistics["cutoff"].values
        svalues = result.final_statistics["svalue"].values
        qvalues = result.final_statistics["qvalue"].values

        pvalues = df[(df.peak_group_rank == 1) & (df.decoy == 0)]["p_value"].values
        top_targets = df[(df.peak_group_rank == 1) & (df.decoy == 0)]["d_score"].values
        top_decoys = df[(df.peak_group_rank == 1) & (df.decoy == 1)]["d_score"].values

        # Check if any of the values are empty, can't create a report if they are
        if not all(
            [
                len(top_targets),
                len(top_decoys),
                len(cutoffs),
                len(svalues),
                len(qvalues),
                len(pvalues),
            ]
        ):
            logger.error("Not enough values to create a report.")
            logger.error(f"top_targets: {len(top_targets)}")
            logger.error(f"top_decoys: {len(top_decoys)}")
            logger.error(f"cutoffs: {len(cutoffs)}")
            logger.error(f"svalues: {len(svalues)}")
            logger.error(f"qvalues: {len(qvalues)}")
            logger.error(f"pvalues: {len(pvalues)}")
            logger.error(f"pi0: {len(pi0)}")
            return

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
        logger.success(f"{pdf_path} written.")

    def _save_tsv_weights(self, weights):
        """
        Save the model weights to a TSV file, ensuring no duplicate levels.

        If weights for the current level already exist, they are removed before saving the new ones.
        """
        weights["level"] = self.level
        trained_weights_path = self.config.extra_writes.get("trained_weights_path")

        if trained_weights_path is not None:

            if os.path.exists(trained_weights_path):
                existing_df = pd.read_csv(trained_weights_path, sep=",")
                existing_df = existing_df[existing_df["level"] != self.level]
                updated_df = pd.concat([existing_df, weights], ignore_index=True)
            else:
                updated_df = weights

            # Always overwrite with a single header
            updated_df.to_csv(trained_weights_path, sep=",", index=False)
            logger.success(f"{trained_weights_path} written.")

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
            logger.success("%s written." % trained_weights_path)
