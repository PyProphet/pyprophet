import os
import pandas as pd
import sqlite3
import click
import pickle
from shutil import copyfile
from ._base import BaseReader, BaseWriter, BaseIOConfig
from ._config import RunnerIOConfig, IPFIOConfig, LevelContextIOConfig
from ..data_handling import check_sqlite_table
from ..report import save_report
from ..glyco.report import save_report as save_report_glyco


class OSWReader(BaseReader):
    """
    Class for reading and processing data from an OpenSWATH workflow OSW-sqlite based file.

    The OSWReader class provides methods to read different levels of data from the file and process it accordingly.
    It supports reading data for semi-supervised learning, IPF analysis, context level analysis.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment'), or context level used peptide/protein/gene inference (e.g., 'global', 'experiment-wide', 'run-specific').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        read(): Read data from the input file based on the alogorithm.
    """

    def __init__(self, config: BaseIOConfig):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        if isinstance(self.config, RunnerIOConfig):
            return self._read_for_semi_supervised()
        elif isinstance(self.config, IPFIOConfig):
            return self._read_for_ipf()
        elif isinstance(self.config, LevelContextIOConfig):
            return self._read_for_context_level()
        else:
            raise NotImplementedError(
                f"Unsupported config type: {type(self.config).__name__}"
            )

    def _read_for_semi_supervised(self) -> pd.DataFrame:
        ss_main_score = self.config.runner.ss_main_score

        con = sqlite3.connect(self.config.infile)

        if self.level in ("ms2", "ms1ms2"):
            if not check_sqlite_table(con, "FEATURE_MS2"):
                raise click.ClickException(
                    "MS2-level feature table not present in file."
                )

            con.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);
                """
            )

            if not self.config.runner.glyco:
                table = pd.read_sql_query(
                    """
                    SELECT *,
                        RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
                    FROM FEATURE_MS2
                    INNER JOIN
                    (SELECT RUN_ID,
                            ID,
                            PRECURSOR_ID,
                            EXP_RT
                    FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID
                    INNER JOIN
                    (SELECT ID,
                            CHARGE AS PRECURSOR_CHARGE,
                            DECOY
                    FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                    INNER JOIN
                    (SELECT PRECURSOR_ID AS ID,
                            COUNT(*) AS TRANSITION_COUNT
                    FROM TRANSITION_PRECURSOR_MAPPING
                    INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
                    WHERE DETECTING==1
                    GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
                    ORDER BY RUN_ID,
                            PRECURSOR.ID ASC,
                            FEATURE.EXP_RT ASC;
                    """,
                    con,
                )
            else:
                table = pd.read_sql_query(
                    """
                    SELECT *,
                        RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
                    FROM FEATURE_MS2
                    INNER JOIN
                    (SELECT RUN_ID,
                            ID,
                            PRECURSOR_ID,
                            EXP_RT
                    FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID
                    INNER JOIN
                    (SELECT ID,
                            CHARGE AS PRECURSOR_CHARGE,
                            DECOY
                    FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                    INNER JOIN
                    (SELECT PRECURSOR_ID AS ID,
                            DECOY_PEPTIDE,
                            DECOY_GLYCAN
                    FROM PRECURSOR_GLYCOPEPTIDE_MAPPING
                    INNER JOIN GLYCOPEPTIDE
                    ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID == GLYCOPEPTIDE.ID) AS DECOY
                    ON FEATURE.PRECURSOR_ID = DECOY.ID
                    INNER JOIN
                    (SELECT PRECURSOR_ID AS ID,
                            COUNT(*) AS TRANSITION_COUNT
                    FROM TRANSITION_PRECURSOR_MAPPING
                    INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
                    WHERE DETECTING==1
                    GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
                    ORDER BY RUN_ID,
                            PRECURSOR.ID ASC,
                            FEATURE.EXP_RT ASC;
                    """,
                    con,
                )
        elif self.level == "ms1":
            if not check_sqlite_table(con, "FEATURE_MS1"):
                raise click.ClickException(
                    "MS1-level feature table not present in file."
                )

            con.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_ms1_feature_id ON FEATURE_MS1 (FEATURE_ID);
                """
            )
            if not self.config.runner.glyco:
                table = pd.read_sql_query(
                    """
                    SELECT *,
                        RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
                    FROM FEATURE_MS1
                    INNER JOIN
                    (SELECT RUN_ID,
                            ID,
                            PRECURSOR_ID,
                            EXP_RT
                    FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID
                    INNER JOIN
                    (SELECT ID,
                            CHARGE AS PRECURSOR_CHARGE,
                            DECOY
                    FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                    ORDER BY RUN_ID,
                            PRECURSOR.ID ASC,
                            FEATURE.EXP_RT ASC;
                    """,
                    con,
                )
            else:
                if not check_sqlite_table(con, "SCORE_MS2"):
                    raise click.ClickException(
                        "MS1-level scoring for glycoform inference requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first."
                    )
                if not check_sqlite_table(con, "FEATURE_MS1"):
                    raise click.ClickException(
                        "MS1-level feature table not present in file."
                    )

                table = pd.read_sql_query(
                    f"""
                    SELECT DECOY.*,
                        FEATURE_MS1.*, 
                        FEATURE.*,
                        PRECURSOR.*,
                        RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
                    FROM FEATURE_MS1
                    INNER JOIN
                    (SELECT RUN_ID,
                        ID,
                        PRECURSOR_ID,
                        EXP_RT
                    FROM FEATURE) AS FEATURE ON FEATURE_MS1.FEATURE_ID = FEATURE.ID
                    
                    INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID

                    INNER JOIN
                    (SELECT ID,
                        CHARGE AS PRECURSOR_CHARGE,
                        DECOY
                    FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                    
                    INNER JOIN
                        (SELECT PRECURSOR_ID AS ID,
                                DECOY_PEPTIDE,
                                DECOY_GLYCAN
                        FROM PRECURSOR_GLYCOPEPTIDE_MAPPING
                        INNER JOIN GLYCOPEPTIDE 
                        ON PRECURSOR_GLYCOPEPTIDE_MAPPING.GLYCOPEPTIDE_ID == GLYCOPEPTIDE.ID) AS DECOY 
                        ON FEATURE.PRECURSOR_ID = DECOY.ID

                    WHERE RANK <= {self.config.runner.ipf_max_peakgroup_rank}
                    ORDER BY RUN_ID,
                        PRECURSOR.ID ASC,
                        FEATURE.EXP_RT ASC;
                    """,
                    con,
                )
        elif self.level == "transition":
            if not check_sqlite_table(con, "SCORE_MS2"):
                raise click.ClickException(
                    "Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' on this file first."
                )
            if not check_sqlite_table(con, "FEATURE_TRANSITION"):
                raise click.ClickException(
                    "Transition-level feature table not present in file."
                )

            con.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_transition_id ON TRANSITION (ID);
                CREATE INDEX IF NOT EXISTS idx_score_ms2_feature_id ON SCORE_MS2 (FEATURE_ID);
                CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_transition_feature_id ON FEATURE_TRANSITION (FEATURE_ID);
                CREATE INDEX IF NOT EXISTS idx_feature_transition_transition_id ON FEATURE_TRANSITION (TRANSITION_ID);
                """
            )

            table = pd.read_sql_query(
                f"""
                SELECT TRANSITION.DECOY AS DECOY,
                    FEATURE_TRANSITION.*,
                    PRECURSOR.CHARGE AS PRECURSOR_CHARGE,
                    TRANSITION.PRODUCT_CHARGE AS PRODUCT_CHARGE,
                    RUN_ID || '_' || FEATURE_TRANSITION.FEATURE_ID || '_' || PRECURSOR_ID || '_' || TRANSITION_ID AS GROUP_ID
                FROM FEATURE_TRANSITION
                INNER JOIN
                (SELECT RUN_ID,
                        ID,
                        PRECURSOR_ID,
                        EXP_RT
                FROM FEATURE) AS FEATURE ON FEATURE_TRANSITION.FEATURE_ID = FEATURE.ID
                INNER JOIN PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN SCORE_MS2 ON FEATURE.ID = SCORE_MS2.FEATURE_ID
                INNER JOIN
                (SELECT ID,
                        CHARGE AS PRODUCT_CHARGE,
                        DECOY
                FROM TRANSITION) AS TRANSITION ON FEATURE_TRANSITION.TRANSITION_ID = TRANSITION.ID
                WHERE RANK <= {self.config.runner.ipf_max_peakgroup_rank}
                AND PEP <= {self.config.runner.ipf_max_peakgroup_pep}
                AND VAR_ISOTOPE_OVERLAP_SCORE <= {self.config.runner.ipf_max_transition_isotope_overlap}
                AND VAR_LOG_SN_SCORE > {self.config.runner.ipf_min_transition_sn}
                AND PRECURSOR.DECOY == 0
                ORDER BY RUN_ID,
                        PRECURSOR.ID,
                        FEATURE.EXP_RT,
                        TRANSITION.ID;
                """,
                con,
            )
        elif self.level == "alignment":
            if not check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
                raise click.ClickException(
                    "MS2-level feature alignemnt table not present in file."
                )

            con.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_precursor_precursor_id ON PRECURSOR (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_precursor_id ON FEATURE (PRECURSOR_ID);
                CREATE INDEX IF NOT EXISTS idx_feature_feature_id ON FEATURE (ID);
                CREATE INDEX IF NOT EXISTS idx_feature_ms2_feature_id ON FEATURE_MS2 (FEATURE_ID);
                """
            )

            table = pd.read_sql_query(
                """
                SELECT
                    ALIGNED_FEATURE_ID AS FEATURE_ID,
                    XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                    XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE_TO_REFERENCE, 
                    MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE, 
                    XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,  
                    XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE, 
                    MI_TO_ALL AS VAR_MI_TO_ALL, 
                    RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION, 
                    PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO,
                    LABEL AS DECOY,
                    ALIGNED_FILENAME || '_' || FEATURE.PRECURSOR_ID AS GROUP_ID
                FROM FEATURE_MS2_ALIGNMENT
                LEFT JOIN
                (SELECT RUN_ID,
                        ID,
                        PRECURSOR_ID,
                        EXP_RT
                FROM FEATURE) AS FEATURE ON REFERENCE_FEATURE_ID = FEATURE.ID
                """,
                con,
            )
            # Map DECOY to 1 and -1 to 0 and 1
            table["DECOY"] = table["DECOY"].map({1: 0, -1: 1})
        else:
            raise click.ClickException("Unspecified data level selected.")

        # Append MS1 scores to MS2 table if selected
        if self.level == "ms1ms2":
            if not check_sqlite_table(con, "FEATURE_MS1"):
                raise click.ClickException(
                    "MS1-level feature table not present in file."
                )
            ms1_table = pd.read_sql_query("SELECT * FROM FEATURE_MS1;", con)

            ms1_scores = [c for c in ms1_table.columns if c.startswith("VAR_")]
            ms1_table = ms1_table[["FEATURE_ID"] + ms1_scores]
            ms1_table.columns = ["FEATURE_ID"] + [
                "VAR_MS1_" + s.split("VAR_")[1] for s in ms1_scores
            ]

            table = pd.merge(table, ms1_table, how="left", on="FEATURE_ID")

        if self.config.runner.add_alignment_features:
            # Append MS2 alignment scores to MS2 table if selected
            if self.level in ("ms2", "ms1ms2"):
                if not check_sqlite_table(con, "FEATURE_MS2_ALIGNMENT"):
                    raise click.ClickException(
                        "MS2-level feature alignment table not present in file."
                    )

                if not check_sqlite_table(con, "SCORE_ALIGNMENT"):
                    raise click.ClickException(
                        "To add MS2-level alignment features, alignment-level first needs to be performed. Please run 'pyprophet score --level=alignment' on this file first."
                    )

                alignment_table = pd.read_sql_query(
                    """SELECT 
                            ALIGNED_FEATURE_ID AS FEATURE_ID,
                            PRECURSOR_ID,
                            XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                            XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE_TO_REFERENCE, 
                            MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE, 
                            XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,  
                            XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE_TO_ALL, 
                            MI_TO_ALL AS VAR_MI_TO_ALL, 
                            RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION, 
                            PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO 
                        FROM 
                        (SELECT DISTINCT * FROM FEATURE_MS2_ALIGNMENT) AS FEATURE_MS2_ALIGNMENT
                        INNER JOIN 
                        (SELECT DISTINCT *, MIN(QVALUE) FROM SCORE_ALIGNMENT GROUP BY FEATURE_ID) AS SCORE_ALIGNMENT 
                        ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_MS2_ALIGNMENT.ALIGNED_FEATURE_ID
                        WHERE LABEL = 1""",
                    con,
                )

                if "PRECURSOR_ID" in table.columns:
                    table = pd.merge(
                        table,
                        alignment_table,
                        how="left",
                        on=["FEATURE_ID", "PRECURSOR_ID"],
                    )
                else:
                    table = pd.merge(
                        table, alignment_table, how="left", on="FEATURE_ID"
                    )

            # Append TRANSITION alignment scores to TRANSITION table if selected
            if self.level == "transition":
                if not check_sqlite_table(con, "FEATURE_TRANSITION_ALIGNMENT"):
                    raise click.ClickException(
                        "Transition-level feature alignment table not present in file."
                    )

                if not check_sqlite_table(con, "SCORE_ALIGNMENT"):
                    raise click.ClickException(
                        "To add Transition-level alignment features, alignment-level first needs to be performed. Please run 'pyprophet score --level=alignment' on this file first."
                    )

                alignment_table = pd.read_sql_query(
                    """SELECT 
                            FEATURE_TRANSITION_ALIGNMENT.FEATURE_ID,
                            TRANSITION_ID,
                            XCORR_COELUTION_TO_REFERENCE AS VAR_XCORR_COELUTION_TO_REFERENCE,
                            XCORR_SHAPE_TO_REFERENCE AS VAR_XCORR_SHAPE_TO_REFERENCE, 
                            MI_TO_REFERENCE AS VAR_MI_TO_REFERENCE, 
                            XCORR_COELUTION_TO_ALL AS VAR_XCORR_COELUTION_TO_ALL,  
                            XCORR_SHAPE_TO_ALL AS VAR_XCORR_SHAPE_TO_ALL, 
                            MI_TO_ALL AS VAR_MI_TO_ALL, 
                            RETENTION_TIME_DEVIATION AS VAR_RETENTION_TIME_DEVIATION, 
                            PEAK_INTENSITY_RATIO AS VAR_PEAK_INTENSITY_RATIO
                        FROM FEATURE_TRANSITION_ALIGNMENT
                        INNER JOIN 
                        (SELECT DISTINCT *, MIN(QVALUE) FROM SCORE_ALIGNMENT GROUP BY FEATURE_ID) AS SCORE_ALIGNMENT 
                        ON SCORE_ALIGNMENT.FEATURE_ID = FEATURE_TRANSITION_ALIGNMENT.FEATURE_ID 
                    """,
                    con,
                )
                table = pd.merge(
                    table,
                    alignment_table,
                    how="left",
                    on=["FEATURE_ID", "TRANSITION_ID"],
                )

            cols = [
                "VAR_XCORR_COELUTION_TO_REFERENCE",
                "VAR_XCORR_SHAPE_TO_REFERENCE",
                "VAR_MI_TO_REFERENCE",
                "VAR_XCORR_COELUTION_TO_ALL",
                "VAR_XCORR_SHAPE_TO_ALL",
                "VAR_MI_TO_ALL",
                "VAR_RETENTION_TIME_DEVIATION",
                "VAR_PEAK_INTENSITY_RATIO",
            ]
            # Fill in missing values for cols2 with -1
            table[cols] = table[cols].fillna(-1)

        # Format table
        table.columns = [col.lower() for col in table.columns]

        # Mark main score column
        if ss_main_score.lower() in table.columns:
            table = table.rename(
                index=str,
                columns={ss_main_score.lower(): "main_" + ss_main_score.lower()},
            )
        elif (
            ss_main_score.lower() == "swath_pretrained"
        ):  # TODO: Do we want to deprecate this?
            # Add a pretrained main score corresponding to the original implementation in OpenSWATH
            # This is optimized for 32-windows SCIEX TripleTOF 5600 data
            table["main_var_pretrained"] = -(
                -0.19011762 * table["var_library_corr"]
                + 2.47298914 * table["var_library_rmsd"]
                + 5.63906731 * table["var_norm_rt_score"]
                + -0.62640133 * table["var_isotope_correlation_score"]
                + 0.36006925 * table["var_isotope_overlap_score"]
                + 0.08814003 * table["var_massdev_score"]
                + 0.13978311 * table["var_xcorr_coelution"]
                + -1.16475032 * table["var_xcorr_shape"]
                + -0.19267813 * table["var_yseries_score"]
                + -0.61712054 * table["var_log_sn_score"]
            )
        else:
            raise click.ClickException(
                f"Main score ({ss_main_score.lower()}) column not present in data. Current columns: {table.columns}"
            )

        # Enable transition count & precursor / product charge scores for XGBoost-based classifier
        if self.config.runner.classifier == "XGBoost" and self.level != "alignment":
            click.echo(
                "Info: Enable number of transitions & precursor / product charge scores for XGBoost-based classifier"
            )
            table = table.rename(
                index=str,
                columns={
                    "precursor_charge": "var_precursor_charge",
                    "product_charge": "var_product_charge",
                    "transition_count": "var_transition_count",
                },
            )

        con.close()
        return table

    def _read_for_ipf(self):
        # implement logic from `ipf.py`
        raise NotImplementedError

    def _read_for_context_level(self):
        # implement logic from `levels_context.py`
        raise NotImplementedError


class OSWWriter(BaseWriter):
    """
    Class for writing OpenSWATH results to an OSW-sqlite based file.

    Attributes:
        infile (str): Input file path.
        outfile (str): Output file path.
        classifier (str): Classifier used for semi-supervised learning.
        level (str): Level used in semi-supervised learning (e.g., 'ms1', 'ms2', 'ms1ms2', 'transition', 'alignment'), or context level used peptide/protein/gene inference (e.g., 'global', 'experiment-wide', 'run-specific').
        glyco (bool): Flag indicating whether analysis is glycoform-specific.

    Methods:
        save_results(result, pi0): Save the results to the output file based on the module using this class.
        save_weights(weights): Save the weights to the output file.
    """

    def __init__(self, config: BaseIOConfig):
        super().__init__(config)

    def save_results(self, result, pi0):
        if isinstance(self.config, RunnerIOConfig):
            return self._save_semi_supervised_results(result, pi0)
        elif isinstance(self.config, IPFIOConfig):
            return self._save_ipf_results(result)
        elif isinstance(self.config, LevelContextIOConfig):
            return self._save_context_level_results(result)
        else:
            raise NotImplementedError(
                f"Mode {self.config.mode} is not supported in OSWWriter."
            )

    def _save_semi_supervised_results(self, result, pi0):
        if self.infile != self.outfile:
            copyfile(self.infile, self.outfile)

        con = sqlite3.connect(self.config.outfile)

        if self.glyco and self.level in ["ms2", "ms1ms2"]:
            table = "SCORE_MS2"
            if self.level in ("ms2", "ms1ms2"):
                c = con.cursor()
                c.execute("DROP TABLE IF EXISTS SCORE_MS2;")
                c.execute("DROP TABLE IF EXISTS SCORE_MS2_PART_PEPTIDE;")
                c.execute("DROP TABLE IF EXISTS SCORE_MS2_PART_GLYCAN;")
                con.commit()
                c.fetchall()

                table = "SCORE_MS2"

            elif self.level == "ms1":
                c = con.cursor()
                c.execute("DROP TABLE IF EXISTS SCORE_MS1;")
                c.execute("DROP TABLE IF EXISTS SCORE_MS1_PART_PEPTIDE;")
                c.execute("DROP TABLE IF EXISTS SCORE_MS1_PART_GLYCAN;")
                con.commit()
                c.fetchall()

                table = "SCORE_MS1"

            df = result.scored_tables
            if "h_score" in df.columns:
                df = df[
                    [
                        "feature_id",
                        "d_score_combined",
                        "h_score",
                        "h0_score",
                        "peak_group_rank",
                        "q_value",
                        "pep",
                    ]
                ]
                df.columns = [
                    "FEATURE_ID",
                    "SCORE",
                    "HSCORE",
                    "H0SCORE",
                    "RANK",
                    "QVALUE",
                    "PEP",
                ]
            else:
                df = df[
                    [
                        "feature_id",
                        "d_score_combined",
                        "peak_group_rank",
                        "q_value",
                        "pep",
                    ]
                ]
                df.columns = ["FEATURE_ID", "SCORE", "RANK", "QVALUE", "PEP"]
            df.to_sql(table, con, index=False)

            for part in ["peptide", "glycan"]:
                df = result.scored_tables
                df = df[["feature_id", "d_score_" + part, "pep_" + part]]
                df.columns = ["FEATURE_ID", "SCORE", "PEP"]
                df.to_sql(table + "_PART_" + part.upper(), con, index=False)
        else:
            if self.level in ("ms2", "ms1ms2"):
                c = con.cursor()
                c.execute("DROP TABLE IF EXISTS SCORE_MS2;")
                con.commit()
                c.fetchall()

                df = result.scored_tables
                if "h_score" in df.columns:
                    df = df[
                        [
                            "feature_id",
                            "d_score",
                            "h_score",
                            "h0_score",
                            "peak_group_rank",
                            "p_value",
                            "q_value",
                            "pep",
                        ]
                    ]
                    df.columns = [
                        "FEATURE_ID",
                        "SCORE",
                        "HSCORE",
                        "H0SCORE",
                        "RANK",
                        "PVALUE",
                        "QVALUE",
                        "PEP",
                    ]
                else:
                    df = df[
                        [
                            "feature_id",
                            "d_score",
                            "peak_group_rank",
                            "p_value",
                            "q_value",
                            "pep",
                        ]
                    ]
                    df.columns = [
                        "FEATURE_ID",
                        "SCORE",
                        "RANK",
                        "PVALUE",
                        "QVALUE",
                        "PEP",
                    ]
                table = "SCORE_MS2"
                df.to_sql(table, con, index=False)
            elif self.level == "ms1":
                c = con.cursor()
                c.execute("DROP TABLE IF EXISTS SCORE_MS1;")
                con.commit()
                c.fetchall()

                df = result.scored_tables
                if "h_score" in df.columns:
                    df = df[
                        [
                            "feature_id",
                            "d_score",
                            "h_score",
                            "h0_score",
                            "peak_group_rank",
                            "p_value",
                            "q_value",
                            "pep",
                        ]
                    ]
                    df.columns = [
                        "FEATURE_ID",
                        "SCORE",
                        "HSCORE",
                        "H0SCORE",
                        "RANK",
                        "PVALUE",
                        "QVALUE",
                        "PEP",
                    ]
                else:
                    df = df[
                        [
                            "feature_id",
                            "d_score",
                            "peak_group_rank",
                            "p_value",
                            "q_value",
                            "pep",
                        ]
                    ]
                    df.columns = [
                        "FEATURE_ID",
                        "SCORE",
                        "RANK",
                        "PVALUE",
                        "QVALUE",
                        "PEP",
                    ]
                table = "SCORE_MS1"
                df.to_sql(table, con, index=False)
            elif self.level == "transition":
                c = con.cursor()
                c.execute("DROP TABLE IF EXISTS SCORE_TRANSITION;")
                con.commit()
                c.fetchall()

                df = result.scored_tables[
                    [
                        "feature_id",
                        "transition_id",
                        "d_score",
                        "peak_group_rank",
                        "p_value",
                        "q_value",
                        "pep",
                    ]
                ]
                df.columns = [
                    "FEATURE_ID",
                    "TRANSITION_ID",
                    "SCORE",
                    "RANK",
                    "PVALUE",
                    "QVALUE",
                    "PEP",
                ]
                table = "SCORE_TRANSITION"
                df.to_sql(table, con, index=False)
            elif self.level == "alignment":
                c = con.cursor()
                c.execute("DROP TABLE IF EXISTS SCORE_ALIGNMENT;")
                con.commit()
                c.fetchall()

                df = result.scored_tables[
                    [
                        "feature_id",
                        "d_score",
                        "peak_group_rank",
                        "p_value",
                        "q_value",
                        "pep",
                    ]
                ]
                df.columns = ["FEATURE_ID", "SCORE", "RANK", "PVALUE", "QVALUE", "PEP"]
                table = "SCORE_ALIGNMENT"
                df.to_sql(table, con, index=False)

        con.close()
        click.echo(f"Info: {self.outfile} written.")

        if result.final_statistics is not None:
            if self.config.glyco and self.level in ["ms2", "ms1ms2"]:
                save_report_glyco(
                    os.path.join(self.config.prefix + "_" + self.level + "_report.pdf"),
                    self.config.outfile + ": " + self.level + "-level scoring",
                    result.scored_tables,
                    result.final_statistics,
                    pi0,
                )
            else:
                cutoffs = result.final_statistics["cutoff"].values
                svalues = result.final_statistics["svalue"].values
                qvalues = result.final_statistics["qvalue"].values

                pvalues = result.scored_tables.loc[
                    (result.scored_tables.peak_group_rank == 1)
                    & (result.scored_tables.decoy == 0)
                ]["p_value"].values
                top_targets = result.scored_tables.loc[
                    (result.scored_tables.peak_group_rank == 1)
                    & (result.scored_tables.decoy == 0)
                ]["d_score"].values
                top_decoys = result.scored_tables.loc[
                    (result.scored_tables.peak_group_rank == 1)
                    & (result.scored_tables.decoy == 1)
                ]["d_score"].values

                save_report(
                    os.path.join(self.config.prefix + "_" + self.level + "_report.pdf"),
                    self.config.outfile,
                    top_decoys,
                    top_targets,
                    cutoffs,
                    svalues,
                    qvalues,
                    pvalues,
                    pi0,
                    self.config.color_palette,
                )
            click.echo(
                f"Info: {os.path.join(self.config.prefix + '_' + self.level + '_report.pdf')} written."
            )

    def _save_ipf_results(self, result):
        # extract logic from ipf.py
        raise NotImplementedError

    def _save_context_level_results(self, result):
        # extract logic from levels_context.py
        raise NotImplementedError

    def save_weights(self, weights):
        if self.classifier == "LDA":
            weights["level"] = self.level
            con = sqlite3.connect(self.outfile)

            c = con.cursor()
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="GLYCOPEPTIDEPROPHET_WEIGHTS";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM GLYCOPEPTIDEPROPHET_WEIGHTS WHERE LEVEL =="%s"'
                        % self.level
                    )
            else:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_WEIGHTS";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM PYPROPHET_WEIGHTS WHERE LEVEL =="%s"' % self.level
                    )
            c.close()

            # print(weights)

            weights.to_sql("PYPROPHET_WEIGHTS", con, index=False, if_exists="append")

        elif self.classifier == "XGBoost":
            con = sqlite3.connect(self.outfile)

            c = con.cursor()
            if self.glyco and self.level in ["ms2", "ms1ms2"]:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="GLYCOPEPTIDEPROPHET_XGB";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM GLYCOPEPTIDEPROPHET_XGB WHERE LEVEL =="%s"'
                        % self.level
                    )
                else:
                    c.execute(
                        "CREATE TABLE GLYCOPEPTIDEPROPHET_XGB (level TEXT, xgb BLOB)"
                    )

                c.execute(
                    "INSERT INTO GLYCOPEPTIDEPROPHET_XGB VALUES(?, ?)",
                    [self.level, pickle.dumps(weights)],
                )
            else:
                c.execute(
                    'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="PYPROPHET_XGB";'
                )
                if c.fetchone()[0] == 1:
                    c.execute(
                        'DELETE FROM PYPROPHET_XGB WHERE LEVEL =="%s"' % self.level
                    )
                else:
                    c.execute("CREATE TABLE PYPROPHET_XGB (level TEXT, xgb BLOB)")

                c.execute(
                    "INSERT INTO PYPROPHET_XGB VALUES(?, ?)",
                    [self.level, pickle.dumps(weights)],
                )
            con.commit()
            c.close()
