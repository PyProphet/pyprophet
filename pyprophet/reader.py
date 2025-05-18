import os
import click
import polars as pl
import pandas as pd
import duckdb
from .data_handling import get_parquet_column_names


def read_parquet_dir(
    infile,
    level,
    classifier,
    ss_main_score,
    ipf_max_peakgroup_rank,
    ipf_max_peakgroup_pep,
    ipf_max_transition_isotope_overlap,
    ipf_min_transition_sn,
):
    '''
    Read the parquet files from the input directory and return a pandas dataframe.
    '''

    precursor_file = os.path.join(infile, "precursors_features.parquet")
    transition_file = os.path.join(infile, "transition_features.parquet")
    alignment_file = os.path.join(infile, "feature_alignment.parquet")

    all_precursor_column_names = get_parquet_column_names(precursor_file)
    all_transition_column_names = get_parquet_column_names(transition_file)

    if level == "alignment":
        if  os.path.exists(alignment_file):
            all_alignment_column_names = get_parquet_column_names(alignment_file)
        else:
            click.echo(click.style(f"Error: Couldn't find: {alignment_file}", fg="red"))
            raise click.ClickException(
                click.style(
                "Alignment-level features are not present in the input parquet directory. Please run ARYCAL first to perform alignment of features. https://github.com/singjc/arycal", fg="red")
            )
        

    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW precursors AS SELECT * FROM read_parquet('{precursor_file}')"
    )
    if (
        os.path.exists(transition_file)
        and os.path.basename(transition_file) == "transition_features.parquet"
    ):
        con.execute(
            f"CREATE VIEW transitions AS SELECT * FROM read_parquet('{transition_file}')"
        )
        
    if (
        os.path.exists(alignment_file)
        and os.path.basename(alignment_file) == "feature_alignment.parquet"
    ):
        con.execute(
            f"CREATE VIEW alignment_features AS SELECT * FROM read_parquet('{alignment_file}')"
        )

    if level == "ms2" or level == "ms1ms2":
        if not any(
            [col.startswith("FEATURE_MS2_") for col in all_precursor_column_names]
        ):
            raise click.ClickException(
                "MS2-level feature columns are not present in precursors_features.parquet file."
            )

        # Filter columes names for FEATURE_MS2_
        feature_ms2_cols = [
            col for col in all_precursor_column_names if col.startswith("FEATURE_MS2_")
        ]
        # prepare feature ms2 columns for sql query
        feature_ms2_cols_sql = ", ".join([f"p.{col}" for col in feature_ms2_cols])

        query = f"""
        SELECT 
            p.RUN_ID, 
            p.PRECURSOR_ID, 
            p.PRECURSOR_CHARGE, 
            p.FEATURE_ID, 
            p.EXP_RT, 
            p.PRECURSOR_DECOY AS DECOY, 
            {feature_ms2_cols_sql},
            COALESCE(t.TRANSITION_COUNT, 0) AS TRANSITION_COUNT,
            p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
        FROM precursors p
        LEFT JOIN (
            SELECT 
                PRECURSOR_ID, 
                COUNT(*) AS TRANSITION_COUNT
            FROM (
                SELECT DISTINCT PRECURSOR_ID, TRANSITION_ID
                FROM transitions
                WHERE TRANSITION_DETECTING = 1
            ) AS sub
            GROUP BY PRECURSOR_ID
        ) AS t
        ON p.PRECURSOR_ID = t.PRECURSOR_ID
        ORDER BY p.RUN_ID, p.PRECURSOR_ID ASC, p.EXP_RT ASC
        """
        table = con.execute(query).pl()

        # Rename columns
        table = table.rename(
            {
                **{
                    col: col.replace("FEATURE_MS2_", "")
                    for col in table.columns
                    if col.startswith("FEATURE_MS2_")
                }
            }
        )

        table = table.to_pandas()

    elif level == "ms1":
        if not any(
            [col.startswith("FEATURE_MS1_") for col in all_precursor_column_names]
        ):
            raise click.ClickException(
                "MS1-level feature columns are not present in precursors_features.parquet file."
            )

        # Filter columes names for FEATURE_MS1_
        feature_ms1_cols = [
            col for col in all_precursor_column_names if col.startswith("FEATURE_MS1_")
        ]
        # prepare feature ms1 columns for sql query
        feature_ms1_cols_sql = ", ".join([f"p.{col}" for col in feature_ms1_cols])

        query = f"""
                SELECT 
                    p.RUN_ID, 
                    p.PRECURSOR_ID, 
                    p.PRECURSOR_CHARGE, 
                    p.FEATURE_ID, 
                    p.EXP_RT, 
                    p.PRECURSOR_DECOY AS DECOY, 
                    {feature_ms1_cols_sql},
                    p.RUN_ID || '_' || p.PRECURSOR_ID AS GROUP_ID
                FROM precursors p
                ORDER BY p.RUN_ID, p.PRECURSOR_ID ASC, p.EXP_RT ASC
                """
        table = con.execute(query).pl()

        # Rename columns
        table = table.rename(
            {
                **{
                    col: col.replace("FEATURE_MS1_", "")
                    for col in table.columns
                    if col.startswith("FEATURE_MS1_")
                }
            }
        )
        table = table.to_pandas()

    elif level == "transition":
        if not any(
            [col.startswith("SCORE_MS2_") for col in all_precursor_column_names]
        ):
            raise click.ClickException(
                "Transition-level scoring for IPF requires prior MS2 or MS1MS2-level scoring. Please run 'pyprophet score --level=ms2' or 'pyprophet score --level=ms1ms2' first."
            )
        if not any(
            [
                col.startswith("FEATURE_TRANSITION_")
                for col in all_transition_column_names
            ]
        ):
            raise click.ClickException(
                "Transition-level feature columns are not present in transition_features.parquet file."
            )

        # Filter columes names for FEATURE_MS1_
        feature_transition_cols = [
            col
            for col in all_transition_column_names
            if col.startswith("FEATURE_TRANSITION_VAR")
        ]
        # prepare feature ms1 columns for sql query
        feature_transition_cols_sql = ", ".join(
            [f"t.{col}" for col in feature_transition_cols]
        )

        query = f"""
        SELECT 
            t.TRANSITION_DECOY AS DECOY,
            t.FEATURE_ID AS FEATURE_ID,
            t.TRANSITION_ID AS TRANSITION_ID,
            {feature_transition_cols_sql},
            p.PRECURSOR_CHARGE AS PRECURSOR_CHARGE,
            t.TRANSITION_CHARGE AS PRODUCT_CHARGE,
            p.RUN_ID || '_' || t.FEATURE_ID || '_' || t.TRANSITION_ID AS GROUP_ID
        FROM transitions t
        INNER JOIN precursors p ON t.PRECURSOR_ID = p.PRECURSOR_ID AND t.FEATURE_ID = p.FEATURE_ID
        WHERE p.SCORE_MS2_RANK <= {ipf_max_peakgroup_rank}
        AND p.SCORE_MS2_PEP <= {ipf_max_peakgroup_pep}
        AND p.PRECURSOR_DECOY = 0
        AND t.FEATURE_TRANSITION_VAR_ISOTOPE_OVERLAP_SCORE <= {ipf_max_transition_isotope_overlap}
        AND t.FEATURE_TRANSITION_VAR_LOG_SN_SCORE > {ipf_min_transition_sn}
        ORDER BY p.RUN_ID, p.PRECURSOR_ID, p.EXP_RT, t.TRANSITION_ID
        """
        table = con.execute(query).pl()

        # Rename columns
        table = table.rename(
            {
                **{
                    col: col.replace("FEATURE_TRANSITION_", "")
                    for col in table.columns
                    if col.startswith("FEATURE_TRANSITION_")
                }
            }
        )
        table = table.to_pandas()

    elif level == "alignment":
        feature_alignment_cols =[
            col
            for col in all_alignment_column_names
            if col.startswith("VAR_")
        ]
        # Prepare alignment query
        feature_alignment_cols_sql = ", ".join(
            [f"a.{col}" for col in feature_alignment_cols]
        )

        query = f"""
        SELECT 
            a.ALIGNMENT_ID,
            a.RUN_ID, 
            a.PRECURSOR_ID, 
            a.FEATURE_ID, 
            a.ALIGNED_RT, 
            a.DECOY, 
            {feature_alignment_cols_sql},
            a.RUN_ID || '_' || a.FEATURE_ID || '_' || a.PRECURSOR_ID AS GROUP_ID
        FROM alignment_features a
        ORDER BY a.RUN_ID, a.PRECURSOR_ID ASC, a.REFERENCE_RT ASC
        """
        table = con.execute(query).pl()

        table = table.to_pandas()
        #  Map DECOY to 1 and -1 to 0 and 1
        table['DECOY'] = table['DECOY'].map({1: 0, -1: 1})

    else:
        raise click.ClickException("Unspecified data level selected.")

    if level == "ms1ms2":
        if not any(
            [col.startswith("FEATURE_MS1_") for col in all_precursor_column_names]
        ):
            raise click.ClickException(
                "MS1-level feature columns are not present in parquet file."
            )

        feature_ms1_cols = [
            col
            for col in all_precursor_column_names
            if col.startswith("FEATURE_MS1_VAR")
        ]
        feature_ms1_cols_sql = ", ".join([f"p.{col}" for col in feature_ms1_cols])
        query = f"""
        SELECT 
            p.FEATURE_ID, 
            {feature_ms1_cols_sql}
        FROM precursors p
        """
        ms1_table = con.execute(query).df()
        table = pd.merge(table, ms1_table, how="left", on="FEATURE_ID")

    table.columns = [col.lower() for col in table.columns]

    if ss_main_score.lower() in table.columns:
        table = table.rename(
            columns={ss_main_score.lower(): "main_" + ss_main_score.lower()}
        )
    elif ss_main_score.lower() == "swath_pretrained":
        # SWATH pretrained score is not really used anymore, so drop support for it in parquet file input workflow
        raise click.ClickException(
            "SWATH pretrained score not available for parquet files workflow"
        )
    else:
        raise click.ClickException(
            f"Main score ({ss_main_score.lower()}) column not present in data. Current columns: {table.columns}"
        )

    if classifier == "XGBoost" and level != "alignment":
        click.echo(
            "Info: Enable number of transitions & precursor / product charge scores for XGBoost-based classifier"
        )
        table = table.rename(
            columns={
                "precursor_charge": "var_precursor_charge",
                "product_charge": "var_product_charge",
                "transition_count": "var_transition_count",
            }
        )

    return table
