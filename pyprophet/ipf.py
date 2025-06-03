"""
This module implements the Inference of PeptidoForms (IPF) workflow.

IPF is a statistical framework for inferring peptidoforms (modified peptides)
and their probabilities from mass spectrometry data. The module includes
functions for precursor-level and peptidoform-level inference, Bayesian modeling,
and signal propagation across aligned runs.

Key Features:
    - Precursor-level inference using MS1 and MS2 data.
    - Peptidoform-level inference using transition-level data.
    - Bayesian modeling for posterior probability computation.
    - Signal propagation across aligned runs.
    - Model-based FDR estimation.

Functions:
    - compute_model_fdr: Computes model-based FDR estimates from posterior error probabilities.
    - prepare_precursor_bm: Prepares Bayesian model data for precursor-level inference.
    - transfer_confident_evidence_across_runs: Propagates confident evidence across aligned runs.
    - prepare_transition_bm: Prepares Bayesian model data for transition-level inference.
    - apply_bm: Applies the Bayesian model to compute posterior probabilities.
    - precursor_inference: Conducts precursor-level inference.
    - peptidoform_inference: Conducts peptidoform-level inference.
    - infer_peptidoforms: Orchestrates the IPF workflow.

Classes:
    None
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rankdata

from ._config import IPFIOConfig
from .io.dispatcher import ReaderDispatcher, WriterDispatcher


def compute_model_fdr(data_in):
    """
    Computes model-based FDR estimates from posterior error probabilities.

    Args:
        data_in (array-like): Input posterior error probabilities.

    Returns:
        np.ndarray: FDR estimates for the input data.
    """
    data = np.asarray(data_in)

    # compute model based FDR estimates from posterior error probabilities
    order = np.argsort(data)

    ranks = np.zeros(data.shape[0], dtype=int)
    fdr = np.zeros(data.shape[0])

    # rank data with with maximum ranks for ties
    ranks[order] = rankdata(data[order], method="max")

    # compute FDR/q-value by using cumulative sum of maximum rank for ties
    fdr[order] = data[order].cumsum()[ranks[order] - 1] / ranks[order]

    return fdr


def prepare_precursor_bm(data):
    """
    Prepares Bayesian model data for precursor-level inference.

    Args:
        data (pd.DataFrame): Input data containing MS1 and MS2 precursor probabilities.

    Returns:
        pd.DataFrame: Bayesian model data for precursor-level inference.
    """
    # MS1-level precursors
    ms1_precursor_data = data[
        ["feature_id", "ms2_peakgroup_pep", "ms1_precursor_pep"]
    ].dropna(axis=0, how="any")
    ms1_bm_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature_id": ms1_precursor_data["feature_id"],
                    "prior": 1 - ms1_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 1 - ms1_precursor_data["ms1_precursor_pep"],
                    "hypothesis": True,
                }
            ),
            pd.DataFrame(
                {
                    "feature_id": ms1_precursor_data["feature_id"],
                    "prior": ms1_precursor_data["ms2_peakgroup_pep"],
                    "evidence": ms1_precursor_data["ms1_precursor_pep"],
                    "hypothesis": False,
                }
            ),
        ]
    )

    # MS2-level precursors
    ms2_precursor_data = data[
        ["feature_id", "ms2_peakgroup_pep", "ms2_precursor_pep"]
    ].dropna(axis=0, how="any")
    ms2_bm_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature_id": ms2_precursor_data["feature_id"],
                    "prior": 1 - ms2_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 1 - ms2_precursor_data["ms2_precursor_pep"],
                    "hypothesis": True,
                }
            ),
            pd.DataFrame(
                {
                    "feature_id": ms2_precursor_data["feature_id"],
                    "prior": ms2_precursor_data["ms2_peakgroup_pep"],
                    "evidence": ms2_precursor_data["ms2_precursor_pep"],
                    "hypothesis": False,
                }
            ),
        ]
    )

    # missing precursor data
    missing_precursor_data = (
        data[["feature_id", "ms2_peakgroup_pep"]]
        .dropna(axis=0, how="any")
        .drop_duplicates()
    )
    missing_bm_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature_id": missing_precursor_data["feature_id"],
                    "prior": 1 - missing_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 0,
                    "hypothesis": True,
                }
            ),
            pd.DataFrame(
                {
                    "feature_id": missing_precursor_data["feature_id"],
                    "prior": missing_precursor_data["ms2_peakgroup_pep"],
                    "evidence": 1,
                    "hypothesis": False,
                }
            ),
        ]
    )

    # combine precursor data
    precursor_bm_data = pd.concat([ms1_bm_data, ms2_bm_data])
    # append missing precursors if no MS1/MS2 evidence is available
    precursor_bm_data = pd.concat(
        [
            precursor_bm_data,
            missing_bm_data.loc[
                ~missing_bm_data["feature_id"].isin(precursor_bm_data["feature_id"])
            ],
        ]
    )

    return precursor_bm_data


def transfer_confident_evidence_across_runs(
    df1,
    across_run_confidence_threshold,
    group_cols=[
        "feature_id",
        "transition_id",
        "peptide_id",
        "bmask",
        "num_peptidoforms",
        "alignment_group_id",
    ],
    value_cols=["pep", "precursor_peakgroup_pep"],
):
    """
    Propagates confident evidence across aligned runs.

    Args:
        df1 (pd.DataFrame): Input data containing feature-level information.
        across_run_confidence_threshold (float): Confidence threshold for propagation.
        group_cols (list): Columns to group by during propagation.
        value_cols (list): Columns to apply the minimum reduction.

    Returns:
        pd.DataFrame: Data with propagated evidence across runs.
    """
    feature_ids = np.unique(df1["feature_id"])
    df_list = []
    for feature_id in feature_ids:
        tmp_df = df1[
            (df1["feature_id"] == feature_id)
            | (
                (df1["feature_id"] != feature_id)
                & (df1["pep"] <= across_run_confidence_threshold)
            )
        ]
        tmp_df["feature_id"] = feature_id
        df_list.append(tmp_df)
    df_filtered = pd.concat(df_list)

    # Group by relevant columns and apply min reduction
    df_result = df_filtered.groupby(group_cols, as_index=False)[value_cols].min()

    return df_result


def prepare_transition_bm(
    data, propagate_signal_across_runs, across_run_confidence_threshold
):
    """
    Prepares Bayesian model data for transition-level inference.

    Args:
        data (pd.DataFrame): Input data containing transition-level information.
        propagate_signal_across_runs (bool): Whether to propagate signal across runs.
        across_run_confidence_threshold (float): Confidence threshold for propagation.

    Returns:
        pd.DataFrame: Bayesian model data for transition-level inference.
    """
    # Propagate peps <= threshold for aligned feature groups across runs
    if propagate_signal_across_runs:
        ## Separate out features that need propagation and those that don't to avoid calling apply on the features that don't need propagated peps
        non_prop_data = data.loc[data["feature_id"] == data["alignment_group_id"]]
        prop_data = data.loc[data["feature_id"] != data["alignment_group_id"]]

        # Group by alignment_group_id and apply function in parallel
        data_with_confidence = (
            prop_data.groupby("alignment_group_id", group_keys=False)
            .apply(
                lambda df: transfer_confident_evidence_across_runs(
                    df, across_run_confidence_threshold
                )
            )
            .reset_index(drop=True)
        )

        logger.info(
            f"Propagating signal for {len(prop_data['feature_id'].unique())} aligned features of total {len(data['feature_id'].unique())} features across runs ..."
        )

        ## Concat non prop data with prop data
        data = pd.concat([non_prop_data, data_with_confidence], ignore_index=True)

    # peptide_id = -1 indicates h0, i.e. the peak group is wrong!
    # initialize priors
    data.loc[data.peptide_id != -1, "prior"] = (
        1 - data.loc[data.peptide_id != -1, "precursor_peakgroup_pep"]
    ) / data.loc[data.peptide_id != -1, "num_peptidoforms"]  # potential peptidoforms
    data.loc[data.peptide_id == -1, "prior"] = data.loc[
        data.peptide_id == -1, "precursor_peakgroup_pep"
    ]  # h0

    # set evidence
    data.loc[data.bmask == 1, "evidence"] = (
        1 - data.loc[data.bmask == 1, "pep"]
    )  # we have evidence FOR this peptidoform or h0
    data.loc[data.bmask == 0, "evidence"] = data.loc[
        data.bmask == 0, "pep"
    ]  # we have evidence AGAINST this peptidoform or h0

    if propagate_signal_across_runs:
        cols = [
            "feature_id",
            "alignment_group_id",
            "num_peptidoforms",
            "prior",
            "evidence",
            "peptide_id",
        ]
    else:
        cols = ["feature_id", "num_peptidoforms", "prior", "evidence", "peptide_id"]
    data = data[cols]
    data = data.rename(columns=lambda x: x.replace("peptide_id", "hypothesis"))

    return data


def apply_bm(data):
    """
    Applies the Bayesian model to compute posterior probabilities.

    Args:
        data (pd.DataFrame): Input Bayesian model data.

    Returns:
        pd.DataFrame: Data with posterior probabilities for each hypothesis.
    """
    # compute likelihood * prior per feature & hypothesis
    # all priors are identical but pandas DF multiplication requires aggregation, so we use min()
    pp_data = (
        data.groupby(["feature_id", "hypothesis"])["evidence"].prod()
        * data.groupby(["feature_id", "hypothesis"])["prior"].min()
    ).reset_index()
    pp_data.columns = ["feature_id", "hypothesis", "likelihood_prior"]

    # compute likelihood sum per feature
    pp_data["likelihood_sum"] = pp_data.groupby("feature_id")[
        "likelihood_prior"
    ].transform("sum")

    # compute posterior hypothesis probability
    pp_data["posterior"] = pp_data["likelihood_prior"] / pp_data["likelihood_sum"]

    return pp_data.fillna(value=0)


def precursor_inference(
    data,
    ipf_ms1_scoring,
    ipf_ms2_scoring,
    ipf_max_precursor_pep,
    ipf_max_precursor_peakgroup_pep,
):
    """
    Conducts precursor-level inference.

    Args:
        data (pd.DataFrame): Input data containing precursor-level information.
        ipf_ms1_scoring (bool): Whether to use MS1-level scoring.
        ipf_ms2_scoring (bool): Whether to use MS2-level scoring.
        ipf_max_precursor_pep (float): Maximum PEP threshold for precursors.
        ipf_max_precursor_peakgroup_pep (float): Maximum PEP threshold for peak groups.

    Returns:
        pd.DataFrame: Inferred precursor probabilities.
    """
    # prepare MS1-level precursor data
    if ipf_ms1_scoring:
        ms1_precursor_data = data[data["ms1_precursor_pep"] < ipf_max_precursor_pep][
            ["feature_id", "ms1_precursor_pep"]
        ].drop_duplicates()
    else:
        ms1_precursor_data = data[["feature_id"]].drop_duplicates()
        ms1_precursor_data["ms1_precursor_pep"] = np.nan

    # prepare MS2-level precursor data
    if ipf_ms2_scoring:
        ms2_precursor_data = data[data["ms2_precursor_pep"] < ipf_max_precursor_pep][
            ["feature_id", "ms2_precursor_pep"]
        ].drop_duplicates()
    else:
        ms2_precursor_data = data[["feature_id"]].drop_duplicates()
        ms2_precursor_data["ms2_precursor_pep"] = np.nan

    # prepare MS2-level peak group data
    ms2_pg_data = data[["feature_id", "ms2_peakgroup_pep"]].drop_duplicates()

    if ipf_ms1_scoring or ipf_ms2_scoring:
        # merge MS1- & MS2-level precursor and peak group data
        precursor_data = ms2_precursor_data.merge(
            ms1_precursor_data, on=["feature_id"], how="outer"
        ).merge(ms2_pg_data, on=["feature_id"], how="outer")

        # prepare precursor-level Bayesian model
        logger.info("Preparing precursor-level data ... ")
        precursor_data_bm = prepare_precursor_bm(precursor_data)

        # compute posterior precursor probability
        logger.info("Conducting precursor-level inference ... ")
        prec_pp_data = apply_bm(precursor_data_bm)
        prec_pp_data["precursor_peakgroup_pep"] = 1 - prec_pp_data["posterior"]

        inferred_precursors = prec_pp_data[prec_pp_data["hypothesis"]][
            ["feature_id", "precursor_peakgroup_pep"]
        ]

    else:
        # no precursor-level data on MS1 and/or MS2 should be used; use peak group-level data
        logger.info("Skipping precursor-level inference.")
        inferred_precursors = ms2_pg_data.rename(
            columns=lambda x: x.replace("ms2_peakgroup_pep", "precursor_peakgroup_pep")
        )

    inferred_precursors = inferred_precursors[
        (
            inferred_precursors["precursor_peakgroup_pep"]
            < ipf_max_precursor_peakgroup_pep
        )
    ]

    return inferred_precursors


def peptidoform_inference(
    transition_table,
    precursor_data,
    ipf_grouped_fdr,
    propagate_signal_across_runs,
    across_run_confidence_threshold,
):
    """
    Conducts peptidoform-level inference.

    Args:
        transition_table (pd.DataFrame): Input data containing transition-level information.
        precursor_data (pd.DataFrame): Precursor-level probabilities.
        ipf_grouped_fdr (bool): Whether to use grouped FDR estimation.
        propagate_signal_across_runs (bool): Whether to propagate signal across runs.
        across_run_confidence_threshold (float): Confidence threshold for propagation.

    Returns:
        pd.DataFrame: Inferred peptidoform probabilities and FDR estimates.
    """
    transition_table = pd.merge(transition_table, precursor_data, on="feature_id")

    # compute transition posterior probabilities
    logger.info("Preparing peptidoform-level data ... ")
    transition_data_bm = prepare_transition_bm(
        transition_table, propagate_signal_across_runs, across_run_confidence_threshold
    )

    # compute posterior peptidoform probability
    logger.info("Conducting peptidoform-level inference ... ")

    pf_pp_data = apply_bm(transition_data_bm)
    pf_pp_data["pep"] = 1 - pf_pp_data["posterior"]

    # compute model-based FDR
    if ipf_grouped_fdr:
        pf_pp_data["qvalue"] = (
            pd.merge(
                pf_pp_data,
                transition_data_bm[
                    ["feature_id", "num_peptidoforms"]
                ].drop_duplicates(),
                on=["feature_id"],
                how="inner",
            )
            .groupby("num_peptidoforms")["pep"]
            .transform(compute_model_fdr)
        )
    else:
        pf_pp_data["qvalue"] = compute_model_fdr(pf_pp_data["pep"])

    # merge precursor-level data with UIS data
    result = pf_pp_data.merge(
        precursor_data[["feature_id", "precursor_peakgroup_pep"]].drop_duplicates(),
        on=["feature_id"],
        how="inner",
    )

    return result


def infer_peptidoforms(config: IPFIOConfig):
    """
    Orchestrates the Inference of PeptidoForms (IPF) workflow.

    Args:
        config (IPFIOConfig): Configuration object for the IPF workflow.

    Returns:
        None
    """
    logger.info("Starting IPF (Inference of PeptidoForms).")
    reader = ReaderDispatcher.get_reader(config)

    # precursor level
    precursor_table = reader.read(level="peakgroup_precursor")
    precursor_data = precursor_inference(
        precursor_table,
        config.ipf_ms1_scoring,
        config.ipf_ms2_scoring,
        config.ipf_max_precursor_pep,
        config.ipf_max_precursor_peakgroup_pep,
    )

    # peptidoform level
    peptidoform_table = reader.read(level="transition")

    ## prepare for propagating signal across runs for aligned features
    if config.propagate_signal_across_runs:
        across_run_feature_map = reader.read(level="alignment")

        peptidoform_table = peptidoform_table.merge(
            across_run_feature_map, how="left", on="feature_id"
        )
        ## Fill missing alignment_group_id with feature_id for those that are not aligned
        peptidoform_table["alignment_group_id"] = peptidoform_table[
            "alignment_group_id"
        ].astype(object)
        mask = peptidoform_table["alignment_group_id"].isna()
        peptidoform_table.loc[mask, "alignment_group_id"] = peptidoform_table.loc[
            mask, "feature_id"
        ].astype(str)

        peptidoform_table = peptidoform_table.astype({"alignment_group_id": "int64"})

    peptidoform_data = peptidoform_inference(
        peptidoform_table,
        precursor_data,
        config.ipf_grouped_fdr,
        config.propagate_signal_across_runs,
        config.across_run_confidence_threshold,
    )

    # finalize results and write to table
    logger.info("Storing results.")
    peptidoform_data = peptidoform_data[peptidoform_data["hypothesis"] != -1][
        ["feature_id", "hypothesis", "precursor_peakgroup_pep", "qvalue", "pep"]
    ]
    peptidoform_data.columns = [
        "FEATURE_ID",
        "PEPTIDE_ID",
        "PRECURSOR_PEAKGROUP_PEP",
        "QVALUE",
        "PEP",
    ]

    # Convert feature_id to int64
    peptidoform_data = peptidoform_data.astype({"FEATURE_ID": "int64"})

    writer = WriterDispatcher.get_writer(config)
    writer.save_results(result=peptidoform_data)
