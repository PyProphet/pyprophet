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
from scipy.special import expit, logsumexp
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


def compute_grouped_model_fdr(data_in, group_keys, log_prefix="Grouped FDR"):
    """
    Compute model-based FDR estimates independently within each group.

    Args:
        data_in (array-like): Input posterior error probabilities.
        group_keys (array-like): Group label per row in ``data_in``.
        log_prefix (str): Prefix used for group-count logging.

    Returns:
        np.ndarray: FDR estimates for the input data, grouped by ``group_keys``.
    """
    data = np.asarray(data_in, dtype=float)
    groups = pd.Series(group_keys).fillna("NA").astype(str)
    if len(groups) != len(data):
        raise ValueError("group_keys must have the same length as data_in.")

    qvalues = np.empty(len(data), dtype=float)
    counts = groups.value_counts().sort_index()
    logger.info(
        f"{log_prefix}: "
        + ", ".join(f"{group}={count}" for group, count in counts.items())
    )

    for group, idx in groups.groupby(groups).groups.items():
        idx_arr = np.fromiter(idx, dtype=int)
        qvalues[idx_arr] = compute_model_fdr(data[idx_arr])

    return qvalues


def _support_phospho_loss_group_keys(metrics):
    """
    Build coarse FDR strata from support strength and phospho-loss support.

    Support is grouped into <=1, 2, and >=3 supporting transitions. Each support
    stratum is then split by whether at least one phospho-loss supporting
    transition is present.
    """
    supporting = metrics["supporting_transitions"].fillna(0).astype(int).to_numpy()
    phospho_loss = (
        metrics["phospho_loss_supporting_transitions"].fillna(0).astype(int).to_numpy()
    )

    support_labels = np.where(
        supporting <= 1,
        "support_le1",
        np.where(supporting == 2, "support_eq2", "support_ge3"),
    )
    phospho_labels = np.where(phospho_loss > 0, "phloss", "no_phloss")
    return pd.Series(
        support_labels + "__" + phospho_labels,
        index=metrics.index,
        dtype="object",
    )


def compute_ipf_qvalues(
    pf_pp_data,
    grouped_fdr=False,
    grouped_fdr_strategy="num_peptidoforms",
    grouping_metrics=None,
):
    """
    Compute IPF q-values using pooled or grouped model-based FDR.

    Grouped FDR currently supports:
    - ``num_peptidoforms``: the legacy grouping by per-feature peptidoform count
    - ``support_phospho_loss``: grouping by supporting-transition bin and
      phospho-loss support
    """
    if not grouped_fdr:
        return compute_model_fdr(pf_pp_data["pep"])

    if grouped_fdr_strategy == "num_peptidoforms":
        if "num_peptidoforms" not in pf_pp_data.columns:
            raise ValueError(
                "num_peptidoforms is required for grouped FDR with strategy 'num_peptidoforms'."
            )
        return compute_grouped_model_fdr(
            pf_pp_data["pep"],
            pf_pp_data["num_peptidoforms"].fillna(-1).astype(int),
            log_prefix="Grouped FDR by num_peptidoforms",
        )

    if grouped_fdr_strategy == "support_phospho_loss":
        if grouping_metrics is None:
            raise ValueError(
                "grouping_metrics is required for grouped FDR with strategy 'support_phospho_loss'."
            )
        required_cols = {
            "feature_id",
            "peptide_id",
            "supporting_transitions",
            "phospho_loss_supporting_transitions",
        }
        missing = required_cols - set(grouping_metrics.columns)
        if missing:
            raise ValueError(
                "grouping_metrics is missing required columns for grouped FDR "
                f"strategy 'support_phospho_loss': {sorted(missing)}"
            )

        merged = pf_pp_data.merge(
            grouping_metrics[
                [
                    "feature_id",
                    "peptide_id",
                    "supporting_transitions",
                    "phospho_loss_supporting_transitions",
                ]
            ].drop_duplicates(),
            left_on=["feature_id", "hypothesis"],
            right_on=["feature_id", "peptide_id"],
            how="left",
        )
        group_keys = _support_phospho_loss_group_keys(merged)
        return compute_grouped_model_fdr(
            merged["pep"],
            group_keys,
            log_prefix="Grouped FDR by support/phospho-loss",
        )

    raise ValueError(
        "Unsupported ipf_grouped_fdr_strategy: "
        f"{grouped_fdr_strategy!r}. Expected 'num_peptidoforms' or 'support_phospho_loss'."
    )


def compute_post_ipf_filter_metrics(transition_table, precursor_table):
    """
    Computes per-feature / per-hypothesis metrics used for optional post-IPF filtering.

    Args:
        transition_table (pd.DataFrame): Transition-level peptidoform table before Bayesian modeling.
        precursor_table (pd.DataFrame): Peakgroup / precursor-level table, optionally including feature_ms2_intensity.

    Returns:
        pd.DataFrame: One row per feature_id + peptide_id with supporting transition
        counts and optional feature_ms2_intensity.
    """
    hypotheses = transition_table.loc[
        transition_table["peptide_id"] != -1, ["feature_id", "peptide_id"]
    ].drop_duplicates()

    supporting_cols = ["feature_id", "peptide_id", "transition_id"]
    supporting_cols.extend(
        [
            col
            for col in [
                "n_mapped_peptides",
                "has_phospho_loss",
                "isotope_overlap_score",
            ]
            if col in transition_table.columns
        ]
    )
    supporting_rows = transition_table.loc[
        (transition_table["peptide_id"] != -1) & (transition_table["bmask"] == 1),
        supporting_cols,
    ].drop_duplicates()

    supporting = (
        supporting_rows.groupby(["feature_id", "peptide_id"], as_index=False)[
            "transition_id"
        ]
        .nunique()
        .rename(columns={"transition_id": "supporting_transitions"})
    )

    if "n_mapped_peptides" in supporting_rows.columns:
        unique_supporting = (
            supporting_rows.loc[supporting_rows["n_mapped_peptides"] == 1]
            .groupby(["feature_id", "peptide_id"], as_index=False)["transition_id"]
            .nunique()
            .rename(columns={"transition_id": "unique_supporting_transitions"})
        )
    else:
        unique_supporting = pd.DataFrame(
            columns=["feature_id", "peptide_id", "unique_supporting_transitions"]
        )

    if "has_phospho_loss" in supporting_rows.columns:
        phospho_loss_supporting = (
            supporting_rows.loc[supporting_rows["has_phospho_loss"] == 1]
            .groupby(["feature_id", "peptide_id"], as_index=False)["transition_id"]
            .nunique()
            .rename(
                columns={"transition_id": "phospho_loss_supporting_transitions"}
            )
        )
    else:
        phospho_loss_supporting = pd.DataFrame(
            columns=[
                "feature_id",
                "peptide_id",
                "phospho_loss_supporting_transitions",
            ]
        )

    if "isotope_overlap_score" in supporting_rows.columns:
        supporting_overlap = (
            supporting_rows.groupby(["feature_id", "peptide_id"], as_index=False)[
                "isotope_overlap_score"
            ]
            .median()
            .rename(
                columns={
                    "isotope_overlap_score": "median_supporting_isotope_overlap"
                }
            )
        )
    else:
        supporting_overlap = pd.DataFrame(
            columns=[
                "feature_id",
                "peptide_id",
                "median_supporting_isotope_overlap",
            ]
        )

    metrics = (
        hypotheses.merge(supporting, on=["feature_id", "peptide_id"], how="left")
        .merge(unique_supporting, on=["feature_id", "peptide_id"], how="left")
        .merge(
            phospho_loss_supporting,
            on=["feature_id", "peptide_id"],
            how="left",
        )
        .merge(supporting_overlap, on=["feature_id", "peptide_id"], how="left")
        .fillna(
            {
                "supporting_transitions": 0,
                "unique_supporting_transitions": 0,
                "phospho_loss_supporting_transitions": 0,
            }
        )
    )
    metrics["supporting_transitions"] = metrics["supporting_transitions"].astype(int)
    metrics["unique_supporting_transitions"] = metrics[
        "unique_supporting_transitions"
    ].astype(int)
    metrics["phospho_loss_supporting_transitions"] = metrics[
        "phospho_loss_supporting_transitions"
    ].astype(int)

    if "feature_ms2_intensity" in precursor_table.columns:
        metrics = metrics.merge(
            precursor_table[["feature_id", "feature_ms2_intensity"]].drop_duplicates(),
            on="feature_id",
            how="left",
        )

    return metrics


def prepare_post_ipf_filter_metrics(
    transition_table,
    precursor_table,
    propagate_signal_across_runs=False,
    across_run_confidence_threshold=0.5,
):
    """
    Prepares post-IPF filter metrics from the same transition evidence state used by IPF.

    When across-run propagation is enabled, supporting-transition counts are computed on the
    propagated evidence table so the post-IPF filter matches the final inference behavior.

    Args:
        transition_table (pd.DataFrame): Transition-level peptidoform table.
        precursor_table (pd.DataFrame): Peakgroup / precursor-level table.
        propagate_signal_across_runs (bool): Whether IPF propagates evidence across runs.
        across_run_confidence_threshold (float): Confidence threshold for signal propagation.

    Returns:
        pd.DataFrame: Metrics from compute_post_ipf_filter_metrics().
    """
    filter_transition_table = transition_table.copy()

    if propagate_signal_across_runs:
        non_prop_data = filter_transition_table.loc[
            filter_transition_table["feature_id"]
            == filter_transition_table["alignment_group_id"]
        ]
        prop_data = filter_transition_table.loc[
            filter_transition_table["feature_id"]
            != filter_transition_table["alignment_group_id"]
        ]

        if len(prop_data) > 0:
            group_cols = [
                "feature_id",
                "transition_id",
                "peptide_id",
                "bmask",
                "num_peptidoforms",
                "alignment_group_id",
            ]
            group_cols.extend(
                [
                    col
                    for col in [
                        "n_mapped_peptides",
                        "has_phospho_loss",
                        "isotope_overlap_score",
                    ]
                    if col in filter_transition_table.columns
                ]
            )
            propagated_data = (
                prop_data.groupby("alignment_group_id", group_keys=False)
                .apply(
                    lambda df: transfer_confident_evidence_across_runs(
                        df,
                        across_run_confidence_threshold,
                        group_cols=group_cols,
                        value_cols=["pep"],
                    )
                )
                .reset_index(drop=True)
            )
            filter_transition_table = pd.concat(
                [non_prop_data, propagated_data], ignore_index=True
            )
        else:
            filter_transition_table = non_prop_data.copy()

    return compute_post_ipf_filter_metrics(filter_transition_table, precursor_table)


def _ipf_filters_active(
    min_supporting_transitions=0,
    min_unique_supporting_transitions=0,
    require_phospho_loss_below_support=0,
    min_peakgroup_intensity=0.0,
    conditional_min_peakgroup_intensity=0.0,
):
    return (
        min_supporting_transitions > 0
        or min_unique_supporting_transitions > 0
        or require_phospho_loss_below_support > 0
        or min_peakgroup_intensity > 0
        or conditional_min_peakgroup_intensity > 0
    )


def _ipf_score_adjustment_active(
    score_support_log_weight=0.0,
    score_unique_support_log_weight=0.0,
    score_log_intensity_weight=0.0,
    score_phospho_loss_bonus=0.0,
    score_overlap_penalty_weight=0.0,
):
    return (
        score_support_log_weight != 0.0
        or score_unique_support_log_weight != 0.0
        or score_log_intensity_weight != 0.0
        or score_phospho_loss_bonus != 0.0
        or score_overlap_penalty_weight != 0.0
    )


def _transition_evidence_adjustment_active(
    transition_evidence_nonunique_logit_scale=1.0,
    transition_evidence_no_phospho_loss_logit_scale=1.0,
    transition_evidence_overlap_logit_scale_weight=0.0,
    transition_evidence_max_isotope_overlap=0.0,
):
    return (
        transition_evidence_nonunique_logit_scale != 1.0
        or transition_evidence_no_phospho_loss_logit_scale != 1.0
        or transition_evidence_overlap_logit_scale_weight != 0.0
        or transition_evidence_max_isotope_overlap > 0.0
    )


def apply_transition_evidence_adjustment(
    transition_table,
    transition_evidence_nonunique_logit_scale=1.0,
    transition_evidence_no_phospho_loss_logit_scale=1.0,
    transition_evidence_overlap_logit_scale_weight=0.0,
    transition_evidence_overlap_reference=0.02,
    transition_evidence_max_isotope_overlap=0.0,
):
    """
    Adjust transition-level PEPs before BM so weak/non-informative identifying transitions
    contribute less extreme evidence.

    The adjustment shrinks the transition PEP logit toward zero, which weakens both supporting
    and opposing evidence symmetrically once the BM evidence is constructed from PEP and BMASK.
    Optionally, high-overlap transitions can be removed entirely before BM.
    """
    if not _transition_evidence_adjustment_active(
        transition_evidence_nonunique_logit_scale=transition_evidence_nonunique_logit_scale,
        transition_evidence_no_phospho_loss_logit_scale=transition_evidence_no_phospho_loss_logit_scale,
        transition_evidence_overlap_logit_scale_weight=transition_evidence_overlap_logit_scale_weight,
        transition_evidence_max_isotope_overlap=transition_evidence_max_isotope_overlap,
    ):
        return transition_table

    if transition_evidence_nonunique_logit_scale <= 0:
        raise ValueError("transition_evidence_nonunique_logit_scale must be > 0.")
    if transition_evidence_no_phospho_loss_logit_scale <= 0:
        raise ValueError(
            "transition_evidence_no_phospho_loss_logit_scale must be > 0."
        )
    if transition_evidence_overlap_logit_scale_weight < 0:
        raise ValueError(
            "transition_evidence_overlap_logit_scale_weight must be >= 0."
        )
    if transition_evidence_overlap_reference < 0:
        raise ValueError("transition_evidence_overlap_reference must be >= 0.")
    if transition_evidence_max_isotope_overlap < 0:
        raise ValueError("transition_evidence_max_isotope_overlap must be >= 0.")

    adjusted = transition_table.copy()

    if transition_evidence_max_isotope_overlap > 0:
        if "isotope_overlap_score" not in adjusted.columns:
            raise ValueError(
                "isotope_overlap_score is required for transition_evidence_max_isotope_overlap filtering."
            )
        before = len(adjusted)
        adjusted = adjusted[
            adjusted["isotope_overlap_score"].fillna(transition_evidence_overlap_reference)
            <= transition_evidence_max_isotope_overlap
        ].copy()
        logger.info(
            "Applied pre-BM transition overlap filter: "
            f"kept {len(adjusted)}/{before} transition rows with isotope_overlap_score <= "
            f"{transition_evidence_max_isotope_overlap}."
        )

    if "pep" not in adjusted.columns:
        raise ValueError("Transition table must contain pep for evidence adjustment.")

    scales = np.ones(len(adjusted), dtype=float)

    if transition_evidence_nonunique_logit_scale != 1.0:
        if "n_mapped_peptides" not in adjusted.columns:
            raise ValueError(
                "n_mapped_peptides is required for transition_evidence_nonunique_logit_scale."
            )
        scales *= np.where(
            adjusted["n_mapped_peptides"].fillna(0).astype(int).to_numpy() > 1,
            transition_evidence_nonunique_logit_scale,
            1.0,
        )

    if transition_evidence_no_phospho_loss_logit_scale != 1.0:
        if "has_phospho_loss" not in adjusted.columns:
            raise ValueError(
                "has_phospho_loss is required for transition_evidence_no_phospho_loss_logit_scale."
            )
        scales *= np.where(
            adjusted["has_phospho_loss"].fillna(0).astype(int).to_numpy() > 0,
            1.0,
            transition_evidence_no_phospho_loss_logit_scale,
        )

    if transition_evidence_overlap_logit_scale_weight != 0.0:
        if "isotope_overlap_score" not in adjusted.columns:
            raise ValueError(
                "isotope_overlap_score is required for transition_evidence_overlap_logit_scale_weight."
            )
        overlap = (
            adjusted["isotope_overlap_score"]
            .fillna(transition_evidence_overlap_reference)
            .astype(float)
            .clip(lower=0.0)
            .to_numpy()
        )
        scales *= np.exp(
            -transition_evidence_overlap_logit_scale_weight
            * np.maximum(0.0, overlap - transition_evidence_overlap_reference)
        )

    eps = np.finfo(float).eps
    pep = adjusted["pep"].astype(float).clip(lower=eps, upper=1 - eps).to_numpy()
    pep_logit = np.log(pep / (1 - pep))
    adjusted["pep"] = expit(pep_logit * scales)

    logger.info(
        "Applied pre-BM transition evidence shaping: "
        f"nonunique_logit_scale={transition_evidence_nonunique_logit_scale}, "
        f"no_phospho_loss_logit_scale={transition_evidence_no_phospho_loss_logit_scale}, "
        f"overlap_logit_scale_weight={transition_evidence_overlap_logit_scale_weight}, "
        f"overlap_reference={transition_evidence_overlap_reference}, "
        f"max_isotope_overlap={transition_evidence_max_isotope_overlap}. "
        f"Adjusted {len(adjusted)} transition rows."
    )

    return adjusted


def apply_ipf_score_adjustment(
    result,
    score_metrics,
    ipf_grouped_fdr=False,
    score_support_log_weight=0.0,
    score_unique_support_log_weight=0.0,
    score_log_intensity_weight=0.0,
    score_phospho_loss_bonus=0.0,
    score_intensity_reference=10000.0,
    score_overlap_penalty_weight=0.0,
    score_overlap_reference=0.02,
    score_adjust_max_supporting_transitions=0,
    score_adjust_no_phospho_loss_only=False,
):
    """
    Experimentally re-rank IPF peptidoform hypotheses by adjusting the PEP in logit space
    using support-count and feature-intensity signals, then recomputing q-values.

    Positive weights improve hypotheses with stronger support / higher intensity. When
    score_adjust_max_supporting_transitions is set, the adjustment only applies to weak-
    support hypotheses at or below that threshold. When score_adjust_no_phospho_loss_only
    is enabled, only hypotheses without phospho-loss support are adjusted.
    """
    if not _ipf_score_adjustment_active(
        score_support_log_weight=score_support_log_weight,
        score_unique_support_log_weight=score_unique_support_log_weight,
        score_log_intensity_weight=score_log_intensity_weight,
        score_phospho_loss_bonus=score_phospho_loss_bonus,
        score_overlap_penalty_weight=score_overlap_penalty_weight,
    ):
        return result

    if score_intensity_reference <= 0:
        raise ValueError("score_intensity_reference must be > 0.")
    if score_overlap_reference < 0:
        raise ValueError("score_overlap_reference must be >= 0.")

    merged = result.merge(
        score_metrics,
        left_on=["feature_id", "hypothesis"],
        right_on=["feature_id", "peptide_id"],
        how="left",
    )

    for metric_col in [
        "supporting_transitions",
        "unique_supporting_transitions",
        "phospho_loss_supporting_transitions",
    ]:
        if metric_col not in merged.columns:
            merged[metric_col] = 0
        merged[metric_col] = merged[metric_col].fillna(0).astype(int)

    if score_log_intensity_weight != 0.0 and "feature_ms2_intensity" not in merged.columns:
        raise ValueError(
            "feature_ms2_intensity is required for support/intensity-aware IPF score adjustment."
        )
    if score_overlap_penalty_weight != 0.0 and "median_supporting_isotope_overlap" not in merged.columns:
        raise ValueError(
            "median_supporting_isotope_overlap is required for overlap-aware IPF score adjustment."
        )

    apply_mask = merged["hypothesis"] != -1
    if score_adjust_max_supporting_transitions > 0:
        apply_mask &= (
            merged["supporting_transitions"] <= score_adjust_max_supporting_transitions
        )
    if score_adjust_no_phospho_loss_only:
        apply_mask &= merged["phospho_loss_supporting_transitions"] == 0

    score_bonus = np.zeros(len(merged), dtype=float)
    if score_support_log_weight != 0.0:
        score_bonus += score_support_log_weight * np.log1p(
            merged["supporting_transitions"].astype(float).to_numpy()
        )
    if score_unique_support_log_weight != 0.0:
        score_bonus += score_unique_support_log_weight * np.log1p(
            merged["unique_supporting_transitions"].astype(float).to_numpy()
        )
    if score_log_intensity_weight != 0.0:
        intensity = (
            merged["feature_ms2_intensity"]
            .fillna(0.0)
            .astype(float)
            .clip(lower=1.0)
            .to_numpy()
        )
        score_bonus += score_log_intensity_weight * (
            np.log10(intensity) - np.log10(score_intensity_reference)
        )
    if score_phospho_loss_bonus != 0.0:
        score_bonus += score_phospho_loss_bonus * (
            merged["phospho_loss_supporting_transitions"].astype(float).to_numpy() > 0
        )
    if score_overlap_penalty_weight != 0.0:
        overlap = (
            merged["median_supporting_isotope_overlap"]
            .fillna(score_overlap_reference)
            .astype(float)
            .clip(lower=0.0)
            .to_numpy()
        )
        score_bonus -= score_overlap_penalty_weight * np.maximum(
            0.0, overlap - score_overlap_reference
        )

    score_bonus = np.where(apply_mask.to_numpy(), score_bonus, 0.0)

    eps = np.finfo(float).eps
    pep = merged["pep"].astype(float).clip(lower=eps, upper=1 - eps).to_numpy()
    pep_logit = np.log(pep / (1 - pep))
    adjusted_pep = expit(pep_logit - score_bonus)
    adjusted_pep = np.clip(adjusted_pep, eps, 1 - eps)

    adjusted = merged.copy()
    adjusted["pep"] = adjusted_pep

    if ipf_grouped_fdr:
        if "num_peptidoforms" not in adjusted.columns:
            raise ValueError(
                "num_peptidoforms is required for grouped FDR after IPF score adjustment."
            )
        adjusted["qvalue"] = adjusted.groupby("num_peptidoforms")["pep"].transform(
            compute_model_fdr
        )
    else:
        adjusted["qvalue"] = compute_model_fdr(adjusted["pep"])

    logger.info(
        "Applied support/intensity-aware IPF score adjustment: "
        f"support_log_weight={score_support_log_weight}, "
        f"unique_support_log_weight={score_unique_support_log_weight}, "
        f"log_intensity_weight={score_log_intensity_weight}, "
        f"phospho_loss_bonus={score_phospho_loss_bonus}, "
        f"intensity_reference={score_intensity_reference}, "
        f"overlap_penalty_weight={score_overlap_penalty_weight}, "
        f"overlap_reference={score_overlap_reference}, "
        f"max_supporting_transitions={score_adjust_max_supporting_transitions}, "
        f"no_phospho_loss_only={score_adjust_no_phospho_loss_only}. "
        f"Adjusted {int(apply_mask.sum())}/{len(adjusted)} hypothesis rows."
    )

    return adjusted


def _apply_ipf_filter_thresholds(
    metrics,
    min_supporting_transitions=0,
    min_unique_supporting_transitions=0,
    require_phospho_loss_below_support=0,
    min_peakgroup_intensity=0.0,
    conditional_min_peakgroup_intensity=0.0,
    conditional_min_peakgroup_intensity_max_supporting_transitions=0,
    conditional_min_peakgroup_intensity_no_phospho_loss_only=False,
    log_prefix="Applied IPF",
):
    filtered = metrics.copy()

    for metric_col in [
        "supporting_transitions",
        "unique_supporting_transitions",
        "phospho_loss_supporting_transitions",
    ]:
        if metric_col not in filtered.columns:
            filtered[metric_col] = 0
        filtered[metric_col] = filtered[metric_col].fillna(0).astype(int)

    if min_supporting_transitions > 0:
        before = len(filtered)
        filtered = filtered[
            filtered["supporting_transitions"] >= min_supporting_transitions
        ].copy()
        logger.info(
            f"{log_prefix} supporting-transition filter: "
            f"kept {len(filtered)}/{before} feature-hypothesis rows "
            f"with supporting_transitions >= {min_supporting_transitions}."
        )

    if min_unique_supporting_transitions > 0:
        before = len(filtered)
        filtered = filtered[
            filtered["unique_supporting_transitions"] >= min_unique_supporting_transitions
        ].copy()
        logger.info(
            f"{log_prefix} unique-supporting-transition filter: "
            f"kept {len(filtered)}/{before} feature-hypothesis rows "
            f"with unique_supporting_transitions >= {min_unique_supporting_transitions}."
        )

    if require_phospho_loss_below_support > 0:
        before = len(filtered)
        filtered = filtered[
            (filtered["supporting_transitions"] >= require_phospho_loss_below_support)
            | (filtered["phospho_loss_supporting_transitions"] > 0)
        ].copy()
        logger.info(
            f"{log_prefix} phospho-loss rescue filter: "
            f"kept {len(filtered)}/{before} feature-hypothesis rows "
            f"with supporting_transitions >= {require_phospho_loss_below_support} "
            "or at least one phospho-loss supporting transition."
        )

    if min_peakgroup_intensity > 0:
        if "feature_ms2_intensity" not in filtered.columns:
            raise ValueError(
                "feature_ms2_intensity is required for ipf_min_peakgroup_intensity filtering."
            )
        before = len(filtered)
        filtered = filtered[
            filtered["feature_ms2_intensity"] >= min_peakgroup_intensity
        ].copy()
        logger.info(
            f"{log_prefix} peakgroup-intensity filter: "
            f"kept {len(filtered)}/{before} feature-hypothesis rows "
            f"with feature_ms2_intensity >= {min_peakgroup_intensity}."
        )

    if conditional_min_peakgroup_intensity > 0:
        if conditional_min_peakgroup_intensity_max_supporting_transitions <= 0:
            raise ValueError(
                "ipf_conditional_min_peakgroup_intensity_max_supporting_transitions must be > 0 "
                "when ipf_conditional_min_peakgroup_intensity is enabled."
            )
        if "feature_ms2_intensity" not in filtered.columns:
            raise ValueError(
                "feature_ms2_intensity is required for ipf_conditional_min_peakgroup_intensity filtering."
            )

        before = len(filtered)
        weak_support_mask = (
            filtered["supporting_transitions"]
            <= conditional_min_peakgroup_intensity_max_supporting_transitions
        )
        if conditional_min_peakgroup_intensity_no_phospho_loss_only:
            weak_support_mask = weak_support_mask & (
                filtered["phospho_loss_supporting_transitions"] == 0
            )

        filtered = filtered[
            (~weak_support_mask)
            | (
                filtered["feature_ms2_intensity"]
                >= conditional_min_peakgroup_intensity
            )
        ].copy()
        logger.info(
            f"{log_prefix} conditional peakgroup-intensity filter: "
            f"kept {len(filtered)}/{before} feature-hypothesis rows "
            f"after requiring feature_ms2_intensity >= {conditional_min_peakgroup_intensity} "
            f"for rows with supporting_transitions <= "
            f"{conditional_min_peakgroup_intensity_max_supporting_transitions}"
            + (
                " and no phospho-loss supporting transitions."
                if conditional_min_peakgroup_intensity_no_phospho_loss_only
                else "."
            )
        )

    return filtered


def apply_pre_bm_hypothesis_filters(
    transition_table,
    precursor_table,
    propagate_signal_across_runs=False,
    across_run_confidence_threshold=0.5,
    min_supporting_transitions=0,
    min_unique_supporting_transitions=0,
    require_phospho_loss_below_support=0,
    min_peakgroup_intensity=0.0,
    conditional_min_peakgroup_intensity=0.0,
    conditional_min_peakgroup_intensity_max_supporting_transitions=0,
    conditional_min_peakgroup_intensity_no_phospho_loss_only=False,
):
    """
    Applies optional structural hypothesis filters before transition-level Bayesian modeling.

    The same feature/peptidoform metrics used for post-IPF filtering are computed on the
    current evidence state and used to prune weak hypotheses before posterior inference.
    Retained rows have their num_peptidoforms recomputed so priors remain coherent.

    Args:
        transition_table (pd.DataFrame): Transition-level peptidoform table.
        precursor_table (pd.DataFrame): Peakgroup / precursor-level table.
        propagate_signal_across_runs (bool): Whether IPF propagates evidence across runs.
        across_run_confidence_threshold (float): Confidence threshold for signal propagation.
        min_supporting_transitions (int): Minimum supporting transitions required.
        min_unique_supporting_transitions (int): Minimum uniquely supporting transitions required.
        require_phospho_loss_below_support (int): Require at least one phospho-loss supporting
            transition when supporting_transitions is below this threshold.
        min_peakgroup_intensity (float): Minimum MS2 feature intensity required.

    Returns:
        pd.DataFrame: Filtered transition table ready for peptidoform inference.
    """
    if not _ipf_filters_active(
        min_supporting_transitions=min_supporting_transitions,
        min_unique_supporting_transitions=min_unique_supporting_transitions,
        require_phospho_loss_below_support=require_phospho_loss_below_support,
        min_peakgroup_intensity=min_peakgroup_intensity,
        conditional_min_peakgroup_intensity=conditional_min_peakgroup_intensity,
    ):
        return transition_table

    filter_metrics = prepare_post_ipf_filter_metrics(
        transition_table,
        precursor_table,
        propagate_signal_across_runs=propagate_signal_across_runs,
        across_run_confidence_threshold=across_run_confidence_threshold,
    )
    kept_hypotheses = _apply_ipf_filter_thresholds(
        filter_metrics,
        min_supporting_transitions=min_supporting_transitions,
        min_unique_supporting_transitions=min_unique_supporting_transitions,
        require_phospho_loss_below_support=require_phospho_loss_below_support,
        min_peakgroup_intensity=min_peakgroup_intensity,
        conditional_min_peakgroup_intensity=conditional_min_peakgroup_intensity,
        conditional_min_peakgroup_intensity_max_supporting_transitions=conditional_min_peakgroup_intensity_max_supporting_transitions,
        conditional_min_peakgroup_intensity_no_phospho_loss_only=conditional_min_peakgroup_intensity_no_phospho_loss_only,
        log_prefix="Applied pre-BM",
    )[["feature_id", "peptide_id"]].drop_duplicates()

    filtered = transition_table.merge(
        kept_hypotheses.assign(_keep_hypothesis=True),
        on=["feature_id", "peptide_id"],
        how="left",
    )
    filtered = filtered[
        (filtered["peptide_id"] == -1) | (filtered["_keep_hypothesis"] == True)
    ].copy()
    filtered = filtered.drop(columns=["_keep_hypothesis"])

    num_peptidoforms = (
        filtered.loc[filtered["peptide_id"] != -1]
        .groupby("feature_id")["peptide_id"]
        .nunique()
        .rename("num_peptidoforms")
    )
    filtered = filtered.drop(columns=["num_peptidoforms"], errors="ignore").merge(
        num_peptidoforms,
        on="feature_id",
        how="left",
    )
    filtered["num_peptidoforms"] = filtered["num_peptidoforms"].fillna(0).astype(int)

    logger.info(
        "Applied pre-BM hypothesis filtering: "
        f"kept {len(filtered)} transition rows spanning "
        f"{filtered.loc[filtered['peptide_id'] != -1, ['feature_id', 'peptide_id']].drop_duplicates().shape[0]} "
        "non-H0 feature-hypothesis candidates."
    )

    return filtered


def apply_post_ipf_filters(
    result,
    filter_metrics,
    min_supporting_transitions=0,
    min_unique_supporting_transitions=0,
    require_phospho_loss_below_support=0,
    min_peakgroup_intensity=0.0,
    conditional_min_peakgroup_intensity=0.0,
    conditional_min_peakgroup_intensity_max_supporting_transitions=0,
    conditional_min_peakgroup_intensity_no_phospho_loss_only=False,
):
    """
    Applies optional post-IPF filters to inferred peptidoform results.

    Args:
        result (pd.DataFrame): Inferred peptidoform results with FEATURE_ID / PEPTIDE_ID.
        filter_metrics (pd.DataFrame): Metrics from compute_post_ipf_filter_metrics().
        min_supporting_transitions (int): Minimum supporting transitions required.
        min_unique_supporting_transitions (int): Minimum uniquely supporting transitions required.
        require_phospho_loss_below_support (int): Require at least one phospho-loss supporting
            transition when supporting_transitions is below this threshold.
        min_peakgroup_intensity (float): Minimum MS2 feature intensity required.

    Returns:
        pd.DataFrame: Filtered peptidoform results.
    """
    if not _ipf_filters_active(
        min_supporting_transitions=min_supporting_transitions,
        min_unique_supporting_transitions=min_unique_supporting_transitions,
        require_phospho_loss_below_support=require_phospho_loss_below_support,
        min_peakgroup_intensity=min_peakgroup_intensity,
        conditional_min_peakgroup_intensity=conditional_min_peakgroup_intensity,
    ):
        return result

    merged = result.merge(
        filter_metrics,
        left_on=["FEATURE_ID", "PEPTIDE_ID"],
        right_on=["feature_id", "peptide_id"],
        how="left",
    )
    merged = _apply_ipf_filter_thresholds(
        merged,
        min_supporting_transitions=min_supporting_transitions,
        min_unique_supporting_transitions=min_unique_supporting_transitions,
        require_phospho_loss_below_support=require_phospho_loss_below_support,
        min_peakgroup_intensity=min_peakgroup_intensity,
        conditional_min_peakgroup_intensity=conditional_min_peakgroup_intensity,
        conditional_min_peakgroup_intensity_max_supporting_transitions=conditional_min_peakgroup_intensity_max_supporting_transitions,
        conditional_min_peakgroup_intensity_no_phospho_loss_only=conditional_min_peakgroup_intensity_no_phospho_loss_only,
        log_prefix="Applied post-IPF",
    )

    return merged[
        ["FEATURE_ID", "PEPTIDE_ID", "PRECURSOR_PEAKGROUP_PEP", "QVALUE", "PEP"]
    ].copy()


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


def apply_bm(data, use_log_space=False, evidence_epsilon=0.0):
    """
    Applies the Bayesian model to compute posterior probabilities.

    Args:
        data (pd.DataFrame): Input Bayesian model data.
        use_log_space (bool): Whether to compute Bayesian posteriors in log-space.
        evidence_epsilon (float): Optional clipping epsilon applied to evidence values
            before inference. 0 disables clipping.

    Returns:
        pd.DataFrame: Data with posterior probabilities for each hypothesis.
    """
    if evidence_epsilon < 0 or evidence_epsilon >= 0.5:
        raise ValueError("evidence_epsilon must satisfy 0 <= epsilon < 0.5.")

    if evidence_epsilon > 0:
        data = data.copy()
        data["evidence"] = data["evidence"].clip(
            lower=evidence_epsilon, upper=1 - evidence_epsilon
        )

    if use_log_space:
        with np.errstate(divide="ignore", invalid="ignore"):
            grouped_logs = (
                data.assign(log_evidence=np.log(data["evidence"]))
                .groupby(["feature_id", "hypothesis"])["log_evidence"]
                .sum()
                .reset_index()
            )
            grouped_prior = (
                data.groupby(["feature_id", "hypothesis"], as_index=False)["prior"]
                .min()
            )
            grouped_logs = grouped_logs.merge(
                grouped_prior, on=["feature_id", "hypothesis"], how="left"
            )
            grouped_logs["log_prior"] = np.log(grouped_logs["prior"])
            grouped_logs["log_likelihood_prior"] = (
                grouped_logs["log_evidence"] + grouped_logs["log_prior"]
            )
            grouped_logs["log_likelihood_sum"] = grouped_logs.groupby("feature_id")[
                "log_likelihood_prior"
            ].transform(logsumexp)
            grouped_logs["posterior"] = np.exp(
                grouped_logs["log_likelihood_prior"]
                - grouped_logs["log_likelihood_sum"]
            )
            grouped_logs["likelihood_prior"] = np.exp(
                grouped_logs["log_likelihood_prior"]
            )
            grouped_logs["likelihood_sum"] = np.exp(
                grouped_logs["log_likelihood_sum"]
            )

        pp_data = grouped_logs[
            [
                "feature_id",
                "hypothesis",
                "likelihood_prior",
                "likelihood_sum",
                "posterior",
            ]
        ]
        return pp_data.fillna(value=0)

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
    use_log_space_bm=False,
    bm_evidence_epsilon=0.0,
):
    """
    Conducts precursor-level inference.

    Args:
        data (pd.DataFrame): Input data containing precursor-level information.
        ipf_ms1_scoring (bool): Whether to use MS1-level scoring.
        ipf_ms2_scoring (bool): Whether to use MS2-level scoring.
        ipf_max_precursor_pep (float): Maximum PEP threshold for precursors.
        ipf_max_precursor_peakgroup_pep (float): Maximum PEP threshold for peak groups.
        use_log_space_bm (bool): Whether to compute Bayesian posteriors in log-space.
        bm_evidence_epsilon (float): Optional clipping epsilon applied to BM evidence.

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
        prec_pp_data = apply_bm(
            precursor_data_bm,
            use_log_space=use_log_space_bm,
            evidence_epsilon=bm_evidence_epsilon,
        )
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
    ipf_grouped_fdr_strategy,
    propagate_signal_across_runs,
    across_run_confidence_threshold,
    grouping_metrics=None,
):
    """
    Conducts peptidoform-level inference.

    Args:
        transition_table (pd.DataFrame): Input data containing transition-level information.
        precursor_data (pd.DataFrame): Precursor-level probabilities.
        ipf_grouped_fdr (bool): Whether to use grouped FDR estimation.
        ipf_grouped_fdr_strategy (str): Grouping strategy when grouped FDR is enabled.
        propagate_signal_across_runs (bool): Whether to propagate signal across runs.
        across_run_confidence_threshold (float): Confidence threshold for propagation.
        grouping_metrics (pd.DataFrame | None): Optional per-feature/per-hypothesis
            metrics required by grouped FDR strategies beyond ``num_peptidoforms``.

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
    pf_pp_data = pf_pp_data.merge(
        transition_data_bm[["feature_id", "num_peptidoforms"]].drop_duplicates(),
        on=["feature_id"],
        how="left",
    )

    # compute model-based FDR
    pf_pp_data["qvalue"] = compute_ipf_qvalues(
        pf_pp_data,
        grouped_fdr=ipf_grouped_fdr,
        grouped_fdr_strategy=ipf_grouped_fdr_strategy,
        grouping_metrics=grouping_metrics,
    )

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

    transition_score_metrics = None
    need_transition_score_metrics = (
        config.ipf_min_supporting_transitions > 0
        or config.ipf_min_peakgroup_intensity > 0
        or (
            config.ipf_grouped_fdr
            and config.ipf_grouped_fdr_strategy == "support_phospho_loss"
        )
    )
    if need_transition_score_metrics:
        transition_score_metrics = prepare_post_ipf_filter_metrics(
            peptidoform_table,
            precursor_table,
            propagate_signal_across_runs=config.propagate_signal_across_runs,
            across_run_confidence_threshold=config.across_run_confidence_threshold,
        )

    peptidoform_data = peptidoform_inference(
        peptidoform_table,
        precursor_data,
        config.ipf_grouped_fdr,
        config.ipf_grouped_fdr_strategy,
        config.propagate_signal_across_runs,
        config.across_run_confidence_threshold,
        grouping_metrics=transition_score_metrics,
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
    peptidoform_data = apply_post_ipf_filters(
        peptidoform_data,
        transition_score_metrics,
        min_supporting_transitions=config.ipf_min_supporting_transitions,
        min_peakgroup_intensity=config.ipf_min_peakgroup_intensity,
    )

    writer = WriterDispatcher.get_writer(config)
    writer.save_results(result=peptidoform_data)
