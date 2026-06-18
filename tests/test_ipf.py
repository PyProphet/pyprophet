# encoding: utf-8
from __future__ import print_function

import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal

from pyprophet.ipf import (
    apply_bm,
    apply_post_ipf_filters,
    compute_model_fdr,
    compute_post_ipf_filter_metrics,
    prepare_precursor_bm,
    prepare_transition_bm,
)


pd.options.display.expand_frame_repr = False
pd.options.display.precision = 4
pd.options.display.max_columns = None


def test_0():
    test_in = pd.DataFrame(
        {
            "feature_id": [0],
            "ms1_precursor_pep": [0.4],
            "ms2_peakgroup_pep": [0.2],
            "ms2_precursor_pep": [0.5],
        }
    )
    test_ref = pd.DataFrame(
        data={
            "evidence": [0.6, 0.4, 0.5, 0.5],
            "feature_id": [0, 0, 0, 0],
            "hypothesis": [True, False, True, False],
            "prior": [0.8, 0.2, 0.8, 0.2],
        },
        index=None,
    )

    test_out = prepare_precursor_bm(test_in)

    assert_frame_equal(
        test_out[["feature_id", "prior", "evidence", "hypothesis"]].reset_index(
            drop=True
        ),
        test_ref[["feature_id", "prior", "evidence", "hypothesis"]].reset_index(
            drop=True
        ),
    )


def test_1():
    tin = np.array([0.5, 0.4, 0.2, 0.1, 0.001, 0.9, 0.7])
    tref = np.array(
        [0.2402000, 0.1752500, 0.1003333, 0.0505000, 0.0010000, 0.4001429, 0.3168333]
    )

    tout = compute_model_fdr(tin)

    assert_almost_equal(tout, tref)


def test_2():
    tin = np.array(
        [
            0.5,
            0.4,
            0.2,
            0.1,
            0.001,
            0.9,
            0.7,
            0.001,
            0,
            0.3,
            0.12,
            0.4,
            0.1111,
            0.2222,
            0.88887,
            1.0,
            0.0000000000001,
        ]
    )
    tref = np.array(
        [
            1.81176923e-01,
            1.54608333e-01,
            6.66375000e-02,
            2.04000000e-02,
            5.00000000e-04,
            3.02760625e-01,
            2.18235714e-01,
            5.00000000e-04,
            0.00000000e00,
            1.05530000e-01,
            4.75857143e-02,
            1.54608333e-01,
            3.55166667e-02,
            8.39222222e-02,
            2.62944667e-01,
            3.43774706e-01,
            5.00155473e-14,
        ]
    )

    tout = compute_model_fdr(tin)

    assert_almost_equal(tout, tref)


def test_3():
    test_in = {
        "feature_id": "id0",
        "evidence": [0.1, 0.1, 0.9, 0.9, 0.2, 0.8, 0.8, 0.2, 0.4, 0.4, 0.6, 0.4],
        "hypothesis": [1, 2, 3, -1, 1, 2, 3, -1, 1, 2, 3, -1],
        "prior": [0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4],
    }
    tin = pd.DataFrame(data=test_in, index=None)
    test_ref = {
        "feature_id": ["id0", "id0", "id0", "id0"],
        "hypothesis": [
            -1,
            1,
            2,
            3,
        ],
        "likelihood_prior": [0.0288, 0.0016, 0.0064, 0.0864],
        "likelihood_sum": [0.1232, 0.1232, 0.1232, 0.1232],
        "posterior": [0.233766, 0.012987, 0.051948, 0.701299],
    }
    tref = pd.DataFrame(data=test_ref, index=None)

    tout = apply_bm(tin)

    print(tref)
    print(tout)

    assert_frame_equal(
        tout[
            [
                "feature_id",
                "hypothesis",
                "likelihood_prior",
                "likelihood_sum",
                "posterior",
            ]
        ],
        tref[
            [
                "feature_id",
                "hypothesis",
                "likelihood_prior",
                "likelihood_sum",
                "posterior",
            ]
        ],
    )


def test_apply_bm_log_space_matches_raw():
    tin = pd.DataFrame(
        {
            "feature_id": ["id0"] * 6,
            "evidence": [0.9, 0.8, 0.7, 0.1, 0.2, 0.3],
            "hypothesis": [1, 1, 1, 2, 2, 2],
            "prior": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )

    tout_raw = apply_bm(tin)
    tout_log = apply_bm(tin, use_log_space=True)

    assert_frame_equal(
        tout_raw[
            [
                "feature_id",
                "hypothesis",
                "likelihood_prior",
                "likelihood_sum",
                "posterior",
            ]
        ].sort_values(["feature_id", "hypothesis"]).reset_index(drop=True),
        tout_log[
            [
                "feature_id",
                "hypothesis",
                "likelihood_prior",
                "likelihood_sum",
                "posterior",
            ]
        ].sort_values(["feature_id", "hypothesis"]).reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_apply_bm_log_space_avoids_raw_underflow():
    n = 1000
    tin = pd.DataFrame(
        {
            "feature_id": ["id0"] * (2 * n),
            "evidence": [0.1] * n + [0.2] * n,
            "hypothesis": [1] * n + [2] * n,
            "prior": [0.5] * (2 * n),
        }
    )

    tout_raw = apply_bm(tin)
    tout_log = apply_bm(tin, use_log_space=True)

    assert tout_raw["posterior"].sum() == 0
    assert np.isclose(tout_log["posterior"].sum(), 1.0)
    assert tout_log.loc[tout_log["hypothesis"] == 2, "posterior"].iloc[0] > 0.999999


def test_apply_bm_evidence_epsilon_clips_extreme_posteriors():
    tin = pd.DataFrame(
        {
            "feature_id": ["id0"] * 4,
            "evidence": [1.0, 1.0, 0.0, 0.0],
            "hypothesis": [1, 1, 2, 2],
            "prior": [0.5, 0.5, 0.5, 0.5],
        }
    )

    tout = apply_bm(tin, evidence_epsilon=1e-6)

    posterior_1 = tout.loc[tout["hypothesis"] == 1, "posterior"].iloc[0]
    posterior_2 = tout.loc[tout["hypothesis"] == 2, "posterior"].iloc[0]

    assert 0 < posterior_1 < 1
    assert 0 < posterior_2 < 1
    assert np.isclose(posterior_1 + posterior_2, 1.0)


def test_compute_post_ipf_filter_metrics():
    transition_table = pd.DataFrame(
        {
            "feature_id": [1, 1, 1, 2, 2, 3],
            "peptide_id": [10, 10, 11, 20, -1, 30],
            "transition_id": [100, 101, 102, 200, 201, 300],
            "bmask": [1, 0, 1, 1, 1, 1],
            "n_mapped_peptides": [1, 2, 1, 2, 2, 1],
            "has_phospho_loss": [1, 0, 0, 1, 0, 0],
            "isotope_overlap_score": [0.01, 0.05, 0.02, 0.03, 0.04, 0.01],
        }
    )
    precursor_table = pd.DataFrame(
        {
            "feature_id": [1, 2, 3],
            "feature_ms2_intensity": [1000.0, 200.0, 50.0],
        }
    )

    tout = compute_post_ipf_filter_metrics(transition_table, precursor_table)
    tref = pd.DataFrame(
        {
            "feature_id": [1, 1, 2, 3],
            "peptide_id": [10, 11, 20, 30],
            "supporting_transitions": [1, 1, 1, 1],
            "unique_supporting_transitions": [1, 1, 0, 1],
            "phospho_loss_supporting_transitions": [1, 0, 1, 0],
            "median_supporting_isotope_overlap": [0.01, 0.02, 0.03, 0.01],
            "feature_ms2_intensity": [1000.0, 1000.0, 200.0, 50.0],
        }
    )

    assert_frame_equal(
        tout.sort_values(["feature_id", "peptide_id"]).reset_index(drop=True),
        tref.sort_values(["feature_id", "peptide_id"]).reset_index(drop=True),
    )

def test_apply_post_ipf_filters():
    result = pd.DataFrame(
        {
            "FEATURE_ID": [1, 1, 2, 3, 4],
            "PEPTIDE_ID": [10, 11, 20, 30, 40],
            "PRECURSOR_PEAKGROUP_PEP": [0.1, 0.1, 0.2, 0.3, 0.4],
            "QVALUE": [0.01, 0.02, 0.03, 0.04, 0.05],
            "PEP": [0.001, 0.002, 0.003, 0.004, 0.005],
        }
    )
    filter_metrics = pd.DataFrame(
        {
            "feature_id": [1, 1, 2, 3, 4],
            "peptide_id": [10, 11, 20, 30, 40],
            "supporting_transitions": [1, 2, 2, 3, 1],
            "unique_supporting_transitions": [1, 2, 0, 3, 1],
            "phospho_loss_supporting_transitions": [1, 0, 1, 0, 0],
            "feature_ms2_intensity": [1000.0, 1000.0, 200.0, 50.0, 1000.0],
        }
    )

    tout = apply_post_ipf_filters(
        result,
        filter_metrics,
        min_supporting_transitions=1,
        min_unique_supporting_transitions=1,
        require_phospho_loss_below_support=2,
        min_peakgroup_intensity=100.0,
    )
    tref = pd.DataFrame(
        {
            "FEATURE_ID": [1, 1],
            "PEPTIDE_ID": [10, 11],
            "PRECURSOR_PEAKGROUP_PEP": [0.1, 0.1],
            "QVALUE": [0.01, 0.02],
            "PEP": [0.001, 0.002],
        }
    )

    assert_frame_equal(
        tout.sort_values(["FEATURE_ID", "PEPTIDE_ID"]).reset_index(drop=True),
        tref.sort_values(["FEATURE_ID", "PEPTIDE_ID"]).reset_index(drop=True),
    )


def test_apply_post_ipf_filters_conditional_intensity():
    result = pd.DataFrame(
        {
            "FEATURE_ID": [1, 2, 3, 4],
            "PEPTIDE_ID": [10, 20, 30, 40],
            "PRECURSOR_PEAKGROUP_PEP": [0.1, 0.1, 0.1, 0.1],
            "QVALUE": [0.01, 0.02, 0.03, 0.04],
            "PEP": [0.001, 0.002, 0.003, 0.004],
        }
    )
    filter_metrics = pd.DataFrame(
        {
            "feature_id": [1, 2, 3, 4],
            "peptide_id": [10, 20, 30, 40],
            "supporting_transitions": [1, 1, 2, 3],
            "unique_supporting_transitions": [1, 1, 2, 3],
            "phospho_loss_supporting_transitions": [0, 1, 0, 0],
            "feature_ms2_intensity": [5000.0, 5000.0, 5000.0, 5000.0],
        }
    )

    tout = apply_post_ipf_filters(
        result,
        filter_metrics,
        conditional_min_peakgroup_intensity=10000.0,
        conditional_min_peakgroup_intensity_max_supporting_transitions=2,
        conditional_min_peakgroup_intensity_no_phospho_loss_only=True,
    )
    tref = pd.DataFrame(
        {
            "FEATURE_ID": [2, 4],
            "PEPTIDE_ID": [20, 40],
            "PRECURSOR_PEAKGROUP_PEP": [0.1, 0.1],
            "QVALUE": [0.02, 0.04],
            "PEP": [0.002, 0.004],
        }
    )

    assert_frame_equal(
        tout.sort_values(["FEATURE_ID", "PEPTIDE_ID"]).reset_index(drop=True),
        tref.sort_values(["FEATURE_ID", "PEPTIDE_ID"]).reset_index(drop=True),
    )
