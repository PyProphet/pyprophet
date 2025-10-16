#!/usr/bin/env python
"""
Quick comparison script for HistGradientBoosting vs XGBoost classifiers.

This script demonstrates how to use both classifiers and compare their performance
on the same dataset.

Usage:
    python compare_classifiers.py --in test_data.osw --level ms2
"""

import argparse
import time
from pathlib import Path

import pandas as pd


def run_comparison(infile: str, level: str = "ms2"):
    """
    Run scoring with both XGBoost and HistGradientBoosting and compare results.
    
    Args:
        infile: Path to input OSW/parquet file
        level: Scoring level (ms1, ms2, ms1ms2, transition)
    """
    from pyprophet._config import RunnerIOConfig
    from pyprophet.scoring.runner import PyProphetLearner
    
    results = {}
    
    for classifier_name in ["XGBoost", "HistGradientBoosting"]:
        print(f"\n{'='*80}")
        print(f"Running with {classifier_name} classifier")
        print(f"{'='*80}\n")
        
        # Create config
        config = RunnerIOConfig.from_cli_args(
            infile=infile,
            outfile=infile.replace(".osw", f"_{classifier_name.lower()}.osw"),
            subsample_ratio=1.0,
            level=level,
            context="score_learn",
            classifier=classifier_name,
            autotune=False,
            xeval_fraction=0.5,
            xeval_num_iter=10,
            ss_initial_fdr=0.15,
            ss_iteration_fdr=0.05,
            ss_num_iter=10,
            ss_main_score="auto",
            ss_score_filter="",
            ss_scale_features=False,
            group_id="group_id",
            parametric=False,
            pfdr=False,
            pi0_lambda=(0.1, 0.5, 0.05),
            pi0_method="bootstrap",
            pi0_smooth_df=3,
            pi0_smooth_log_pi0=False,
            lfdr_truncate=True,
            lfdr_monotone=True,
            lfdr_transformation="probit",
            lfdr_adj=1.5,
            lfdr_eps=1e-8,
            ipf_max_peakgroup_rank=1,
            ipf_max_peakgroup_pep=0.7,
            ipf_max_transition_isotope_overlap=0.5,
            ipf_min_transition_sn=0.0,
            add_alignment_features=False,
            glyco=False,
            density_estimator="gmm",
            grid_size=256,
            tric_chromprob=False,
            threads=1,
            test=True,
            color_palette="normal",
            main_score_selection_report=False,
        )
        
        # Run scoring
        start_time = time.time()
        learner = PyProphetLearner(config)
        learner.run()
        elapsed_time = time.time() - start_time
        
        # Store results
        results[classifier_name] = {
            "time": elapsed_time,
            "config": config,
        }
        
        print(f"\n{classifier_name} completed in {elapsed_time:.2f} seconds")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print("Runtime:")
    for name, res in results.items():
        print(f"  {name:25s}: {res['time']:.2f} seconds")
    
    speedup = results["XGBoost"]["time"] / results["HistGradientBoosting"]["time"]
    print(f"\nSpeedup: {speedup:.2f}x")
    
    print("\nTo compare FDR and q-values, check the output files:")
    for name in results.keys():
        print(f"  - {results[name]['config'].outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare XGBoost and HistGradientBoosting classifiers"
    )
    parser.add_argument(
        "--in",
        dest="infile",
        required=True,
        help="Input OSW/parquet file",
    )
    parser.add_argument(
        "--level",
        default="ms2",
        choices=["ms1", "ms2", "ms1ms2", "transition"],
        help="Scoring level",
    )
    
    args = parser.parse_args()
    run_comparison(args.infile, args.level)
