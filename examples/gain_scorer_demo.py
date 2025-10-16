#!/usr/bin/env python
"""
Example script demonstrating the gain-like permutation importance scorer.

This script shows:
1. How to train a HistGBCLearner
2. How feature importances are computed using the gain-like metric
3. How to compare with XGBoost's gain metric

Usage:
    python examples/gain_scorer_demo.py
"""

import numpy as np
from sklearn.datasets import make_classification


def create_mock_experiment(X_data):
    """Create a mock Experiment object for testing"""
    class MockExperiment:
        def __init__(self, X):
            self._X = X
        
        def get_feature_matrix(self, use_main_score=True):
            return self._X
    
    return MockExperiment(X_data)


def main():
    print("=" * 80)
    print("Gain-Like Permutation Importance Scorer Demo")
    print("=" * 80)
    print()
    
    # Import classifiers
    from pyprophet.scoring.classifiers import HistGBCLearner, XGBLearner
    
    # Create synthetic dataset
    print("1. Creating synthetic classification dataset...")
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=42
    )
    
    # Split into decoy (y=0) and target (y=1)
    decoy_mask = y == 0
    target_mask = y == 1
    
    decoy_exp = create_mock_experiment(X[decoy_mask])
    target_exp = create_mock_experiment(X[target_mask])
    
    print(f"   - Total samples: {len(X)}")
    print(f"   - Decoy samples: {sum(decoy_mask)}")
    print(f"   - Target samples: {sum(target_mask)}")
    print(f"   - Features: {X.shape[1]}")
    print()
    
    # Train HistGradientBoosting with gain-like scorer
    print("2. Training HistGradientBoosting classifier...")
    print("   - Using gain-like permutation importance scorer")
    hgb_learner = HistGBCLearner()
    hgb_learner.learn(decoy_exp, target_exp)
    print("   ✓ Training complete")
    print()
    
    # Train XGBoost for comparison
    print("3. Training XGBoost classifier for comparison...")
    print("   - Using default 'gain' importance metric")
    xgb_params = {
        "max_depth": 6,
        "learning_rate": 0.3,
        "objective": "binary:logitraw",
        "eval_metric": "auc",
    }
    xgb_learner = XGBLearner(autotune=False, xgb_params=xgb_params, threads=1)
    xgb_learner.learn(decoy_exp, target_exp)
    print("   ✓ Training complete")
    print()
    
    # Compare feature importances
    print("4. Comparing Feature Importances:")
    print()
    print(f"{'Feature':<10} {'HistGBC (Gain-Like)':<25} {'XGBoost (Gain)':<20}")
    print("-" * 80)
    
    for i in range(10):
        feat_name = f"f{i}"
        hgb_imp = hgb_learner.importance.get(feat_name, 0.0)
        xgb_imp = xgb_learner.importance.get(feat_name, 0.0)
        print(f"{feat_name:<10} {hgb_imp:>20.4f}     {xgb_imp:>15.4f}")
    
    print()
    
    # Rank features by importance
    print("5. Top 5 Features by Importance:")
    print()
    
    # Rank HistGBC features
    hgb_ranked = sorted(
        hgb_learner.importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Rank XGBoost features
    xgb_ranked = sorted(
        xgb_learner.importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    print("   HistGradientBoosting (Gain-Like):")
    for feat, imp in hgb_ranked:
        print(f"      {feat}: {imp:.4f}")
    
    print()
    print("   XGBoost (Gain):")
    for feat, imp in xgb_ranked:
        print(f"      {feat}: {imp:.4f}")
    
    print()
    
    # Summary
    print("6. Summary:")
    print()
    print("   - Both classifiers use gain-based importance metrics")
    print("   - HistGBC uses permutation importance with log-loss scoring")
    print("   - XGBoost uses average gain per split")
    print("   - Both metrics measure feature contribution to model performance")
    print("   - Rankings should be similar (but not identical)")
    print()
    
    # Compare rankings
    hgb_top3 = {feat for feat, _ in hgb_ranked[:3]}
    xgb_top3 = {feat for feat, _ in xgb_ranked[:3]}
    overlap = len(hgb_top3 & xgb_top3)
    
    print(f"   - Top 3 features overlap: {overlap}/3")
    print()
    
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
