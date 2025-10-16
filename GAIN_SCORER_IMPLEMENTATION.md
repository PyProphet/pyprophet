# Gain-Like Permutation Importance Implementation

## Overview

This document describes the implementation of a gain-like permutation importance scorer for the `HistGBCLearner` class in PyProphet. This enhancement aligns the feature importance metric used by HistGradientBoosting with XGBoost's default "gain" metric.

## Problem Statement

Previously, `HistGBCLearner` would use scikit-learn's native `feature_importances_` attribute when available. However, this attribute uses a **split-based** importance metric (similar to XGBoost's "weight" or "cover" importance), which counts how many times a feature is used to split data. This is fundamentally different from XGBoost's default **"gain"** metric, which measures the average improvement in the loss function when a feature is used for splitting.

## Solution

The implementation now **always computes permutation importance** using a custom gain-like scorer, even though `HistGradientBoostingClassifier` has a native `feature_importances_` attribute. This ensures consistency with XGBoost's gain metric.

### Key Components

1. **Custom Loss-Gain Scorer** (lines 427-436 in `classifiers.py`):
   ```python
   def loss_gain_score(y_true, y_pred_proba, **kwargs):
       """Score based on negative log loss - higher is better, like XGBoost gain."""
       return -log_loss(y_true, y_pred_proba, labels=[0, 1])
   
   gain_scorer = make_scorer(loss_gain_score, greater_is_better=True, needs_proba=True)
   ```

2. **Permutation Importance Computation** (lines 437-444):
   - Uses stratified subsampling (up to 2000 samples) for computational efficiency
   - Computes permutation importance with 5 repeats
   - Uses the gain-like scorer to measure feature importance

3. **Post-Processing** (lines 445-451):
   - Clamps negative values to zero (permutation importance can be slightly negative due to noise)
   - Scales importances by 100x to match XGBoost's gain magnitude (typically 0-100 range)
   - Formats as dictionary with keys `f0`, `f1`, etc. (XGBoost-compatible format)

4. **Persistence** (lines 452-453):
   - Stores importance on the classifier object itself using `_pyprophet_importance` attribute
   - Ensures importance survives serialization/deserialization through `set_parameters()`

## Why Gain-Like Metric?

The gain metric has several advantages:

1. **Sensitivity**: Measures actual contribution to model performance, not just usage frequency
2. **Interpretability**: Shows which features most reduce prediction error
3. **Consistency**: Aligns with XGBoost's default metric, making results comparable
4. **Robustness**: Less affected by model structure (e.g., feature usage in early vs. late splits)

## Implementation Details

### File: `pyprophet/scoring/classifiers.py`

**Modified**: `HistGBCLearner.learn()` method (lines 406-455)
- Removed conditional that would use sklearn's native `feature_importances_`
- Now always computes permutation importance with gain-like scorer
- Added detailed comments explaining the rationale

**Modified**: `HistGBCLearner.set_parameters()` method (lines 478-481)
- Updated comments to reflect that no importance info is a rare edge case

### File: `tests/test_gain_scorer.py`

**Added**: Comprehensive unit tests
- `test_histgbc_gain_scorer_computation()`: Validates importance computation
- `test_histgbc_scorer_produces_reasonable_scores()`: Tests feature ranking quality
- `test_histgbc_importance_stored_on_classifier()`: Validates persistence
- `test_histgbc_importance_format_matches_xgboost()`: Ensures XGBoost compatibility

## Usage

The gain-like importance is computed automatically during training:

```python
from pyprophet.scoring.classifiers import HistGBCLearner

learner = HistGBCLearner()
learner.learn(decoy_peaks, target_peaks)

# Access feature importances (gain-like metric)
print(learner.importance)
# Output: {'f0': 12.34, 'f1': 8.76, 'f2': 5.43, ...}
```

## Performance Considerations

- **Subsampling**: Limited to 2000 samples for speed while maintaining statistical reliability
- **Stratification**: Maintains class balance in subsample
- **Parallelization**: Uses all available cores (`n_jobs=-1`)
- **Repeats**: 5 repeats balance stability vs. computation time

## Testing

Run the unit tests with:

```bash
pytest tests/test_gain_scorer.py -v
```

Or run all scoring tests:

```bash
pytest tests/test_pyprophet_score.py -v
```

## Comparison with XGBoost

| Aspect | XGBoost Gain | HistGBC Gain-Like |
|--------|-------------|-------------------|
| **Metric** | Average gain per split | Permutation importance with log-loss |
| **Computation** | During training | Post-training |
| **Scale** | 0-100+ | Scaled to 0-100+ |
| **Format** | `{'f0': val, ...}` | `{'f0': val, ...}` |
| **Interpretation** | Loss reduction per feature | Loss reduction per feature |

## Future Enhancements

Possible improvements:
1. Make number of repeats configurable
2. Allow custom scoring metrics
3. Add support for feature interaction importance
4. Cache importances for faster re-use

## References

- scikit-learn permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- XGBoost feature importance: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
