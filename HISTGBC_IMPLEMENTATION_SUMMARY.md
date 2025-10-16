# HistGradientBoostingClassifier Implementation Summary

## Overview
This PR adds support for scikit-learn's `HistGradientBoostingClassifier` as an alternative to XGBoost in PyProphet. This allows users to optionally replace the XGBoost dependency with a native sklearn classifier that offers similar performance.

## Changes Made

### 1. Core Classifier Implementation (`pyprophet/scoring/classifiers.py`)
- **Completed `HistGBCLearner` class** with full implementation:
  - `__init__`: Initialize with autotune, parameters dict, and thread count
  - `learn()`: Train the HistGradientBoosting model with early stopping
  - `score()`: Score peaks using decision_function for XGBoost compatibility
  - `get_parameters()`: Return the trained classifier
  - `set_parameters()`: Set classifier and extract feature importances
  - `tune()`: Hyperparameter tuning via RandomizedSearchCV
- **Feature importance** stored in XGBoost-compatible format (`{f0: value, f1: value, ...}`)
- **Default parameters**: max_iter=100, early_stopping=True, validation_fraction=0.1

### 2. CLI Integration (`pyprophet/cli/score.py`)
- Added `"HistGradientBoosting"` to classifier choices
- Updated help text to mention HistGradientBoosting
- Updated autotune help text to include HistGradientBoosting

### 3. Configuration (`pyprophet/_config.py`)
- Updated `RunnerConfig.classifier` Literal type to include `"HistGradientBoosting"`
- Updated docstrings to mention HistGradientBoosting
- Renamed `xgb_params` comment to be generic for both XGBoost and HistGradientBoosting
- Updated conditional parameter display in `__str__` method

### 4. PyProphet Runner (`pyprophet/scoring/pyprophet.py`)
- **Imported `HistGBCLearner`** from classifiers module
- **Instantiate HistGBCLearner** when classifier="HistGradientBoosting"
- **apply_weights()**: Handle HistGradientBoosting like XGBoost for weight loading
- **_apply_weights_on_exp()**: Use set_learner for HistGradientBoosting
- **_build_result()**: Importance logging works via existing else clause (same as XGBoost)

### 5. Weight Application (`pyprophet/scoring/runner.py`)
- **PyProphetRunner.run()**: Return trained model path for HistGradientBoosting
- **PyProphetWeightApplier.__init__()**: Load pickled models for HistGradientBoosting
- Updated error messages to mention HistGradientBoosting

## Usage

Users can now train and score with HistGradientBoosting:

```bash
# Basic usage
pyprophet score --in data.osw --classifier=HistGradientBoosting

# With hyperparameter tuning
pyprophet score --in data.osw --classifier=HistGradientBoosting --autotune

# Apply pre-trained weights
pyprophet score --in data.osw --classifier=HistGradientBoosting --apply_weights model.bin
```

## Benefits

1. **Removes XGBoost dependency**: Users can use HistGradientBoosting and avoid XGBoost installation
2. **Native sklearn integration**: Better compatibility with existing sklearn pipelines
3. **Similar performance**: HistGradientBoosting uses histogram-based algorithm similar to XGBoost
4. **Consistent API**: Implemented to match XGBoost learner interface exactly

## Testing Recommendations

To verify the implementation and compare performance:

```bash
# Run existing XGBoost tests with HistGradientBoosting
pytest tests/test_pyprophet_score.py -k test_score_xgboost -v

# Compare performance on test data
pyprophet score --in test_data.osw --classifier=XGBoost --out xgb_results.osw
pyprophet score --in test_data.osw --classifier=HistGradientBoosting --out hgb_results.osw

# Compare FDR, q-values, and runtime between the two
```

## Next Steps

1. ✅ Complete HistGBCLearner implementation
2. ✅ Wire into CLI and configuration
3. ✅ Wire into PyProphet runner and weight handling
4. ✅ Syntax validation (all files pass)
5. ⏳ Performance comparison testing (requires runtime environment)
6. ⏳ Update documentation (README.md)
7. ⏳ Consider making HistGradientBoosting the default if performance is equivalent

## Files Modified

- `pyprophet/scoring/classifiers.py` (+101 lines, -24 lines)
- `pyprophet/cli/score.py` (+6 lines, -6 lines)
- `pyprophet/_config.py` (+14 lines, -14 lines)
- `pyprophet/scoring/pyprophet.py` (+12 lines, -12 lines)
- `pyprophet/scoring/runner.py` (+8 lines, -8 lines)

**Total**: +141 insertions, -64 deletions

## Commit Message

```
Complete HistGradientBoosting classifier implementation

- Add complete HistGBCLearner class with all required methods
- Wire HistGradientBoosting into CLI as a classifier choice
- Update RunnerConfig to support HistGradientBoosting classifier
- Update PyProphet to instantiate and handle HistGBCLearner
- Update runner.py to handle HistGradientBoosting in weight loading/saving
- Feature importances stored in XGBoost-compatible format
- Hyperparameter tuning support via RandomizedSearchCV
```
