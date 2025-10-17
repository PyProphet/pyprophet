# HistGradientBoosting Classifier Test Additions

## Summary
Added comprehensive test coverage for the HistGradientBoosting classifier in `tests/test_pyprophet_score.py`.

## Changes Made

### 1. Updated Test Strategies
All test strategies now support the HistGradientBoosting classifier via new parameters:
- `histgbc=True` - Enable HistGradientBoosting classifier
- `histgbc_tune=True` - Enable autotuning for HistGradientBoosting

Modified strategies:
- `OSWTestStrategy.execute()` - Added histgbc and histgbc_tune support
- `ParquetTestStrategy.execute()` - Added histgbc and histgbc_tune support
- `SplitParquetTestStrategy.execute()` - Added histgbc and histgbc_tune support
- `MultiSplitParquetTestStrategy.execute()` - Added histgbc and histgbc_tune support

### 2. New Test Cases

#### OSW Format Tests
- **`test_osw_histgbc()`** - Basic HistGradientBoosting classifier test with OSW files
  - Tests MS2 level scoring
  - Uses PFDR and pi0_lambda=0 0 0
  
- **`test_osw_histgbc_tune()`** - HistGradientBoosting with autotuning
  - Tests MS2 level scoring with hyperparameter optimization
  - Uses --autotune flag
  
- **`test_osw_histgbc_multilevel()`** - Multi-level test for HistGBC
  - Tests MS1, MS2, and transition levels
  - Uses PFDR and pi0_lambda=0 0 0

#### Parquet Format Tests
- **`test_parquet_histgbc()`** - HistGradientBoosting with Parquet files
  - Tests all levels (MS1, MS2, transition)
  - Uses PFDR and pi0_lambda=0 0 0
  
- **`test_parquet_histgbc_tune()`** - HistGBC with autotuning on Parquet
  - Tests with hyperparameter optimization
  - Full multi-level scoring

#### Split Parquet Format Tests
- **`test_split_parquet_histgbc()`** - HistGBC with split Parquet format
  - Tests directory-based parquet structure
  - Multi-level scoring with PFDR

#### Multi-Split Parquet Format Tests
- **`test_multi_split_parquet_histgbc()`** - HistGBC with multi-split Parquet
  - Tests multiple directory parquet structure
  - Complete workflow validation

## Test Execution

### Run All HistGBC Tests
```bash
pytest tests/test_pyprophet_score.py -k histgbc -v
```

### Run Individual Tests
```bash
# OSW tests
pytest tests/test_pyprophet_score.py::test_osw_histgbc -v
pytest tests/test_pyprophet_score.py::test_osw_histgbc_tune -v
pytest tests/test_pyprophet_score.py::test_osw_histgbc_multilevel -v

# Parquet tests
pytest tests/test_pyprophet_score.py::test_parquet_histgbc -v
pytest tests/test_pyprophet_score.py::test_parquet_histgbc_tune -v

# Split Parquet tests
pytest tests/test_pyprophet_score.py::test_split_parquet_histgbc -v

# Multi-Split Parquet tests
pytest tests/test_pyprophet_score.py::test_multi_split_parquet_histgbc -v
```

## Coverage

The new tests provide comprehensive coverage for:

1. **Classifier Functionality**
   - Basic HistGradientBoosting operation
   - Feature importance computation (permutation-based)
   - Hyperparameter autotuning

2. **File Formats**
   - OSW (SQLite) format
   - Parquet (single file) format
   - Split Parquet (directory) format
   - Multi-Split Parquet (nested directories) format

3. **Scoring Levels**
   - MS1 level
   - MS2 level
   - Transition level
   - Multi-level workflows

4. **Configuration Options**
   - PFDR (peptide-level FDR)
   - Custom pi0_lambda values
   - Autotuning with RandomizedSearchCV
   - Score filtering

## Implementation Details

### HistGradientBoosting Classifier Features
- Uses sklearn's HistGradientBoostingClassifier
- Permutation importance with custom gain-like scorer (negative log loss)
- Stratified subsampling (max 2000 samples) for performance
- Importance values scaled 100× to match XGBoost magnitude
- Thread management support via OMP_NUM_THREADS

### Key Parameters Tested
- `--classifier=HistGradientBoosting` - Select HistGBC classifier
- `--autotune` - Enable hyperparameter optimization
- `--pfdr` - Enable peptide-level FDR
- `--pi0_lambda` - Control pi0 estimation
- `--level` - Specify scoring level (ms1/ms2/transition)

## Related Files

### Modified Files
- `tests/test_pyprophet_score.py` - Added HistGBC test cases and strategy updates

### Implementation Files (from PR #144)
- `pyprophet/scoring/classifiers.py` - HistGBCLearner class
- `pyprophet/cli/score.py` - CLI integration
- `pyprophet/scoring/runner.py` - Thread management
- `pyprophet/scoring/semi_supervised.py` - Semi-supervised learning integration

## Notes

1. **Performance**: HistGradientBoosting with permutation importance is slower than XGBoost (~12× slower due to permutation overhead)

2. **Thread Management**: When using HistGBC, consider setting `OMP_NUM_THREADS` to avoid CPU over-subscription:
   ```bash
   OMP_NUM_THREADS=6 pyprophet score --in input.osw --classifier HistGradientBoosting --threads 3
   ```

3. **Feature Importances**: HistGBC uses permutation importance with a gain-like scorer (negative log loss) to approximate XGBoost's gain metric

4. **Compatibility**: All existing test infrastructure works seamlessly with HistGBC through the strategy pattern implementation
