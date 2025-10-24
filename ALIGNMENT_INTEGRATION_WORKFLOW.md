# SCORE_ALIGNMENT Integration Workflow

## Overview

This diagram illustrates how the SCORE_ALIGNMENT integration works to recover peaks with weak MS2 signals but good alignment scores.

## High-Level Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PyProphet Export Command                          │
│  pyprophet export tsv --in data.osw --out results.tsv               │
│  (use_alignment=True by default)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    1. Configuration Check                            │
│  • use_alignment = True (default)                                   │
│  • max_alignment_pep = 0.7 (default)                                │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    2. Auto-Detection Phase                           │
│                                                                      │
│  OSW Files:                                                          │
│  ├─ Check FEATURE_MS2_ALIGNMENT table exists?                       │
│  └─ Check SCORE_ALIGNMENT table exists?                             │
│                                                                      │
│  Parquet Files:                                                      │
│  └─ Check for {basename}_feature_alignment.parquet?                 │
│                                                                      │
│  Split Parquet Files:                                                │
│  └─ Check for {infile}/feature_alignment.parquet?                   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
                    ┌─────────────┴─────────────┐
                    │                           │
          ┌─────────▼──────────┐    ┌──────────▼─────────┐
          │  Alignment Present │    │ Alignment Missing  │
          │   use_alignment=T  │    │  use_alignment=T   │
          └─────────┬──────────┘    └──────────┬─────────┘
                    │                           │
                    │                           ↓
                    │              ┌────────────────────────┐
                    │              │ Standard Export Only   │
                    │              │ (no alignment used)    │
                    │              └────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    3. Data Reading Phase                             │
│                                                                      │
│  Step A: Fetch Base Features (MS2 QVALUE filter)                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ SELECT * FROM FEATURES                                        │ │
│  │ WHERE SCORE_MS2.QVALUE < max_rs_peakgroup_qvalue (e.g., 0.05)│ │
│  │ → Base Features (passed MS2 threshold)                        │ │
│  │ → Mark with from_alignment=0                                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Step B: Fetch Aligned Features (Alignment PEP filter)              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ SELECT ALIGNED_FEATURE_ID FROM FEATURE_MS2_ALIGNMENT          │ │
│  │ JOIN SCORE_ALIGNMENT                                          │ │
│  │ WHERE LABEL = 1 (target)                                      │ │
│  │ AND SCORE_ALIGNMENT.PEP < max_alignment_pep (e.g., 0.7)      │ │
│  │ → Aligned Features (good alignment scores)                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    4. Feature Recovery Logic                         │
│                                                                      │
│  ┌─────────────────┐        ┌──────────────────┐                   │
│  │ Base Features   │        │ Aligned Features │                   │
│  │ (MS2 passed)    │        │ (Alignment good) │                   │
│  │ IDs: 1,2,3,4,5  │        │ IDs: 3,4,6,7,8   │                   │
│  └────────┬────────┘        └────────┬─────────┘                   │
│           │                          │                              │
│           └──────────┬───────────────┘                              │
│                      ↓                                              │
│           ┌──────────────────────┐                                  │
│           │ Find NEW features:   │                                  │
│           │ aligned - base       │                                  │
│           │ = {6, 7, 8}         │                                  │
│           └──────────┬───────────┘                                  │
│                      ↓                                              │
│           ┌──────────────────────┐                                  │
│           │ Fetch full data for  │                                  │
│           │ recovered features   │                                  │
│           │ 6, 7, 8              │                                  │
│           │ Mark: from_alignment=1│                                 │
│           │ Add: alignment_pep   │                                  │
│           │ Add: alignment_qvalue│                                  │
│           └──────────┬───────────┘                                  │
│                      ↓                                              │
│           ┌──────────────────────┐                                  │
│           │ Combine:             │                                  │
│           │ Base (1,2,3,4,5) +   │                                  │
│           │ Recovered (6,7,8)    │                                  │
│           │ = Final (1-8)        │                                  │
│           └──────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    5. Export Results                                 │
│                                                                      │
│  Final TSV/Matrix includes:                                          │
│  • Original features (from_alignment=0)                              │
│  • Recovered features (from_alignment=1, with alignment scores)      │
│  • More complete quantification with fewer missing values            │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Workflow

### A. Reader Classes (OSW, Parquet, Split Parquet)

```
┌──────────────────────────────────────────────────────────────┐
│                    Reader.__init__()                         │
│                                                              │
│  OSWReader:                                                  │
│    N/A - checks at read time                                │
│                                                              │
│  ParquetReader:                                              │
│    self._has_alignment = _check_alignment_file_exists()     │
│    • Checks: {basename}_feature_alignment.parquet           │
│                                                              │
│  SplitParquetReader:                                         │
│    self._has_alignment = _check_alignment_file_exists()     │
│    • Checks: {infile}/feature_alignment.parquet             │
└──────────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                    Reader.read()                             │
│                                                              │
│  → _read_standard_data()                                     │
│    if config.use_alignment AND alignment_present:           │
│      → _fetch_alignment_features()                           │
│      → Merge with base features                              │
└──────────────────────────────────────────────────────────────┘
```

### B. Alignment Detection Methods

```
OSW Files (.osw):
┌─────────────────────────────────────────┐
│  _check_alignment_presence(con)        │
│                                         │
│  return:                                │
│    check_sqlite_table(                 │
│      con, "FEATURE_MS2_ALIGNMENT"      │
│    ) AND                                │
│    check_sqlite_table(                 │
│      con, "SCORE_ALIGNMENT"            │
│    )                                    │
└─────────────────────────────────────────┘

Parquet Files (.parquet):
┌─────────────────────────────────────────┐
│  _check_alignment_file_exists()        │
│                                         │
│  if infile.endswith('.parquet'):       │
│    base = infile[:-8]                  │
│    alignment_file =                    │
│      f"{base}_feature_alignment.parquet"│
│    return os.path.exists(alignment_file)│
└─────────────────────────────────────────┘

Split Parquet Files (directory with .oswpq):
┌─────────────────────────────────────────┐
│  _check_alignment_file_exists()        │
│                                         │
│  if os.path.isdir(infile):             │
│    alignment_file = os.path.join(      │
│      infile, "feature_alignment.parquet"│
│    )                                    │
│    return os.path.exists(alignment_file)│
└─────────────────────────────────────────┘
```

### C. Feature Recovery Decision Tree

```
                    Start Export
                         │
                         ↓
              ┌──────────────────────┐
              │ use_alignment=True?  │
              └──────────┬───────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
          YES                         NO
           │                           │
           ↓                           ↓
    ┌──────────────┐          ┌──────────────┐
    │ Alignment    │          │ Standard     │
    │ data exists? │          │ Export Only  │
    └──────┬───────┘          └──────────────┘
           │
     ┌─────┴─────┐
     │           │
    YES         NO
     │           │
     ↓           ↓
┌─────────┐  ┌─────────┐
│ Use     │  │Standard │
│Alignment│  │Export   │
└─────────┘  └─────────┘
     │           │
     └─────┬─────┘
           ↓
    Export Results
```

## Example Scenario

### Before Alignment Integration:

```
Run 1: Feature detected with MS2 QVALUE = 0.02 ✓ (exported)
Run 2: Feature detected with MS2 QVALUE = 0.08 ✗ (not exported - weak signal)
Run 3: Feature detected with MS2 QVALUE = 0.03 ✓ (exported)

Result: Missing quantification in Run 2
```

### After Alignment Integration:

```
Run 1: Feature detected with MS2 QVALUE = 0.02 ✓ (exported, from_alignment=0)
Run 2: Feature detected with MS2 QVALUE = 0.08 ✗ (weak MS2)
       BUT: Alignment PEP = 0.4 ✓ (good alignment!)
       → Recovered via alignment (exported, from_alignment=1)
Run 3: Feature detected with MS2 QVALUE = 0.03 ✓ (exported, from_alignment=0)

Result: Complete quantification across all runs
```

## File Structure Examples

### OSW Format:
```
data.osw (SQLite database)
├─ FEATURE_MS2_ALIGNMENT table
└─ SCORE_ALIGNMENT table
```

### Parquet Format:
```
data.parquet                    ← Main file
data_feature_alignment.parquet  ← Alignment file
```

### Split Parquet Format:
```
experiment/
├─ run1.oswpq/
│  ├─ precursors_features.parquet
│  └─ transition_features.parquet
├─ run2.oswpq/
│  ├─ precursors_features.parquet
│  └─ transition_features.parquet
└─ feature_alignment.parquet    ← Alignment file (parent level)
```

## Key Benefits

1. **Increased Coverage**: Recovers peaks with weak MS2 but good alignment
2. **Better Quantification**: Fewer missing values in matrices
3. **Quality Control**: Uses alignment PEP/QVALUE thresholds
4. **Backwards Compatible**: Disabled by default via auto-detection
5. **Transparent**: Features marked with `from_alignment` flag

## Configuration Options

```bash
# Use default (enabled with auto-detection)
pyprophet export tsv --in data.osw --out results.tsv

# Customize threshold
pyprophet export tsv --in data.osw --out results.tsv \
  --max_alignment_pep 0.5

# Explicitly disable
pyprophet export tsv --in data.osw --out results.tsv \
  --no-use_alignment
```

## Output Columns

Recovered features include additional columns:

- `from_alignment`: 0 (base) or 1 (recovered)
- `alignment_pep`: Alignment posterior error probability
- `alignment_qvalue`: Alignment q-value

These allow users to:
- Identify which features were recovered
- Assess alignment quality
- Filter or analyze separately if needed
