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
│  │ SELECT CAST(FEATURE.ID AS INTEGER) AS id,                     │ │
│  │        ... (other columns)                                     │ │
│  │ FROM FEATURES                                                  │ │
│  │ WHERE SCORE_MS2.QVALUE < max_rs_peakgroup_qvalue (e.g., 0.05)│ │
│  │ → Base Features (passed MS2 threshold)                        │ │
│  │ → Mark with from_alignment=0                                  │ │
│  │ → CAST preserves precision for large feature IDs             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Step B: Fetch Aligned Features (Alignment PEP filter)              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ SELECT DENSE_RANK() OVER (...) AS alignment_group_id,        │ │
│  │        ALIGNED_FEATURE_ID AS id,                              │ │
│  │        CAST(REFERENCE_FEATURE_ID AS INTEGER)                  │ │
│  │          AS alignment_reference_feature_id,                   │ │
│  │        REFERENCE_RT AS alignment_reference_rt                 │ │
│  │ FROM FEATURE_MS2_ALIGNMENT                                    │ │
│  │ JOIN SCORE_ALIGNMENT                                          │ │
│  │ WHERE LABEL = 1 (target)                                      │ │
│  │ AND SCORE_ALIGNMENT.PEP < max_alignment_pep (e.g., 0.7)      │ │
│  │ AND REF FEATURE passes MS2 QVALUE threshold                  │ │
│  │ → Aligned Features (good alignment scores)                    │ │
│  │ → Includes alignment_group_id and reference info              │ │
│  │ → CAST preserves precision for large feature IDs             │ │
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
│           ┌──────────────────────────────────────┐                  │
│           │ Fetch full data for recovered        │                  │
│           │ features: 6, 7, 8                    │                  │
│           │ Mark: from_alignment=1               │                  │
│           │ Add: alignment_pep                   │                  │
│           │ Add: alignment_qvalue                │                  │
│           │ Add: alignment_group_id              │                  │
│           │ Add: alignment_reference_feature_id  │                  │
│           │ Add: alignment_reference_rt          │                  │
│           └──────────┬───────────────────────────┘                  │
│                      ↓                                              │
│           ┌──────────────────────────────────────┐                  │
│           │ Assign alignment_group_id to         │                  │
│           │ reference features                   │                  │
│           │ (features pointed to by aligned IDs) │                  │
│           └──────────┬───────────────────────────┘                  │
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
- `alignment_group_id`: Group identifier linking aligned features together
- `alignment_reference_feature_id`: ID of the reference feature used for alignment
- `alignment_reference_rt`: Retention time of the reference feature

These allow users to:
- Identify which features were recovered
- Assess alignment quality
- Track which features are aligned together via `alignment_group_id`
- Find the reference feature that was used for alignment
- Filter or analyze separately if needed

## Technical Implementation Details

### Precision Preservation for Large Feature IDs

Large integer feature IDs (e.g., `5,405,272,318,039,692,409`) require special handling to prevent precision loss during database operations and pandas DataFrame creation.

#### The Problem
- Feature IDs can exceed 2^53, the maximum integer that float64 can represent precisely
- When pandas reads INTEGER columns from databases without explicit typing, it may infer float64 dtype
- This causes precision loss: `5,405,272,318,039,692,409` → `5,405,272,318,039,692,288`

#### The Solution
SQL queries use explicit CAST operations in SELECT clauses (but NOT in JOIN conditions):

```sql
-- OSW (SQLite)
SELECT CAST(FEATURE.ID AS INTEGER) AS id,
       CAST(FEATURE_MS2_ALIGNMENT.REFERENCE_FEATURE_ID AS INTEGER) AS alignment_reference_feature_id
FROM ...

-- Parquet (DuckDB)  
SELECT CAST(fa.REFERENCE_FEATURE_ID AS BIGINT) AS REFERENCE_FEATURE_ID
FROM ...
```

**Key Design Principles:**
1. **CAST in SELECT**: Ensures pandas reads columns as integers, preserving precision
2. **No CAST in JOIN**: Database can use indexes for fast lookups (~16 seconds vs 50 minutes)
3. **Post-query conversion**: After reading, convert to pandas Int64 dtype for nullable integer support

```python
# After reading from database
if "alignment_reference_feature_id" in df.columns:
    df["alignment_reference_feature_id"] = df["alignment_reference_feature_id"].astype("Int64")
if "id" in data.columns:
    data["id"] = data["id"].astype("Int64")
```

### Alignment Group ID Assignment

The `alignment_group_id` is computed using `DENSE_RANK()` to assign a unique identifier to each alignment group:

```sql
SELECT DENSE_RANK() OVER (ORDER BY PRECURSOR_ID, ALIGNMENT_ID) AS alignment_group_id,
       ALIGNED_FEATURE_ID AS id,
       REFERENCE_FEATURE_ID AS alignment_reference_feature_id
FROM FEATURE_MS2_ALIGNMENT
```

#### Assigning Group IDs to Reference Features

Reference features (those that aligned features point to) also need to receive their `alignment_group_id`. This is handled in post-processing:

```python
# 1. Extract mapping: reference_feature_id -> alignment_group_id
ref_mapping = data[
    data["alignment_reference_feature_id"].notna()
][["alignment_reference_feature_id", "alignment_group_id"]].drop_duplicates()

# 2. Create reverse mapping: id -> alignment_group_id for references
ref_group_mapping = ref_mapping.rename(
    columns={"alignment_reference_feature_id": "id", 
             "alignment_group_id": "ref_alignment_group_id"}
)

# 3. Merge to assign group IDs to reference features
data = pd.merge(data, ref_group_mapping, on="id", how="left")

# 4. Fill in alignment_group_id where it's null but ref_alignment_group_id exists
mask = data["alignment_group_id"].isna() & data["ref_alignment_group_id"].notna()
data.loc[mask, "alignment_group_id"] = data.loc[mask, "ref_alignment_group_id"]
```

**Result:** All features in an alignment group (both aligned and reference features) share the same `alignment_group_id`, enabling:
- Tracking which features are aligned together
- Identifying the reference feature for each alignment group
- Analyzing alignment quality across related features

### Performance Considerations

| Approach | Query Time | Precision | Index Usage |
|----------|-----------|-----------|-------------|
| No CAST | ~16 sec | ❌ Lost | ✅ Yes |
| CAST in JOIN | ~50 min | ✅ Preserved | ❌ No |
| CAST in SELECT | ~16 sec | ✅ Preserved | ✅ Yes |

**Conclusion:** CAST in SELECT clause provides both precision preservation and optimal performance.
