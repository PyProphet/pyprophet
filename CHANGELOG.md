# Changelog

All notable changes to this project will be documented in this file.

## [3.0.5] - 2025-11-27

### 🚀 Features

- Add native installer creation scripts (DEB, RPM, DMG, EXE)
- Add native installer creation scripts (DEB, RPM, DMG, EXE)
- Enhance artifact upload process in GitHub Actions for Linux, Windows, and macOS
- *(build)* Improve DMG testing and installation process for macOS
- *(build)* Enhance build scripts for Windows and macOS with UPX compression and runtime dependency installation
- *(build)* Add --onefile option to PyInstaller for all platforms
- *(build)* Enhance runtime dependency installation in build scripts for Linux, macOS, and Windows
- *(build)* Streamline PyInstaller build process with onefile mode and module exclusions for macOS and Linux
- *(build)* Implement UPX compression for final binaries in build scripts for Linux, macOS, and Windows
- *(build)* Update packaging scripts to use single-file executable for Linux, macOS, and Windows
- *(build)* Enforce Python 3.11+ requirement in build scripts for Linux, macOS, and Windows
- *(build)* Create minimal entry point script to avoid path conflicts in PyInstaller builds for Linux, macOS, and Windows
- *(build)* Install pyprophet as a regular package to avoid import conflicts in build scripts for Linux, macOS, and Windows
- *(build)* Build and install pyprophet as a wheel for cleaner installation in Linux and Windows scripts
- *(build)* Optimize PyInstaller build process with temporary directory for cleaner execution
- *(build)* Enhance build script to save original directory and clean up previous builds
- *(build)* Clean up build artifacts and optimize wheel installation process in Linux script
- *(build)* Modify installation process to ensure pyprophet is installed in site-packages and verify dependencies
- *(build)* Enhance installation script to uninstall existing pyprophet and optimize wheel build process
- *(build)* Create isolated virtual environment for building and streamline pyprophet installation process
- *(build)* Streamline build script by cleaning artifacts and optimizing virtual environment setup
- *(build)* Optimize Linux build script by simplifying installation steps and improving error handling
- *(build)* Enhance Windows build script to clean Python cache and improve installation process
- *(build)* Update GitHub Actions workflow to improve error handling and streamline command execution
- *(build)* Enhance Windows build script to optimize PyInstaller process and add self-extracting archive support
- Implement lazy loading for pyarrow imports and improve error handling for parquet support
- Enhance version retrieval and sanitization for macOS build scripts
- *(export)* Add support for exporting minimal scored-report columns from split Parquet files

### 🐛 Bug Fixes

- *(build)* Handle errors in file permission setting and listing during package creation
- *(tests)* Fix bug with restructuring of ipf score to precursor mapping and IM boundaries
- *(osw)* Remove debug print statement for temporary table query in OSWWriter
- *(report)* Ensure axes are consistently indexed in plot_score_distributions function

### 💼 Other

- Requirements.txt
- Only add from_alignment column when alignment is actually used
- Extract helper method to reduce code duplication
- Filter out columns without types when building temp table schema
- Add RUN_ID to score peptide/protein views for proper joining
- Use LEFT JOIN for score views to avoid filtering out rows without matching RUN_ID
- Regtest for new test
- Regtest output
- Extract pyarrow import check to helper function to reduce code duplication
- Move file type reader methods into IO export classes
- Use existing _ensure_pyarrow() instead of custom function
- Regtest outputs

### 🚜 Refactor

- Simplify parquet reader and writer initialization by using utility functions
- Update pyarrow import to improve parquet file handling
- Streamline parquet file imports and initialization using utility functions
- Avoid circular imports by using type names for config checks
- Reorganize parquet file imports for improved clarity and initialization
- Enhance lazy loading for pyarrow imports and improve parquet initialization
- *(parquet)* Improve alignment group ID assignment and update score column references

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG.md
- Replace xgboost with xgboost-cpu to reduce package size
- Update Python version to 3.11 in GitHub Actions workflows
- Remove local path for pyprophet in requirements.txt
- Simplify command paths in GitHub Actions workflow for pyprophet
- Remove tomli module exclusion from PyInstaller build scripts for Linux, macOS, and Windows
- Remove pip install command from GitHub Actions workflow to resolve numpy source directory issue
- Remove sklearn metadata copy from PyInstaller build script for Linux
- Add error handling to cleanup commands in Windows build script
- Remove pyarrow from main dependencies and add to parquet group
- Update pyarrow dependency to optional for Parquet support
- Exclude unnecessary modules from PyInstaller builds for Linux, macOS, and Windows

## [3.0.4] - 2025-10-21

### 🚀 Features

- Enhance PyInstaller scripts for macOS and Windows with metadata collection and binary inclusion
- Expand GitHub Actions workflow to include comprehensive testing for all commands across platforms
- Add concurrency control to GitHub Actions workflows for improved efficiency
- Add concurrency control to multiple GitHub Actions workflows for improved efficiency
- Add PyInstaller hook for pyprophet package to ensure proper module collection

### 🐛 Bug Fixes

- *(build)* Fix archiving of windows build release to use zip instead of tar

### 💼 Other

- Pyinstaller build scripts
- Gha's for package building
- Create launcher for PyInstaller bundles with threading support
- Windows dist script
- Create PyInstaller hooks for duckdb and xgboost packages
- Implement PyInstaller hook for pyopenms package with dynamic library collection
- Move build scripts to a dedicated directory and update paths
- Update build script paths to reflect new directory structure

### 🚜 Refactor

- Remove tabulate dependency and update related code in main.py
- Remove numexpr dependency from project configuration
- Remove seaborn dependency from project configuration
- Reorganize imports and clean up unused code in report.py
- Remove statsmodels dependency and update related code in stats.py
- Reduce decimal precision in lfdr tests for consistency
- Update expected output values in lfdr tests for consistency
- Add Gaussian kernel density estimation comparison script
- Update expected output values in pyprophet score tests for consistency with change of kernel density estimate
- Filter script
- Reorganize imports and optimize database query in split.py
- Update expected output values in pyprophet IPF scoring tests for consistency
- Implement statsmodels KDEUnivariate method
- Update expected output values in pyprophet score tests for consistency
- Add comparison script for FFT-based KDE and statsmodels KDEUnivariate
- Replace DCT-based KDE with FFT-based KDE implementation
- Update expected output values in pyprophet IPF scoring tests for consistency
- Update expected output values in test_stats for consistency
- Update expected output values in pyprophet export test for consistency
- Update build and test workflows for improved clarity and manual trigger

### 📚 Documentation

- Update installation instructions and add options for pre-built executables and Docker
- Add contributing guidelines and commit message conventions

### ⚙️ Miscellaneous Tasks

- Remove feature branch from GitHub Actions workflow triggers
- Update Python version support and refine package discovery settings
- Add GitHub Actions workflow for automatic changelog generation
- Remove concurrency settings from GitHub Actions workflow
- Update CHANGELOG.md
- Refactor GitHub Actions workflow for improved build and artifact handling

## [3.0.3] - 2025-10-18

### 🚀 Features

- Update Dockerfile to use Ubuntu 24.04 and set up Python virtual environment
- Enhance HistGradientBoostingClassifier with improved parameter tuning and parallel processing
- Add HistGBCLearner support for autotuning in semi-supervised learning
- Add support for thread management in HistGradientBoosting to prevent oversubscription
- Add documentation for thread management in HistGradientBoosting and HistGBCLearner
- Add version checking utilities for PyProphet
- Add update check and message display in CLI

### 🐛 Bug Fixes

- Filter out invalid params for HistGradientBoostingClassifier to prevent TypeError
- Remove stray return and fix indentation in HistGBCLearner.learn
- Use permutation_importance fallback for HistGradientBoostingClassifier feature importances
- Add check for importance attribute before logging feature importances
- Store feature importance on classifier object to persist through set_parameters
- Pi0 when passing single value as lambda
- Update l2_regularization default value in HistGBCLearner to improve model performance
- Initialize ll variable in pi0est function to prevent potential errors
- Correct typo in comment regarding OMP_NUM_THREADS and numpy import order
- Set OMP_NUM_THREADS before sklearn import for HistGBC
- Ensure OMP_NUM_THREADS is set before any numpy imports for HistGradientBoosting
- Set OMP_NUM_THREADS to 1 for pytest execution in CI workflow
- Add 'packaging' to runtime dependencies in pyproject.toml

### 💼 Other

- Dockerfile
- Defaults for HistGBC
- Impl. summary docs
- Robust thread control\n- Safely derive total_jobs from TOTAL_CPUS or os.cpu_count() with fallback and clamping\n- Coordinate outer (RandomizedSearchCV n_jobs) vs inner (OpenMP) parallelism\n- Use threadpoolctl to enforce OpenMP limits during tune() and learn() fits\n- Avoid OMP_NUM_THREADS=0 and respect user-provided OMP_NUM_THREADS if set
- Delete HISTGBC_TEST_ADDITIONS.md as part of documentation cleanup

### 🚜 Refactor

- Move compare_classifiers.py to sandbox and update parameters for optimization
- Clean up code formatting and improve readability in BaseWriter class
- Remove unused pandas import from compare_classifiers.py
- Move OMP_NUM_THREADS setup to main.py for HistGradientBoosting and update logging in runner

### 📚 Documentation

- Clarify OMP_NUM_THREADS setup requirements in main.py

### 🧪 Testing

- Add HistGradientBoosting classifier coverage across OSW/Parquet/Split/Multisplit; enable autotune and PFDR paths; extend strategies to support --classifier=HistGradientBoosting
- Add regression test outputs for HistGradientBoosting classifier across various configurations
- Add tests for version checking functionality

## [3.0.2] - 2025-10-09

### 🚀 Features

- Add log colorization option for improved logging visibility
- Add exporting "score_peptide" and "score_protein" from OSW files
- Start implementation of lib export with pyprophet
- Option to keep significant decoys in lib refinement
- Add option to export rt unit in non iRT
- Sort by intensity if q value tie
- Add calibration report export functionality
- Support tuple input for lambda in pi0est function
- More efficient sqMass to parquet export
- Add transition ordinal and type to export
- Implement export functionality for scored reports with parquet and osw support

### 🐛 Bug Fixes

- Refactor SQL subquery for feature alignment in OSWReader
- Alignment table fetching for ipf osw reader
- QVALUE->Q_VALUE
- Export protein info in lib
- Lib export compute annotation  col if empty
- Error description
- Bug in sql query
- Bug resulting from code suggestions
- Ensure score_ipf_qvalue is checked for NaN before filtering

### 💼 Other

- _optimized cython bindings
- Note that keep_decoys in lib gen is experimental feature
- Chrom parquet schema documentation
- Dependencies

### 📚 Documentation

- Update schema with new columns

### 🧪 Testing

- Add test for lib generation
- Update tests with new snapshots

## [3.0.1] - 2025-07-24

### 🐛 Bug Fixes

- Ipf osw alignment data fetching

### 💼 Other

- Test nb

## [3.0.0] - 2025-06-13

### 🚀 Features

- Add option to split transition data in convert_osw_to_parquet function
- Add memory usage information to parquet export
- Improve memory usage reporting in parquet export
- Enhance feature transition data extraction in convert_osw_to_parquet
- Add functionality to extract and export memory usage information in convert_osw_to_parquet
- Move format_bytes function to data handling
- Improve coalesce handling and data processing in convert_osw_to_parquet
- Remove batch_size option and optimize memory usage in parquet export
- Include FEATURE_ID in feature transition columns for improved data extraction in convert_osw_to_parquet
- Include RUN_ID in feature transition data extraction for improved data processing in convert_osw_to_parquet
- Introduce normalized peptides and IPF peptide mapping in convert_osw_to_parquet function
- Implement RunnerConfig and RunnerIOConfig dataclasses for handling scoring and configuration in IO operations
- Add ErrorEstimationConfig dataclass for error estimation configuration in IO operations
- Add properties for Base reader and writer for common used fields from config
- Implement different io reader and writer classes
- Add method to write PDF report with scoring results
- Add subsample_ratio parameter for data subsampling in PyProphet scoring
- Add function to retrieve feature mapping across runs in glycoform.py
- Add LevelContextIOConfig for context-based configuration in peptide inference
- Add trained model path for alignment in RunnerIOConfig
- Introduce LevelContextIOConfig for context-based configuration in ReaderDispatcher and WriterDispatcher
- Implement feature to scale features
- Add support for autotuning hyperparameters in SVM classifier and XGBoost in RunnerConfig
- Add new commands for glycopeptide, gene, and protein in levels context
- Add export_scored_report function to export PyProphet scored reports
- Add method to merge precursor and transition features files into single parquet files
- Install seaborn and psutil dependencies in Dockerfile
- Add HistGradientBoostingClassifier for improved learning performance
- Add new test data files for PyProphet score split parquet outputs
- Collapse IPF peptide IDs to avoid duplicating feature data in DataFrame
- Add function to convert a sequence with unimod modifications to a codename
- Enhance export functionality with logging for legacy formats
- Enhance peptide and protein data merging with improved logging and handling for run-specific and global scores
- Improve logger formatting based on log level for easier debbugging and tracing
- Enhance logging for legacy export format with data size and shape information
- Add score column selection and joins for osw to parquet export

### 🐛 Bug Fixes

- How weights are saved for xgboost and how they're loaded
- Parquet ms1 data fetching
- Parquet transition data fetching
- Score parquet reader, ensure scores get added to proper block
- Peptide and protein data merging with improved handling of run-specific and global scores for tsv export
- Change debug logs to info level for legacy export formats
- Update test output for peptide and protein data export
- Correct SQL queries to use appropriate scoring values for peptide and protein data

### 💼 Other

- Click logs
- Scoring with parquet directory
- Ipf to accept split parquet files
- Fix xgboost feat importance printing
- Logging to loguru, and handle multi directory of split parquet runs
- Option to split by run
- Test io scoring module
- Test pyp score
- Pyp test score regtest output
- Pyp score tests
- Io reading tests
- Subsampling in parquet scoring reader
- Report for post scored OSW-PYP results
- Optimized c
- Init to glyco module
- Apply ruff formatting
- Format with ruff
- Apply ruff formatting
- Cli
- Test io for levels context
- Io export module
- Assets logo
- Parquet test data
- Pyp call in glyco scoring

### 🚜 Refactor

- Optimize join and column selection in convert_osw_to_parquet function
- Remove unused code related to writing parquet batches
- Repalce monolith passed params with runner config
- Runner, abstract data reading and writer to separate classes
- Update data_handling module with new check_sqlite_table function and refactor PyProphetRunner class to use runner_config properties
- Update import paths for configuration modules in various IO-related files
- Update error estimation configuration access in StandardSemiSupervisedLearner
- Update runner config to use specific properties for dynamic main score setting
- Update data handling module with new check_sqlite_table function and refactor PyProphetRunner class to use runner config properties
- Replace Exception with ValueError for zero standard deviation in decoy scores
- Update base class methods for file type detection and handling in IO-related files
- Update import paths for IO-related files to use new utility functions
- Update import paths for IO-related files to use new utility functions
- Update import paths for IO-related files to use new utility functions and io.util module
- Refactor read_parquet_dir function for improved data handling and code organization
- Improve data handling and code organization in read_parquet_dir function
- Update DuckDB table checking to be case-insensitive
- Update alignment_table view query to include new columns and ordering
- Update BaseReader class with additional module specific config for reading and deprecate swath_pretrained main score option
- Split io into context modules for reading/writing and use a single dispatcher for delegating
- Add config io dispatcher to IPF
- Filter persisted weights by current level in PyProphetWeightApplier
- Comment out unnecessary exception for missing transition-level feature table
- Remove unnecessary code for creating views in SplitParquetReader
- Update SplitParquetReader and SplitParquetWriter to handle alignemnt when multi run
- Update IPF CLI to use IPFIOConfig
- Update ErrorEstimationConfig with default values and numpy import
- Update feature selection query in convert_osw_to_parquet function
- Update logging messages for semi-supervised learning levels in main.py
- Remove redundant SQL queries for gene and protein inference
- Update IO configurations to use specific config classes for runners and IPF
- Update context_fdr usage in ParquetReader and SplitParquetReader
- Update context_fdr assignment in ParquetReader and SplitParquetReader
- Update logging and setup_logger usage in various modules
- Group level context subcommands and add global stats decoratory for shared stat params
- Scoring to single parquet file with blocked nulls
- Introduce export command group for different formats and update TSV export functionality
- Update column renaming logic in BaseWriter for different file types
- Update column data types to BIGINT in convert_osw_to_parquet function
- Update xgb param space
- Update product charge field name in OSWReader class
- Merge ms1 data into feature table for ms1ms2 level in OSWReader class
- Finalize feature table column names for semi-supervised scoring
- Update convert_osw_to_parquet to include necessary fields for mapping to TRANSITION data
- Validate row count consistency after joining scores in ParquetWriter classes
- Update table references from 'transitions' to 'transition' in SplitParquetReader class
- Include FEATURE_ID field in convert_osw_to_parquet for mapping to TRANSITION data
- Validate row count consistency after joining scores in ParquetWriter and SplitParquetWriter classes
- Replace click.echo with logger for informative messages in scoring classes
- Add subsampling option in test functions for Parquet scoring
- Add SVM classifier support and autotuning in PyProphetRunner and SemiSupervisedLearner classes
- Improve logging format in setup_logger function
- Report
- Level context stats, move report writing to base writer
- Update logging level to 'trace' for creating views in SplitParquetReader
- Remove optional model retraining in SVMLearner class
- Update logging level to 'debug' for generating main panel plot
- Remove redundant comments and unnecessary code in BaseWriter class
- Update logging message to remove f-string usage in export_parquet.py
- Update BaseIOConfig class to include __str__ and __repr__ methods
- Implement __str__ and __repr__ methods for ErrorEstimationConfig, RunnerConfig, and RunnerIOConfig classes
- Improve logging for scaling columns in Experiment class
- Add setup_logger function for initializing logging with specified log level
- Update CombinedGroup class and shared statistics options for PyProphet CLI
- Improve error handling and logging for classifier weights validation
- Update PyProphetWeightApplier to handle SVM classifier and improve code readability
- Update classifier handling in LinearLearner and SVMLearner classes
- Update data source references in ParquetReader and ParquetWriter classes
- Update SVMLearner class to include class_weight parameter for hyperparameter tuning
- Update logging functionality to include header information for PyProphet CLI commands
- Update dependencies in pyproject.toml, optimize imports, and restructure code in various modules
- Update Base classes to use BaseOSWReader and BaseOSWWriter for consistency across modules
- Separate out subcommands into separate files in cli module to clean up main
- Add export_scored_report command and measure memory usage and time for various export functions
- Add measure_memory_usage_and_time decorator to score function
- Remove unused format_bytes function for better code cleanliness
- Update module paths for optimized C files
- Improve readability and documentation in scoring classes
- Update CLI module paths for improved organization
- Improve CLI option descriptions for better clarity and understanding
- Imports
- Update file paths in read_parquet method for consistency
- Update CLI option descriptions and variable names for consistency and clarity
- Enhance logging and diagnostics for row mismatch in score DataFrame
- Add 'PROTEIN_ID' column handling for 'ms1', 'ms2', and 'ms1ms2' levels in parquet scoring
- Collapse protein IDs to avoid duplicating feature data in DataFrames
- Set alignment_file to None if it doesn't exist to handle missing file gracefully
- Remove unnecessary print statement in SplitParquetWriter class
- Update print_alignment_file handling in BaseSplitParquetReader class
- Update handling of SCORE_MS2_SCORE column in ParquetReader and SplitParquetReader classes
- Improve handling of multiple runs in report generation
- Update handling of run_id for different levels in report generation
- Improve handling of file paths and deep copy in BaseIOConfig and score functions
- Update handling of score and key columns based on file type in BaseWriter class
- Update handling of group_id formatting in OSWReader class
- Improve test fixtures and validation functions for peptide and protein level analysis
- Enhance utility functions for SQLite and Parquet file handling
- Update data handling imports to pyprophet.scoring module
- Update test utilities for reading and comparing data across different levels and contexts
- Update level data extraction for IPF in ParquetReader class
- Add ExportIOConfig class for exporting results to various formats
- Update column names for P-value and Q-value in OSWWriter class
- Update import paths for SQLite table checking in export modules
- Update export module to include ExportIOConfig for flexible result exporting
- Update ParquetReader to include 'export' context for printing parquet tree
- Update ordering of columns in SQL queries for consistency in SplitParquetReader class
- Pyp export test
- Improve clarity of help messages in export_tsv function
- Generalize ipf unimod - codename peptide id mapping
- Update SQL queries to include additional conditions for transition scoring data
- Update logging format for summary display in PlotGenerator class
- Collapse IPF peptide IDs to avoid duplicating feature data in DataFrame
- Update BaseSplitParquetReader to inherit from BaseParquetReader to share similar methods
- Drop duplicates in ParquetReader and SplitParquetReader
- Update compare_dataframes function to accept sort_cols parameter for flexible sorting
- Align column naming for consistency in OSW file handling across modules
- Update levels_contexts.py to pass a copy of data to _write_levels_context_pdf_report
- Improve SQLite table column checking functions for better code organization and clarity
- Enhance file type inference with detailed logging for unsupported formats
- Standardize Q_VALUE to QVALUE in SQL queries for consistency
- Add function to check if a file is likely a TSV file based on its content
- Standardize Q_VALUE to QVALUE in SQL queries for consistency
- Add debug logging for reader class and file type in PyProphetRunner
- Replace xgboost autotune flag with standardized flag for consistency
- Add support for sqmass file type in WriterDispatcher and update export logic
- Move export_parquet.py to sandbox
- Add parquet export configuration options for compression and file splitting
- Add pyopenms to project dependencies
- Remove unused import for parquet conversion in export module
- Update memory usage and execution time measurement decorator
- Update XGBoost configuration by removing hyperparameters and standardizing parameters, change hyperopt to use sklearn randomized grid search
- Improve basename extraction logic in OSWWriter for consistency
- Remove scoring_format option and update help text for input file formats
- Remove unused XGBoost hyperparameters and standardize parameter configuration
- Update test output files for consistency and add new parquet output
- Add new test output file for unscored OSW data in parquet format
- Update import statement for merge function in CLI module

### 📚 Documentation

- Add doc-string to runner configs
- Update README with examples of running specific tests using pytest -k option
- Add API reference documentation for core module and configuration data classes
- Add Makefile for Sphinx documentation
- Add instructions for running tests in parallel with pytest-xdist

### ⚙️ Miscellaneous Tasks

- Comment out unnecessary print statement in SplitParquetReader
- Update SplitParquetReader to log column extraction and retrieved columns
- Add loguru and seaborn dependencies
- Check for and log duplicate entries in score DataFrame on join column
- Add Sphinx documentation configuration and update dependencies for documentation tools
- Mv some test data to sandbox
- Add pytest-xdist to testing dependencies for parallel test execution
- Update dependencies in requirements.txt to latest versions
- Update test output files for consistency and add new parquet output
- Enable parallel test execution with pytest-xdist
- Doc string documentation
- Update .gitignore to exclude generated documentation files
- Remove unused import statements from configuration and test files
- Update documentation to reflect change from --split_by_run to --split_runs option
- Add user guide and update documentation structure
- Refactor test file to use dynamic data folder paths
- Add scoring documentation and update API reference
- Update test output files to reflect new data structure
- Update documentation to include new scoring parameters
- Sort DataFrame columns in test cases for consistency
- Update test output files to reflect new scoring results
- Update test output files to reflect new scoring results
- Update subsample_ratio in pyprophet scoring test for consistency
- Add new API documentation for IPF and levels context, and update CLI documentation for merging files
- Update API documentation for levels context and scoring modules
- Update subsample_ratio in scoring tests for consistency
- Update Python version to 3.10 and adjust requirements in ReadTheDocs configuration
- Update README to include logo and CI badge
- Fix formatting in ReadTheDocs configuration for requirements
- Update subsample_ratio in scoring tests for consistency
- Update Sphinx documentation dependencies for consistency
- Reorganize Sphinx extensions for improved documentation structure
- Update documentation for consistency and clarity
- Specify Python version requirements in pyproject.toml
- Bump version to 3.0.0 in pyproject.toml
- Update README with additional PyPI and Docker badges
- Bump version to 3.0.0 in documentation configuration
- Update version retrieval in documentation configuration
- Add custom CSS and badges to documentation homepage
- Update README to improve structure and add references section
- Remove unnecessary split_runs option from pyprophet export command
- Add hidden div to have a hidden title in home page to avoid <no title> in rendered docs when not having the title
- Rename levels context command to infer
- Update help text for memory profiling option in score command
- Mark deprecated commands for future removal and update command structure
- Update CLI documentation to reflect command renaming to 'infer'
- Update tests to use 'infer' command instead of 'levels-context' for consistency
- Add Parquet schema and compression types documentation for XIC Parquet format
- Add Precursors and Transition Features Parquet schema documentation
- Add concurrency to CI workflow to cancel previous runs
- Update Python version requirement to support 3.9 and above

## [2.3.3] - 2025-04-30

### 🚀 Features

- Add support for converting sqMass files to Parquet format
- Refactor code to load and use the `sqlite_scanner` extension
- Update Dockerfile to include `sqlite_scanner` extension
- Add detecting transition column to Parquet export

### 🐛 Bug Fixes

- Bug in convert_osw_to_parquet

### ⚙️ Miscellaneous Tasks

- Update PyProphet CLI options for Parquet export

## [2.3.2] - 2025-04-29

### 🚀 Features

- Add duckdb and its extensions to Dockerfile

### 🐛 Bug Fixes

- Dockerfile

### 💼 Other

- Function to create indices if they do no exist when using duckdb connections

## [2.3.1] - 2025-04-28

### 💼 Other

- Dependencies

## [2.3.0] - 2025-04-07

### 🚀 Features

- Transition lvl string in prec level data
- Option to exlcude decoys
- Add level parameter to prepare_data_table function
- Add peptide mass calculators for glyco peptides
- Add glycopeptide scoring deom GproDIA
- Add glycopeptide export from GproDIA
- Update PyProphet score CLI help doc for input and output files
- Merge transition_df and transition_peptide_df on TRANSITION_ID
- Add parquet input support for levels context inferences
- Update PyProphet CLI options for input and output files
- Update numpy and polars dependencies in pyproject.toml
- Update PyProphet WeightApplier with Glyco args

### 🐛 Bug Fixes

- Stats tests
- Fix level context tests
- ValueError: Buffer dtype mismatch, expected 'DATA_TYPE' but got 'double'
- Export parquet peptide global level
- Export parqeut add FEATURE_MS1.AREA_INTENSITY column
- Failing tests
- Control for if propagation across runs is no desired
- Transfer_confident_evidence_across_runs function. ffil converts the feature_id to a float, which causes downtream areas when downcasting back to int64
- Optimize save_parquet_results for transition level
- Add try-catch exception handling for loading sqlite_scanner

### 💼 Other

- Mergeps method to merge feature alignment tables if present
- Alignment scoring
- Propagate signal for ipf if using aligned mapping
- Across run ipf
- Tests
- Glyco scoring from GproDIA
- LDAGlycoPeptideScorer class for glyco scoring from GproDIA
- Glycoform inference and add across run signal propagation
- Glyco report
- Osw to parquet exporting for scoring
- Function to check if file is a parquet file
- Scorer parquet input reader and parquet score results writer
- Parquet input for IPF scoring
- Convert_osw_to_parquet for backward compatability with older OSW files that have missing Gene tables and IM related columns
- Tests for scoring with parquet input
- Tests for osw to parquet scoring format
- Optimized cython code

### 🚜 Refactor

- New function for normlaizing score to decoys
- Combine queries into one
- Merge across_run_feature_map into peptidoform_table
- Update IPF code to use DuckDB for database connection

### 📚 Documentation

- Decrease size of merged_osw_zstd_level_11_33s.parquet by 62%

### 🧪 Testing

- Set tree method as exact for tests
- Add tests for no-decoys flag
- Fix parquet tests
- Feature_id is long integer

### ⚙️ Miscellaneous Tasks

- Add removed comments back

## [2.2.9] - 2024-10-30

### 🐛 Bug Fixes

- Export parquet bugs

## [2.2.8] - 2024-10-12

### 💼 Other

- D_score cuttoff line at 1% qvalue and add PP plot

## [2.2.6] - 2024-10-09

### 🚀 Features

- Option to only export precursors with features
- Auto populate var columns

### 🐛 Bug Fixes

- Minor bug fixes
- Featureless transition not appearing properly

### 💼 Other

- Bug with experiment wide context

### 📚 Documentation

- Fix doc strings

### 🧪 Testing

- Add test files for .osw

## [2.2.2] - 2022-12-15

### 💼 Other

- Export before scoring with MS1MS2 scores

## [2.1.5] - 2020-04-07

### 💼 Other

- Merge UPDATE]: Added merging scored osw output files option
- Merge UPDATE]: Correct else if statement for def merge_osw
- Merge UPDATE]: Correcting indentation for def merge_osw
- Merge UPDATE]: Correcting indentation for def merge_osw
- Merge UPDATE]: Make merged_post_scored_runs boolean flag for def merge
- Merge UPDATE]: Debugging def merge_osw
- Merge UPDATE]: Debugging def merge_osw
- Merge UPDATE]: Debugging def merge_osw
- Merge UPDATE]: Debugging def merge_osw
- Merge UPDATE]: Debugging def merge_osw
- Subsample Merged File]: Subsampling a merged file, and appending PRECURSOR table for scoring

## [2.1.0] - 2019-04-02

### 💼 Other

- Support for MAYU export
- Compute d_score & q-value based on all peak groups
- Remove fdr_all_pg option

<!-- generated by git-cliff -->
