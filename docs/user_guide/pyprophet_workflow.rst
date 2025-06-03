Running a Full PyProphet Workflow (Parquet)
===========================================

In this section, we will walk through a complete PyProphet workflow using parquet files as the input, from scoring OpenSWATH results to exporting the final results. This guide assumes you have already run OpenSWATH and have an OSW file ready for processing.

Feature File Conversion
-----------------------------------------

Let's take a look at how many OSW feature files we have:

.. code-block:: bash

    $ ls -lh ./osw/*.osw | wc -l
    247 # subtract 1 for the header line
    $ du -sh ./osw/*.osw
    3.9T osw/

We can see just how large the OSW files are. Since we have a lot of runs as well, lets convert the OSW files to a split parquet format, which will split the precursor and transition data into separate parquet files for each run. This can be done using the following command:

.. code-block:: bash

    cd osw
    # get all .osw files in the current directory
    osw_files=$(find . -maxdepth 1 -name "*.osw")

    for osw_file in $osw_files; do
        pyprophet export parquet --in "$osw_file" --out "${osw_file%.osw}.oswpq" --split_transition_data --split_runs
    done

    # Lets move the run parquet directories into a single directory for easier management
    mkdir all_runs.oswpqd
    mv *.oswpq all_runs.oswpqd

After converting the OSW files to parquet format, you should have a directory structure like this:

.. code-block:: text

    └── 📁 all_runs.oswpqd
        ├── 📁 20200626_erli_phos_10.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_101.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_102.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_103.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_104.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_105.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_106.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_107.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_108.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        ├── 📁 20200626_erli_phos_109.oswpq
        │   ├── 📄 precursors_features.parquet
        │   └── 📄 transition_features.parquet
        │   ... (236 more run(s) collapsed)

Let's check how many parquet files we have now and how large they are:

.. code-block:: bash

    $ find ./all_runs.oswpqd -type d -name "*.oswpq" | wc -l
    246
    $ du -sh ./all_runs.oswpqd/*/*.parquet
    1.6T	oswpq/
    $ du -sh ./all_runs.oswpqd/20200902_Erli_pro_144.oswpq/*
    287M	all_runs.oswpqd/20200902_Erli_pro_144.oswpq/precursors_features.parquet
    6.2G	all_runs.oswpqd/20200902_Erli_pro_144.oswpq/transition_features.parquet
    ...

As we can see, the parquet files are much smaller than the original OSW files, and we have split the precursor and transition data into separate files for each run. This will make it easier to work with the data in subsequent steps.

Scoring OpenSWATH Results
-----------------------------------------

Now that we have our data in a more manageable format, we can proceed with scoring the OpenSWATH results using PyProphet. We will use the `pyprophet score` command to score the features in the parquet files.

Peak-Group Level MS1-MS2 Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pyprophet score --in all_runs.oswpqd --level ms1ms2 --classifier XGBoost 

This command will score the features in the parquet files at the peak-group level (MS1-MS2) using the XGBoost classifier. All the runs in the `all_runs.oswpqd` directory get processed together during the scoring, which allows PyProphet to learn the weights for the features across all runs. The results scores are then written back to the corresponding parquet files in the `all_runs.oswpqd` directory.

.. note::
    There are different classifiers available for scoring, such as LDA, SVM and XGBoost. 

    If your dataset is really large, you may want to subsample the data to speed up the scoring process, and then apply the learned weights to the rest of the data. This can be done using the `--subsample_ratio` parameter, which allows you to specify the ratio of the data to use for training the classifier. For example, `--subsample_ratio 0.1` will use 10% of the data during the learning and scoring process. 

    See the :ref:`CLI documentation <cli_score>` for more information on the available classifiers and the additional parameters you can use to customize the scoring process.

Transition Level Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are running and IPF workflow, then you want to perform additional transition level scoring after the peak-group level scoring. 

.. code-block:: bash

    pyprophet score --in all_runs.oswpqd --level transition --classifier XGBoost --ss_initial_fdr 0.2 --ss_iteration_fdr 0.01 --ipf_min_transition_sn=-1

.. note::
    The transition level data is larger than the peak-group level data, so it will likely help to use the `--subsample_ratio` parameter to speed up the scoring process, depending on how much compute resources you have available.

Alignment Level Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you performed chromatogram feature alignment using `ARYCAL <https://github.com/singjc/arycal>`_, you can optionally perform scoring at the alignment level to estimate the quality of the aligned features. 

.. code-block:: bash

    pyprophet score --in all_runs.oswpqd --level alignment --classifier XGBoost

Inference of Peptidoforms (IPF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are running an IPF workflow, you can perform inference of peptidoforms using the `pyprophet ipf` command. 

.. code-block:: bash

    pyprophet ipf --in all_runs.oswpqd --no-ipf_ms1_scoring --no-ipf_ms2_scoring --propagate_signal_across_runs --ipf_max_alignment_pep 0.7 --across_run_confidence_threshold 0.5

.. note::
    To use the `--propagate_signal_across_runs` parameter, you need to have performed chromatogram feature alignment, and have scored the aligned features at the alignment level. This will allow PyProphet to propagate the signal across runs and improve the inference of peptidoforms.

    See the :ref:`CLI documentation <cli_ipf>` for more information on the available parameters and how to customize the IPF workflow.

Contexts and FDR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To conduct peptide / protein / gene inference, you can use the `pyprophet levels-context` command. This command will infer the levels of peptides, proteins, and genes from the scored features in the parquet files. 

.. code-block:: bash

    pyprophet levels-context peptide --in all_runs.oswpqd --context global
    pyprophet levels-context peptide --in all_runs.oswpqd --context experiment-wide
    pyprophet levels-context peptide --in all_runs.oswpqd --context run-specific
    pyprophet levels-context protein --in all_runs.oswpqd --context global
    pyprophet levels-context protein --in all_runs.oswpqd --context experiment-wide
    pyprophet levels-context protein --in all_runs.oswpqd --context run-specific

See the :ref:`CLI documentation <cli_levels_context>` for more information on the available contexts and parameters.

Exporting Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, you can export the results to a variety of formats using the `pyprophet export` command. 

To export a filtered version of the results to a TSV file, you can use the following command:

.. code-block:: bash

    pyprophet export tsv --in all_runs.oswpqd --out results.tsv 

This will export the results to a TSV file, which can be easily read and processed by other tools or scripts. See the :ref:`CLI documentation <export_tsv_proteomics>` for more information on the available parameters and how to customize the export process.

.. warning::
    By default, IPF results on peptidoform-level will be used if available. This can be disabled by setting --ipf=disable.

To export quantification matrices, you can use the following command:

.. code-block:: bash

    pyprophet export matrix --in all_runs.oswpqd --out peptide_matrix.tsv --level peptide
    pyprophet export matrix --in all_runs.oswpqd --out protein_matrix.tsv --level protein

.. note::
    You can change how the intensities are collapsed and summarized at different levels by setting the `--top_n` and `--consistent_top` parameters to your liking. By default it will use the top 3 intense features for summarization, that are consistent across runs. You can also apply optional normalization to the intensities using the `--normalization` parameter, which can be set to `none`, `median`, `medianmedian`, or `quantile`, the default is `none`. See the :ref:`CLI documentation <export_matrix_proteomics>` for more information on the available parameters and how to customize the export process.

To generate a high-level pdf report summary of the results, you can use the following command:

.. code-block:: bash

    pyprophet export score-report --in all_runs.oswpqd

If you want to inspect the distributions of the target-decoy features used during scoring, you can use the following command:

.. code-block:: bash

    pyprophet export score-plots --in all_runs.oswpqd 