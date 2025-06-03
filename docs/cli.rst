.. _command_line_interface:

Command Line Interface
=========================

.. automodule:: pyprophet.cli
   :members:
   :undoc-members:
   :show-inheritance:


`pyprophet` is the main command-line interface for PyProphet, with subcommands for scoring, IPF, levels context inference, and other utility functions. 

.. click:: pyprophet.main:cli
   :prog: pyprophet
   :nested: none

Semi-Supervised Scoring of Peak-Groups
--------------------------------------
.. _cli_score:

PyProphet provides a command-line interface for scoring peak-groups using the `score` subcommand. This provides a re-implementation of the original `mProphet <http://dx.doi.org/10.1093/bioinformatics/btu686>`_ algorithm, which is a semi-supervised machine learning approach for scoring peak-groups in SRM mass spectrometry data.

.. currentmodule:: pyprophet.cli.score

.. click:: pyprophet.cli.score:score
   :prog: pyprophet score
   :nested: full


The :program:`score` command has several advanced options that can be seen using the :option:`--helphelp` flag. 


Inference of Peptidoforms
-------------------------
.. _cli_ipf:

For PTM analyses, PyProphet provides the :program:`ipf` subcommand. This command allows you to perform inference of peptidoforms, for site-localization of peptidoforms in large-scale DIA experiments.

Refer to `Rosenberger, G. et. al. (2017) <https://www.nature.com/articles/nbt.3908#Abs2>`_ to learn more about the inference of peptidoforms workflow.

.. currentmodule:: pyprophet.cli.ipf

.. click:: pyprophet.cli.ipf:ipf
   :prog: pyprophet ipf
   :nested: none

For glycoform inference, you can use the :program:`glycoform` subcommand, which is specifically designed for glycopeptide analyses.

.. currentmodule:: pyprophet.cli.ipf

.. click:: pyprophet.cli.ipf:glycoform
   :prog: pyprophet glycoform
   :nested: none

Refer to `Yang, Y. et. al. (2021) <https://www.nature.com/articles/s41467-021-26246-3#Abs1>`_ to learn more about the glycoform inference workflow.

Peptide / Protein / Gene Inference
----------------------------------
.. _cli_levels_context:

To perform inference at different levels of biological context and different experimental contexts (global, experiment-wide and run-specific), PyProphet provides the :program:`levels-context` subcommand. This command allows you to infer peptide, glycopeptide, protein, and gene levels from your data.

Refer to `Rosenberger, G. et. al. (2017) <https://www.nature.com/articles/nmeth.4398>`_ to learn more about the levels context inference.

For more information about glycopeptide inference, refer to `Yang, Y.. et. al. (2021) <https://www.nature.com/articles/s41467-021-26246-3#Abs1>`_.

.. currentmodule:: pyprophet.cli.levels_context

.. click:: pyprophet.cli.levels_context:peptide
   :prog: pyprophet levels-context peptide
   :nested: none

The :program:`peptide` command accepts a :option:`helphelp` argument to display its advanced options that are not shown here.

.. click:: pyprophet.cli.levels_context:glycopeptide
   :prog: pyprophet levels-context glycopeptide
   :nested: none

.. click:: pyprophet.cli.levels_context:protein
   :prog: pyprophet levels-context protein
   :nested: none

The :program:`protein` command accepts a :option:`helphelp` argument to display its advanced options that are not shown here.

.. click:: pyprophet.cli.levels_context:gene
   :prog: pyprophet levels-context gene
   :nested: none

The :program:`gene` command accepts a :option:`helphelp` argument to display its advanced options that are not shown here.

Exporters
---------

PyProphet provides several export utilities to export between different file formats for OpenSwath's (*.osw* / *.sqMass*sqlite-based formats) and experimental parquet formats, as well as exporting PDF reports of the data. 

.. currentmodule:: pyprophet.cli.export

TSV Results (Proteomics)
^^^^^^^^^^^^^^^^^^^^^^^^
.. _export_tsv_proteomics:

To export results from a post-scoring workflow (using the *.osw* input workflow) to a tab-separated values (TSV) file, you can use the :program:`export tsv` subcommand. This is useful for exporting results in a format that can be easily read and processed by other tools or scripts.

.. click:: pyprophet.cli.export:export_tsv
   :prog: pyprophet export tsv
   :nested: none

TSV Results (Small Molecules)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is similar to the TSV export for proteomics, but specifically designed for small molecule data. It allows you to export results in a tab-separated values (TSV) format, which can be easily read and processed by other tools or scripts.

.. click:: pyprophet.cli.export:export_compound
   :prog: pyprophet export compound
   :nested: none

TSV Results (Glycoform)
^^^^^^^^^^^^^^^^^^^^^^^

This is similar to the TSV export for proteomics, but specifically designed for glycoform data. It allows you to export results in a tab-separated values (TSV) format, which can be easily read and processed by other tools or scripts.

.. click:: pyprophet.cli.export:export_glyco
   :prog: pyprophet export glyco
   :nested: none

TSV Quantification Matrices (Proteomics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _export_matrix_proteomics:

To export quantification matrices from a post-scoring workflow to a tab-separated values (TSV) file, you can use the :program:`export matrix` subcommand. This is useful for exporting quantification data in a format that can be easily read and processed by other tools or scripts.

.. click:: pyprophet.cli.export:export_matrix
   :prog: pyprophet export matrix
   :nested: none


Convert OSW to Parquet
^^^^^^^^^^^^^^^^^^^^^^

To convert OpenSwath's *.osw* / *.sqMass* format to a parquet format, you can use the :program:`export parquet` subcommand. This is useful for converting results from the *.osw* / *.sqMass* format to a more efficient and space saving data storage format. This subcommand has the option to convert the entire *.osw* file to a snigle parquet file (with both precursor and transition data) or to split the parquet file into a separate precursors_features.parquet file and a transition_features.parquet file. There is the option to further split by run, which is useful for large datasets.

.. click:: pyprophet.cli.export:export_parquet
   :prog: pyprophet export parquet
   :nested: none

Export Score Plots
^^^^^^^^^^^^^^^^^^

It may be useful to export the distribution of scores for the different input features. This can help you investigate the distribution and quality of scores for target-decoy separation.

.. click:: pyprophet.cli.export:export_score_plots
   :prog: pyprophet export score-plots
   :nested: none

Export Results Report
^^^^^^^^^^^^^^^^^^^^

To export a PDF report of the results, you can use the :program:`export score-report` subcommand. This is useful for generating a report that summarizes the results of your analysis, including scores and identifications, and other relevant information.

.. click:: pyprophet.cli.export:export_scored_report
   :prog: pyprophet export score-report
   :nested: none