Semi-Supervised Scoring Documentation
==========================

.. automodule:: pyprophet.scoring
   :no-members:
   :no-inherited-members:

Runner
----------------

.. currentmodule:: pyprophet.scoring

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   runner.PyProphetRunner
   runner.PyProphetLearner
   runner.PyProphetWeightApplier

PyProphet
----------------

.. currentmodule:: pyprophet.scoring

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   pyprophet.PyProphet
   pyprophet.Scorer

Semi-Supervised
----------------

.. currentmodule:: pyprophet.scoring

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   semi_supervised.AbstractSemiSupervisedLearner
   semi_supervised.StandardSemiSupervisedLearner

Classifiers
----------------

.. currentmodule:: pyprophet.scoring

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   classifiers.AbstractLearner
   classifiers.LinearLearner
   classifiers.LDALearner
   classifiers.SVMLearner
   classifiers.XGBLearner

Data Handling
----------------

.. currentmodule:: pyprophet.scoring

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   data_handling.Experiment
   data_handling.prepare_data_table
   data_handling.cleanup_and_check
   data_handling.check_for_unique_blocks
   data_handling.update_chosen_main_score_in_table
   data_handling.use_metabolomics_scores