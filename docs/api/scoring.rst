Semi-Supervised Scoring Documentation
==========================

.. automodule:: pyprophet.scoring
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyprophet.scoring

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   scoring

Runner
----------------

.. currentmodule:: pyprophet.scoring.runner

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   PyProphetRunner
   PyProphetLearner
   PyProphetWeightApplier

PyProphet
----------------

.. currentmodule:: pyprophet.scoring.pyprophet

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   PyProphet
   Scorer

Semi-Supervised
----------------

.. currentmodule:: pyprophet.scoring.semi_supervised

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   AbstractSemiSupervisedLearner
   StandardSemiSupervisedLearner

Classifiers
----------------

.. currentmodule:: pyprophet.scoring.classifiers

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   AbstractLearner
   LinearLearner
   LDALearner
   SVMLearner
   XGBLearner
   HistGBCLearner

Data Handling
----------------

.. currentmodule:: pyprophet.scoring.data_handling

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   Experiment
   prepare_data_table
   cleanup_and_check
   check_for_unique_blocks
   update_chosen_main_score_in_table
   use_metabolomics_scores