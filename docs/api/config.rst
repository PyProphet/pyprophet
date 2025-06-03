Configuration Data Classes
=====================================================

For scoring, IPF and levels context inference, PyProphet uses configuration data classes to manage settings and parameters. These classes are designed to be easily extensible and provide a structured way to handle configuration options.

.. automodule:: pyprophet._base
   :no-members:
   :no-inherited-members:

Abstract Base Classes
---------------------

.. currentmodule:: pyprophet._base

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   BaseIOConfig


Scoring Configuration
----------------------

.. automodule:: pyprophet._config
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyprophet._config

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst
 
   RunnerIOConfig
   RunnerConfig
   ErrorEstimationConfig

IPF Configuration
-----------------

.. currentmodule:: pyprophet._config

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   IPFIOConfig

Levels Context Configuration
----------------------------

.. currentmodule:: pyprophet._config

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   LevelContextIOConfig

Export Configuration
----------------------

.. currentmodule:: pyprophet._config

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   ExportIOConfig