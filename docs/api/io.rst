IO: Reading and Writing Data
=====================================================



.. automodule:: pyprophet.io
   :members:
   :inherited-members:

Abstract Base Classes
----------------------

.. currentmodule:: pyprophet.io

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   _base.BaseReader
   _base.BaseWriter
   _base.BaseOSWReader
   _base.BaseOSWWriter
   _base.BaseParquetReader
   _base.BaseParquetWriter
   _base.BaseSplitParquetReader
   _base.BaseSplitParquetWriter
   dispatcher.ReaderDispatcher
   dispatcher.WriterDispatcher

These submodules provide specific implementations for reading and writing data for specific algorithms.

Scoring
----------------------

.. automodule:: pyprophet.io.scoring
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyprophet.io.scoring

.. autosummary:: 
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   osw.OSWReader
   osw.OSWWriter
   parquet.ParquetReader
   parquet.ParquetWriter
   split_parquet.SplitParquetReader
   split_parquet.SplitParquetWriter
   tsv.TSVReader
   tsv.TSVWriter

IPF
----------------------

.. automodule:: pyprophet.io.ipf
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyprophet.io.ipf

.. autosummary:: 
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   osw.OSWReader
   osw.OSWWriter
   parquet.ParquetReader
   parquet.ParquetWriter
   split_parquet.SplitParquetReader
   split_parquet.SplitParquetWriter

Levels Context
----------------------

.. automodule:: pyprophet.io.levels_context
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyprophet.io.levels_context

.. autosummary:: 
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   osw.OSWReader
   osw.OSWWriter
   parquet.ParquetReader
   parquet.ParquetWriter
   split_parquet.SplitParquetReader
   split_parquet.SplitParquetWriter

Export
----------------------

.. automodule:: pyprophet.io.export
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyprophet.io.export

.. autosummary:: 
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   osw.OSWReader
   osw.OSWWriter
   parquet.ParquetReader
   parquet.ParquetWriter
   split_parquet.SplitParquetReader
   split_parquet.SplitParquetWriter