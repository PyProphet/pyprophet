File Conversion
=========================================


Feature File Conversion
-----------------------------------------

The output from OpenSWATH is usually an sqlite-based (.osw) file, which is great for storing different levels of information and data. However, if you have a large input peptide query parameter and a lot of data, the OSW file can become very large and unwieldy. In such cases, you may benefit from converting the OSW file to a parquet file, which is a columnar storage format that is more efficient for large datasets. This can be acheived using with pyprophet using the following command:

.. code-block:: bash
    
    $ pyprophet export parquet --in input.osw --out output.parquet


This will convert the OSW file to a single parquet file, containing precursor metadata and feature data, as well as transition metadata and feature data. See the :ref:`Parquet file format documentation <parquet_format>` for more information on the structure of the parquet file.

If your OSW file is really large, you may want to split the precursor and transition data into separate parquet files. This can be done using the following command:

.. code-block:: bash
    
    $ pyprophet export parquet --in input.osw --out output.oswpq --split_transition_data

This will create two parquet files in the `output.oswpq` directory: `precursors_features.parquet` and `transition_features.parquet`. The `precursors_features.parquet` file contains precursor metadata and feature data, while the `transition_features.parquet` file contains transition metadata and feature data. This can be useful if you want to work with the precursor and transition data separately. The transition data will generally be much larger than the precursor data, so splitting the data can help with performance and memory usage. See the :ref:`Split parquet file format documentation <split_parquet_format>` for more information on the structure of the parquet files.

If you have multiple runs in your OSW file, you can further split the data by run. This can be done using the following command:

.. code-block:: bash
    
    $ pyprophet export parquet --in input.osw --out output.oswpqd --split_transition_data --split_runs

This will create a directory for each run in the `output.oswpqd` directory, with the precursor and transition data split into separate parquet files for each run. See the :ref:`Split parquet file format documentation <split_parquet_format>` for more information on the structure of the parquet files.


Extracted Ion Chromatgoram File Conversion
------------------------------------------

If you run OpenSWATH and output an XIC file, it will typically be in an sqlite-based format (.sqMass). Similar to the OSW file, this file can become quite large if you have a lot of data. PyProphet provides a way to convert this XIC file to a parquet file, which can be more efficient for large datasets. This can be done using the following command:

.. code-block:: bash
    
    $ pyprophet export parquet --in input.sqMass --out output.parquet