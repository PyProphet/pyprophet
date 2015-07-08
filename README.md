pyprophet
=========

python reimplementation of mProphet algorithm. For more information, see the following publication:

Reiter L, Rinner O, Picotti P, Hüttenhain R, Beck M, Brusniak MY, Hengartner MO, Aebersold R.
*mProphet: automated data processing and statistical validation for large-scale
SRM experiments.* **Nat Methods.** 2011 May;8(5):430-5. [doi:
10.1038/nmeth.1584.](http://dx.doi.org/10.1038/nmeth.1584) Epub 2011 Mar 20.

In short, the algorithm can take targeted proteomics data, learn a linear
separation between true signal and the noise signal and then compute a q-value
(false discovery rate) to achieve experiment-wide cutoffs.  


Installation
============

Install *pyprophet* from Python package index:

````
    $ pip install numpy
    $ pip install pyprophet
````

or:

````
   $ easy_install numpy
   $ easy_install pyprophet
````


Running pyprophet
=================

*pyoprophet* is not only a Python package, but also a command line tool:

````
   $ pyprophet --help
````

or:

````
   $ pyprophet --delim=tab tests/test_data.txt
````


Running tests
=============

The *pyprophet* tests are best executed using `py.test`, to run the tests use:

````
    $ pip install pytest
    $ py.test tests
````

