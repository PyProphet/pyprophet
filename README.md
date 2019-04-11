[![Build Status](https://travis-ci.org/PyProphet/pyprophet.svg?branch=master)](https://travis-ci.org/PyProphet/pyprophet) [![Project Stats](https://www.openhub.net/p/PyProphet/widgets/project_thin_badge.gif)](https://www.openhub.net/p/PyProphet)

PyProphet
=========

PyProphet: Semi-supervised learning and scoring of OpenSWATH results.

PyProphet is a Python re-implementation of the mProphet algorithm [1] optimized for SWATH-MS data acquired by data-independent acquisition (DIA). The algorithm was originally published in [2] and has since been extended to support new data types and analysis modes [3,4].

Please consult the [OpenSWATH website](http://openswath.org) for usage instructions and help.

1. Reiter L, Rinner O, Picotti P, Hüttenhain R, Beck M, Brusniak MY, Hengartner MO, Aebersold R.
*mProphet: automated data processing and statistical validation for large-scale
SRM experiments.* **Nat Methods.** 2011 May;8(5):430-5. [doi:
10.1038/nmeth.1584.](http://dx.doi.org/10.1038/nmeth.1584) Epub 2011 Mar 20.

2. Teleman J, Röst HL, Rosenberger G, Schmitt U, Malmström L, Malmström J, Levander F.
*DIANA--algorithmic improvements for analysis of data-independent acquisition MS data.* **Bioinformatics.** 2015 Feb 15;31(4):555-62. [doi: 10.1093/bioinformatics/btu686.](http://dx.doi.org/10.1093/bioinformatics/btu686) Epub 2014 Oct 27.

3. Rosenberger G, Liu Y, Röst HL, Ludwig C, Buil A, Bensimon A, Soste M, Spector TD, Dermitzakis ET, Collins BC, Malmström L, Aebersold R. *Inference and quantification of peptidoforms in large sample cohorts by SWATH-MS.* **Nat Biotechnol** 2017 Aug;35(8):781-788. [doi: 10.1038/nbt.3908.](http://dx.doi.org/10.1038/nbt.3908) Epub 2017 Jun 12.

4. Rosenberger G, Bludau I, Schmitt U, Heusel M, Hunter CL, Liu Y, MacCoss MJ, MacLean BX, Nesvizhskii AI, Pedrioli PGA, Reiter L, Röst HL, Tate S, Ting YS, Collins BC, Aebersold R.
*Statistical control of peptide and protein error rates in large-scale targeted data-independent acquisition analyses.* **Nat Methods.** 2017 Sep;14(9):921-927. [doi: 10.1038/nmeth.4398.](http://dx.doi.org/10.1038/nmeth.4398) Epub 2017 Aug 21. 

Installation
============

We strongly advice to install PyProphet in a Python [*virtualenv*](https://virtualenv.pypa.io/en/stable/). PyProphet is compatible with Python 3.

Install the development version of *pyprophet* from GitHub:

````
    $ pip install git+https://github.com/PyProphet/pyprophet.git@master
````

Install the stable version of *pyprophet* from the Python Package Index (PyPI):

````
    $ pip install pyprophet
````

Running pyprophet
=================

*pyprophet* is not only a Python package, but also a command line tool:

````
   $ pyprophet --help
````

or:

````
   $ pyprophet score --in=tests/test_data.txt
````

Docker
=================

PyProphet is also available from Docker (automated builds):

Pull the development version of *pyprophet* from DockerHub (synced with GitHub):

````
    $ docker pull pyprophet/master:latest
````

Pull the stable version (e.g. 2.1.2) of *pyprophet* from DockerHub (synced with releases):

````
    $ docker pull pyprophet/pyprophet:2.1.2
````

Running tests
=============

The *pyprophet* tests are best executed using `py.test` and the `pytest-regtest` plugin:

````
    $ pip install pytest
    $ pip install pytest-regtest
    $ py.test ./tests
````

