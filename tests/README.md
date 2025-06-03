README
======

The scripts should be run with `py.test` (>=3.4.1) with installed plugin `pytest-regest`
(>=1.0.14 see https://pypi.python.org/pypi/pytest-regtest).

The plugin allows recording of approved output so that later test runs will check if
the output is still the same. It is simple to use as you can see in `test_via_regression.py`.

In order to record output you have to use the `regtest` fixture like in the following example.
This `regtest` behaves like a file handle, so you can write to it as usual:

````
    def test_0(regtest):
        print >> regtest, "this is the recorded output"
````

If you now create a new test function `test_0` in a file `test_xyz.py`, first run

````
    $ py.test tests/test_xyz.py::test_0
````

which will show you the yet not approved output. You can approve this output using

````
    $ py.test --regtest-reset tests/test_xyz.py::test_0
````

Which will create a file in `tests/_regtest_outputs/test_xyz.test_0.out` which you should not forget to
commit with `git`.


Later runs like
````
    $ py.test tests/test_xyz.py
````

will then check if the recorded output is still the same.

If you want to only run certain tests and certain combination of fixture paramaters, you can use the `-k` option of `py.test`:

````
# Run all combinations for OSW input only
pytest -k "test_ipf_scoring and osw"

# Run specific parameter combination
pytest -k "test_ipf_scoring and ms1_on and ms2_off and h0_on"

# Run all peptide tests for OSW input
pytest -k "test_peptide_levels and osw"

# Run protein tests for specific context
pytest -k "test_protein_levels and experiment-wide"

# Run all tests for parquet input
pytest -k "parquet"
````

To run tests in parallel, you need the `pytest-xdist` plugin. You can then use the `-n` option to specify the number of parallel workers:

````
pytest -n 4
````