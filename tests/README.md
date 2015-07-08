README
======

The scripts should be run with `py.test` with installed plugin `pytest-regest`
(see https://pypi.python.org/pypi/pytest-regtest).

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
    $ py.test --regtest-rest tests/test_xyz.py::test_0
````

Which will create a file in `tests/_regtest_outputs/test_xyz.test_0.out` which you should not forget to
commit with `git`.


Later runs like
````
    $ py.test tests/test_xyz.py
````

will then check if the recorded output is still the same.
