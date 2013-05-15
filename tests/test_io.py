from nose.tools import *

import pyprophet.io

import os.path

def data_path(fname):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", fname)


def test_read():
    load_from = data_path("testfile.csv")
    table = pyprophet.io.read_csv(load_from, sep="\t")
    assert_equals(len(table), 9165)
    assert_equals(len(table.columns.values), 41)


