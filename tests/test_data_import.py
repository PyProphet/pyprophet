from nose.tools import *

import pyprophet.io
import pyprophet.data_import

import os.path

def data_path(fname):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", fname)


def test_prepare_data_table():
    load_from = data_path("testfile.csv")
    table = pyprophet.io.read_csv(load_from, sep="\t")
    assert_equals(len(table), 9165)
    assert_equals(len(table.columns.values), 41)

    (var_columns,
     df_orig, prepared) =  pyprophet.data_import.prepare_data_table(table)

    assert_equals(len(var_columns), 16)
    assert_equals(len(df_orig), 9165)
    assert_equals(len(prepared), 9165)

    one_run = prepared[prepared["transition_group_id"] == "100_run0"]
    assert_sequence_equal(list(one_run.peak_id), [0, 2, 1, 4, 3, 5, 6])
    assert_sequence_equal(list(one_run.peak_rank), range(7))



