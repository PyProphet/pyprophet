
import pyprophet.optimized as o
import numpy as np


def test_rank():

    groups = [1, 1, 1, 1, 1, 2, 3, 3, 3, 0]
    values = [2, 7, 0, 5, 3, 7, 2, 1, 3, 9]

    groups = np.array(groups)
    values = np.array(values, dtype=float)

    ranks = o.rank(groups, values)
    assert list(ranks) == [4, 1, 5, 2, 3, 1, 2, 3, 1, 1], ranks

    # corner cases:

    groups = np.array([1])
    values = np.array([1.0])
    assert list(o.rank(groups, values)) == [1]

    groups = np.array([], dtype=int)
    values = np.array([], dtype=float)
    assert list(o.rank(groups, values)) == []


def _test_match(values):
    values = np.array(values, dtype=float)
    ix = list(o.find_nearest_matches(values, values))
    assert [values[i] for i in ix] == list(values)

    values2 = values + 0.1
    ix2 = list(o.find_nearest_matches(values2, values))
    assert [values2[i] for i in ix] == [values2[i] for i in ix2]
    values2 = values - 0.1
    ix2 = list(o.find_nearest_matches(values2, values))
    assert [values2[i] for i in ix] == [values2[i] for i in ix2]


def test_find_neared_matches():

    ix = o.find_nearest_matches(np.arange(4.0), np.arange(2.0))
    assert list(ix) == [0, 1]

    ix = o.find_nearest_matches(np.arange(2.0), np.arange(4.0))
    assert list(ix) == [0, 1, 1, 1]

    # unsorted with duplicates
    _test_match([2, 7, 0, 5, 3, 7, 2, 1, 3, 9])

    # unsorted without duplicates
    _test_match([7, 0, 5, 2, 1, 3, 9])

    # sorted
    _test_match([1, 2, 5, 6, 9, 11])

    # reverse sorted
    _test_match([11, 9, 6, 5, 2, 1, 0])

    # one element basis
    _test_match([2])

    # two element basis const
    _test_match([2, 2])

    # two element basis asc
    _test_match([2, 3])

    # two element basis desc
    _test_match([3, 2])

    # constant sequence
    _test_match([2, 2, 2])

    # nearly constant sequence 1
    _test_match([3, 2, 2, 2])

    # nearly constant sequence 2
    _test_match([3, 3, 3, 2])

    # nearly constant sequence 3
    _test_match([1, 2, 2, 2])

    # nearly constant sequence 4
    _test_match([1, 1, 1, 2])

def test_count_num_positives():
    assert list(o.count_num_positives(np.array((9.0, 8, 8, 7, 5)))) == [5, 4, 4, 2, 1]
    assert list(o.count_num_positives(np.array((9.0, 8, 8, 7, 7)))) == [5, 4, 4, 2, 2]
    assert list(o.count_num_positives(np.array((9.0, 8, 7, 6, 5)))) == [5, 4, 3, 2, 1]
    assert list(o.count_num_positives(np.array((9.0, 9, 9, 9, 9)))) == [5, 5, 5, 5, 5]
    assert list(o.count_num_positives(np.array((9.0,)))) == [1]
    assert list(o.count_num_positives(np.array(()))) == []
