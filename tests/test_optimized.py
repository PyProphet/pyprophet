
import pyprophet.optimized as o
import numpy as np


def test_rank():

    groups = [1, 1, 1, 1, 1, 2, 3, 3, 3, 0]
    values = [2, 7, 0, 5, 3, 7, 2, 1, 3, 9]

    groups = np.array(groups, dtype=np.int64)
    values = array32(values)

    ranks = o.rank(groups, values)
    assert list(ranks) == [4, 1, 5, 2, 3, 1, 2, 3, 1, 1], ranks

    # corner cases:

    groups = np.array([1], dtype=np.int64)
    values = array32([1.0])
    assert list(o.rank(groups, values)) == [1]

    groups = np.array([], dtype=np.int64)
    values = array32([])
    assert list(o.rank(groups, values)) == []


def test_rank32():

    groups = [1, 1, 1, 1, 1, 2, 3, 3, 3, 0]
    values = [2, 7, 0, 5, 3, 7, 2, 1, 3, 9]

    groups = np.array(groups, dtype=np.uint32)
    values = array32(values)

    ranks = o.rank32(groups, values)
    assert list(ranks) == [4, 1, 5, 2, 3, 1, 2, 3, 1, 1], ranks

    # corner cases:

    groups = np.array([1], dtype=np.uint32)
    values = array32([1.0])
    assert list(o.rank32(groups, values)) == [1]

    groups = np.array([], dtype=np.uint32)
    values = array32([])
    assert list(o.rank32(groups, values)) == []


def test_single_chromatogram_hypothesis_fast():

    prior_chrom_null = 0.2
    prior_pg = 0.1

    probabilities = array64([.7])  # probability that the peaks are false

    result = o.single_chromatogram_hypothesis_fast(probabilities, prior_chrom_null, prior_pg)
    result_h0 = result[0]
    result = result[1:]

    np.testing.assert_array_almost_equal(result, [0.17647059])
    np.testing.assert_array_almost_equal([result_h0], [0.823529411765])

    probabilities = array64([0.5, 0.7, 0.1, 0.01])  # probability that the peaks are false

    result = o.single_chromatogram_hypothesis_fast(probabilities, prior_chrom_null, prior_pg)
    result_h0 = result[0]
    result = result[1:]

    np.testing.assert_array_almost_equal(result, [0.00897436, 0.00384615, 0.08076923, 0.88846154])
    np.testing.assert_array_almost_equal([result_h0], [0.0179487179487])


def _test_match(values):
    values = array32(values)
    ix = list(o.find_nearest_matches(values, values))
    assert [values[i] for i in ix] == list(values)

    values2 = values + 0.1
    ix2 = list(o.find_nearest_matches(values2, values))
    assert [values2[i] for i in ix] == [values2[i] for i in ix2]
    values2 = values - 0.1
    ix2 = list(o.find_nearest_matches(values2, values))
    assert [values2[i] for i in ix] == [values2[i] for i in ix2]


def test_find_neared_matches():

    ix = o.find_nearest_matches(arange32(4.0), arange32(2.0))
    assert list(ix) == [0, 1]

    ix = o.find_nearest_matches(arange32(2.0), arange32(4.0))
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


def array32(values):
    return np.array(values, dtype=np.float32)


def array64(values):
    return np.array(values, dtype=np.float64)


def arange32(up_to):
    return np.arange(up_to, dtype=np.float32)


def test_count_num_positives():
    assert list(o.count_num_positives(array64((9.0, 8, 8, 7, 5)))) == [5, 4, 4, 2, 1]
    assert list(o.count_num_positives(array64((9.0, 8, 8, 7, 7)))) == [5, 4, 4, 2, 2]
    assert list(o.count_num_positives(array64((9.0, 8, 7, 6, 5)))) == [5, 4, 3, 2, 1]
    assert list(o.count_num_positives(array64((9.0, 9, 9, 9, 9)))) == [5, 5, 5, 5, 5]
    assert list(o.count_num_positives(array64((9.0,)))) == [1]
    assert list(o.count_num_positives(array64(()))) == []


def _test_find_neared_matches_fuzzy():
    for l in range(1, 100):
        for i in range(100):

            basis = np.random.random((l,))
            basis.sort()

            search = np.random.random((l,))

            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)

            basis = basis[::-1]

            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)

            search = basis

            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)

            basis = basis[::-1]

            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)

            # introduct dupplicates
            basis = np.hstack((basis, basis))
            basis.sort()
            search = np.random.random((l,))
            search = basis
            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)

            basis = basis[::-1]
            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)

            # introduct more dupplicates
            basis = np.hstack((basis, basis))
            basis.sort()
            search = np.random.random((l,))
            search = basis
            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)

            basis = basis[::-1]
            tobe = o.find_nearest_matches(basis, search, 0)
            optim = o.find_nearest_matches(basis, search)
            assert np.all(tobe == optim)



