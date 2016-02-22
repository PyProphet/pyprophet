# encoding: utf-8
from __future__ import print_function

from pyprophet.data_handling import check_for_unique_blocks


def test_ok():

    assert check_for_unique_blocks([1]) is True
    assert check_for_unique_blocks([1, 1]) is True
    assert check_for_unique_blocks([1, 2]) is True
    assert check_for_unique_blocks([1, 2, 3, 4, 5]) is True
    assert check_for_unique_blocks([1, 2, 3, 4, 5, 5]) is True
    assert check_for_unique_blocks([1, 1, 1]) is True
    assert check_for_unique_blocks([1, 1, 1, 2]) is True
    assert check_for_unique_blocks([1, 2, 2, 2]) is True
    assert check_for_unique_blocks([1, 1, 2, 2, 3, 3, 4, 4]) is True
    assert check_for_unique_blocks([1, 1, 2, 2, 3, 3, 4, 4, 5]) is True

    assert check_for_unique_blocks(map(str, [1])) is True
    assert check_for_unique_blocks(map(str, [1, 1])) is True
    assert check_for_unique_blocks(map(str, [1, 2])) is True
    assert check_for_unique_blocks(map(str, [1, 2, 3, 4, 5])) is True
    assert check_for_unique_blocks(map(str, [1, 2, 3, 4, 5, 5])) is True
    assert check_for_unique_blocks(map(str, [1, 1, 1])) is True
    assert check_for_unique_blocks(map(str, [1, 1, 1, 2])) is True
    assert check_for_unique_blocks(map(str, [1, 2, 2, 2])) is True
    assert check_for_unique_blocks(map(str, [1, 1, 2, 2, 3, 3, 4, 4])) is True
    assert check_for_unique_blocks(map(str, [1, 1, 2, 2, 3, 3, 4, 4, 5])) is True


def test_not_ok():

    assert check_for_unique_blocks([1, 2, 1]) is False
    assert check_for_unique_blocks([1, 2, 3, 4, 1]) is False
    assert check_for_unique_blocks([1, 2, 3, 4, 5, 1]) is False
    assert check_for_unique_blocks([1, 1, 2, 2, 1]) is False
    assert check_for_unique_blocks([1, 1, 1, 2, 1]) is False
    assert check_for_unique_blocks([1, 2, 2, 2, 1]) is False
    assert check_for_unique_blocks([1, 1, 2, 2, 3, 3, 4, 4, 3]) is False
    assert check_for_unique_blocks([1, 1, 2, 2, 3, 3, 4, 4, 5, 4]) is False

    assert check_for_unique_blocks(map(str, [1, 2, 1])) is False
    assert check_for_unique_blocks(map(str, [1, 2, 3, 4, 1])) is False
    assert check_for_unique_blocks(map(str, [1, 2, 3, 4, 5, 1])) is False
    assert check_for_unique_blocks(map(str, [1, 1, 2, 2, 1])) is False
    assert check_for_unique_blocks(map(str, [1, 1, 1, 2, 1])) is False
    assert check_for_unique_blocks(map(str, [1, 2, 2, 2, 1])) is False
    assert check_for_unique_blocks(map(str, [1, 1, 2, 2, 3, 3, 4, 4, 3])) is False
    assert check_for_unique_blocks(map(str, [1, 1, 2, 2, 3, 3, 4, 4, 5, 4])) is False
