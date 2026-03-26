# encoding: utf-8
from __future__ import print_function

import numpy as np

from pyprophet.scoring.data_handling import check_for_unique_blocks
from pyprophet.scoring.data_handling import _to_writable_c_array


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


def test_to_writable_c_array_handles_read_only_input():
    source = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    source.setflags(write=False)

    out = _to_writable_c_array(source, np.float32)

    assert out.flags.writeable is True
    assert out.flags.c_contiguous is True
    out[0] = 42.0
    assert out[0] == 42.0
