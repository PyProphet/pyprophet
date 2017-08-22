# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import multiprocessing
import random
import sys

import numpy as np
import pandas as pd


def _standard_config():

    config = {}

    config["semi_supervised_learner.use_best"] = True
    config["semi_supervised_learner.stat_best"] = True

    config["delim.in"] = "\t"
    config["delim.out"] = "\t"

    config["multiple_files.merge_results"] = True

    config["target.overwrite"] = True
    config["target.compress_results"] = False

    config["ignore.invalid_score_columns"] = True

    config["apply_scorer"] = None

    config["compute.probabilities"] = False

    config["d_score.cutoff"] = -1000.0

    return config


def set_pandas_print_options():
    # w, h = pd.util.terminal.get_terminal_size()

    # set output options for regression tests on a wide terminal
    pd.set_option('display.width', 100)
    # reduce precision to avoid to sensitive tests because of roundings:
    pd.set_option('display.precision', 6)


class _ConfigHolder(object):

    def __init__(self):
        self.config = _standard_config()

    def update(self, dd):
        self.config.update(dd)

    def get(self, name, default=None):
        if default is not None:
            raise RuntimeError("default value not allowed")
        value = self.config.get(name, default)
        return self._translate(name, value)

    def _translate(self, name, value):
        if name == "num_processes" and value == -1:
            value = multiprocessing.cpu_count()
        return value

    def __getitem__(self, name):
        return self.config[name]

    def __setitem__(self, name, value):
        self.config[name] = value

CONFIG = _ConfigHolder()
