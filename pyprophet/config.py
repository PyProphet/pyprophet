# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import multiprocessing
import random
import sys

import pandas as pd


def _standard_config(n_cpus=1):

    config = {}
    info = {}

    config["is_test"] = 0
    info["is_test"] = "[switches randomness off]"

    config["random_seed"] = None
    info[
        "random_seed"] = "[seed for pseude random generator. can be used to get reproducable results]"

    config["xeval.fraction"] = 0.5
    config["xeval.num_iter"] = 5

    lambda_ = 0.4

    config["semi_supervised_learner.initial_fdr"] = 0.15
    config["semi_supervised_learner.initial_lambda"] = lambda_

    config["semi_supervised_learner.iteration_fdr"] = 0.02
    config["semi_supervised_learner.iteration_lambda"] = lambda_

    config["semi_supervised_learner.num_iter"] = 5

    config["semi_supervised_learner.use_best"] = 0
    config["semi_supervised_learner.stat_best"] = 0

    info[
        "semi_supervised_learner.use_best"] = """[use only weights from last iteration for final classifier]"""
    info["semi_supervised_learner.stat_best"] = """[use only stats from final for statistics]"""

    config["final_statistics.lambda"] = lambda_

    config["final_statistics.emp_p"] = False
    info["final_statistics.emp_p"] = """[use empirical p-values for scoring]"""

    config["final_statistics.pfdr"] = False
    info["final_statistics.pfdr"] = """[compute positive FDR (pFDR) instead of FDR]"""

    config["num_processes"] = n_cpus
    info["num_processes"] = "[-1 means 'all available cpus']"

    config["delim.in"] = "tab"
    info["delim.in"] = r"""[you can eg use 'tab' or ',']"""

    config["delim.out"] = "tab"
    info["delim.out"] = r"""[you can eg use 'tab' or ',']"""

    config["multiple_files.merge_results"] = 0
    info["multiple_files.merge_results"] = r"""[create one merged result table over all inputs]"""

    config["target.dir"] = None
    config["target.prefix"] = None
    config["target.overwrite"] = 0
    config["target.compress_results"] = 0
    info["target.compress_results"] = """[remove var_ and main_ columns in output files]"""

    config["ignore.invalid_score_columns"] = False
    info["ignore.invalid_score_columns"] =\
        """[ignore score columns which only contain NaN or infinity values]"""

    config["apply_scorer"] = None
    info["apply_scorer"] = r"""[name of *_scorer.bin file of existing classifier]"""

    config["apply_weights"] = None
    info["apply_weights"] = r"""[name of *_weights.txt file of existing LDA weights]"""

    config["export.mayu"] = False
    info["export.mayu"] = """[export input files for MAYU]"""

    config["compute.probabilities"] = False
    info["compute.probabilities"] = """[Compute approximate binned probability values]"""

    config["d_score.cutoff"] = -1000.0
    info[
        "d_score.cutoff"] = """[Filter output such that only results with a d_score higher than this value are reported]"""

    config["out_of_core"] = 0
    info["out_of_core"] = """[handle large data files by slower out of core computation]"""

    config["out_of_core.sampling_rate"] = 0.1
    info[
        "out_of_core.sampling_rate"] = """[handle large data files by random sampling. this value is between 0.0 and 1.0]"""

    return config, info


def _fix_config_types(dd):
    for k in ["xeval.num_iter",
              "semi_supervised_learner.num_iter",
              "semi_supervised_learner.stat_best",
              "semi_supervised_learner.use_best",
              "is_test",
              "ignore.invalid_score_columns",
              "target.overwrite",
              "out_of_core",
              "num_processes",
              "target.compress_results",
              ]:
        dd[k] = int(dd[k])

    for k in ["xeval.fraction",
              "d_score.cutoff",
              "semi_supervised_learner.initial_fdr",
              "semi_supervised_learner.initial_lambda",
              "semi_supervised_learner.iteration_lambda",
              "semi_supervised_learner.initial_fdr",
              "semi_supervised_learner.iteration_fdr",
              "out_of_core.sampling_rate",
              "final_statistics.lambda",
              ]:
        dd[k] = float(dd[k])

    if dd["delim.in"] == "tab":
        dd["delim.in"] = "\t"

    if dd["delim.out"] == "tab":
        dd["delim.out"] = "\t"

    if dd["random_seed"] is not None:
        dd["random_seed"] = int(dd["random_seed"])
    else:
        dd["random_seed"] = random.randint(0, sys.maxint)


def set_pandas_print_options():
    # w, h = pd.util.terminal.get_terminal_size()

    # set output options for regression tests on a wide terminal
    pd.set_option('display.width', 100)
    # reduce precision to avoid to sensitive tests because of roundings:
    pd.set_option('display.precision', 6)


class _ConfigHolder(object):

    def __init__(self):
        self.config, self.info = _standard_config()

    def update(self, dd):
        self.config.update(dd)
        _fix_config_types(self.config)

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
        _fix_config_types(self.config)


CONFIG = _ConfigHolder()
