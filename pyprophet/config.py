# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import multiprocessing

import pandas as pd


def _standard_config(n_cpus=1):

    config = dict(is_test=0)
    info = dict(is_test="[switches randomness off]")

    lambda_ = 0.4

    if n_cpus == -1:
        n_cpus = multiprocessing.cpu_count()

    config["xeval.fraction"] = 0.5
    config["xeval.num_iter"] = 5

    config["semi_supervised_learner.initial_fdr"] = 0.15
    config["semi_supervised_learner.initial_lambda"] = lambda_

    config["semi_supervised_learner.iteration_fdr"] = 0.02
    config["semi_supervised_learner.iteration_lambda"] = lambda_

    config["semi_supervised_learner.num_iter"] = 5

    config["final_statistics.lambda"] = lambda_

    config["final_statistics.fdr_all_pg"] = False
    info["final_statistics.fdr_all_pg"] = """[use all peak groups for score & q-value calculation]"""

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
    info["d_score.cutoff"] = """[Filter output such that only results with a d_score higher than this value are reported]"""

    return config, info


def _fix_config_types(dd):
    for k in ["xeval.num_iter",
              "semi_supervised_learner.num_iter",
              "is_test",
              "ignore.invalid_score_columns",
              "target.overwrite",
              "num_processes"]:
        dd[k] = int(dd[k])

    for k in ["xeval.fraction",
              "d_score.cutoff",
              "semi_supervised_learner.initial_fdr",
              "semi_supervised_learner.initial_lambda",
              "semi_supervised_learner.iteration_lambda",
              "semi_supervised_learner.initial_fdr",
              "semi_supervised_learner.iteration_fdr",
              "final_statistics.lambda"]:
        dd[k] = float(dd[k])

    if dd["delim.in"] == "tab":
        dd["delim.in"] = "\t"

    if dd["delim.out"] == "tab":
        dd["delim.out"] = "\t"


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
        return self.config.get(name, default)

    def __getitem__(self, name):
        return self.config[name]

    def __setitem__(self, name, value):
        self.config[name] = value
        _fix_config_types(self.config)


CONFIG = _ConfigHolder()

