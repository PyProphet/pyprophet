# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import multiprocessing


CONFIG = dict(is_test=0)


def standard_config(n_cpus=1):
    info = dict(is_test="[switches randomness off]")

    lambda_ = 0.4

    if n_cpus == -1:
        n_cpus = multiprocessing.cpu_count()

    CONFIG["xeval.fraction"] = 0.5
    CONFIG["xeval.num_iter"] = 5

    CONFIG["semi_supervised_learner.initial_fdr"] = 0.15
    CONFIG["semi_supervised_learner.initial_lambda"] = lambda_

    CONFIG["semi_supervised_learner.iteration_fdr"] = 0.02
    CONFIG["semi_supervised_learner.iteration_lambda"] = lambda_

    CONFIG["semi_supervised_learner.num_iter"] = 5

    CONFIG["final_statistics.lambda"] = lambda_

    CONFIG["num_processes"] = n_cpus
    info["num_processes"] = "[-1 means 'all available cpus']"

    CONFIG["delim.in"] = "tab"
    info["delim.in"] = r"""[you can eg use 'tab' or ',']"""

    CONFIG["delim.out"] = "tab"
    info["delim.out"] = r"""[you can eg use 'tab' or ',']"""

    CONFIG["target.dir"] = None
    CONFIG["target.overwrite"] = 0

    CONFIG["ignore.invalid_score_columns"] = False
    info["ignore.invalid_score_columns"] =\
        """ignore score columns which conly contain NaN of infinity values"""

    CONFIG["apply"] = None
    info["apply"] = r"""[name of *_scorer.bin file of existing classifier]"""

    return CONFIG, info

CONFIG, __ = standard_config()


def fix_config_types(dd):
    for k in ["xeval.num_iter",
              "semi_supervised_learner.num_iter",
              "is_test",
              "ignore.invalid_score_columns",
              "target.overwrite",
              "num_processes"]:
        dd[k] = int(dd[k])

    for k in ["xeval.fraction",
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
