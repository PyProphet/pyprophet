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

    CONFIG["final_statistics.fdr_all_pg"] = False
    info["final_statistics.fdr_all_pg"] = """[use all peak groups for score & q-value calculation]"""

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
        """[ignore score columns which only contain NaN or infinity values]"""

    CONFIG["apply_scorer"] = None
    info["apply_scorer"] = r"""[name of *_scorer.bin file of existing classifier]"""

    CONFIG["apply_weights"] = None
    info["apply_weights"] = r"""[name of *_weights.txt file of existing LDA weights]"""

    CONFIG["export.mayu"] = False
    info["export.mayu"] = """[export input files for MAYU]"""

    CONFIG["compute.probabilities"] = False
    info["compute.probabilities"] = """[Compute approximate binned probability values]"""

    CONFIG["d_score.cutoff"] = -1000.0
    info["d_score.cutoff"] = """[Filter output such that only results with a d_score higher than this value are reported]"""

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
