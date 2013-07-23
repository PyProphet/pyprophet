#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")


import multiprocessing

def standard_config(n_cpus=1):
    config = dict(is_test=0)
    info  = dict(is_test="[switches randomness off]")

    lambda_ = 0.4

    if n_cpus == -1:
        n_cpus = multiprocessing.cpu_count()

    config["xeval.fraction"] = 0.5
    config["xeval.num_iter"] = 5

    config["semi_supervised_learner.initial_fdr"] = 0.15
    config["semi_supervised_learner.initial_lambda"] = lambda_

    config["semi_supervised_learner.iteration_fdr"] = 0.02
    config["semi_supervised_learner.iteration_lambda"] =  lambda_

    config["semi_supervised_learner.num_iter"] = 5

    config["final_statistics.lambda"] =  lambda_

    config["xeval.num_processes"] = n_cpus
    info["xeval.num_processes"] = "[-1 means 'all available cpus']"


    config["delim.in"] = ","
    info["delim.in"] = r"""[you can use 'tab' or "\t" (with these quote marks)]"""

    config["delim.out"] = ","
    info["delim.out"] = r"""[you can use 'tab' or "\t" (with these quote marks)]"""

    config["target.dir"] = None
    config["target.overwrite"] = 0

    return config, info

def fix_config_types(dd):
    for k in ["xeval.num_iter",
              "semi_supervised_learner.num_iter",
              "is_test",
              "target.overwrite",
              "xeval.num_processes"]:
        dd[k] = int(dd[k])

    for k in ["xeval.fraction",
              "semi_supervised_learner.initial_fdr",
              "semi_supervised_learner.initial_lambda",
              "semi_supervised_learner.iteration_lambda",
              "semi_supervised_learner.initial_fdr",
              "semi_supervised_learner.iteration_fdr",
              "final_statistics.lambda"]:
        dd[k] = float(dd[k])
