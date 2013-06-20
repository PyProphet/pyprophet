#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")


import multiprocessing

def standard_config(n_cpus=1):
    config = dict(is_test=0)

    lambda_ = 0.4

    if n_cpus == -1:
        n_cpus = multiprocessing.cpu_count()

    config["xeval.fraction"] = 0.5
    config["xeval.num_iter"] = 5

    config["semi_supervised_learner.initial_fdr"] = 0.15
    config["semi_supervised_learner.initial_lambda"] = lambda_

    config["semi_supervised_learner.iteration_fdr"] = 0.02
    config["semi_supervised_learner.iteration_lambda"] =  lambda_

    config["final_statistics.lambda"] =  lambda_

    config["xeval.num_processes"] = n_cpus

    config["delim.in"] = ","
    config["delim.out"] = ","

    config["target.dir"] = None
    config["target.overwrite"] = False

    return config
