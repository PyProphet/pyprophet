# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import multiprocessing
import random
import sys

import numpy as np
import pandas as pd

# Parameter transformation functions
def transform_pi0_lambda(ctx, param, value):
  if value[1] == 0 and value[2] == 0:
      pi0_lambda = value[0]
  elif 0 <= value[0] < 1 and value[0] <= value[1] <= 1 and 0 < value[2] < 1:
      pi0_lambda = np.arange(value[0], value[1], value[2])
  else:
      sys.exit('Error: Wrong input values for pi0_lambda. pi0_lambda must be within [0,1).')
  return(pi0_lambda)

def transform_threads(ctx, param, value):
    if value == -1:
        value = multiprocessing.cpu_count()
    return(value)

def transform_random_seed(ctx, param, value):
    if value is None:
        value = np.random.randint(0, sys.maxsize)
    return(value)

def transform_subsample_ratio(ctx, param, value):
    if value < 0 or value > 1:
      sys.exit('Error: Wrong input values for subsample_ratio. subsample_ratio must be within [0,1].')
    return(value)

def set_parameters(xeval_fraction, xeval_iterations, initial_fdr, iteration_fdr, ss_iterations, group_id, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, lfdr_truncate, lfdr_monotone, lfdr_transformation, lfdr_adj, lfdr_eps, tric_chromprob, threads, test, random_seed):

    options = dict()

    # Semi-supervised learning
    options['xeval.fraction'] = xeval_fraction
    options['xeval.num_iter'] = xeval_iterations
    options['semi_supervised_learner.initial_fdr'] = initial_fdr
    options['semi_supervised_learner.initial_lambda'] = pi0_lambda
    options['semi_supervised_learner.iteration_fdr'] = iteration_fdr
    options['semi_supervised_learner.iteration_lambda'] = pi0_lambda
    options['semi_supervised_learner.num_iter'] = ss_iterations

    # Statistics
    options['group_id'] = group_id
    options['final_statistics.emp_p'] = parametric
    options['final_statistics.pfdr'] = pfdr
    options['final_statistics.lambda'] = pi0_lambda
    options['final_statistics.pi0_method'] = pi0_method
    options['final_statistics.pi0_smooth_df'] = pi0_smooth_df
    options['final_statistics.pi0_smooth_log_pi0'] = pi0_smooth_log_pi0
    options['final_statistics.lfdr_trunc'] = lfdr_truncate
    options['final_statistics.lfdr_monotone'] = lfdr_monotone
    options['final_statistics.lfdr_transf'] = lfdr_transformation
    options['final_statistics.lfdr_adj'] = lfdr_adj
    options['final_statistics.lfdr_eps'] = lfdr_eps

    # TRIC
    options['tric_chromprob'] = tric_chromprob

    # Processing
    options['num_processes'] = threads
    options['is_test'] = test
    options['random_seed'] = random_seed

    CONFIG.update(options)

def set_pandas_print_options():
    # w, h = pd.util.terminal.get_terminal_size()

    # set output options for regression tests on a wide terminal
    pd.set_option('display.width', 100)
    # reduce precision to avoid to sensitive tests because of roundings:
    pd.set_option('display.precision', 6)


class _ConfigHolder(object):

    def __init__(self):
        self.config = {'num_processes': 1}

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
