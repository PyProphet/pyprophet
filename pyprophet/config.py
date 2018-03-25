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


def transform_subsample_ratio(ctx, param, value):
    if value < 0 or value > 1:
      sys.exit('Error: Wrong input values for subsample_ratio. subsample_ratio must be within [0,1].')
    return(value)

