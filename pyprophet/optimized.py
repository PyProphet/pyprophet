#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

from numba import jit, i8, f8
import numpy as np


@jit(restype=i8[:], argtypes=(f8[:], f8[:]))
def find_nearest_matches(basis, sample_points):
    num_basis = len(basis)
    num_samples = len(sample_points)
    result = np.zeros((num_samples,), dtype=int)
    for i in range(num_samples):
        sp_i = sample_points[i]
        best_dist = abs(basis[0] - sp_i)
        best_j = 0
        for j in range(1, num_basis):
            dist = abs(basis[j] - sp_i)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        result[i] = best_j
    return result


@jit(restype=i8[:], argtypes=[i8[:], f8[:]])
def find_top_ranked(tg_ids, scores):
    n = len(scores)
    flags = np.zeros((n,), dtype=int)
    current_max = scores[0]
    current_imax = 0
    current_tg_id = tg_ids[0]
    current_write_i=0
    for i in range(len(tg_ids)):
        id_ = tg_ids[i]
        sc = scores[i]
        if id_ != current_tg_id:
            current_tg_id = id_
            flags[current_imax] = 1
            current_write_i += 1
            current_max = sc
            current_imax = i
            continue
        if sc > current_max:
            current_max = sc
            current_imax = i
    flags[current_imax] = 1
    return flags
