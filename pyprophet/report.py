import pdb
# needed for headless environment:
import matplotlib
matplotlib.use('Agg')

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv


def save_report(report_path, in_file_name, scored_table, final_stat):
    cutoffs = final_stat["cutoff"].values
    svalues = final_stat["svalue"].values
    qvalues = final_stat["qvalue"].values

    tops = scored_table[scored_table["peak_group_rank"] == 1]
    top_decoys = tops[tops["decoy"] == 1]["d_score"].values
    top_target = tops[tops["decoy"] == 0]["d_score"].values

    # thanks to lorenz blum for the plotting code below:

    plt.figure(figsize=(8.27, 11.69))
    plt.subplots_adjust(hspace=.5)

    plt.subplot(311)
    plt.title(in_file_name + "\n\nROC")
    plt.xlabel('False Positive Rate (qvalue)')
    plt.ylabel('True Positive Rate (svalue)')

    plt.scatter(qvalues, svalues, s=3)
    plt.plot(qvalues, svalues)

    plt.subplot(312)
    plt.title('d_score Performance')
    plt.xlabel('dscore cutoff')
    plt.ylabel('rates')

    plt.scatter(cutoffs, svalues, color='g', s=3)
    plt.plot(cutoffs, svalues, color='g', label="TPR (svalue)")
    plt.scatter(cutoffs, qvalues, color='r', s=3)
    plt.plot(cutoffs, qvalues, color='r', label="FPR (qvalue)")

    plt.subplot(313)
    plt.title("Top Peak Groups' d-score Distributions")
    plt.xlabel("d_score")
    plt.ylabel("# of groups")
    plt.hist(
        [top_target, top_decoys], 20, color=['w', 'r'], label=['target', 'decoy'], histtype='bar')
    plt.legend(loc=2)

    plt.savefig(report_path)

    return cutoffs, svalues, qvalues, top_target, top_decoys
