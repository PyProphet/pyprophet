# needed for headless environment:

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from scipy.stats import gaussian_kde
from numpy import linspace, concatenate, around

from config import CONFIG, set_pandas_print_options


class Protein:

    def __init__(self, name):
        self.peptides = set()
        self.name = name

    def add_peptide(self, peptide):
        self.peptides.update([peptide])

    def get_concat_peptides(self):
        return "".join(self.peptides)


def save_report(report_path, prefix, decoys, targets, top_decoys, top_targets, cutoffs, svalues,
                qvalues, pvalues, pi0):

    if plt is None:
        raise ImportError("you need matplotlib package to create a report")

    plt.figure(figsize=(10, 15))
    plt.subplots_adjust(hspace=.5)

    plt.subplot(321)
    plt.title(prefix + "\n\nROC")
    plt.xlabel('false positive rate (q-value)')
    plt.ylabel('true positive rate (s-value)')

    plt.scatter(qvalues, svalues, s=3)
    plt.plot(qvalues, svalues)

    plt.subplot(322)
    plt.title('d-score performance')
    plt.xlabel('d-score cutoff')
    plt.ylabel('rates')

    plt.scatter(cutoffs, svalues, color='g', s=3)
    plt.plot(cutoffs, svalues, color='g', label="TPR (s-value)")
    plt.scatter(cutoffs, qvalues, color='r', s=3)
    plt.plot(cutoffs, qvalues, color='r', label="FPR (q-value)")

    plt.subplot(323)
    plt.title(CONFIG.get("group_id") + " d-score distributions")
    plt.xlabel("d-score")
    plt.ylabel("# of " + CONFIG.get("group_id"))
    plt.hist(
        [top_targets, top_decoys], 20, color=['g', 'r'], label=['target', 'decoy'], histtype='bar')
    plt.legend(loc=2)

    plt.subplot(324)
    tdensity = gaussian_kde(top_targets)
    tdensity.covariance_factor = lambda: .25
    tdensity._compute_covariance()
    ddensity = gaussian_kde(top_decoys)
    ddensity.covariance_factor = lambda: .25
    ddensity._compute_covariance()
    xs = linspace(min(concatenate((top_targets, top_decoys))), max(
        concatenate((top_targets, top_decoys))), 200)
    plt.title(CONFIG.get("group_id") + " d-score densities")
    plt.xlabel("d-score")
    plt.ylabel("density")
    plt.plot(xs, tdensity(xs), color='g', label='target')
    plt.plot(xs, ddensity(xs), color='r', label='decoy')
    plt.legend(loc=2)

    plt.subplot(325)
    if pvalues is not None:
        counts, __, __ = plt.hist(pvalues, bins=40, normed=True)
        plt.plot([0, 1], [pi0['pi0'], pi0['pi0']], "r")
        plt.title("p-value density histogram: $\pi_0$ = " + str(around(pi0['pi0'], decimals=3)))
        plt.xlabel("p-value")
        plt.ylabel("density histogram")

    if pi0['pi0_smooth'] is not False:
        plt.subplot(326)
        plt.plot(pi0['lambda_'], pi0['pi0_lambda'], ".")
        plt.plot(pi0['lambda_'], pi0['pi0_smooth'], "r")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title("$\pi_0$ smoothing fit plot")
        plt.xlabel("lambda")
        plt.ylabel("$\pi_0$($\lambda$)")

    plt.savefig(report_path)

    return cutoffs, svalues, qvalues, top_targets, top_decoys
