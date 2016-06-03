# needed for headless environment:

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from scipy.stats import gaussian_kde
from numpy import linspace, concatenate


class Protein:

    def __init__(self, name):
        self.peptides = set()
        self.name = name

    def add_peptide(self, peptide):
        self.peptides.update([peptide])

    def get_concat_peptides(self):
        return "".join(self.peptides)


def save_report(report_path, prefix, decoys, targets, top_decoys, top_targets, cutoffs, svalues,
                qvalues, pvalues, lambda_):

    if plt is None:
        raise ImportError("you need matplotlib package to create a report")

    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(hspace=.5)

    plt.subplot(511)
    plt.title(prefix + "\n\nROC")
    plt.xlabel('False Positive Rate (qvalue)')
    plt.ylabel('True Positive Rate (svalue)')

    plt.scatter(qvalues, svalues, s=3)
    plt.plot(qvalues, svalues)

    plt.subplot(512)
    plt.title('d_score Performance')
    plt.xlabel('dscore cutoff')
    plt.ylabel('rates')

    plt.scatter(cutoffs, svalues, color='g', s=3)
    plt.plot(cutoffs, svalues, color='g', label="TPR (svalue)")
    plt.scatter(cutoffs, qvalues, color='r', s=3)
    plt.plot(cutoffs, qvalues, color='r', label="FPR (qvalue)")

    plt.subplot(513)
    plt.title("Top Peak Groups' d_score Distributions")
    plt.xlabel("d_score")
    plt.ylabel("# of groups")
    plt.hist(
        [top_targets, top_decoys], 20, color=['w', 'r'], label=['target', 'decoy'], histtype='bar')
    plt.legend(loc=2)

    plt.subplot(514)
    tdensity = gaussian_kde(top_targets)
    tdensity.covariance_factor = lambda: .25
    tdensity._compute_covariance()
    ddensity = gaussian_kde(top_decoys)
    ddensity.covariance_factor = lambda: .25
    ddensity._compute_covariance()
    xs = linspace(min(concatenate((top_targets, top_decoys))), max(
        concatenate((top_targets, top_decoys))), 200)
    plt.title("Top Peak Groups' d_score Density")
    plt.xlabel("d_score")
    plt.ylabel("density")
    plt.plot(xs, tdensity(xs), color='g', label='target')
    plt.plot(xs, ddensity(xs), color='r', label='decoy')
    plt.legend(loc=2)

    plt.subplot(515)
    if pvalues is not None:
        counts, __, __ = plt.hist(pvalues, bins=40)
        y_max = max(counts)
        plt.plot([lambda_, lambda_], [0, y_max], "r")
        plt.title("histogram pvalues")

    plt.savefig(report_path)

    return cutoffs, svalues, qvalues, top_targets, top_decoys


def mayu_cols():
    interesting_cols = ['run_id', 'transition_group_id', 'Sequence', 'ProteinName', 'm_score',
                        'Charge']
    return interesting_cols


def export_mayu(mayu_cutoff_file, mayu_fasta_file, mayu_csv_file, scored_table, final_stat):

    interesting_cols = mayu_cols()
    # write MAYU CSV input file
    mayu_csv = scored_table.df[scored_table.df["peak_group_rank"] == 1][interesting_cols]
    row_index = [str(i) for i in range(len(mayu_csv.index))]
    mayu_csv['Identifier'] = ("run" + mayu_csv['run_id'].astype('|S10') + "." + row_index
                              + "." + row_index + "." + mayu_csv['Charge'].astype('|S10'))
    mayu_csv['Mod'] = ''
    mayu_csv['m_score'] = 1 - mayu_csv['m_score']
    mayu_csv = mayu_csv[['Identifier', 'Sequence', 'ProteinName', 'Mod', 'm_score']]
    mayu_csv.columns = ['Identifier', 'Sequence', 'Protein', 'Mod', 'MScore']
    mayu_csv.to_csv(mayu_csv_file, sep=",", index=False)

    # write MAYU FASTA input file
    mayu_fasta = scored_table.df[scored_table.df["peak_group_rank"] == 1]
    mayu_fasta_file_out = open(mayu_fasta_file, "w")

    protein_dic = {}
    for entry in mayu_fasta[['ProteinName', 'Sequence']].iterrows():
        peptide = entry[1]['Sequence']
        protein = entry[1]['ProteinName']
        if protein not in protein_dic:
            p = Protein(protein)
            protein_dic[protein] = p
        protein_dic[protein].add_peptide(peptide)

    for k in protein_dic:
        protein = protein_dic[k]
        mayu_fasta_file_out.write(">%s\n" % protein.name)
        mayu_fasta_file_out.write(protein.get_concat_peptides())
        mayu_fasta_file_out.write("\n")

    # write MAYU cutoff input file
    mayu_cutoff = (final_stat.ix[0]['FP'] + final_stat.ix[0]['TN']) / \
                  (final_stat.ix[0]['TP'] + final_stat.ix[0]['FN']
                   + final_stat.ix[0]['FP'] + final_stat.ix[0]['TN'])

    mayu_cutoff_file_out = open(mayu_cutoff_file, "w")
    mayu_cutoff_file_out.write("%s" % mayu_cutoff)

    return True
