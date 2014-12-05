# needed for headless environment:
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy import linspace, concatenate

from config import CONFIG

class Protein:

    def __init__(self, name):
        self.peptides = set()
        self.name = name

    def add_peptide(self, peptide):
        self.peptides.update([peptide])

    def get_concat_peptides(self):
        return "".join(self.peptides)


def save_report(report_path, in_file_name, scored_table, final_stat):

    cutoffs = final_stat["cutoff"].values
    svalues = final_stat["svalue"].values
    qvalues = final_stat["qvalue"].values

    decoys = scored_table[scored_table["decoy"] == 1]["d_score"].values
    targets = scored_table[scored_table["decoy"] == 0]["d_score"].values

    tops = scored_table[scored_table["peak_group_rank"] == 1]
    top_decoys = tops[tops["decoy"] == 1]["d_score"].values
    top_target = tops[tops["decoy"] == 0]["d_score"].values

    # thanks to lorenz blum for the plotting code below:

    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(hspace=.5)

    plt.subplot(411)
    plt.title(in_file_name + "\n\nROC")
    plt.xlabel('False Positive Rate (qvalue)')
    plt.ylabel('True Positive Rate (svalue)')

    plt.scatter(qvalues, svalues, s=3)
    plt.plot(qvalues, svalues)

    plt.subplot(412)
    plt.title('d_score Performance')
    plt.xlabel('dscore cutoff')
    plt.ylabel('rates')

    plt.scatter(cutoffs, svalues, color='g', s=3)
    plt.plot(cutoffs, svalues, color='g', label="TPR (svalue)")
    plt.scatter(cutoffs, qvalues, color='r', s=3)
    plt.plot(cutoffs, qvalues, color='r', label="FPR (qvalue)")

    plt.subplot(413)
    if (CONFIG.get("final_statistics.fdr_all_pg")):
        plt.title("All Peak Groups' d_score Distributions")
        plt.xlabel("d_score")
        plt.ylabel("# of groups")
        plt.hist(
            [targets, decoys], 20, color=['w', 'r'], label=['target', 'decoy'], histtype='bar')
        plt.legend(loc=2)
    else:
        plt.title("Top Peak Groups' d_score Distributions")
        plt.xlabel("d_score")
        plt.ylabel("# of groups")
        plt.hist(
            [top_target, top_decoys], 20, color=['w', 'r'], label=['target', 'decoy'], histtype='bar')
        plt.legend(loc=2)

    plt.subplot(414)
    if (CONFIG.get("final_statistics.fdr_all_pg")):
        atdensity = gaussian_kde(targets)
        atdensity.covariance_factor = lambda : .25
        atdensity._compute_covariance()
        addensity = gaussian_kde(decoys)
        addensity.covariance_factor = lambda : .25
        addensity._compute_covariance()
        axs = linspace(min(concatenate((targets,decoys))),max(concatenate((targets,decoys))),200)
        plt.title("All Peak Groups' d_score Density")
        plt.xlabel("d_score")
        plt.ylabel("density")
        plt.plot(axs, atdensity(axs), color='g', label='target')
        plt.plot(axs, addensity(axs), color='r', label='decoy')
        plt.legend(loc=2)
    else:
        tdensity = gaussian_kde(top_target)
        tdensity.covariance_factor = lambda : .25
        tdensity._compute_covariance()
        ddensity = gaussian_kde(top_decoys)
        ddensity.covariance_factor = lambda : .25
        ddensity._compute_covariance()
        xs = linspace(min(concatenate((top_target,top_decoys))),max(concatenate((top_target,top_decoys))),200)
        plt.title("Top Peak Groups' d_score Density")
        plt.xlabel("d_score")
        plt.ylabel("density")
        plt.plot(xs, tdensity(xs), color='g', label='target')
        plt.plot(xs, ddensity(xs), color='r', label='decoy')
        plt.legend(loc=2)

    plt.savefig(report_path)

    return cutoffs, svalues, qvalues, top_target, top_decoys


def export_mayu(mayu_cutoff_file, mayu_fasta_file, mayu_csv_file, scored_table, final_stat):

    # write MAYU CSV input file
    interesting_cols = ['run_id', 'transition_group_id', 'Sequence', 'ProteinName', 'm_score',
                        'Charge']
    mayu_csv = scored_table[scored_table["peak_group_rank"] == 1][interesting_cols]
    row_index = [str(i) for i in range(len(mayu_csv.index))]
    mayu_csv['Identifier'] = ("run" + mayu_csv['run_id'].astype('|S10') + "." + row_index
                              + "." + row_index + "." + mayu_csv['Charge'].astype('|S10'))
    mayu_csv['Mod'] = ''
    mayu_csv['m_score'] = 1 - mayu_csv['m_score']
    mayu_csv = mayu_csv[['Identifier', 'Sequence', 'ProteinName', 'Mod', 'm_score']]
    mayu_csv.columns = ['Identifier', 'Sequence', 'Protein', 'Mod', 'MScore']
    mayu_csv.to_csv(mayu_csv_file, sep=",", index=False)

    # write MAYU FASTA input file
    mayu_fasta = scored_table[scored_table["peak_group_rank"] == 1]
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
