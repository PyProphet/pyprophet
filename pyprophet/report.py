try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from PyPDF2 import PdfMerger, PdfReader

import os

import click
from scipy.stats import gaussian_kde
from numpy import linspace, concatenate, around

def color_blind_friendly(color_palette):
    
    color_dict = {"normal":["#F5793A", "#0F2080"], "protan":["#AE9C45", "#052955"], "deutran":["#C59434", "#092C48"], "tritan":["#1ECBE1", "#E1341E"]}
    
    if color_palette not in color_dict:
        click.echo(f"WARN: {color_palette} is not a valid color_palette, must be one of 'normal'. 'protan', 'deutran', or 'tritan'. Using default 'normal'.")
        color_palette = "normal"
    return color_dict[color_palette][0], color_dict[color_palette][1]

def save_report(pdf_path, title, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0, color_palette="normal"):

    if plt is None:
        raise ImportError("Error: The matplotlib package is required to create a report.")

    t_col, d_col = color_blind_friendly(color_palette)

    plt.figure(figsize=(10, 15))
    plt.subplots_adjust(hspace=.5)

    plt.subplot(321)
    plt.title("q-/s-value")
    plt.xlabel('false positive rate (q-value)')
    plt.ylabel('true positive rate (s-value)')

    plt.scatter(qvalues, svalues, s=3)
    plt.plot(qvalues, svalues)

    plt.subplot(322)
    plt.title('d-score performance')
    plt.xlabel('d-score cutoff')
    plt.ylabel('rates')

    plt.scatter(cutoffs, svalues, color=t_col, s=3)
    plt.plot(cutoffs, svalues, color=t_col, label="TPR (s-value)")
    plt.scatter(cutoffs, qvalues, color=d_col, s=3)
    plt.plot(cutoffs, qvalues, color=d_col, label="FPR (q-value)")

    plt.subplot(323)
    plt.title("group d-score distributions")
    plt.xlabel("d-score")
    plt.ylabel("# of groups")
    plt.hist(
        [top_targets, top_decoys], 20, color=[t_col, d_col], label=['target', 'decoy'], histtype='bar')
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
    plt.title("group d-score densities")
    plt.xlabel("d-score")
    plt.ylabel("density")
    plt.plot(xs, tdensity(xs), color=t_col, label='target')
    plt.plot(xs, ddensity(xs), color=d_col, label='decoy')
    plt.legend(loc=2)

    plt.subplot(325)
    if pvalues is not None:
        counts, __, __ = plt.hist(pvalues, bins=20, density=True)
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
        plt.xlabel("$\lambda$")
        plt.ylabel("$\pi_0$($\lambda$)")

    plt.suptitle(title)
    plt.savefig(pdf_path)

def plot_scores(df, out, color_palette="normal"):

    if plt is None:
        raise ImportError("Error: The matplotlib package is required to create a report.")

    score_columns = ["SCORE"] + [c for c in df.columns if c.startswith("MAIN_VAR_")] + [c for c in df.columns if c.startswith("VAR_")]

    t_col, d_col = color_blind_friendly(color_palette)

    with PdfPages(out) as pdf:
        for idx in score_columns:
            top_targets = df[df["DECOY"] == 0][idx]
            top_decoys = df[df["DECOY"] == 1][idx]

            if not (top_targets.isnull().values.any() or top_targets.isnull().values.any()):
                plt.figure(figsize=(10, 10))
                plt.subplots_adjust(hspace=.5)

                plt.subplot(211)
                plt.title(idx)
                plt.xlabel(idx)
                plt.ylabel("# of groups")
                plt.hist(
                    [top_targets, top_decoys], 20, color=[t_col, d_col], label=['target', 'decoy'], histtype='bar')
                plt.legend(loc=2)

                try:
                    tdensity = gaussian_kde(top_targets)
                    tdensity.covariance_factor = lambda: .25
                    tdensity._compute_covariance()
                    ddensity = gaussian_kde(top_decoys)
                    ddensity.covariance_factor = lambda: .25
                    ddensity._compute_covariance()
                    xs = linspace(min(concatenate((top_targets, top_decoys))), max(
                        concatenate((top_targets, top_decoys))), 200)
                    plt.subplot(212)
                    plt.xlabel(idx)
                    plt.ylabel("density")
                    plt.plot(xs, tdensity(xs), color=t_col, label='target')
                    plt.plot(xs, ddensity(xs), color=d_col, label='decoy')
                    plt.legend(loc=2)
                except:
                    plt.subplot(212)

                pdf.savefig()
                plt.close()

def plot_hist(x, title, xlabel, ylabel, pdf_path="histogram_plot.png"):

    if plt is not None:
        # Clear figures
        plt.close('all')
        counts, __, __ = plt.hist(x, bins=20, density=True)
        plt.title(title, wrap=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(pdf_path)

        # Clear figures
        plt.close('all')

def main_score_selection_report(title, sel_column, mapper, decoy_scores, target_scores, target_pvalues, pi0, color_palette="normal", pdf_path="main_score_selection_report.pdf", worker_num=1):
    
    if plt is None:
        raise ImportError("Error: The matplotlib package is required to create a report.")
    
    # Create output to merge pdges
    output = PdfMerger() 
    # Generate colors
    t_col, d_col = color_blind_friendly(color_palette)

    # Clear figures
    plt.close('all')

    plt.figure(figsize=(10, 11))
    plt.subplots_adjust(hspace=.5)

    # Plot Score Distribution
    plt.subplot(121)
    plt.hist([target_scores, decoy_scores], 20, color=[t_col, d_col], label=['target', 'decoy'], histtype='bar')
    plt.title(f"histogram of scores")
    plt.xlabel("score")
    plt.ylabel("density histogram")
    plt.legend(loc=1)
    # Plot P-value Distribution
    plt.subplot(122)
    if target_pvalues is not None:
        counts, __, __ = plt.hist(target_pvalues, bins=20, density=True)
        if pi0 is not None:
            plt.plot([0, 1], [pi0['pi0'], pi0['pi0']], "r")
            plt.title("p-value density histogram: $\pi_0$ = " + str(around(pi0['pi0'], decimals=3)))
        else:
            plt.title("p-value density histogram: $\pi_0$ estimation failed.")
        plt.xlabel("target p-values")
        plt.ylabel("density histogram")
    # Finalize figure
    plt.suptitle(f"{title}: {mapper[sel_column]}")
    plt.tight_layout()
    # Append to existing master report if exists, otherwise write to a new master report pdf.
    if os.path.isfile(pdf_path):
        temp_pdf_path = f"temp_main_score_selection_report_thread__{worker_num}.pdf"
        # Save current plot in temporary pdf
        plt.savefig(temp_pdf_path)
        # Append master pdf and temp pdf to output merger
        output.append(PdfReader(open(pdf_path, "rb")))
        output.append(PdfReader(open(temp_pdf_path, "rb")))
        # Write to master pdf
        output.write(pdf_path)
        # Remove temporary pdf
        os.remove(temp_pdf_path)
    else:
        # Save as master pdf
        plt.savefig(pdf_path)
    # Clear figures
    plt.close('all')