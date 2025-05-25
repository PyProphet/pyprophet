try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from pypdf import PdfMerger, PdfReader

import os

import click
from loguru import logger
import warnings
from functools import wraps
from sklearn.metrics import jaccard_score
from scipy.stats import gaussian_kde
import numpy as np
from numpy import (
    linspace,
    concatenate,
    around,
    argmin,
    sort,
    arange,
    interp,
    array,
    degrees,
    arctan,
)


# ======================
# Utility Functions
# ======================


def handle_plot_errors(func):
    """Decorator to handle plot generation errors gracefully."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to generate plot {func.__name__}: {str(e)}")
            # Create an empty plot with error message
            if len(args) > 0 and hasattr(args[0], "figure"):
                ax = args[0]
                ax.clear()
                ax.text(
                    0.5,
                    0.5,
                    f"Plot Error:\n{str(e)}",
                    ha="center",
                    va="center",
                    color="red",
                )
                ax.set_title(f"Failed to generate plot: {func.__name__}")
            return None

    return wrapper


def color_blind_friendly(color_palette):

    color_dict = {
        "normal": ["#F5793A", "#0F2080"],
        "protan": ["#AE9C45", "#052955"],
        "deutran": ["#C59434", "#092C48"],
        "tritan": ["#1ECBE1", "#E1341E"],
    }

    if color_palette not in color_dict:
        logger.warning(
            f"{color_palette} is not a valid color_palette, must be one of 'normal'. 'protan', 'deutran', or 'tritan'. Using default 'normal'."
        )
        color_palette = "normal"
    return color_dict[color_palette][0], color_dict[color_palette][1]


# ======================
# Main Plotting class
# ======================


class PlotGenerator:
    """Class to handle all plotting operations with error handling."""

    def __init__(self, color_palette="normal"):
        self.t_col, self.d_col = color_blind_friendly(color_palette)

    @staticmethod
    def filter_identifications(df, id_key, q_thresh=0.05):
        base = (df.q_value <= q_thresh) & (df.decoy == 0)
        if id_key in ("precursor_id", "transition_id"):
            base &= df.peak_group_rank == 1
        return df[base]

    @staticmethod
    def ecdf(data):
        data = sort(data)
        n = len(data)
        return data, arange(1, n + 1) / n

    @staticmethod
    def compute_jaccard_matrix(binary_matrix):
        runs = binary_matrix.columns
        n = len(runs)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                a = binary_matrix.iloc[:, i].values
                b = binary_matrix.iloc[:, j].values
                intersection = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                score = intersection / union if union != 0 else 0.0
                matrix[i, j] = matrix[j, i] = score

        return pd.DataFrame(matrix, index=runs, columns=runs)

    @handle_plot_errors
    def add_id_barplot(self, ax, df, id_key):
        logger.debug(f"Generating identification barplot for {id_key}")
        id_counts = (
            self.filter_identifications(df, id_key)
            .groupby("run_id")[id_key]
            .nunique()
            .reset_index()
        )
        id_counts.rename(columns={id_key: "num_identified"}, inplace=True)

        sns.barplot(data=id_counts, x="run_id", y="num_identified", ax=ax)
        ax.set(
            title=f"Identified {id_key} per Run (q ≤ 0.05)",
            ylabel=f"Number of {id_key}s",
            xlabel="Run ID",
        )
        ax.tick_params(axis="x", rotation=60, labelsize=6)

    @handle_plot_errors
    def plot_identification_consistency(self, ax, df, id_key, iterations=100):
        logger.debug(f"Generating identification consistency plot for {id_key}")
        df_filtered = self.filter_identifications(df, id_key)
        run_ids = df_filtered["run_id"].unique()
        max_k = len(run_ids)

        intersect_counts = []
        for k in range(1, max_k + 1):
            intersect_sizes = []
            for _ in range(iterations):
                sampled_runs = np.random.choice(run_ids, k, replace=False)
                sets = [
                    set(df_filtered[df_filtered["run_id"] == run][id_key].unique())
                    for run in sampled_runs
                ]
                intersection = set.intersection(*sets) if sets else set()
                intersect_sizes.append(len(intersection))
            intersect_counts.append((k, int(np.mean(intersect_sizes))))
        x, y = zip(*intersect_counts)
        ax.plot(x, y, marker="o")
        ax.set(
            title=f"Intersection of {id_key}s Across Runs",
            xlabel="Number of Random Runs Sampled",
            ylabel=f"Mean # of Common {id_key}s",
        )
        ax.grid(True)
        ax.set_xticks(range(1, max_k + 1))

    @handle_plot_errors
    def add_violinplot(self, ax, df, id_key):
        if "area_intensity" not in df:
            raise ValueError("Missing 'area_intensity' column.")

        logger.debug(f"Generating quantification distribution plot for {id_key}")

        filtered = self.filter_identifications(df, id_key)
        sns.violinplot(
            data=filtered,
            x="run_id",
            y="area_intensity",
            inner="box",
            ax=ax,
            log_scale=True,
        )
        ax.set(
            title="Area Intensity per Run (q ≤ 0.05)",
            ylabel="Area Intensity (log scale)",
            xlabel="Run ID",
        )
        ax.tick_params(axis="x", rotation=60, labelsize=6)

    @handle_plot_errors
    def plot_cv_distribution(self, ax, df, id_key):
        if "area_intensity" not in df:
            raise ValueError("Missing 'area_intensity' column.")

        logger.debug(f"Generating CV distribution plot for {id_key}")

        df_filtered = self.filter_identifications(df, id_key)
        pivot = df_filtered.pivot_table(
            index=id_key, columns="run_id", values="area_intensity"
        )

        cv = pivot.std(axis=1) / pivot.mean(axis=1)
        logger.info(f"CV: min={cv.min()}, max={cv.max()}, mean={cv.mean()}")
        sns.histplot(cv.dropna(), bins=50, kde=True, ax=ax)
        ax.set(
            title="Coefficient of Variation (CV) Distribution",
            xlabel="CV",
            ylabel=f"Number of {id_key}s",
        )
        ax.grid(True)

    @handle_plot_errors
    def plot_jaccard_similarity(self, ax, df, id_key):
        df_filtered = self.filter_identifications(df, id_key)
        if df_filtered.empty:
            raise ValueError("No data for Jaccard similarity plot.")

        logger.debug(f"Generating idetnfication Jaccard similarity plot for {id_key}")

        pivot = df_filtered.assign(present=1).pivot_table(
            index=id_key, columns="run_id", values="present", fill_value=0
        )
        jaccard_matrix = self.compute_jaccard_matrix(pivot)
        sns.heatmap(jaccard_matrix, cmap="viridis", ax=ax)
        ax.set(title=f"Jaccard Similarity of {id_key} IDs Across Runs")

    @handle_plot_errors
    def plot_intensity_correlation(self, ax, df, id_key):
        if "area_intensity" not in df:
            raise ValueError("Missing 'area_intensity' column.")

        logger.debug(f"Generating intensity correlation plot for {id_key}")

        df_filtered = self.filter_identifications(df, id_key)
        if df_filtered.empty:
            raise ValueError("No data for intensity correlation plot.")

        pivot = df_filtered.pivot_table(
            index=id_key, columns="run_id", values="area_intensity"
        )
        corr = pivot.corr(method="spearman")
        sns.heatmap(corr, cmap="coolwarm", center=1.0, ax=ax)
        ax.set(title="Run-to-Run Intensity Correlation (Spearman)")

    @handle_plot_errors
    def plot_main_panel(
        self, title, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0
    ):
        plt.figure(figsize=(10, 15))
        plt.subplots_adjust(hspace=0.3)

        # Plot q-value vs s-value
        plt.subplot(321)
        plt.title("q-/s-value")
        plt.xlabel("false positive rate (q-value)")
        plt.ylabel("true positive rate (s-value)")
        plt.scatter(qvalues, svalues, s=3)
        plt.plot(qvalues, svalues)

        # Plot q-value vs d-score
        plt.subplot(322)
        plt.title("d-score performance")
        plt.xlabel("d-score cutoff")
        plt.ylabel("rates")
        plt.scatter(cutoffs, svalues, color=self.t_col, s=3)
        plt.plot(cutoffs, svalues, color=self.t_col, label="TPR (s-value)")
        plt.scatter(cutoffs, qvalues, color=self.d_col, s=3)
        plt.plot(cutoffs, qvalues, color=self.d_col, label="FPR (q-value)")
        plt.legend()

        # Plot group d-score distributions
        plt.subplot(323)
        plt.title("group d-score distributions")
        plt.xlabel("d-score")
        plt.ylabel("# of groups")
        plt.hist(
            [top_targets, top_decoys],
            bins=20,
            color=[self.t_col, self.d_col],
            label=["target", "decoy"],
            histtype="bar",
        )

        s_value_cutoff = svalues[argmin(abs(qvalues - 0.01))]
        d_cutoff_at_1_pcnt = cutoffs[svalues == s_value_cutoff][-1]
        plt.axvline(x=d_cutoff_at_1_pcnt, color="grey", linestyle="--", linewidth=2)
        y_max = plt.gca().get_ylim()[1]
        plt.text(
            d_cutoff_at_1_pcnt + 0.05,
            y_max * 0.95,
            f"Cutoff @ 1%: {d_cutoff_at_1_pcnt:.2f}",
            fontsize=7,
            fontweight="bold",
            bbox=dict(facecolor="lightgray", alpha=0.5),
        )
        plt.legend(loc=2)

        # Plot group d-score densities
        plt.subplot(324)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tdensity = gaussian_kde(top_targets)
                tdensity.covariance_factor = lambda: 0.25
                tdensity._compute_covariance()
                ddensity = gaussian_kde(top_decoys)
                ddensity.covariance_factor = lambda: 0.25
                ddensity._compute_covariance()

                xs = linspace(
                    min(concatenate((top_targets, top_decoys))),
                    max(concatenate((top_targets, top_decoys))),
                    200,
                )

                plt.plot(xs, tdensity(xs), color=self.t_col, label="target")
                plt.plot(xs, ddensity(xs), color=self.d_col, label="decoy")
                plt.axvline(
                    x=d_cutoff_at_1_pcnt, color="grey", linestyle="--", linewidth=2
                )
                y_max = plt.gca().get_ylim()[1]
                plt.text(
                    d_cutoff_at_1_pcnt + 0.05,
                    y_max * 0.95,
                    f"Cutoff @ 1%: {d_cutoff_at_1_pcnt:.2f}",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(facecolor="lightgray", alpha=0.5),
                )
                plt.legend(loc=2)
            except Exception as e:
                plt.text(
                    0.5,
                    0.5,
                    f"Could not plot densities:\n{str(e)}",
                    ha="center",
                    va="center",
                    color="red",
                )
                plt.title("Density Plot Failed")

        plt.title("group d-score densities")
        plt.xlabel("d-score")
        plt.ylabel("density")

        # Plot p-value density histogram
        plt.subplot(325)
        if pvalues is not None:
            plt.hist(pvalues, bins=20, density=True)
            plt.plot([0, 1], [pi0["pi0"], pi0["pi0"]], "r")
            plt.title(
                r"p-value density histogram: $\pi_0$ = " + str(around(pi0["pi0"], 3))
            )
            plt.xlabel("p-value")
            plt.ylabel("density")

        # Plot pi0 smoothing fit plot or P-P plot
        plt.subplot(326)
        if pi0["pi0_smooth"] is not False:
            plt.plot(pi0["lambda_"], pi0["pi0_lambda"], ".")
            plt.plot(pi0["lambda_"], pi0["pi0_smooth"], "r")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.title(r"$\pi_0$ smoothing fit plot")
            plt.xlabel(r"$\lambda$")
            plt.ylabel(r"$\pi_0(\lambda)$")
        else:
            x_target, y_target = self.ecdf(array(top_targets))
            x_decoy, y_decoy = self.ecdf(array(top_decoys))
            x_seq = linspace(
                min(x_target.min(), x_decoy.min()),
                max(x_target.max(), x_decoy.max()),
                1000,
            )
            y_target_interp = interp(x_seq, x_target, y_target)
            y_decoy_interp = interp(x_seq, x_decoy, y_decoy)
            plt.scatter(
                y_decoy_interp,
                y_target_interp,
                s=3,
                alpha=0.5,
                label="Target vs Decoy ECDF",
            )
            plt.plot([0, 1], [0, 1], "r--", label="y = x (Perfect match)")
            plt.plot([0, 1], [0, pi0["pi0"]], "b:", label=f"y = {pi0['pi0']:.2f} * x")
            plt.title("P-P Plot")
            plt.xlabel("Decoy ECDF")
            plt.ylabel("Target ECDF")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.legend()

        plt.suptitle(title)

    @handle_plot_errors
    def add_summary_table_with_spanners(self, pdf, df, id_key="precursor_id"):
        logger.debug(f"Generating summary table for {id_key}")

        thresholds = [0.01, 0.05, 0.10]
        col_labels = ["run_id", "num_ids", "min_area", "mean_area", "max_area"]
        table_blocks = []

        for thresh in thresholds:
            filtered = self.filter_identifications(df, id_key, q_thresh=thresh).copy()
            if "area_intensity" not in filtered.columns:
                filtered.loc[:, "area_intensity"] = np.nan

            summary = (
                filtered.groupby("run_id")
                .agg(
                    num_ids=(id_key, "nunique"),
                    min_area=("area_intensity", "min"),
                    mean_area=("area_intensity", "mean"),
                    max_area=("area_intensity", "max"),
                )
                .reset_index()
            )
            if summary.empty:
                continue
            summary_rows = summary.round(2).astype(str).values.tolist()
            logger.info(
                f"Summary for q-value ≤ {thresh}:\n{pd.DataFrame(summary_rows).rename(columns={0: 'run_id', 1: 'num_ids', 2: 'min_area', 3: 'mean_area', 4: 'max_area'})}"
            )
            table_blocks.append((f"Q-Value ≤ {int(thresh * 100)}%", summary_rows))

        # Layout
        font_size = 8
        spanner_height = 0.030
        row_height = 0.028
        table_width = 0.75
        padding = 0.012
        min_margin = 0.08

        def new_page():
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            return fig, ax, 0.95

        fig, ax, y_cursor = new_page()

        for header, rows in table_blocks:
            n_rows = max(len(rows), 2)
            block_height = spanner_height + (row_height * n_rows) + padding

            if y_cursor - block_height < min_margin:
                plt.title(
                    "Summary of Identifications and Area Intensity per Run", fontsize=10
                )
                plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax, y_cursor = new_page()

            plt.table(
                cellText=[[header]],
                colWidths=[1.0],
                loc="center",
                bbox=[0.15, y_cursor - spanner_height, table_width, spanner_height],
                cellLoc="center",
            ).set_fontsize(font_size + 1)

            y_cursor -= spanner_height + 0.004

            plt.table(
                cellText=rows,
                colLabels=col_labels,
                loc="center",
                bbox=[
                    0.15,
                    y_cursor - (n_rows * row_height),
                    table_width,
                    n_rows * row_height,
                ],
                cellLoc="center",
            ).set_fontsize(font_size)

            y_cursor -= (n_rows * row_height) + padding

        plt.title("Summary of Identifications and Area Intensity per Run", fontsize=10)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        pdf.savefig(fig)
        plt.close(fig)


# ======================
# Plotting Functions
# ======================


def save_report(
    pdf_path,
    title,
    top_decoys,
    top_targets,
    cutoffs,
    svalues,
    qvalues,
    pvalues,
    pi0,
    color_palette="normal",
    level=None,
    df=None,
):
    """Main function to generate and save the report."""

    plotter = PlotGenerator(color_palette)

    with PdfPages(pdf_path) as pdf:
        logger.trace("Generating main panel plot")
        try:
            # Plot main panel
            plotter.plot_main_panel(
                title, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0
            )
            pdf.savefig()
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate main panel: {str(e)}")
            # Create an error page
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(
                0.5,
                0.5,
                f"Failed to generate main panel:\n{str(e)}",
                ha="center",
                va="center",
                color="red",
            )
            ax.set_title("Report Generation Error")
            pdf.savefig(fig)
            plt.close(fig)

        if df is not None and level != "alignment":
            if df[(df.q_value <= 0.05) & (df.decoy == 0)].empty:
                logger.warning(
                    "No significant identifications found. Skipping downstream plots."
                )
                return

            if level in ("ms1", "ms2", "ms1ms2"):
                id_key = "precursor_id"
            elif level == "transition":
                id_key = "transition_id"
            elif level == "peptide":
                id_key = "peptide_id"
            elif level == "protein":
                id_key = "protein_id"
            elif level == "gene":
                id_key = "gene_id"
            else:
                logger.warning("Unknown level specified. Defaulting to precursor_id.")
                id_key = "precursor_id"

            skip_quant_dist = level in ("peptide", "protein", "gene")
            skip_quant_corr = skip_quant_dist

            if df["run_id"].isna().all() and level in ("peptide", "protein", "gene"):
                df["run_id"] = "global"
                skip_consistency = skip_jaccard = True
            else:
                skip_consistency = skip_jaccard = False

            # First set of plots
            try:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                plotter.add_id_barplot(axes[0, 0], df, id_key)
                if not skip_consistency:
                    plotter.plot_identification_consistency(axes[0, 1], df, id_key)
                if not skip_quant_dist:
                    plotter.add_violinplot(axes[1, 0], df, id_key)
                    plotter.plot_cv_distribution(axes[1, 1], df, id_key)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to generate first set of plots: {str(e)}")

            # Second set of plots
            try:
                fig, axes = plt.subplots(2, 1, figsize=(10, 12))
                if not skip_jaccard:
                    plotter.plot_jaccard_similarity(axes[0], df, id_key)
                if not skip_quant_corr:
                    plotter.plot_intensity_correlation(axes[1], df, id_key)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to generate second set of plots: {str(e)}")

            # Summary tables
            try:
                plotter.add_summary_table_with_spanners(pdf, df, id_key)
            except Exception as e:
                logger.error(f"Failed to generate summary tables: {str(e)}")


def plot_scores(df, out, color_palette="normal"):

    if plt is None:
        raise ImportError(
            "Error: The matplotlib package is required to create a report."
        )

    score_columns = (
        ["SCORE"]
        + [c for c in df.columns if c.startswith("MAIN_VAR_")]
        + [c for c in df.columns if c.startswith("VAR_")]
    )

    t_col, d_col = color_blind_friendly(color_palette)

    with PdfPages(out) as pdf:
        for idx in score_columns:
            top_targets = df[df["DECOY"] == 0][idx]
            top_decoys = df[df["DECOY"] == 1][idx]

            if not (
                top_targets.isnull().values.any() or top_targets.isnull().values.any()
            ):
                plt.figure(figsize=(10, 10))
                plt.subplots_adjust(hspace=0.5)

                plt.subplot(211)
                plt.title(idx)
                plt.xlabel(idx)
                plt.ylabel("# of groups")
                plt.hist(
                    [top_targets, top_decoys],
                    20,
                    color=[t_col, d_col],
                    label=["target", "decoy"],
                    histtype="bar",
                )
                plt.legend(loc=2)

                try:
                    tdensity = gaussian_kde(top_targets)
                    tdensity.covariance_factor = lambda: 0.25
                    tdensity._compute_covariance()
                    ddensity = gaussian_kde(top_decoys)
                    ddensity.covariance_factor = lambda: 0.25
                    ddensity._compute_covariance()
                    xs = linspace(
                        min(concatenate((top_targets, top_decoys))),
                        max(concatenate((top_targets, top_decoys))),
                        200,
                    )
                    plt.subplot(212)
                    plt.xlabel(idx)
                    plt.ylabel("density")
                    plt.plot(xs, tdensity(xs), color=t_col, label="target")
                    plt.plot(xs, ddensity(xs), color=d_col, label="decoy")
                    plt.legend(loc=2)
                except:
                    plt.subplot(212)

                pdf.savefig()
                plt.close()


def plot_hist(x, title, xlabel, ylabel, pdf_path="histogram_plot.png"):

    if plt is not None:
        # Clear figures
        plt.close("all")
        counts, __, __ = plt.hist(x, bins=20, density=True)
        plt.title(title, wrap=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(pdf_path)

        # Clear figures
        plt.close("all")


def main_score_selection_report(
    title,
    sel_column,
    mapper,
    decoy_scores,
    target_scores,
    target_pvalues,
    pi0,
    color_palette="normal",
    pdf_path="main_score_selection_report.pdf",
    worker_num=1,
):

    if plt is None:
        raise ImportError(
            "Error: The matplotlib package is required to create a report."
        )

    # Create output to merge pdges
    output = PdfMerger()
    # Generate colors
    t_col, d_col = color_blind_friendly(color_palette)

    # Clear figures
    plt.close("all")

    plt.figure(figsize=(10, 11))
    plt.subplots_adjust(hspace=0.5)

    # Plot Score Distribution
    plt.subplot(121)
    plt.hist(
        [target_scores, decoy_scores],
        20,
        color=[t_col, d_col],
        label=["target", "decoy"],
        histtype="bar",
    )
    plt.title(f"histogram of scores")
    plt.xlabel("score")
    plt.ylabel("density histogram")
    plt.legend(loc=1)
    # Plot P-value Distribution
    plt.subplot(122)
    if target_pvalues is not None:
        counts, __, __ = plt.hist(target_pvalues, bins=20, density=True)
        if pi0 is not None:
            plt.plot([0, 1], [pi0["pi0"], pi0["pi0"]], "r")
            plt.title(
                r"p-value density histogram: $\pi_0$ = "
                + str(around(pi0["pi0"], decimals=3))
            )
        else:
            plt.title(r"p-value density histogram: $\pi_0$ estimation failed.")
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
    plt.close("all")
