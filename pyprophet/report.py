try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from pypdf import PdfWriter, PdfReader
import sys
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


def filter_identifications(df, id_key, q_thresh=0.05):
    base = (df.q_value <= q_thresh) & (df.decoy == 0)
    if id_key in ("precursor_id", "transition_id"):
        base &= df.peak_group_rank == 1
    return df[base]


# ======================
# Main Plotting class
# ======================


class PlotGenerator:
    """Class to handle all plotting operations with error handling."""

    def __init__(self, color_palette="normal"):
        self.t_col, self.d_col = color_blind_friendly(color_palette)

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
    def add_id_barplot(self, ax, df, id_key, title=None, ylabel=None, xlabel=None):
        if title is None:
            title = f"Identified {id_key} per Run"
        if ylabel is None:
            ylabel = f"Number of {id_key}s"
        if xlabel is None:
            xlabel = "Run ID"

        logger.debug(f"Generating identification barplot for {id_key}")
        id_counts = df.groupby("run_id")[id_key].nunique().reset_index()
        id_counts.rename(columns={id_key: "num_identified"}, inplace=True)

        sns.barplot(data=id_counts, x="run_id", y="num_identified", ax=ax)
        ax.set(
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
        )
        ax.tick_params(axis="x", rotation=60, labelsize=6)

        # Check how many runs there are, if there are more than 10 runs, then turn off the x-axis labels
        if len(id_counts) > 10:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xticklabels([])

    @handle_plot_errors
    def plot_identification_consistency(
        self, ax, df, id_key, iterations=100, title=None, ylabel=None, xlabel=None
    ):
        if title is None:
            title = f"Intersection of {id_key}s Across Runs"
        if ylabel is None:
            ylabel = f"Mean # of Common {id_key}s"
        if xlabel is None:
            xlabel = "Number of Random Runs Sampled"

        logger.debug(f"Generating identification consistency plot for {id_key}")
        run_ids = df["run_id"].unique()
        max_k = len(run_ids)

        intersect_counts = []
        for k in range(1, max_k + 1):
            intersect_sizes = []
            for _ in range(iterations):
                sampled_runs = np.random.choice(run_ids, k, replace=False)
                sets = [
                    set(df[df["run_id"] == run][id_key].unique())
                    for run in sampled_runs
                ]
                intersection = set.intersection(*sets) if sets else set()
                intersect_sizes.append(len(intersection))
            intersect_counts.append((k, int(np.mean(intersect_sizes))))
        x, y = zip(*intersect_counts)
        ax.plot(x, y, marker="o")
        ax.set(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        ax.grid(True)
        ax.set_xticks(range(1, max_k + 1))

    @handle_plot_errors
    def add_violinplot(self, ax, df, id_key, title=None, ylabel=None, xlabel=None):
        if "area_intensity" not in df:
            raise ValueError("Missing 'area_intensity' column.")

        if title is None:
            title = "Quantification Distribution per Run"
        if ylabel is None:
            ylabel = "Area Intensity (log scale)"
        if xlabel is None:
            xlabel = "Run ID"

        logger.debug(f"Generating quantification distribution plot for {id_key}")

        sns.violinplot(
            data=df,
            x="run_id",
            y="area_intensity",
            inner="box",
            ax=ax,
            log_scale=True,
        )
        ax.set(
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
        )
        ax.tick_params(axis="x", rotation=60, labelsize=6)

        # Check how many runs there are, if there are more than 10 runs, then turn off the x-axis labels
        if len(df["run_id"].unique()) > 10:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xticklabels([])

    @handle_plot_errors
    def plot_cv_distribution(
        self, ax, df, id_key, title=None, ylabel=None, xlabel=None
    ):
        if "area_intensity" not in df:
            raise ValueError("Missing 'area_intensity' column.")

        if title is None:
            title = "Coefficient of Variation (CV) Distribution"
        if ylabel is None:
            ylabel = "Number of Identifications"
        if xlabel is None:
            xlabel = "CV"

        logger.debug(f"Generating CV distribution plot for {id_key}")

        pivot = df.pivot_table(index=id_key, columns="run_id", values="area_intensity")

        cv = pivot.std(axis=1) / pivot.mean(axis=1)
        logger.info(f"CV: min={cv.min()}, max={cv.max()}, mean={cv.mean()}")
        sns.histplot(cv.dropna(), bins=50, kde=True, ax=ax)
        ax.set(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        ax.grid(True)

    @handle_plot_errors
    def plot_jaccard_similarity(self, ax, df, id_key, title=None):
        if title is None:
            title = f"Jaccard Similarity of {id_key} IDs Across Runs"
        if df.empty:
            raise ValueError("No data for Jaccard similarity plot.")

        logger.debug(f"Generating idetnfication Jaccard similarity plot for {id_key}")

        pivot = df.assign(present=1).pivot_table(
            index=id_key, columns="run_id", values="present", fill_value=0
        )
        jaccard_matrix = self.compute_jaccard_matrix(pivot)
        sns.heatmap(jaccard_matrix, cmap="viridis", ax=ax)
        ax.set(title=title)

        # Check how many runs there are, if there are more than 10 runs, then turn off the x-axis labels
        if len(jaccard_matrix) > 10:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xticklabels([])
            ax.tick_params(axis="y", labelleft=False)
            ax.set_yticklabels([])

    @handle_plot_errors
    def plot_intensity_correlation(self, ax, df, id_key, title=None):
        if title is None:
            title = f"Run-to-Run Intensity Correlation (Spearman) for {id_key}"

        if "area_intensity" not in df:
            raise ValueError("Missing 'area_intensity' column.")

        logger.debug(f"Generating intensity correlation plot for {id_key}")

        if df.empty:
            raise ValueError("No data for intensity correlation plot.")

        pivot = df.pivot_table(index=id_key, columns="run_id", values="area_intensity")
        corr = pivot.corr(method="spearman")
        sns.heatmap(corr, cmap="coolwarm", center=1.0, ax=ax)
        ax.set(title=title)

        # Check how many runs there are, if there are more than 10 runs, then turn off the x-axis labels
        if len(corr) > 10:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xticklabels([])
            ax.tick_params(axis="y", labelleft=False)
            ax.set_yticklabels([])

    @handle_plot_errors
    def plot_q_vs_s(self, ax, qvalues, svalues):
        """Plot q-value vs s-value relationship."""
        ax.set_title("q-/s-value")
        ax.set_xlabel("false positive rate (q-value)")
        ax.set_ylabel("true positive rate (s-value)")
        ax.scatter(qvalues, svalues, s=3)
        ax.plot(qvalues, svalues)

    @handle_plot_errors
    def plot_dscore_performance(self, ax, cutoffs, svalues, qvalues):
        """Plot d-score performance showing TPR and FPR."""
        ax.set_title("d-score performance")
        ax.set_xlabel("d-score cutoff")
        ax.set_ylabel("rates")
        ax.scatter(cutoffs, svalues, color=self.t_col, s=3)
        ax.plot(cutoffs, svalues, color=self.t_col, label="TPR (s-value)")
        ax.scatter(cutoffs, qvalues, color=self.d_col, s=3)
        ax.plot(cutoffs, qvalues, color=self.d_col, label="FPR (q-value)")
        ax.legend()

    @handle_plot_errors
    def plot_dscore_distributions(
        self, ax, top_targets, top_decoys, cutoffs=None, svalues=None, qvalues=None
    ):
        """Plot histogram of target and decoy d-score distributions."""
        ax.set_title("group d-score distributions")
        ax.set_xlabel("d-score")
        ax.set_ylabel("# of groups")
        ax.hist(
            [top_targets, top_decoys],
            bins=20,
            color=[self.t_col, self.d_col],
            label=["target", "decoy"],
            histtype="bar",
        )

        if cutoffs is not None:
            s_value_cutoff = svalues[argmin(abs(qvalues - 0.01))]
            d_cutoff_at_1_pcnt = cutoffs[svalues == s_value_cutoff][-1]
            ax.axvline(x=d_cutoff_at_1_pcnt, color="grey", linestyle="--", linewidth=2)
            y_max = ax.get_ylim()[1]
            ax.text(
                d_cutoff_at_1_pcnt + 0.05,
                y_max * 0.95,
                f"Cutoff @ 1%: {d_cutoff_at_1_pcnt:.2f}",
                fontsize=7,
                fontweight="bold",
                bbox=dict(facecolor="lightgray", alpha=0.5),
            )

        ax.legend(loc=2)

    @handle_plot_errors
    def plot_dscore_densities(
        self, ax, top_targets, top_decoys, cutoffs, svalues, qvalues
    ):
        """Plot density curves of target and decoy d-scores."""
        ax.set_title("group d-score densities")
        ax.set_xlabel("d-score")
        ax.set_ylabel("density")

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

                ax.plot(xs, tdensity(xs), color=self.t_col, label="target")
                ax.plot(xs, ddensity(xs), color=self.d_col, label="decoy")

                s_value_cutoff = svalues[argmin(abs(qvalues - 0.01))]
                d_cutoff_at_1_pcnt = cutoffs[svalues == s_value_cutoff][-1]
                ax.axvline(
                    x=d_cutoff_at_1_pcnt, color="grey", linestyle="--", linewidth=2
                )
                y_max = ax.get_ylim()[1]
                ax.text(
                    d_cutoff_at_1_pcnt + 0.05,
                    y_max * 0.95,
                    f"Cutoff @ 1%: {d_cutoff_at_1_pcnt:.2f}",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(facecolor="lightgray", alpha=0.5),
                )
                ax.legend(loc=2)
            except Exception as e:
                n_nans = f"Number of NaNs in top_targets:\n{np.isnan(top_targets).sum()}\ntop_decoys: {np.isnan(top_decoys).sum()}"
                ax.text(
                    0.5,
                    0.5,
                    f"Could not plot densities:\n{str(e)}\n{n_nans}",
                    ha="center",
                    va="center",
                    color="red",
                )
                ax.set_title("Density Plot Failed")

    @handle_plot_errors
    def plot_pvalue_histogram(self, ax, pvalues, pi0):
        """Plot p-value density histogram with pi0 estimate."""
        if pvalues is not None:
            ax.hist(pvalues, bins=20, density=True)
            ax.plot([0, 1], [pi0["pi0"], pi0["pi0"]], "r")
            ax.set_title(
                r"p-value density histogram: $\pi_0$ = " + str(around(pi0["pi0"], 3))
            )
            ax.set_xlabel("p-value")
            ax.set_ylabel("density")

    @handle_plot_errors
    def plot_pi0_or_pp(self, ax, top_targets, top_decoys, pi0):
        """Plot either pi0 smoothing fit or P-P plot."""
        if pi0["pi0_smooth"] is not False:
            ax.plot(pi0["lambda_"], pi0["pi0_lambda"], ".")
            ax.plot(pi0["lambda_"], pi0["pi0_smooth"], "r")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title(r"$\pi_0$ smoothing fit plot")
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(r"$\pi_0(\lambda)$")
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
            ax.scatter(
                y_decoy_interp,
                y_target_interp,
                s=3,
                alpha=0.5,
                label="Target vs Decoy ECDF",
            )
            ax.plot([0, 1], [0, 1], "r--", label="y = x (Perfect match)")
            ax.plot([0, 1], [0, pi0["pi0"]], "b:", label=f"y = {pi0['pi0']:.2f} * x")
            ax.set_title("P-P Plot")
            ax.set_xlabel("Decoy ECDF")
            ax.set_ylabel("Target ECDF")
            ax.set_aspect("equal", adjustable="box")
            ax.legend()

    @handle_plot_errors
    def plot_main_panel(
        self, title, top_decoys, top_targets, cutoffs, svalues, qvalues, pvalues, pi0
    ):
        """Create main panel with all diagnostic plots."""
        plt.figure(figsize=(10, 15))
        plt.subplots_adjust(hspace=0.3)

        # Plot q-value vs s-value
        ax1 = plt.subplot(321)
        self.plot_q_vs_s(ax1, qvalues, svalues)

        # Plot q-value vs d-score
        ax2 = plt.subplot(322)
        self.plot_dscore_performance(ax2, cutoffs, svalues, qvalues)

        # Plot group d-score distributions
        ax3 = plt.subplot(323)
        self.plot_dscore_distributions(
            ax3, top_targets, top_decoys, cutoffs, svalues, qvalues
        )

        # Plot group d-score densities
        ax4 = plt.subplot(324)
        self.plot_dscore_densities(
            ax4, top_targets, top_decoys, cutoffs, svalues, qvalues
        )

        # Plot p-value density histogram
        ax5 = plt.subplot(325)
        self.plot_pvalue_histogram(ax5, pvalues, pi0)

        # Plot pi0 smoothing fit plot or P-P plot
        ax6 = plt.subplot(326)
        self.plot_pi0_or_pp(ax6, top_targets, top_decoys, pi0)

        plt.suptitle(title)

    @handle_plot_errors
    def add_summary_table_with_spanners(self, pdf, df_dict, id_key="precursor_id"):
        logger.debug(f"Generating summary table for {id_key}")

        col_labels = ["run_id", "num_ids", "min_area", "mean_area", "max_area"]
        table_blocks = []

        for thresh_label, df in df_dict.items():
            if "area_intensity" not in df.columns:
                df = df.copy()
                df.loc[:, "area_intensity"] = np.nan

            summary = (
                df.groupby("run_id")
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
            logger.opt(raw=True, colors=True).info("=" * 80)
            logger.opt(raw=True, colors=True).info(f"\n  Summary for {thresh_label}:\n")
            logger.opt(raw=True, colors=True).info("=" * 80)
            logger.opt(raw=True, colors=True).info("\n")
            logger.opt(raw=True, colors=True).info(
                f"{pd.DataFrame(summary_rows).rename(columns={0: 'run_id', 1: 'num_ids', 2: 'min_area', 3: 'mean_area', 4: 'max_area'})}"
            )
            logger.opt(raw=True, colors=True).info("\n")
            table_blocks.append((thresh_label, summary_rows))

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
        logger.debug("Generating main panel plot")
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

            # Check how many runs there are
            n_runs = df["run_id"].nunique()

            if df["run_id"].isna().all() and level in (
                "peptide",
                "protein",
                "gene",
            ):
                df["run_id"] = "global"

            if n_runs > 1:
                skip_quant_dist = level in ("peptide", "protein", "gene")
                skip_quant_corr = skip_quant_dist

                if df["run_id"].isna().all() and level in (
                    "peptide",
                    "protein",
                    "gene",
                ):
                    skip_consistency = skip_jaccard = True
                else:
                    skip_consistency = skip_jaccard = False

                filtered_df = filter_identifications(df, id_key, 0.05)

                # First set of plots
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    plotter.add_id_barplot(axes[0, 0], filtered_df, id_key)
                    if not skip_consistency:
                        plotter.plot_identification_consistency(
                            axes[0, 1], filtered_df, id_key
                        )
                    if not skip_quant_dist:
                        plotter.add_violinplot(axes[1, 0], filtered_df, id_key)
                        plotter.plot_cv_distribution(axes[1, 1], filtered_df, id_key)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to generate first set of plots: {str(e)}")

                # Second set of plots
                try:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
                    if not skip_jaccard:
                        plotter.plot_jaccard_similarity(axes[0], filtered_df, id_key)
                    if not skip_quant_corr:
                        plotter.plot_intensity_correlation(axes[1], filtered_df, id_key)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to generate second set of plots: {str(e)}")

            # Summary tables
            try:
                # Create filtered DataFrames
                df_1pcnt = filter_identifications(df, id_key, 0.01)
                df_5pcnt = filter_identifications(df, id_key, 0.05)
                df_10pcnt = filter_identifications(df, id_key, 0.10)
                summary_dfs = {
                    "Q-Value ≤ 1%": df_1pcnt,
                    "Q-Value ≤ 5%": df_5pcnt,
                    "Q-Value ≤ 10%": df_10pcnt,
                }
                plotter.add_summary_table_with_spanners(pdf, summary_dfs, id_key)
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


def plot_score_distributions(pdf, plotter, df, score_mapping):
    """Helper function to plot score distributions."""
    n_scores = len(score_mapping)
    n_cols = 2  # Number of columns in subplot grid
    n_rows = (n_scores + n_cols - 1) // n_cols  # Calculate needed rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle("Score Distributions", y=1.02, fontsize=14)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Flatten axes array for easy iteration
    if n_scores > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable even for single plot

    for i, (base_key, base_dict) in enumerate(score_mapping.items()):
        score_col = base_dict["score"]
        base_id = base_dict["id"]
        logger.info(f"Plotting {base_key} scores using column {score_col}")
        ax = axes[i]
        try:
            # Filter valid scores (non-null and finite)
            valid_scores = df[score_col].notna() & np.isfinite(df[score_col])
            df_valid = df[valid_scores]

            if base_key in ("ms1", "ms2"):
                top_targets = df_valid[
                    (df_valid["precursor_decoy"] == 0)
                    & (df_valid["score_ms2_peak_group_rank"] == 1)
                ][score_col].to_numpy()
                top_decoys = df_valid[
                    (df_valid["precursor_decoy"] == 1)
                    & (df_valid["score_ms2_peak_group_rank"] == 1)
                ][score_col].to_numpy()
            elif base_key == "ipf":
                top_targets = df_valid[
                    (df_valid["precursor_decoy"] == 0)
                    & (df_valid["score_ms2_peak_group_rank"] == 1)
                    & (df_valid["score_ipf_qvalue"] <= 0.01)
                ]["score_ms2_score"].to_numpy()
                top_decoys = df_valid[
                    (df_valid["precursor_decoy"] == 1)
                    & (df_valid["score_ms2_peak_group_rank"] == 1)
                    & (df_valid["score_ipf_qvalue"] <= 0.01)
                ]["score_ms2_score"].to_numpy()
            elif base_key == "peptide_global":
                top_targets = (
                    df_valid[df_valid["precursor_decoy"] == 0]
                    .groupby("peptide_id")
                    .first()["score_ms2_score"]
                    .to_numpy()
                )
                top_decoys = (
                    df_valid[df_valid["precursor_decoy"] == 1]
                    .groupby("peptide_id")
                    .first()["score_ms2_score"]
                    .to_numpy()
                )
            elif base_key in ("peptide_experiment_wide", "peptide_run_specific"):
                top_targets = (
                    df_valid[df_valid["precursor_decoy"] == 0]
                    .groupby(["peptide_id", "run_id"])
                    .first()[score_col]
                    .to_numpy()
                )
                top_decoys = (
                    df_valid[df_valid["precursor_decoy"] == 1]
                    .groupby(["peptide_id", "run_id"])
                    .first()[score_col]
                    .to_numpy()
                )
            elif base_key == "protein_global":
                top_targets = (
                    df_valid[df_valid["precursor_decoy"] == 0]
                    .groupby("peptide_id")
                    .filter(lambda x: len(x["protein_id"].unique()) == 1)
                    .groupby("protein_id")
                    .first()[score_col]
                    .to_numpy()
                )
                top_decoys = (
                    df_valid[df_valid["precursor_decoy"] == 1]
                    .groupby("peptide_id")
                    .filter(lambda x: len(x["protein_id"].unique()) == 1)
                    .groupby("protein_id")
                    .first()[score_col]
                    .to_numpy()
                )
            elif base_key in ("protein_experiment_wide", "protein_run_specific"):
                top_targets = (
                    df_valid[df_valid["precursor_decoy"] == 0]
                    .groupby(["protein_id", "run_id"])
                    .first()[score_col]
                    .to_numpy()
                )
                top_decoys = (
                    df_valid[df_valid["precursor_decoy"] == 1]
                    .groupby(["protein_id", "run_id"])
                    .first()[score_col]
                    .to_numpy()
                )

            if len(top_targets) == 0 or len(top_decoys) == 0:
                logger.warning(
                    f"Skipping {base_key} - insufficient data (targets: {len(top_targets)}, decoys: {len(top_decoys)})"
                )
                ax.text(0.5, 0.5, f"No data for {base_key}", ha="center", va="center")
                ax.set_title(f"{base_key} (no data)")
                continue

            # Plot the distributions
            plotter.plot_dscore_distributions(ax, top_targets, top_decoys)
            ax.set_title(f"{base_key} Scores")

            # Add some statistics to the plot
            q_value_col = f"score_{base_key}_q_value"
            if q_value_col not in df_valid.columns:
                logger.warning(f"Q-value column {q_value_col} not found for {base_key}")
                stats_text = (
                    f"Targets: {len(top_targets):,}\nDecoys: {len(top_decoys):,}\n"
                )
            else:
                if base_key in ("ms1", "ms2"):
                    top_targets_df = df_valid[
                        (df_valid["precursor_decoy"] == 0)
                        & (df_valid[q_value_col] <= 0.01)
                        & (df_valid["score_ms2_peak_group_rank"] == 1)
                    ]
                    d_cutoff_at_1_pcnt = top_targets_df[score_col].to_numpy().min()
                    top_targets = top_targets_df[base_id]
                    top_decoys = df_valid[
                        (df_valid["precursor_decoy"] == 1)
                        & (df_valid[q_value_col] <= 0.01)
                        & (df_valid["score_ms2_peak_group_rank"] == 1)
                    ][base_id].to_numpy()
                elif base_key == "peptide_global":
                    top_targets_df = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 0)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby("peptide_id")
                        .first()
                        .reset_index()
                    )
                    d_cutoff_at_1_pcnt = top_targets_df[score_col].to_numpy().min()
                    top_targets = top_targets_df[base_id]
                    top_decoys = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 1)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby("peptide_id")
                        .first()["score_ms2_score"]
                        .to_numpy()
                    )
                elif base_key in ("peptide_experiment_wide", "peptide_run_specific"):
                    top_targets_df = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 0)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby(["peptide_id", "run_id"])
                        .first()
                        .reset_index()
                    )
                    d_cutoff_at_1_pcnt = top_targets_df[score_col].to_numpy().min()
                    top_targets = top_targets_df[base_id]
                    top_decoys = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 1)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby(["peptide_id", "run_id"])
                        .first()[score_col]
                        .to_numpy()
                    )
                elif base_key == "protein_global":
                    top_targets_df = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 0)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby("peptide_id")
                        .filter(lambda x: len(x["protein_id"].unique()) == 1)
                        .groupby("protein_id")
                        .first()
                        .reset_index()
                    )
                    d_cutoff_at_1_pcnt = top_targets_df[score_col].to_numpy().min()
                    top_targets = top_targets_df[base_id]
                    top_decoys = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 1)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby("peptide_id")
                        .filter(lambda x: len(x["protein_id"].unique()) == 1)
                        .groupby("protein_id")
                        .first()[score_col]
                        .to_numpy()
                    )
                elif base_key in ("protein_experiment_wide", "protein_run_specific"):
                    top_targets_df = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 0)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby(["protein_id", "run_id"])
                        .first()
                        .reset_index()
                    )
                    d_cutoff_at_1_pcnt = top_targets_df[score_col].to_numpy().min()
                    top_targets = top_targets_df[base_id]
                    top_decoys = (
                        df_valid[
                            (df_valid["precursor_decoy"] == 1)
                            & (df_valid[q_value_col] <= 0.01)
                            & (df_valid["score_ms2_peak_group_rank"] == 1)
                        ]
                        .groupby(["protein_id", "run_id"])
                        .first()[score_col]
                        .to_numpy()
                    )

                logger.info(
                    f"Total assays: {df_valid.shape[0]}, top targets at 1%: {len(top_targets)}, top decoys at 1%: {len(top_decoys)}, cutoff: {d_cutoff_at_1_pcnt:.2f}"
                )
                stats_text = f"Targets: {len(top_targets):,}\nDecoys: {len(top_decoys):,}\nCutoff @ 1%: {d_cutoff_at_1_pcnt:.2f}\n"
                ax.axvline(
                    x=d_cutoff_at_1_pcnt, color="grey", linestyle="--", linewidth=2
                )
                y_max = ax.get_ylim()[1]
                ax.text(
                    d_cutoff_at_1_pcnt + 0.05,
                    y_max * 0.95,
                    f"Cutoff @ 1%: {d_cutoff_at_1_pcnt:.2f}",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(facecolor="lightgray", alpha=0.5),
                )

            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        except Exception as e:
            logger.error(
                f"Failed to plot {base_key} distribution ({score_col}): {str(e)}"
            )
            ax.text(
                0.5,
                0.5,
                f"Error plotting {base_key}\n({score_col})",
                ha="center",
                va="center",
                color="red",
            )
            ax.set_title(f"{base_key} (error)")

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def prepare_filtered_df(df, level, q_value_cols):
    """Helper function to prepare filtered dataframe for a given level."""
    # Check which q-value column is available
    available_q_value_col = None
    for col in q_value_cols:
        if col in df.columns:
            available_q_value_col = col
            break

    if not available_q_value_col:
        raise ValueError(
            f"No q-value columns found for {level} level. Tried: {q_value_cols}"
        )

    # Filter the dataframe
    filtered_df = df[
        (df["precursor_decoy"] == 0)
        & (df["score_ms2_peak_group_rank"] == 1)
        & (df["score_ms2_q_value"] <= 0.01)
        & (df[available_q_value_col] <= 0.01)
    ].copy()

    logger.trace(
        f"Filtered DataFrame for {level} level with the following filters:\nprecursor_decoy == 0\nscore_ms2_peak_group_rank == 1\nscore_ms2_q_value <= 0.01\n{available_q_value_col} <= 0.01"
    )

    if (
        "score_ipf_qvalue" in filtered_df.columns
        and filtered_df["score_ipf_qvalue"].notna().all()
    ):
        logger.opt(raw=True, colors=True).trace("score_ipf_qvalue <= 0.01\n")
        filtered_df = filtered_df[filtered_df["score_ipf_qvalue"] <= 0.01].copy()

    # Get the most intense feature per group
    if level in ("peptide", "ipf"):
        max_indices = filtered_df.groupby(["peptide_id", "run_id"])[
            "feature_ms2_area_intensity"
        ].idxmax()
        filtered_df = filtered_df.loc[max_indices].reset_index(drop=True)
        id_key = "peptide_id"
    elif level == "protein":
        max_indices = filtered_df.groupby(["protein_id", "run_id"])[
            "feature_ms2_area_intensity"
        ].idxmax()
        filtered_df = filtered_df.loc[max_indices].reset_index(drop=True)
        id_key = "protein_id"
    else:  # precursor level
        id_key = "precursor_id"

    # Rename intensity column if present
    if "feature_ms2_area_intensity" in filtered_df.columns:
        filtered_df.rename(
            columns={"feature_ms2_area_intensity": "area_intensity"}, inplace=True
        )

    return filtered_df, id_key


def plot_identification_quantification(pdf, plotter, df, level):
    """Helper function to plot identification and quantification results."""
    try:
        if level == "peptide":
            q_value_cols = [
                "score_peptide_global_q_value",
                "score_peptide_experiment_wide_q_value",
                "score_peptide_run_specific_q_value",
            ]
            title = "Peptide Identification and Quantification Results"
        elif level == "ipf":
            q_value_cols = [
                "score_ipf_qvalue",
            ]
            title = "IPF Identification and Quantification Results"
        elif level == "protein":
            q_value_cols = [
                "score_protein_global_q_value",
                "score_protein_experiment_wide_q_value",
                "score_protein_run_specific_q_value",
            ]
            title = "Protein Identification and Quantification Results"
        else:  # precursor
            q_value_cols = ["score_ms2_q_value"]
            title = "Precursor Identification and Quantification Results"

        try:
            filtered_df, id_key = prepare_filtered_df(df, level, q_value_cols)
        except ValueError as e:
            logger.warning(f"Skipping {level} level plots: {str(e)}")
            return

        # First set of plots
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(title, fontsize=14)
            plotter.add_id_barplot(axes[0, 0], filtered_df, id_key)
            plotter.plot_identification_consistency(axes[0, 1], filtered_df, id_key)
            plotter.add_violinplot(axes[1, 0], filtered_df, id_key)
            plotter.plot_cv_distribution(axes[1, 1], filtered_df, id_key)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate first set of {level} plots: {str(e)}")

        # Second set of plots
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            plotter.plot_jaccard_similarity(axes[0], filtered_df, id_key)
            plotter.plot_intensity_correlation(axes[1], filtered_df, id_key)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to generate second set of {level} plots: {str(e)}")

    except Exception as e:
        logger.error(f"Failed to generate {level} level plots: {str(e)}")


def create_summary_table(pdf, df):
    """Create a summary table of ID counts per run and overall statistics."""
    try:
        # Prepare filtered dataframes for each level
        precursor_df = df[
            (df["precursor_decoy"] == 0)
            & (df["score_ms2_peak_group_rank"] == 1)
            & (df["score_ms2_q_value"] <= 0.01)
        ].copy()

        logger.trace(
            "Filtered precursor_df with the following conditions:\nprecursor_decoy == 0\nscore_ms2_peak_group_rank == 1\nscore_ms2_q_value <= 0.01"
        )

        if "score_ipf_qvalue" in df.columns and df["score_ipf_qvalue"].notna().all():
            logger.opt(raw=True, colors=True).trace("score_ipf_qvalue <= 0.01\n")
            precursor_df = precursor_df[precursor_df["score_ipf_qvalue"] <= 0.01].copy()

        # Try to get peptide level with best available q-value
        peptide_q_cols = [
            "score_peptide_global_q_value",
            "score_peptide_experiment_wide_q_value",
            "score_peptide_run_specific_q_value",
        ]
        peptide_q_col = next((col for col in peptide_q_cols if col in df.columns), None)
        if peptide_q_col:
            peptide_df = df[
                (df["precursor_decoy"] == 0)
                & (df["score_ms2_peak_group_rank"] == 1)
                & (df["score_ms2_q_value"] <= 0.01)
                & (df[peptide_q_col] <= 0.01)
            ].copy()

            logger.trace(
                f"Filtered peptide_df with the following conditions:\nprecursor_decoy == 0\nscore_ms2_peak_group_rank == 1\nscore_ms2_q_value <= 0.01\n{peptide_q_col} <= 0.01"
            )

            if (
                "score_ipf_qvalue" in df.columns
                and df["score_ipf_qvalue"].notna().all()
            ):
                logger.opt(raw=True, colors=True).trace("score_ipf_qvalue <= 0.01\n")
                peptide_df = peptide_df[peptide_df["score_ipf_qvalue"] <= 0.01].copy()

            peptide_df = peptide_df.loc[
                peptide_df.groupby(["peptide_id", "run_id"])[
                    "feature_ms2_area_intensity"
                ].idxmax()
            ].reset_index(drop=True)
        else:
            peptide_df = pd.DataFrame(columns=["peptide_id", "run_id"])

        # Try to get protein level with best available q-value
        protein_q_cols = [
            "score_protein_global_q_value",
            "score_protein_experiment_wide_q_value",
            "score_protein_run_specific_q_value",
        ]
        protein_q_col = next((col for col in protein_q_cols if col in df.columns), None)
        if protein_q_col:
            protein_df = df[
                (
                    (df["precursor_decoy"] == 0)
                    & (df["score_ms2_peak_group_rank"] == 1)
                    & (df["score_ms2_q_value"] <= 0.01)
                    & (df[peptide_q_col] <= 0.01)
                    if peptide_q_col
                    else True & (df[protein_q_col] <= 0.01)
                )
            ].copy()

            logger.trace(
                f"Filtered protein_df with the following conditions:\nprecursor_decoy == 0\nscore_ms2_peak_group_rank == 1\nscore_ms2_q_value <= 0.01\n{protein_q_col} <= 0.01"
            )
            if (
                "score_ipf_qvalue" in df.columns
                and df["score_ipf_qvalue"].notna().all()
            ):
                logger.opt(raw=True, colors=True).trace("score_ipf_qvalue <= 0.01\n")
                protein_df = protein_df[protein_df["score_ipf_qvalue"] <= 0.01].copy()

            protein_df = protein_df.loc[
                protein_df.groupby(["protein_id", "run_id"])[
                    "feature_ms2_area_intensity"
                ].idxmax()
            ].reset_index(drop=True)
        else:
            protein_df = pd.DataFrame(columns=["protein_id", "run_id"])

        # Get unique run IDs
        run_ids = sorted(df["run_id"].unique())

        # Create summary data
        summary_data = []

        # Add per-run counts
        for run_id in run_ids:
            precursor_count = precursor_df[precursor_df["run_id"] == run_id][
                "precursor_id"
            ].nunique()
            peptide_count = (
                peptide_df[peptide_df["run_id"] == run_id]["peptide_id"].nunique()
                if "peptide_id" in peptide_df.columns
                else 0
            )
            protein_count = (
                protein_df[protein_df["run_id"] == run_id]["protein_id"].nunique()
                if "protein_id" in protein_df.columns
                else 0
            )
            logger.info(
                f"Run ID: {run_id}, Precursors: {precursor_count}, Peptides: {peptide_count}, Proteins: {protein_count}"
            )

            summary_data.append(
                {
                    "Run ID": run_id,
                    "Precursors": precursor_count,
                    "Peptides": peptide_count,
                    "Proteins": protein_count,
                }
            )

        # Add total unique counts across all runs
        total_precursors = precursor_df["precursor_id"].nunique()
        total_peptides = (
            peptide_df["peptide_id"].nunique()
            if "peptide_id" in peptide_df.columns
            else 0
        )
        total_proteins = (
            protein_df["protein_id"].nunique()
            if "protein_id" in protein_df.columns
            else 0
        )
        summary_data.append(
            {
                "Run ID": "Total Unique",
                "Precursors": total_precursors,
                "Peptides": total_peptides,
                "Proteins": total_proteins,
            }
        )

        # Add intersection counts (IDs present in all runs)
        if len(run_ids) > 1:
            # Precursor intersection
            precursor_sets = [
                set(precursor_df[precursor_df["run_id"] == run_id]["precursor_id"])
                for run_id in run_ids
            ]
            precursor_intersection = len(set.intersection(*precursor_sets))

            # Peptide intersection
            if "peptide_id" in peptide_df.columns:
                peptide_sets = [
                    set(peptide_df[peptide_df["run_id"] == run_id]["peptide_id"])
                    for run_id in run_ids
                ]
                peptide_intersection = len(set.intersection(*peptide_sets))
            else:
                peptide_intersection = 0

            # Protein intersection
            if "protein_id" in protein_df.columns:
                protein_sets = [
                    set(protein_df[protein_df["run_id"] == run_id]["protein_id"])
                    for run_id in run_ids
                ]
                protein_intersection = len(set.intersection(*protein_sets))
            else:
                protein_intersection = 0

            summary_data.append(
                {
                    "Run ID": "Intersection (All Runs)",
                    "Precursors": precursor_intersection,
                    "Peptides": peptide_intersection,
                    "Proteins": protein_intersection,
                }
            )

        # Create DataFrame from summary data
        summary_df = pd.DataFrame(summary_data)

        # Create figure for the table
        fig, ax = plt.subplots(figsize=(15, max(4, len(summary_data) * 0.5)))
        ax.axis("tight")
        ax.axis("off")

        # Create table
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
            colColours=["#f7f7f7"] * len(summary_df.columns),
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # Add title
        plt.title("ID Count Summary (FDR ≤ 1%)", pad=20)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to generate summary table: {str(e)}")


def post_scoring_report(df, outfile, color_palette="normal"):
    # Convert all column names to lowercase at the beginning
    df.columns = df.columns.str.lower()

    valid_scores = df["precursor_id"].notna()
    df = df[valid_scores]

    # Check which scores are present in the DataFrame
    score_columns = [col for col in df.columns if col.startswith("score_")]

    if not score_columns:
        logger.critical(
            "No score columns found in the DataFrame. Cannot generate post-scoring report."
        )
        sys.exit(1)

    logger.trace(f"Found score columns: {score_columns}")

    plotter = PlotGenerator(color_palette)

    # Create a mapping of score types to their preferred column names (all lowercase)
    score_preferences = {
        "ms1": {"score": "score_ms1_score", "id": "precursor_id"},
        "ms2": {"score": "score_ms2_score", "id": "precursor_id"},
        "ipf": {"score": "score_ipf_qvalue", "id": "peptide_id"},
        "peptide_global": {"score": "score_peptide_global_score", "id": "peptide_id"},
        "peptide_experiment_wide": {
            "score": "score_peptide_experiment_wide_score",
            "id": "peptide_id",
        },
        "peptide_run_specific": {
            "score": "score_peptide_run_specific_score",
            "id": "peptide_id",
        },
        "protein_global": {"score": "score_protein_global_score", "id": "protein_id"},
        "protein_experiment_wide": {
            "score": "score_protein_experiment_wide_score",
            "id": "protein_id",
        },
        "protein_run_specific": {
            "score": "score_protein_run_specific_score",
            "id": "protein_id",
        },
    }

    # Find available scores based on preferences
    score_mapping = {}
    for score_type, preferences in score_preferences.items():
        logger.trace(f"Checking for score type: {score_type}")
        if preferences["score"] in df.columns:
            logger.trace(f"Found score column for {score_type}: {preferences['score']}")
            # Verify the ID column exists too
            if preferences["id"] in df.columns:
                score_mapping[score_type] = {
                    "score": preferences["score"],
                    "id": preferences["id"],
                }
            else:
                logger.warning(
                    f"ID column {preferences['id']} not found for {score_type}"
                )

    if not score_mapping:
        logger.critical("No valid score columns found for plotting")
        sys.exit(1)

    with PdfPages(outfile) as pdf:
        # Add summary table
        create_summary_table(pdf, df)

        # Plot score distributions
        plot_score_distributions(pdf, plotter, df, score_mapping)

        # Plot identification and quantification results for each level
        plot_identification_quantification(pdf, plotter, df, "precursor")
        # Check if ipf scores are present in score_mapping
        if "ipf" in score_mapping:
            plot_identification_quantification(pdf, plotter, df, "ipf")
        plot_identification_quantification(pdf, plotter, df, "peptide")
        plot_identification_quantification(pdf, plotter, df, "protein")

    logger.info(f"Report generated successfully: {outfile}")


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
    output = PdfWriter()
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
