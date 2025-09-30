#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description:
    Generate a PDF report with m/z, ion mobility, and retention time calibration plots
    from OpenSwathWorkflow debugging files.

CLI:
    python openswathworkflow_calibration_report.py --help
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Optional imports (fail early with a clear message) ---
try:
    import pyopenms as pom
except Exception as e:
    pom = None
    _PYOPENMS_ERR = e


# ----------------------------
# Utilities & helper functions
# ----------------------------


def verbose_print(vlevel: int, threshold: int, msg: str):
    if vlevel >= threshold:
        print(msg, flush=True)


def not_length_0(x) -> bool:
    try:
        return len(x) > 0
    except Exception:
        return False


def read_pairs_from_trafoxml(filename: str) -> pd.DataFrame:
    """
    Read (from, to) RT pairs from a trafoXML file (OpenMS TransformationDescription).
    Returns DataFrame with columns: ['from', 'to'].
    """
    # The OpenMS trafoXML encodes pairs as elements with attributes from/to
    parser = etree.XMLParser(remove_blank_text=True, huge_tree=True)
    tree = etree.parse(filename, parser)
    pairs = []
    for elm in tree.findall(".//Pair"):
        f = float(elm.get("from"))
        t = float(elm.get("to"))
        pairs.append((f, t))
    if not pairs:
        return pd.DataFrame(columns=["from", "to"])
    return pd.DataFrame(pairs, columns=["from", "to"])


def argmax_peaks(
    rt: np.ndarray, inten: np.ndarray, w: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rough equivalent of the R argmax logic:
      - Smooth with Savitzky–Golay (p=3, n=11)
      - Rolling max with window 2*w+1
      - Peak indices where smoothed equals local max (plateaus allowed)
    Returns (peak_indices, y_smooth)
    """
    if len(rt) == 0:
        return np.array([], dtype=int), inten

    # Smooth, clamp negatives to 0 (to match R side)
    yhat = savgol_filter(
        inten,
        window_length=min(11, len(inten) - (len(inten) + 1) % 2),
        polyorder=3,
        mode="interp",
    )
    yhat = np.clip(yhat, 0, None)

    # Rolling maximum
    k = 2 * w + 1
    if k <= 1 or k > len(yhat):
        # trivial window — pick global max
        return np.array([int(np.argmax(yhat))]), yhat

    # compute rolling max using a simple approach
    # center-aligned: positions from w .. n-w-1 valid
    y_roll_max = np.maximum.reduce(
        [np.r_[yhat[i:], np.full(i, -np.inf)] for i in range(k)]
    )
    # Shift to center alignment: our construction gives a causal max; approximate center by shifting by w
    y_roll_max = y_roll_max[: len(yhat)]
    # Peak where smoothed == local max (within epsilon)
    eps = 1e-12
    peak_mask = yhat >= y_roll_max - eps
    # To prevent long plateaus -> pick local ridge indices
    peak_idx = np.where(peak_mask)[0]

    if peak_idx.size == 0:
        return np.array([], dtype=int), yhat

    # Thin peaks: keep only indices that are local maxima vs neighbors
    keep = []
    for i in peak_idx:
        l = yhat[i - 1] if i - 1 >= 0 else -np.inf
        r = yhat[i + 1] if i + 1 < len(yhat) else -np.inf
        if yhat[i] >= l and yhat[i] >= r:
            keep.append(i)
    if not keep:
        keep = [int(np.argmax(yhat))]
    return np.array(sorted(set(keep)), dtype=int), yhat


@dataclass
class XICRow:
    rt: float
    intensity: float
    native_id: str
    peptide_sequence: str
    precursor_mz: float
    precursor_charge: int
    product_mz: float


def load_xics_from_mzml(
    mzml_path: str,
    peptide_list: Optional[List[str]],
    prec_mz_filter: Optional[List[float]],
    prod_mz_filter: Optional[List[float]],
    max_precursors: int = 12,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Use pyopenms to read chromatograms and build a tidy DataFrame.
    Columns: ['rt','int','native_id','peptide_sequence','precursor_mz',
              'precursor_charge','product_mz'].
    Picks up to `max_precursors` distinct precursors (peptide, mz, charge),
    first honoring any peptide_list, then topping up randomly.
    """
    if pom is None:
        raise RuntimeError(
            f"pyopenms import failed. Ensure pyopenms is installed and importable. Original error: {_PYOPENMS_ERR}"
        )

    exp = pom.MSExperiment()
    pom.MzMLFile().load(mzml_path, exp)
    num_chroms = exp.getNrChromatograms()
    verbose_print(verbose, 1, f"INFO: There are {num_chroms} chromatograms found!")
    chroms = list(exp.getChromatograms())

    # ------- metadata over all chromatograms
    meta_rows = []
    for idx, ch in enumerate(chroms, start=1):
        prec = ch.getPrecursor()
        prod = ch.getProduct()
        native_id = ch.getNativeID()
        pep = (
            str(prec.getMetaValue("peptide_sequence"))
            if prec.metaValueExists("peptide_sequence")
            else ""
        )
        if verbose >= 2:
            verbose_print(
                verbose,
                2,
                f"INFO: chrom {idx}/{num_chroms} - {native_id} - {pep} ({prec.getMZ()})",
            )
        meta_rows.append(
            (
                idx,
                native_id,
                pep,
                float(prec.getMZ()),
                int(prec.getCharge()),
                float(prod.getMZ()),
            )
        )

    meta = pd.DataFrame(
        meta_rows,
        columns=[
            "chrom_index",
            "native_id",
            "peptide_sequence",
            "precursor_mz",
            "precursor_charge",
            "product_mz",
        ],
    )

    # Optional product m/z filtering at the chromatogram level
    if prod_mz_filter:
        _pf = np.round(np.asarray(prod_mz_filter, dtype=float), 6)
        meta = meta[np.round(meta["product_mz"], 6).isin(_pf)]

    # ------- select distinct precursors (peptide+mz+charge)
    prec_table = (
        meta[["chrom_index", "peptide_sequence", "precursor_mz", "precursor_charge"]]
        .drop_duplicates(
            subset=["peptide_sequence", "precursor_mz", "precursor_charge"]
        )
        .copy()
    )

    # 1) keep ones whose peptide is in the provided list
    selected = pd.DataFrame(columns=prec_table.columns)
    if peptide_list:
        selected = prec_table[prec_table["peptide_sequence"].isin(peptide_list)].copy()

    # 2) top up with random distinct precursors
    need = max(0, max_precursors - len(selected))
    if need > 0:
        pool = prec_table.copy()

        # apply precursor m/z filter if provided
        if prec_mz_filter:
            _mf = np.round(np.asarray(prec_mz_filter, dtype=float), 6)
            pool = pool[np.round(pool["precursor_mz"], 6).isin(_mf)]
            # if filtering wipes out the pool, fall back to full table
            if pool.empty:
                pool = prec_table.copy()

        # remove ones already selected
        if not selected.empty:
            already = set(
                zip(
                    selected["peptide_sequence"],
                    selected["precursor_mz"],
                    selected["precursor_charge"],
                )
            )
            mask = ~pool.apply(
                lambda r: (
                    r["peptide_sequence"],
                    r["precursor_mz"],
                    r["precursor_charge"],
                )
                in already,
                axis=1,
            )
            pool = pool[mask]

        if len(pool) > 0:
            add_idx = np.random.choice(
                pool.index, size=min(need, len(pool)), replace=False
            )
            selected = pd.concat([selected, pool.loc[add_idx]], ignore_index=True)

    if selected.empty:
        selected = prec_table.head(max_precursors).copy()

    # Bring back all chromatograms (product ions) that belong to the selected precursors
    meta_sel = meta.merge(
        selected[["peptide_sequence", "precursor_mz", "precursor_charge"]],
        on=["peptide_sequence", "precursor_mz", "precursor_charge"],
        how="inner",
    )
    selected_chroms = meta_sel["chrom_index"].astype(int).unique().tolist()

    # ------- extract XICs
    rows: List[XICRow] = []  # IMPORTANT: initialize before appending
    for idx in selected_chroms:
        ch = chroms[idx - 1]
        rt, ints = map(np.asarray, ch.get_peaks())

        # Smooth and clamp negatives
        if len(ints) >= 5:
            win = (
                11
                if len(ints) >= 11
                else (len(ints) if len(ints) % 2 == 1 else len(ints) - 1)
            )
            if win >= 5:
                ints = savgol_filter(
                    ints, window_length=win, polyorder=min(3, win - 2), mode="interp"
                )
        ints = np.clip(ints, 0, None)

        prec = ch.getPrecursor()
        prod = ch.getProduct()
        native_id = ch.getNativeID()
        pep = (
            str(prec.getMetaValue("peptide_sequence"))
            if prec.metaValueExists("peptide_sequence")
            else ""
        )
        prec_mz = float(prec.getMZ())
        charge = int(prec.getCharge())
        prod_mz = float(prod.getMZ())

        for rti, ii in zip(rt.astype(float), ints.astype(float)):
            rows.append(
                XICRow(
                    rt=float(rti),
                    intensity=float(ii),
                    native_id=str(native_id),
                    peptide_sequence=pep,
                    precursor_mz=prec_mz,
                    precursor_charge=charge,
                    product_mz=prod_mz,
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "rt",
                "int",
                "native_id",
                "peptide_sequence",
                "precursor_mz",
                "precursor_charge",
                "product_mz",
            ]
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.rename(columns={"intensity": "int"}, inplace=True)
    return df


def find_rt_apex_by_coelution(peaks_df: pd.DataFrame, verbose: int = 0) -> pd.DataFrame:
    """
    For each (peptide_sequence, precursor_mz, precursor_charge), find co-eluting apex:
      - Within each product_mz group, pick argmax peaks on smoothed XIC
      - Build a table of apex RTs; group by RT bins (35 s)
      - Choose the bin with highest count (co-elution). If none, RT=NaN.
    Returns a DataFrame with columns: peptide_sequence, precursor_mz, precursor_charge, RT
    """
    if peaks_df.empty:
        return pd.DataFrame(
            columns=["peptide_sequence", "precursor_mz", "precursor_charge", "RT"]
        )

    # Group by precursor (peptide + precMz + charge)
    out_rows = []
    grouped_prec = peaks_df.groupby(
        ["peptide_sequence", "precursor_mz", "precursor_charge"], dropna=False
    )
    verbose_print(
        verbose, 1, f"INFO: Finding Peak Apexes for {len(grouped_prec)} precursors"
    )
    for (pep, precmz, chg), g in grouped_prec:
        # For each product_mz, compute peaks
        apex_records = []
        for prod_mz, gg in g.groupby("product_mz", dropna=False):
            rt = gg["rt"].to_numpy()
            it = gg["int"].to_numpy()
            idx, yhat = argmax_peaks(rt, it, w=1)
            if idx.size > 0:
                # Take max intensity among the found indices
                pick = idx[np.argmax(yhat[idx])]
                apex_records.append((rt[pick], it[pick], prod_mz))
        if not apex_records:
            out_rows.append((pep, precmz, chg, np.nan))
            continue

        apex_df = pd.DataFrame(apex_records, columns=["RT", "Int", "product_mz"])
        apex_df = apex_df[apex_df["Int"] > 0]

        if apex_df.empty:
            out_rows.append((pep, precmz, chg, np.nan))
            continue

        # Co-elution binning @ 35 sec
        if len(apex_df) >= 6:
            bin_edges = np.arange(
                apex_df["RT"].min() - 1, apex_df["RT"].max() + 35, 35.0
            )
            if len(bin_edges) < 2:
                # fallback: no binning possible
                best_rt = float(apex_df.loc[apex_df["Int"].idxmax(), "RT"])
            else:
                cats = pd.cut(apex_df["RT"], bins=bin_edges, include_lowest=True)
                counts = apex_df.groupby(cats, observed=False).size()
                if counts.empty or counts.max() < 3:
                    # fallback
                    best_rt = float(apex_df.loc[apex_df["Int"].idxmax(), "RT"])
                else:
                    # use center of the densest bin
                    densest = counts.idxmax()
                    best_rt = float((densest.left + densest.right) / 2.0)
        else:
            # too few, just take the max-intensity apex
            best_rt = float(apex_df.loc[apex_df["Int"].idxmax(), "RT"])
        out_rows.append((pep, precmz, chg, best_rt))
    return pd.DataFrame(
        out_rows, columns=["peptide_sequence", "precursor_mz", "precursor_charge", "RT"]
    )


def zoom_xic_around_rt(df: pd.DataFrame, window_sec: float = 50.0) -> pd.DataFrame:
    """
    If an 'RT' column exists (per precursor apex), shrink each precursor's traces
    to [RT-window, RT+window]. If RT is NaN for a group, keep as-is.
    """
    if "RT" not in df.columns or df.empty:
        return df
    out = []
    for (pep, precmz, chg), g in df.groupby(
        ["peptide_sequence", "precursor_mz", "precursor_charge"], dropna=False
    ):
        rtvals = g["RT"].dropna().unique()
        if rtvals.size > 0:
            rt_center = float(np.mean(rtvals))
            sub = g[
                (g["rt"] > rt_center - window_sec) & (g["rt"] < rt_center + window_sec)
            ].copy()
            out.append(sub)
        else:
            out.append(g)
    if not out:
        return df
    return pd.concat(out, ignore_index=True)


def linfit_and_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Linear regression y = a*x + b and Pearson R (from covariance).
    Returns (a, b, r).
    """
    if x.size < 2 or y.size < 2:
        return (np.nan, np.nan, np.nan)
    a, b = np.polyfit(x, y, 1)
    # Pearson r
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    r = float(np.sum(xm * ym) / denom) if denom > 0 else np.nan
    return a, b, r


def annotate_fit(ax, a: float, b: float, r: float, loc: str = "upper left"):
    """
    Write 'R' and the regression equation in axes coords with a light
    background box so they never overlap points. 'loc' can be:
    'upper left', 'upper right', 'lower left', 'lower right'.
    """
    if any(np.isnan(v) for v in (a, b, r)):
        return

    # axes-fraction anchors (x,y)
    anchors = {
        "upper left": (0.02, 0.98),
        "upper right": (0.98, 0.98),
        "lower left": (0.02, 0.02),
        "lower right": (0.98, 0.02),
    }
    x, y = anchors.get(loc, (0.02, 0.98))

    # two-line label; use a narrow box to avoid covering points
    txt = f"R = {r:.3f}\ny = {a:.4f} × x + {b:.4f}"
    ax.text(
        x,
        y,
        txt,
        transform=ax.transAxes,
        ha="right" if "right" in loc else "left",
        va="top" if "upper" in loc else "bottom",
        fontsize=9,
        bbox=dict(
            facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.25", alpha=0.9
        ),
        zorder=5,
    )


def add_hist_inset(
    ax,
    data,
    title="Δ",
    width=0.63,
    height=0.6,
    loc="lower right",
    *,
    pad=0.02,
    dx=0.0,
    dy=-0.18,
):
    """
    width/height are fractions of the parent axes (0..1).
    pad is the margin from the edges in axes-fraction.
    dx, dy shift the inset position (also axes-fraction).
    Positive dy moves up; negative dy moves down.
    """
    # base anchor by corner
    corners = {
        "lower right": (1.0 - width - pad, pad),
        "lower left": (pad, pad),
        "upper right": (1.0 - width - pad, 1.0 - height - pad),
        "upper left": (pad, 1.0 - height - pad),
    }
    x0, y0 = corners[loc]
    x0 += dx
    y0 += dy  # <-- lower the inset by making dy negative

    ins = inset_axes(
        ax,
        width=f"{int(width * 100)}%",
        height=f"{int(height * 100)}%",
        bbox_to_anchor=(x0, y0, width, height),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ins.hist(np.asarray(data, dtype=float), bins="auto")
    ins.set_title(title, fontsize=8, pad=2)
    ins.tick_params(labelsize=7)
    for s in ins.spines.values():
        s.set_linewidth(0.8)
    return ins


# ----------------------------
# Main report generation
# ----------------------------


def generate_report(
    wd: str = "./",
    im_cal_pattern: str = "_debug_im.txt",
    mz_cal_pattern: str = "_debug_mz.txt",
    irt_trafo_pattern: str = ".trafoXML",
    irt_mzml_pattern: str = "_irt_chrom.mzML",
    report_file: str = "calibration_report.pdf",
    zoom_in_xic: bool = True,
    verbose: int = 0,
) -> None:
    """
    Generate a PDF report with calibration plots from OpenSwathWorkflow debug files.

    Parameters
    ----------
    wd : str
        Working directory to search for files.
    im_cal_pattern : str
        Pattern for IM calibration files.
    mz_cal_pattern : str
        Pattern for m/z calibration files.
    irt_trafo_pattern : str
        Pattern for iRT transformation files.
    irt_mzml_pattern : str
        Pattern for iRT chromatogram mzML files.
    report_file : str
        Output PDF report filename.
    zoom_in_xic : bool
        Whether to zoom into XICs around detected apexes.
    verbose : int
        Verbosity level (0=silent, 1=info, 2=debug).

    Returns
    -------
    None
    """
    start_t = pd.Timestamp.now()
    verbose_print(verbose, 1, f"INFO: Working directory: {wd}")

    # Find files
    file_im_cal_list = sorted(glob.glob(os.path.join(wd, f"*{im_cal_pattern}")))
    file_mz_cal_list = sorted(glob.glob(os.path.join(wd, f"*{mz_cal_pattern}")))
    file_irt_trafo_list = sorted(glob.glob(os.path.join(wd, f"*{irt_trafo_pattern}")))
    file_irt_mzml_list = sorted(glob.glob(os.path.join(wd, f"*{irt_mzml_pattern}")))

    # Deduce unique run names by stripping any of the patterns from the filenames
    def strip_patterns(fp: str) -> str:
        base = os.path.basename(fp)
        for pat in [
            im_cal_pattern,
            mz_cal_pattern,
            irt_trafo_pattern,
            irt_mzml_pattern,
        ]:
            base = base.replace(pat, "")
        return base

    unique_runs = sorted(
        set(
            map(
                strip_patterns,
                file_im_cal_list
                + file_mz_cal_list
                + file_irt_trafo_list
                + file_irt_mzml_list,
            )
        )
    )
    print(
        f"INFO: Found {len(unique_runs)} unique runs to generate reports for",
        flush=True,
    )

    # iRT peptide lists
    biognosis = [
        "ADVTPADFSEWSK",
        "DGLDAASYYAPVR",
        "GAGSSEPVTGLDAK",
        "GTFIIDPAAVIR",
        "GTFIIDPGGVIR",
        "LFLQFGAQGSPFLK",
        "LGGNEQVTR",
        "TPVISGGPYEYR",
        "TPVITGAPYEYR",
        "VEATFGVDESNAK",
        "YILAGVENSK",
    ]
    thermo_pierce = [
        "SSAAPPPPPR",
        "GISNEGQNASIK",
        "HVLTSIGEK",
        "DIPVPKPK",
        "IGDYAGIK",
        "TASEFDSAIAQDK",
        "SAAGAFGPELSR",
        "ELGQSGVDTYLQTK",
        "GLILVGGYGTR",
        "GILFVGSGVSGGEEGAR",
        "SFANQPLEVVYSK",
        "LTILEELR",
        "NGFILDGFPR",
        "ELASGLSFPVGFK",
        "LSSEAPALFQFDLK",
    ]
    irt_standards = list(dict.fromkeys(biognosis + thermo_pierce))

    with PdfPages(report_file) as pdf:
        for run in unique_runs:
            # Resolve matching files
            f_im = [p for p in file_im_cal_list if os.path.basename(p).startswith(run)]
            f_mz = [p for p in file_mz_cal_list if os.path.basename(p).startswith(run)]
            f_tr = [
                p for p in file_irt_trafo_list if os.path.basename(p).startswith(run)
            ]
            f_xic = [
                p for p in file_irt_mzml_list if os.path.basename(p).startswith(run)
            ]

            f_im = f_im[0] if f_im else None
            f_mz = f_mz[0] if f_mz else None
            f_tr = f_tr[0] if f_tr else None
            f_xic = f_xic[0] if f_xic else None

            print(
                "---------------------------------------------------------------------",
                flush=True,
            )
            print(f"INFO: Processing run - {run}", flush=True)
            print(
                f"INFO: IM calibration file - {f_im if f_im else 'Not Found!'}",
                flush=True,
            )
            print(
                f"INFO: MZ calibration file - {f_mz if f_mz else 'Not Found!'}",
                flush=True,
            )
            print(
                f"INFO: iRT transformation file - {f_tr if f_tr else 'Not Found!'}",
                flush=True,
            )
            print(
                f"INFO: iRT XIC mzML - {f_xic if f_xic else 'Not Found!'}", flush=True
            )

            # --- Read IM calibration ---
            dt_im = pd.DataFrame()
            if f_im and os.path.isfile(f_im):
                dt_im = pd.read_csv(f_im, sep=r"\s+|,|\t", engine="python")
                if {"im", "theo_im"}.issubset(dt_im.columns):
                    dt_im["delta_im"] = dt_im["im"] - dt_im["theo_im"]

            # --- Read m/z calibration ---
            dt_mz = pd.DataFrame()
            if f_mz and os.path.isfile(f_mz):
                dt_mz = pd.read_csv(f_mz, sep=r"\s+|,|\t", engine="python")
                if {"mz", "theo_mz"}.issubset(dt_mz.columns):
                    dt_mz["delta_mz"] = dt_mz["mz"] - dt_mz["theo_mz"]

            # --- Read iRT trafo ---
            dt_tr = pd.DataFrame()
            if f_tr and os.path.isfile(f_tr):
                dt_tr = read_pairs_from_trafoxml(f_tr)
                if not dt_tr.empty:
                    dt_tr["delta"] = dt_tr["to"] - dt_tr["from"]
                    dt_tr.rename(columns={"from": "RT"}, inplace=True)

            # If IM table has RT + mz, enrich trafo with mz
            if (
                not dt_tr.empty
                and not dt_im.empty
                and {"RT", "mz"}.issubset(dt_im.columns)
            ):
                # Use unique RT↔mz mapping (drop duplicated RTs)
                rt_mz = dt_im[["RT", "mz"]].drop_duplicates()
                dt_tr = rt_mz.merge(dt_tr, on="RT", how="right")
            elif not dt_tr.empty:
                # ensure 'mz' exists
                dt_tr["mz"] = np.nan

            # --- XIC extraction & linking ---
            dt_xic = pd.DataFrame()
            if f_xic and os.path.isfile(f_xic):
                # Try to provide helpful product_mz filter from m/z calibration
                prod_mz_filter = (
                    dt_mz["theo_mz"].unique().tolist()
                    if "theo_mz" in dt_mz.columns and not dt_mz.empty
                    else None
                )
                # Precursor m/z candidates from im/trafo enrichment
                prec_mz_filter = None
                if not dt_tr.empty and "mz" in dt_tr.columns:
                    # Exclude mzs with multiple RTs like the R script
                    multi_rt_mz = (
                        dt_tr[["RT", "mz"]]
                        .dropna()
                        .drop_duplicates()
                        .groupby("mz")
                        .size()
                    )
                    exclude = set(multi_rt_mz[multi_rt_mz > 1].index.tolist())
                    cand = [
                        float(x)
                        for x in dt_tr["mz"].dropna().unique().tolist()
                        if x not in exclude
                    ]
                    prec_mz_filter = cand if cand else None

                dt_xic = load_xics_from_mzml(
                    f_xic,
                    peptide_list=biognosis,  # prefer biognosis set default
                    prec_mz_filter=prec_mz_filter,
                    prod_mz_filter=prod_mz_filter,
                    verbose=verbose,
                )

                # Merge a better RT per precursor: prefer trafo RT if available, else argmax
                if not dt_xic.empty:
                    # Compute apex by co-elution if needed
                    apex = find_rt_apex_by_coelution(dt_xic, verbose=verbose)
                    apex.rename(columns={"RT": "RT_argmax"}, inplace=True)
                    dt_xic = dt_xic.merge(
                        apex,
                        on=["peptide_sequence", "precursor_mz", "precursor_charge"],
                        how="left",
                    )

                    if (
                        not dt_tr.empty
                        and "mz" in dt_tr.columns
                        and not dt_tr["mz"].isna().all()
                    ):
                        # trafo merged may contain multiple rows per RT; dedupe
                        m = dt_tr[["RT", "mz"]].dropna().drop_duplicates()
                        # prefer trafo RT mapped by precursor_mz
                        dt_xic = dt_xic.merge(
                            m, left_on="precursor_mz", right_on="mz", how="left"
                        )
                        # choose RT: trafo RT if present, else argmax
                        dt_xic["RT"] = np.where(
                            dt_xic["RT"].notna(), dt_xic["RT"], dt_xic["RT_argmax"]
                        )
                    else:
                        dt_xic["RT"] = dt_xic["RT_argmax"]

                    # Zoom into XICs ±50 s if requested
                    if zoom_in_xic and "RT" in dt_xic.columns:
                        print("INFO: Zooming into XICs if possible...", flush=True)
                        dt_xic = zoom_xic_around_rt(dt_xic, window_sec=50.0)

            # ----------------
            # Build the figure
            # ----------------
            # Top row: IM, m/z, iRT-Transformation
            # Bottom: XIC facets

            # Prepare facet keys
            if not dt_xic.empty:
                facets = dt_xic.assign(
                    facet=lambda d: d["peptide_sequence"].astype(str)
                    + " : "
                    + d["precursor_mz"].round(4).astype(str)
                )
                keys_all = sorted(facets["facet"].unique().tolist())
            else:
                facets = None
                keys_all = []

            # force exactly 12 slots (4x3); cap extras, pad empties
            XIC_ROWS, XIC_COLS = 4, 3
            SLOTS = XIC_ROWS * XIC_COLS  # 12
            keys = keys_all[:SLOTS]  # cap at 12
            pad_count = SLOTS - len(keys)  # how many empties to add

            # figure
            fig = plt.figure(figsize=(16, 12))
            outer = fig.add_gridspec(
                nrows=2,
                ncols=1,
                height_ratios=[1, 2.2],  # more space for the 12 chromatograms
                left=0.04,
                right=0.99,
                top=0.93,
                bottom=(
                    0.09 if zoom_in_xic else 0.06
                ),  # <-- more space when every panel shows x-ticks
                hspace=0.22,
                wspace=0.20,
            )
            fig.suptitle(run, color="red", fontweight="bold", fontsize=16)

            # --- top row: 3 calibration panels ---
            top = outer[0].subgridspec(nrows=1, ncols=3, wspace=0.24)
            ax_im = fig.add_subplot(top[0, 0])
            ax_mz = fig.add_subplot(top[0, 1])
            ax_tr = fig.add_subplot(top[0, 2])

            # IM calibration
            if not dt_im.empty and {"im", "theo_im"}.issubset(dt_im.columns):
                x = dt_im["theo_im"].to_numpy(float)
                y = dt_im["im"].to_numpy(float)
                ax_im.scatter(x, y, s=9, alpha=0.7)
                a, b, r = linfit_and_r2(x, y)
                if not np.isnan(a):
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 101)
                    ax_im.plot(xfit, a * xfit + b, linewidth=2, alpha=0.9)
                annotate_fit(
                    ax_im,
                    a,
                    b,
                    r,
                    loc="upper left",
                )
                ax_im.set_xlabel("theoretical Ion Mobility")
                ax_im.set_ylabel("experimental Ion Mobility")
                ax_im.set_title("ion mobility calibration")
                # inset (upper-left so it doesn’t sit on the trend line)
                if "delta_im" in dt_im.columns:
                    add_hist_inset(ax_im, dt_im["delta_im"], "ΔIM", loc="lower right")
            else:
                ax_im.set_title("ion mobility calibration (empty)")
                ax_im.set_xlabel("theoretical Ion Mobility")
                ax_im.set_ylabel("experimental Ion Mobility")
                ax_im.set_xticks([])
                ax_im.set_yticks([])

            # m/z calibration
            if not dt_mz.empty and {"mz", "theo_mz"}.issubset(dt_mz.columns):
                x = dt_mz["theo_mz"].to_numpy(float)
                y = dt_mz["mz"].to_numpy(float)
                ax_mz.scatter(x, y, s=9, alpha=0.7)
                a, b, r = linfit_and_r2(x, y)
                if not np.isnan(a):
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 101)
                    ax_mz.plot(xfit, a * xfit + b, linewidth=2, alpha=0.9)
                annotate_fit(ax_mz, a, b, r, loc="upper left")
                ax_mz.set_xlabel("theoretical m/z")
                ax_mz.set_ylabel("experimental m/z")
                ax_mz.set_title("m/z calibration")
                if "delta_mz" in dt_mz.columns:
                    add_hist_inset(ax_mz, dt_mz["delta_mz"], "Δm/z", loc="lower right")
            else:
                ax_mz.set_title("m/z calibration (empty)")
                ax_mz.set_xlabel("theoretical m/z")
                ax_mz.set_ylabel("experimental m/z")
                ax_mz.set_xticks([])
                ax_mz.set_yticks([])

            # iRT Transformation
            if not dt_tr.empty and {"RT", "delta"}.issubset(dt_tr.columns):
                x = dt_tr["RT"].to_numpy(float)
                y = dt_tr["delta"].to_numpy(float)
                ax_tr.scatter(x, y, s=9, alpha=0.8)
                a, b, r = linfit_and_r2(x, y)
                if not np.isnan(a):
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 101)
                    ax_tr.plot(xfit, a * xfit + b, linewidth=2, alpha=0.9)
                annotate_fit(ax_tr, a, b, r, loc="upper right")
                ax_tr.set_xlabel("original RT [s]")
                ax_tr.set_ylabel("ΔRT [s]")
                ax_tr.set_title("iRT Transformation")
            else:
                ax_tr.set_title("iRT Transformation (empty)")
                ax_tr.set_xlabel("original RT [s]")
                ax_tr.set_ylabel("ΔRT [s]")
                ax_tr.set_xticks([])
                ax_tr.set_yticks([])

            # --- bottom row: fixed 12 chromatogram slots (4 x 3) ---
            bottom = outer[1].subgridspec(nrows=1, ncols=1)
            host = fig.add_subplot(bottom[0, 0])
            host.axis("off")

            # When zooming, each panel has a different x-range → give each its own ticks
            SHOW_ALL_X_TICKS = bool(zoom_in_xic)

            # tighter spacing normally; more breathing room if all panels show ticks
            xic_wspace = 0.12
            xic_hspace = (
                0.28 if SHOW_ALL_X_TICKS else 0.14
            )  # <-- more space for tick labels when zooming

            grid = bottom[0, 0].subgridspec(
                XIC_ROWS,
                XIC_COLS,
                wspace=xic_wspace,
                hspace=xic_hspace,
            )

            def _wrap_title(s: str, width: int = 50) -> str:
                import textwrap as _tw

                return "\n".join(_tw.wrap(s, width=width))

            for i, fk in enumerate(keys):
                rr, cc = divmod(i, XIC_COLS)
                ax = fig.add_subplot(grid[rr, cc])

                # trim per-axes whitespace
                ax.margins(x=0.01, y=0.06)
                ax.tick_params(
                    length=2, pad=1, labelsize=7
                )  # smaller tick font for dense grids
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)

                sub = (
                    facets[facets["facet"] == fk]
                    if facets is not None
                    else pd.DataFrame()
                )
                if not sub.empty:
                    for _, gg in sub.groupby("product_mz", dropna=False):
                        ax.plot(gg["rt"].to_numpy(), gg["int"].to_numpy(), linewidth=1)
                    if "RT" in sub.columns and sub["RT"].notna().any():
                        ax.axvline(
                            float(np.nanmean(sub["RT"].to_numpy())),
                            linestyle="--",
                            linewidth=1,
                            alpha=0.8,
                        )

                ax.set_title(_wrap_title(fk, width=50), fontsize=9, pad=2)

                # x/y labels: all x ticks if zooming, otherwise only bottom row
                if SHOW_ALL_X_TICKS:
                    if rr < XIC_ROWS - 1:
                        ax.set_xlabel("")
                    else:
                        ax.set_xlabel("Retention Time [s]", fontsize=8)
                else:
                    if rr < XIC_ROWS - 1:
                        ax.set_xlabel("")
                        ax.set_xticklabels([])
                    else:
                        ax.set_xlabel("Retention Time [s]")

                if cc > 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel("Intensity", fontsize=8)

            # pad empties (if fewer than 12)
            for j in range(len(keys), SLOTS):
                rr, cc = divmod(j, XIC_COLS)
                ax = fig.add_subplot(grid[rr, cc])
                ax.axis("off")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    end_t = pd.Timestamp.now()
    print(
        f"INFO: Done. Saved report to '{report_file}'. Elapsed: {end_t - start_t}",
        flush=True,
    )


# ----------------------------
# CLI
# ----------------------------


def main():
    desc = """\
    Generate calibration reports for m/z, ion mobility and retention time using
    OpenSwathWorkflow debugging files.

    Example:
      python openswathworkflow_calibration_report.py \\
        --wd ./ \\
        --im-cal-pattern _debug_im.txt \\
        --mz-cal-pattern _debug_mz.txt \\
        --irt-trafo-pattern .trafoXML \\
        --irt-mzml-pattern _irt_chrom.mzML \\
        --report-file calibration_report.pdf \\
        --zoom-in-xic true
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(desc),
    )
    p.add_argument(
        "--wd",
        default=".",
        help="Working directory containing debugging files. Default: current dir",
    )
    p.add_argument(
        "--im-cal-pattern",
        default="_debug_calibration_im.txt",
        help="Ion mobility debug file pattern. Default: _debug_calibration_im.txt",
    )
    p.add_argument(
        "--mz-cal-pattern",
        default="_debug_calibration_mz.txt",
        help="m/z debug file pattern. Default: _debug_calibration_mz.txt",
    )
    p.add_argument(
        "--irt-trafo-pattern",
        default="_debug_calibration_irt.trafoXML",
        help="iRT trafoXML pattern. Default: _debug_calibration_irt.trafoXML",
    )
    p.add_argument(
        "--irt-mzml-pattern",
        default="_debug_calibration_irt_chrom.mzML",
        help="iRT XIC mzML pattern. Default: _debug_calibration_irt_chrom.mzML",
    )
    p.add_argument(
        "--report-file",
        default="calibration_report.pdf",
        help="Output PDF path. Default: calibration_report.pdf",
    )
    p.add_argument(
        "--zoom-in-xic",
        default="true",
        choices=["true", "false"],
        help="Zoom into XICs ±50 s around RT apex. Default: true",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Verbosity level 0/1/2. Default: 0",
    )

    args = p.parse_args()

    # Normalize inputs
    wd = os.path.abspath(os.path.expanduser(args.wd))
    if not os.path.isdir(wd):
        print(f"ERROR: Working directory does not exist: {wd}", file=sys.stderr)
        sys.exit(2)

    zoom = args.zoom_in_xic.lower() == "true"

    try:
        generate_report(
            wd=wd,
            im_cal_pattern=args.im_cal_pattern,
            mz_cal_pattern=args.mz_cal_pattern,
            irt_trafo_pattern=args.irt_trafo_pattern,
            irt_mzml_pattern=args.irt_mzml_pattern,
            report_file=os.path.abspath(os.path.expanduser(args.report_file)),
            zoom_in_xic=zoom,
            verbose=args.verbose,
        )
    except Exception as e:
        # brief traceback + clear message (similar to R script's CLI behavior)
        import traceback

        traceback.print_exc(limit=2)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
