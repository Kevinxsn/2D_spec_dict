#!/usr/bin/env python3
"""
Combined Deconvolution Pipeline
================================
Two-stage deconvolution for 2D-PC-MS (FFC) datasets:

  Stage 1 — FFC-line-based deconvolution (deconv_combine_greedy.py)
      Uses the greedy line-finder (greedy_line.py) to detect FFC correlation
      lines; each line carries its charge split (i, j), so the charge of each
      FFC side is read off directly — no expected-offset table.  The
      isotope-envelope deconvolution that follows (monoisotopic refinement via
      theo_patt) is the same machinery as deconv_combine, reused unchanged.

  Stage 2 — TopFD fallback for residual peaks
      Collects all m/z values from the selected FFC subset that were NOT
      assigned to any line in Stage 1.  These residual m/z values are
      written as a 1D peak list, deconvolved by TopFD, and the results
      are mapped back onto the original FFC rows — exactly like pepline.sh
      but operating only on the sparser residual spectrum.

  Merge — The line-deconvolved and TopFD-deconvolved FFC rows are combined
      into a single output with a consistent column layout.

Usage
-----
    python combined_deconv_pipeline.py config.yaml

Or import and call ``run_combined_pipeline(...)`` from Python.

The script expects TopFD (topfd binary) to be available on the system.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Imports from existing modules ──────────────────────────────────────────
# These must be on PYTHONPATH or in the same directory.
#
# Stage 1 now uses the greedy line-finder for charge assignment
# (deconv_combine_greedy) instead of the expected_offsets / detect_line_clusters
# front end in deconv_combine.  The isotope-envelope deconvolution back end
# (charge-1 transform, theo_patt monoisotopic refinement, background
# augmentation, "replaced" view) is the SAME — it is reused unchanged.
# Stage 2 (TopFD) and the merge are untouched.
from deconv_combine_greedy import deconvolute_ffc_by_lines_greedy
from deconv_combine import PROTON_MASS


# =====================================================================
# STAGE 2 HELPERS: TopFD fallback for residual peaks
# =====================================================================

def _collect_residual_mz(
    full_selected: pd.DataFrame,
    line_assigned_indices: set,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = None,
    intensity_col_b: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect unique m/z values from selected FFCs that were NOT assigned
    to any line.

    Returns a DataFrame with columns ["mz", "intensity"] suitable for
    writing as a TopFD input peak list.  Intensities are taken from the
    FFC table if available, otherwise set to 1.0.
    """
    # Rows not claimed by any line
    residual_mask = ~full_selected.index.isin(line_assigned_indices)
    residual = full_selected[residual_mask]

    if residual.empty:
        return pd.DataFrame(columns=["mz", "intensity"])

    # Gather m/z + intensity from both columns
    mz_a = residual[mz_col_a].values.astype(float)
    mz_b = residual[mz_col_b].values.astype(float)

    if intensity_col_a and intensity_col_a in residual.columns:
        int_a = residual[intensity_col_a].values.astype(float)
    else:
        int_a = np.ones(len(residual), dtype=float)

    if intensity_col_b and intensity_col_b in residual.columns:
        int_b = residual[intensity_col_b].values.astype(float)
    else:
        int_b = np.ones(len(residual), dtype=float)

    all_mz = np.concatenate([mz_a, mz_b])
    all_int = np.concatenate([int_a, int_b])

    # Deduplicate: group m/z values within a tight tolerance and keep
    # the highest intensity for each unique peak.
    order = np.argsort(all_mz)
    all_mz = all_mz[order]
    all_int = all_int[order]

    MERGE_TOL = 0.001  # 1 mDa
    unique_mz = [all_mz[0]]
    unique_int = [all_int[0]]
    for k in range(1, len(all_mz)):
        if all_mz[k] - unique_mz[-1] <= MERGE_TOL:
            if all_int[k] > unique_int[-1]:
                unique_int[-1] = all_int[k]
                unique_mz[-1] = all_mz[k]
        else:
            unique_mz.append(all_mz[k])
            unique_int.append(all_int[k])

    return pd.DataFrame({"mz": unique_mz, "intensity": unique_int})


def _write_topfd_input(peaks_df: pd.DataFrame, output_path: str) -> None:
    """Write a peak list in the space-separated format TopFD expects."""
    peaks_df[["mz", "intensity"]].to_csv(
        output_path, sep=" ", index=False, header=False
    )


def _run_topfd(
    input_path: str,
    topfd_bin: str = "topfd",
    max_charge: int = 4,
    error_tol: float = 0.005,
    env_extra_args: Optional[Dict[str, str]] = None,
) -> str:
    """
    Run TopFD on a peak list and return the path to the .env output.

    Parameters
    ----------
    input_path : str
        Path to the space-separated peak list (mz intensity).
    topfd_bin : str
        Path to the topfd binary.
    max_charge : int
        Maximum charge state for TopFD (-c flag).
    error_tol : float
        Error tolerance for TopFD (-e flag).
    env_extra_args : dict, optional
        Additional environment variables to set (e.g. LD_LIBRARY_PATH).

    Returns
    -------
    Path to the deconv_ms2.env file produced by TopFD.
    """
    cmd = [
        topfd_bin,
        "-T",                          # treat as single MS2 scan
        "-e", str(error_tol),
        "-c", str(max_charge),
        input_path,
    ]

    env = os.environ.copy()
    if env_extra_args:
        for key, val in env_extra_args.items():
            if key in env:
                env[key] = val + ":" + env[key]
            else:
                env[key] = val

    # Run TopFD with cwd set to the input file's directory so that
    # its output (deconv_ms2.env) lands alongside the input, not in
    # whatever directory the caller happened to invoke Python from.
    input_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    print(f"[TopFD] Running: {' '.join(cmd)}")
    print(f"[TopFD] Working directory: {input_dir}")
    subprocess.run(cmd, check=True, env=env, cwd=input_dir)

    # The .env file name is derived from the input filename.
    env_path = os.path.join(input_dir, "deconv_ms2.env")

    if not os.path.exists(env_path):
        # Try alternative naming: TopFD sometimes uses the input stem
        stem = Path(input_path).stem
        alt = os.path.join(input_dir, f"{stem}_ms2.env")
        if os.path.exists(alt):
            env_path = alt
        else:
            raise FileNotFoundError(
                f"TopFD did not produce expected .env file. "
                f"Looked for: {env_path} and {alt}"
            )

    return env_path


def _dedup_env_by_peak_idx(env_path: str) -> pd.DataFrame:
    """
    Deduplicate a TopFD .env file by PEAK_IDX, keeping the row with
    the highest THEO_INTE for each peak.  Mirrors dedup_by_peak_idx.py.
    """
    import csv

    best = {}
    with open(env_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        for row in reader:
            key = row["PEAK_IDX"]
            if key not in best or float(row["THEO_INTE"]) > float(best[key]["THEO_INTE"]):
                best[key] = row

    return pd.DataFrame(list(best.values()), columns=fieldnames)


def _annotate_residual_ffcs(
    residual_ffcs: pd.DataFrame,
    env_df: pd.DataFrame,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    mz_tol: float = 1e-4,
) -> pd.DataFrame:
    """
    Annotate residual FFC rows with TopFD deconvolution results.
    Mirrors annotate_mms.py logic: for each m/z in col_a and col_b,
    look up the matching ORIG_MZ in the env table and append
    THEO_MONO_MZ, THEO_MONO_MASS, THEO_INTE_SUM, THEO_CHARGE.

    Returns the annotated DataFrame with suffixed columns (_1 for A, _2 for B).
    """
    ANNO_COLS = ["THEO_MONO_MZ", "THEO_MONO_MASS", "THEO_INTE_SUM", "THEO_CHARGE"]

    if env_df.empty:
        # No deconvolution results — return with NaN annotation columns
        for col in ANNO_COLS:
            residual_ffcs[col + "_1"] = float("nan")
            residual_ffcs[col + "_2"] = float("nan")
        return residual_ffcs

    # Build indexed lookup (same approach as annotate_mms.py)
    env = env_df.copy()
    env["ORIG_MZ"] = env["ORIG_MZ"].astype(float)
    env["_mz_key"] = env["ORIG_MZ"].round(6)
    env_indexed = env.set_index("_mz_key")

    def lookup(mz: float) -> dict:
        key = round(mz, 6)
        if key in env_indexed.index:
            row = env_indexed.loc[key]
            # Handle possible duplicate index entries
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {col: row[col] for col in ANNO_COLS}
        # Tolerance-based fallback
        candidates = env_indexed[abs(env_indexed["ORIG_MZ"] - mz) < mz_tol]
        if not candidates.empty:
            row = candidates.iloc[0]
            return {col: row[col] for col in ANNO_COLS}
        return {col: float("nan") for col in ANNO_COLS}

    # Annotate side A
    anno1 = [lookup(mz) for mz in residual_ffcs[mz_col_a].values.astype(float)]
    anno1_df = pd.DataFrame(anno1).rename(columns={c: c + "_1" for c in ANNO_COLS})

    # Annotate side B
    anno2 = [lookup(mz) for mz in residual_ffcs[mz_col_b].values.astype(float)]
    anno2_df = pd.DataFrame(anno2).rename(columns={c: c + "_2" for c in ANNO_COLS})

    result = pd.concat([
        residual_ffcs.reset_index(drop=True),
        anno1_df.reset_index(drop=True),
        anno2_df.reset_index(drop=True),
    ], axis=1)

    return result


# =====================================================================
# MERGE: combine Stage 1 + Stage 2 into a unified output
# =====================================================================

def _merge_results(
    stage1_replaced: pd.DataFrame,
    stage1_annotated: pd.DataFrame,
    stage2_annotated: pd.DataFrame,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
) -> pd.DataFrame:
    """
    Merge line-deconvolved (Stage 1) and TopFD-deconvolved (Stage 2) FFCs
    into a single output with a consistent column layout.

    Output columns
    --------------
    The merged table contains:
    * The original FFC columns (covariance, score, ranking, intensities, ...)
    * ``m/z A`` and ``m/z B`` replaced by deconvolved monoisotopic m/z at
      the original charge state: ``mono_mass / charge + proton_mass``.
      This is consistent with how ``deconv_combine.py`` builds its
      "replaced" view.
    * ``charge_A`` and ``charge_B``: the inferred charge for each side.
      - Stage 1: from FFC line geometry (i, j).
      - Stage 2: from TopFD's THEO_CHARGE.
      NaN if the charge could not be determined for that side.
    * ``deconv_method``: ``"ffc_line"`` or ``"topfd"``.
    """
    merged_parts = []

    # ── Stage 1: line-based ──────────────────────────────────────────────
    # The "replaced" df already has m/z A and m/z B set to
    # mono_mass / charge + proton.  Pull charge_A / charge_B from the
    # annotated df (same row order, same length).
    if not stage1_replaced.empty:
        s1 = stage1_replaced.copy()
        # Original m/z from the annotated df (before line-based replacement)
        s1["mz_A_original"] = stage1_annotated[mz_col_a].values
        s1["mz_B_original"] = stage1_annotated[mz_col_b].values
        s1["charge_A"] = stage1_annotated["charge_A"].values
        s1["charge_B"] = stage1_annotated["charge_B"].values
        s1["deconv_method"] = "ffc_line"
        merged_parts.append(s1)

    # ── Stage 2: TopFD fallback ──────────────────────────────────────────
    # THEO_MONO_MZ from TopFD is already the monoisotopic m/z at the
    # detected charge — the same convention as Stage 1's replaced view.
    # THEO_CHARGE gives the charge for each side.
    if not stage2_annotated.empty:
        s2 = stage2_annotated.copy()

        # Preserve original m/z before replacement
        s2["mz_A_original"] = s2[mz_col_a].copy()
        s2["mz_B_original"] = s2[mz_col_b].copy()

        # Replace m/z with TopFD's monoisotopic m/z where available
        if "THEO_MONO_MZ_1" in s2.columns:
            s2[mz_col_a] = s2["THEO_MONO_MZ_1"].fillna(s2[mz_col_a])
        if "THEO_MONO_MZ_2" in s2.columns:
            s2[mz_col_b] = s2["THEO_MONO_MZ_2"].fillna(s2[mz_col_b])

        # Carry charge information from TopFD
        s2["charge_A"] = (
            s2["THEO_CHARGE_1"].astype(float) if "THEO_CHARGE_1" in s2.columns
            else float("nan")
        )
        s2["charge_B"] = (
            s2["THEO_CHARGE_2"].astype(float) if "THEO_CHARGE_2" in s2.columns
            else float("nan")
        )

        s2["deconv_method"] = "topfd"
        merged_parts.append(s2)

    if not merged_parts:
        return pd.DataFrame()

    # ── Select output columns ────────────────────────────────────────────
    # Keep only columns common to both stages so the concat is clean.
    if len(merged_parts) == 2:
        common_cols = list(
            set(merged_parts[0].columns) & set(merged_parts[1].columns)
        )
        # Preserve a sensible column order: follow Stage 1's column order
        orig_cols = [c for c in merged_parts[0].columns if c in common_cols]
        merged = pd.concat(
            [p[orig_cols] for p in merged_parts],
            ignore_index=True,
        )
    else:
        merged = merged_parts[0].reset_index(drop=True)

    return merged


# =====================================================================
# PUBLIC ENTRY POINT
# =====================================================================

def run_combined_pipeline(
    ffc_path: str,
    parent_charge: int,
    parent_mass: float,
    output_dir: str,
    output_prefix: str = "combined",
    # ── FFC column mapping ──
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = "intensity A",
    intensity_col_b: Optional[str] = "intensity B",
    ranking_col: str = "Ranking",
    # ── Stage 1 line-finding (greedy) parameters ──
    delta: float = 0.02,
    min_ffc_number: int = 3,
    line_tol: Optional[float] = None,
    use_master_lines: bool = True,
    # ── Stage 1 deconvolution back-end parameters (reused from deconv_combine) ──
    theo_patt_path: Optional[str] = "theo_patt.txt",
    mz_tol: float = 0.02,
    max_shift: Optional[int] = None,
    top_selected: int = 2000,
    top_bg: int = 8000,
    # ── Stage 2 (TopFD) parameters ──
    run_topfd: bool = True,
    topfd_bin: str = "topfd",
    topfd_max_charge: int = 4,
    topfd_error_tol: float = 0.005,
    topfd_env_vars: Optional[Dict[str, str]] = None,
    # ── Input format ──
    skiprows: int = 1,
    col_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full two-stage deconvolution pipeline.

    Parameters
    ----------
    ffc_path : str
        Path to the FFC data file (TSV, with one header row to skip).
    parent_charge : int
        Precursor charge state.
    parent_mass : float
        Neutral monoisotopic mass of the precursor (Da).
    delta : float
        Sort-and-Split gap threshold for greedy line detection (Da).
    min_ffc_number : int
        Minimum number of FFCs required to call a line.
    theo_patt_path : str, optional
        Path to TopPIC theo_patt.txt for monoisotopic refinement (Stage 1
        deconvolution back end).
    mz_tol : float
        Isotope-spacing tolerance on the charge-1-equivalent axis.
    max_shift : int, optional
        Max isotope steps for monoisotopic selection.
    top_selected, top_bg : int
        FFCs deconvolved vs. FFCs used for line detection + background
        envelope augmentation (top_bg >= top_selected).
    output_dir : str
        Directory for all output files.
    run_topfd : bool
        If False, skip Stage 2 entirely (only line-based results).
    topfd_bin : str
        Path to the topfd binary.
    topfd_max_charge : int
        Max charge for TopFD's -c flag.
    topfd_error_tol : float
        Error tolerance for TopFD's -e flag.
    topfd_env_vars : dict, optional
        Extra environment variables for TopFD (e.g. LD_LIBRARY_PATH).

    Returns
    -------
    dict with keys:
        "stage1_annotated"  -- full annotation from line-based deconv
        "stage1_replaced"   -- line-based deconv with replaced m/z
        "stage1_line_map"   -- line assignment details
        "stage2_annotated"  -- TopFD annotation of residual FFCs (or empty)
        "merged"            -- combined output from both stages
        "stats"             -- dict of summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load the FFC data ────────────────────────────────────────────────
    if col_names is None:
        col_names = [
            "m/z A", "m/z B", "Covariance", "Partial Cov.",
            "Score", "Ranking", "intensity A", "intensity B",
        ]

    df = pd.read_csv(
        ffc_path, sep="\t", skiprows=skiprows,
        header=None, engine="python",
    )
    df.columns = col_names[:len(df.columns)]

    print(f"Loaded {len(df)} FFC rows from {ffc_path}")

    # ── Prepare the selected subset ──────────────────────────────────────
    # Reset index so ffc_index from Stage 1 (positional into the selected
    # subset) lines up with the residual masks below.
    full = df[df[ranking_col] != -1].sort_values(ranking_col).copy()
    full = full.dropna(subset=[mz_col_a, mz_col_b])
    selected_df = full.head(top_selected).reset_index(drop=True)

    print(f"Selected subset: {len(selected_df)} rows (top {top_selected})")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: FFC-line-based deconvolution (greedy charge assignment)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 1: FFC-line-based deconvolution (greedy lines)")
    print("=" * 60)

    result_s1 = deconvolute_ffc_by_lines_greedy(
        df,
        parent_charge=parent_charge,
        parent_mass=parent_mass,
        mz_col_a=mz_col_a,
        mz_col_b=mz_col_b,
        intensity_col_a=intensity_col_a,
        intensity_col_b=intensity_col_b,
        # greedy line-finding
        delta=delta,
        min_ffc_number=min_ffc_number,
        line_tol=line_tol,
        use_master_lines=use_master_lines,
        # deconvolution back end (reused from deconv_combine)
        theo_patt_path=theo_patt_path,
        mz_tol=mz_tol,
        max_shift=max_shift,
        top_selected=top_selected,
        top_bg=top_bg,
        ranking_col=ranking_col,
    )

    s1_annotated = result_s1["annotated"]
    s1_replaced = result_s1["replaced"]
    s1_line_map = result_s1["line_map"]

    # Indices (positional into the selected subset) that Stage 1 claimed.
    line_assigned_indices = set(result_s1.get("assigned_indices", set()))
    if not line_assigned_indices and not s1_annotated.empty \
            and "ffc_index" in s1_annotated.columns:
        line_assigned_indices = set(s1_annotated["ffc_index"].unique())

    n_line_assigned = len(line_assigned_indices)
    n_residual = len(selected_df) - n_line_assigned

    print(f"  Line-assigned FFCs: {n_line_assigned}")
    print(f"  Residual FFCs:      {n_residual}")

    # Save Stage 1 outputs
    s1_annotated.to_csv(
        os.path.join(output_dir, f"{output_prefix}_s1_annotated.tsv"),
        sep="\t", index=False,
    )
    s1_replaced.to_csv(
        os.path.join(output_dir, f"{output_prefix}_s1_replaced.tsv"),
        sep="\t", index=False,
    )
    s1_line_map.to_csv(
        os.path.join(output_dir, f"{output_prefix}_s1_line_map.tsv"),
        sep="\t", index=False,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: TopFD fallback on residual peaks
    # ══════════════════════════════════════════════════════════════════════
    s2_annotated = pd.DataFrame()

    if run_topfd and n_residual > 0:
        print("\n" + "=" * 60)
        print("STAGE 2: TopFD fallback on residual spectrum")
        print("=" * 60)

        # 2a. Collect residual m/z values (deduped 1D peak list)
        residual_peaks = _collect_residual_mz(
            selected_df,
            line_assigned_indices,
            mz_col_a=mz_col_a,
            mz_col_b=mz_col_b,
            intensity_col_a=intensity_col_a,
            intensity_col_b=intensity_col_b,
        )
        print(f"  Residual unique peaks: {len(residual_peaks)}")

        if not residual_peaks.empty:
            # 2b. Write peak list for TopFD
            topfd_workdir = os.path.join(output_dir, "topfd_residual")
            os.makedirs(topfd_workdir, exist_ok=True)
            peak_list_path = os.path.join(topfd_workdir, "residual.pkl")
            _write_topfd_input(residual_peaks, peak_list_path)
            print(f"  Peak list written: {peak_list_path}")

            # 2c. Run TopFD
            try:
                env_path = _run_topfd(
                    peak_list_path,
                    topfd_bin=topfd_bin,
                    max_charge=topfd_max_charge,
                    error_tol=topfd_error_tol,
                    env_extra_args=topfd_env_vars,
                )
                print(f"  TopFD env file: {env_path}")

                # 2d. Deduplicate by PEAK_IDX
                env_dedup = _dedup_env_by_peak_idx(env_path)
                print(f"  Deduped env entries: {len(env_dedup)}")

                # Save deduped env for inspection
                env_dedup_path = os.path.join(
                    topfd_workdir, "deconv_dedup_residual.env"
                )
                env_dedup.to_csv(env_dedup_path, sep="\t", index=False)

                # 2e. Annotate residual FFC rows with TopFD results
                residual_ffcs = selected_df[
                    ~selected_df.index.isin(line_assigned_indices)
                ].copy().reset_index(drop=True)

                s2_annotated = _annotate_residual_ffcs(
                    residual_ffcs,
                    env_dedup,
                    mz_col_a=mz_col_a,
                    mz_col_b=mz_col_b,
                )

                n_both_matched = 0
                if "THEO_MONO_MZ_1" in s2_annotated.columns:
                    both = (
                        s2_annotated["THEO_MONO_MZ_1"].notna()
                        & s2_annotated["THEO_MONO_MZ_2"].notna()
                    )
                    n_both_matched = both.sum()

                print(f"  Residual FFCs annotated: {len(s2_annotated)}")
                print(f"  Both sides matched:      {n_both_matched}")

            except FileNotFoundError as e:
                print(f"  [WARNING] TopFD failed: {e}")
                print("  Skipping Stage 2.")
            except subprocess.CalledProcessError as e:
                print(f"  [WARNING] TopFD returned error: {e}")
                print("  Skipping Stage 2.")

        # Save Stage 2 output
        if not s2_annotated.empty:
            s2_annotated.to_csv(
                os.path.join(output_dir, f"{output_prefix}_s2_annotated.tsv"),
                sep="\t", index=False,
            )
    elif not run_topfd:
        print("\n[Stage 2 skipped: run_topfd=False]")
    else:
        print("\n[Stage 2 skipped: no residual FFCs]")

    # ══════════════════════════════════════════════════════════════════════
    # MERGE: combine both stages
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MERGE: combining Stage 1 + Stage 2")
    print("=" * 60)

    merged = _merge_results(
        s1_replaced,
        s1_annotated,
        s2_annotated,
        mz_col_a=mz_col_a,
        mz_col_b=mz_col_b,
    )

    print(f"  Total merged rows: {len(merged)}")
    if "deconv_method" in merged.columns:
        for method, count in merged["deconv_method"].value_counts().items():
            print(f"    {method}: {count}")

    merged.to_csv(
        os.path.join(output_dir, f"{output_prefix}_merged.tsv"),
        sep="\t", index=False,
    )

    # ── Summary stats ────────────────────────────────────────────────────
    stats = {
        "total_ffc_rows": len(df),
        "selected_subset": len(selected_df),
        "stage1_line_assigned": n_line_assigned,
        "stage1_output_rows": len(s1_annotated),
        "stage2_residual": n_residual,
        "stage2_output_rows": len(s2_annotated),
        "merged_total": len(merged),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return {
        "stage1_annotated": s1_annotated,
        "stage1_replaced": s1_replaced,
        "stage1_line_map": s1_line_map,
        "stage2_annotated": s2_annotated,
        "merged": merged,
        "stats": stats,
    }


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined FFC-line + TopFD deconvolution pipeline"
    )
    parser.add_argument("ffc_file", help="Path to the FFC data file (TSV)")
    parser.add_argument("--output-dir", default="./deconv_output",
                        help="Output directory (default: ./deconv_output)")
    parser.add_argument("--prefix", default="combined",
                        help="Output file prefix (default: combined)")

    # Precursor info
    parser.add_argument("--parent-charge", type=int, required=True,
                        help="Precursor charge state")
    parser.add_argument("--parent-mass", type=float, required=True,
                        help="Neutral monoisotopic mass of precursor (Da)")

    # Stage 1 line-finding (greedy) params
    parser.add_argument("--delta", type=float, default=0.02,
                        help="Sort-and-Split gap threshold for line detection "
                             "(Da, default: 0.02)")
    parser.add_argument("--min-ffc", type=int, default=3,
                        help="Minimum FFCs to call a line (default: 3)")
    parser.add_argument("--line-tol", type=float, default=None,
                        help="FFC-on-line tolerance for removal (default: delta)")
    parser.add_argument("--no-master-lines", action="store_true",
                        help="Skip parental/satellite pre-clearing in greedy lines")

    # Stage 1 deconvolution back-end params (reused from deconv_combine)
    parser.add_argument("--theo-patt", default="theo_patt.txt",
                        help="Path to theo_patt.txt (default: theo_patt.txt)")
    parser.add_argument("--mz-tol", type=float, default=0.02,
                        help="Isotope-spacing tolerance on the charge-1 axis "
                             "(default: 0.02)")
    parser.add_argument("--max-shift", type=int, default=None,
                        help="Max isotope steps for monoisotopic selection")
    parser.add_argument("--top-selected", type=int, default=2000)
    parser.add_argument("--top-bg", type=int, default=8000)
    parser.add_argument("--ranking-col", default="Ranking")

    # Stage 2 params
    parser.add_argument("--no-topfd", action="store_true",
                        help="Skip Stage 2 (TopFD fallback)")
    parser.add_argument("--topfd-bin", default="topfd",
                        help="Path to topfd binary")
    parser.add_argument("--topfd-max-charge", type=int, default=4)
    parser.add_argument("--topfd-error-tol", type=float, default=0.005)

    # Input format
    parser.add_argument("--skiprows", type=int, default=1)
    parser.add_argument("--intensity-col-a", default="intensity A")
    parser.add_argument("--intensity-col-b", default="intensity B")

    args = parser.parse_args()

    result = run_combined_pipeline(
        ffc_path=args.ffc_file,
        parent_charge=args.parent_charge,
        parent_mass=args.parent_mass,
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        intensity_col_a=args.intensity_col_a,
        intensity_col_b=args.intensity_col_b,
        ranking_col=args.ranking_col,
        # greedy line-finding
        delta=args.delta,
        min_ffc_number=args.min_ffc,
        line_tol=args.line_tol,
        use_master_lines=not args.no_master_lines,
        # deconvolution back end
        theo_patt_path=args.theo_patt,
        mz_tol=args.mz_tol,
        max_shift=args.max_shift,
        top_selected=args.top_selected,
        top_bg=args.top_bg,
        skiprows=args.skiprows,
        run_topfd=not args.no_topfd,
        topfd_bin=args.topfd_bin,
        topfd_max_charge=args.topfd_max_charge,
        topfd_error_tol=args.topfd_error_tol,
    )