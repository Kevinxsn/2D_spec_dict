"""
Spurious Parental FFC Finder
============================
Identifies FFCs that lie on the parental line (shift ≈ 0) but cannot be
annotated to any b/y ion breaking point.

A "breaking point" requires BOTH fragment masses to match a known b/y ion
(within ``threshold`` Da) for at least one (i, j) charge-split assignment
on any detected parental line.  FFCs that never satisfy this condition on
any parental line are called "spurious".

Workflow
--------
1. Run find_parental_lines() to detect all parental lines.
2. Keep only lines with  |line.mass - parent_mass| < parental_shift_threshold.
3. Collect every FFC that appears in any of those lines' member_indices.
4. For each FFC, try every parental-line assignment (i, j):
       adj_A = i * mz_A - (i-1) * H
       adj_B = j * mz_B - (j-1) * H
   If any (i, j, ion_A, ion_B) triple gives a full match → not spurious.
5. Return the Rankings of all spurious FFCs and the corresponding rows.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from annotation import (
    MASS_H,
    _build_theoretical_ions,
    _find_all_matches,
)
from greedy_line import Line, find_parental_lines


# =============================================================================
# Core function
# =============================================================================

def find_spurious_parental_ffcs(
    ffc_df: pd.DataFrame,
    pep,
    parent_charge: int,
    parent_mass: float,
    delta: float = 0.02,
    min_ffc_number: int = 3,
    line_tol: Optional[float] = None,
    iso_range: int = 0,
    threshold: float = 0.05,
    parental_shift_threshold: float = 0.05,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
) -> Tuple[List[int], pd.DataFrame]:
    """
    Find spurious FFCs on the parental line.

    Parameters
    ----------
    ffc_df : DataFrame
        FFC data (pre-filtered/sorted as needed).
    pep : Pep
        Peptide object with ion_mass() and AA_array.
    parent_charge : int
        Precursor charge state.
    parent_mass : float
        Precursor reconstructed mass used as reference for shift = 0.
        Same convention as annotation_greedy.py:
            mass = i*mz_A + j*mz_B ≈ M + parent_charge * H
    delta : float
        Sort-and-Split gap threshold for line detection.
    min_ffc_number : int
        Minimum FFCs required to call a valid line.
    line_tol : float, optional
        Tolerance for on-line membership during greedy removal.
        Defaults to delta.
    iso_range : int
        Number of isotope variants to consider during b/y matching.
    threshold : float
        Mass tolerance (Da) for b/y ion matching.
    parental_shift_threshold : float
        Maximum |line.mass - parent_mass| for a line to be considered
        parental (i.e. shift ≈ 0).  Default 0.05 Da captures the ±0.02 Da
        typical reconstruction spread.
    col_a, col_b : str
        Column names for the two m/z values in ffc_df.
    ranking_col : str
        Column name for FFC ranking (lower = better).

    Returns
    -------
    spurious_rankings : list[int]
        Sorted Rankings of all spurious FFCs (empty if none found).
    spurious_df : DataFrame
        Subset of ffc_df rows for spurious FFCs, with extra columns:
            n_parental_lines  – how many parental lines claimed this FFC
            repr_i, repr_j    – (i, j) of the first claiming line
            repr_line_mass    – line.mass of the first claiming line
            adj_mass_A        – singly-charged mass of fragment A (repr line)
            adj_mass_B        – singly-charged mass of fragment B (repr line)
    """
    _EXTRA_COLS = [
        "n_parental_lines", "repr_i", "repr_j",
        "repr_line_mass", "adj_mass_A", "adj_mass_B",
    ]
    empty_df = pd.DataFrame(columns=list(ffc_df.columns) + _EXTRA_COLS)

    # ── 1. Detect parental lines ───────────────────────────────────────────
    parental_lines = find_parental_lines(
        ffc_df,
        parent_charge=parent_charge,
        delta=delta,
        min_ffc_number=min_ffc_number,
        col_a=col_a,
        col_b=col_b,
        ranking_col=ranking_col,
        line_tol=line_tol,
    )

    # ── 2. Keep only shift ≈ 0 lines ──────────────────────────────────────
    true_parental: List[Line] = [
        ln for ln in parental_lines
        if abs(ln.mass - parent_mass) < parental_shift_threshold
    ]

    if not true_parental:
        return [], empty_df

    # ── 3. Collect FFCs → [lines that claim them] ─────────────────────────
    ffc_to_lines: Dict[int, List[Line]] = {}
    for ln in true_parental:
        for idx in ln.member_indices:
            ffc_to_lines.setdefault(int(idx), []).append(ln)

    if not ffc_to_lines:
        return [], empty_df

    # ── 4. Build theoretical ions (no neutral-loss extension for parental) ─
    ions = _build_theoretical_ions(pep, iso_range)

    has_ranking = ranking_col in ffc_df.columns

    # ── 5. Classify each FFC ──────────────────────────────────────────────
    spurious_rows = []

    for idx, lines in ffc_to_lines.items():
        if idx not in ffc_df.index:
            continue

        ffc_row = ffc_df.loc[idx]
        mz_a = float(ffc_row[col_a])
        mz_b = float(ffc_row[col_b])

        fully_annotated = False
        for ln in lines:
            i, j = ln.i, ln.j
            adj_a = i * mz_a - (i - 1) * MASS_H
            adj_b = j * mz_b - (j - 1) * MASS_H

            matches_a = _find_all_matches(adj_a, ions, threshold)
            matches_b = _find_all_matches(adj_b, ions, threshold)

            for ma in matches_a:
                if ma["base_name"] is None:
                    continue
                for mb in matches_b:
                    if mb["base_name"] is not None:
                        fully_annotated = True
                        break
                if fully_annotated:
                    break
            if fully_annotated:
                break

        if not fully_annotated:
            repr_ln = lines[0]
            repr_i, repr_j = repr_ln.i, repr_ln.j
            adj_a_repr = repr_i * mz_a - (repr_i - 1) * MASS_H
            adj_b_repr = repr_j * mz_b - (repr_j - 1) * MASS_H

            extra: dict = {
                "n_parental_lines": len(lines),
                "repr_i":           repr_i,
                "repr_j":           repr_j,
                "repr_line_mass":   round(repr_ln.mass, 4),
                "adj_mass_A":       round(adj_a_repr, 4),
                "adj_mass_B":       round(adj_b_repr, 4),
            }
            spurious_rows.append(ffc_row.to_dict() | extra)

    if not spurious_rows:
        return [], empty_df

    spurious_df = pd.DataFrame(spurious_rows).reset_index(drop=True)
    spurious_rankings: List[int] = []
    if has_ranking:
        spurious_rankings = sorted(int(r) for r in spurious_df[ranking_col])

    return spurious_rankings, spurious_df


# =============================================================================
# Example / quick test
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    import interpreter_modify  # noqa: F401
    import peptide
    from line_finding import prepare_ffc_data
    from merge_ffcs import merge_duplicate_ffcs

    # ── Configuration ─────────────────────────────────────────────────────
    #DATA_PATH   = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/VEA_merged.tsv"
    DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.GLP2_Z4_NCE15_200_ions"
    PEP_SEQ     = "VEADIAGHGQEVLIR"
    #PEP_SEQ      = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
    CHARGE      = 3
    #CHARGE      = 4
    PARENT_MASS = 1608.87      # i*mzA + j*mzB reference (M + Z*H)
    #PARENT_MASS  = 941.96162 * 4
    TOP_N       = 50000
    DELTA       = 0.005
    MIN_FFC     = 3
    ISO_RANGE   = 1
    THRESHOLD   = 0.05
    PARENTAL_SHIFT_THRESHOLD = 0.05

    # ── Build peptide & load data ─────────────────────────────────────────
    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
    print(f"Peptide: {PEP_SEQ}  charge={CHARGE}  parent_mass={PARENT_MASS}")

    
    ffc_df = pd.read_csv(DATA_PATH, sep=r"\s+", skiprows=1, header=None, engine="python")
    ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    #ffc_df = pd.read_csv(DATA_PATH, sep="\t")
    ffc_df = prepare_ffc_data(ffc_df, top_n=TOP_N)
    #ffc_df = merge_duplicate_ffcs(ffc_df)
    print(f"FFC rows after filter: {len(ffc_df)}")

    # ── Run ───────────────────────────────────────────────────────────────
    rankings, spurious_df = find_spurious_parental_ffcs(
        ffc_df,
        pep,
        parent_charge=CHARGE,
        parent_mass=PARENT_MASS,
        delta=DELTA,
        min_ffc_number=MIN_FFC,
        iso_range=ISO_RANGE,
        threshold=THRESHOLD,
        parental_shift_threshold=PARENTAL_SHIFT_THRESHOLD,
    )

    print(f"\nSpurious parental FFCs: {len(spurious_df)}")
    print(f"Rankings: {rankings}")

    if not spurious_df.empty:
        print("\nSpurious FFC table:")
        show_cols = [
            c for c in [
                "m/z A", "m/z B", "Ranking",
                "repr_i", "repr_j", "repr_line_mass",
                "adj_mass_A", "adj_mass_B", "n_parental_lines",
            ]
            if c in spurious_df.columns
        ]
        print(spurious_df[show_cols].to_string(index=False))
