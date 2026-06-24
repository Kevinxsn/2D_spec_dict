"""
Greedy-Line Annotation Pipeline
================================
Combines greedy_line.py with annotation.py.

Key difference from annotation.py
-----------------------------------
annotation.py keys each coverage-table column by *shift* only.
For a charge-4 precursor at shift ≈ 0 (parental), FFCs from the (1,3)
line and the (2,2) line are mixed into one "parent" column.

Here, each column is keyed by **(i, j, shift)**.  The (i, j) comes
directly from the greedy line — no re-fitting of charge states is needed.
The result is that "(1,3) parent" and "(2,2) parent" are separate columns.

Pipeline
--------
1. Run greedy_lines() on the FFC map.
2. For each detected line (i, j, mass):
     shift = round(mass - parent_mass, decimals)
     col   = "(i,j) shift_str"
3. For each FFC assigned to that line (member_indices):
     adj_mass_A = i * mz_A - (i-1) * MASS_H   ← singly-charged mass
     adj_mass_B = j * mz_B - (j-1) * MASS_H
     match adj_mass_A/B against theoretical b/y ions
     record in coverage table under column col
4. Assemble the coverage table (same format as annotation.py output).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ── Re-use low-level helpers from annotation.py ───────────────────────────────
from annotation import (
    MASS_H,
    ISOTOPE_OFFSET,
    _build_row_labels,
    _build_theoretical_ions,
    _find_all_matches,
    _record_full_match,
    _record_partial_match,
    _record_example,
    get_complementary_ion,
    process_ffc_annotations,
    _combine_coverage,
)

# ── Greedy-line detection ─────────────────────────────────────────────────────
from greedy_line import Line, greedy_lines


# =============================================================================
# 1. COLUMN KEY
# =============================================================================

def _col_key(i: int, j: int, shift: float, decimals: int = 2) -> str:
    """
    Format a (i, j, shift) triple as a readable column label.

    Examples
    --------
    (1, 3, 0.0)    -> "(1,3) parent"
    (2, 2, 1.002)  -> "(2,2) +1.00"
    (1, 3, -18.01) -> "(1,3) -18.01"
    """
    half_step = 0.5 * 10 ** (-decimals)
    shift_str = "parent" if abs(shift) < half_step else f"{shift:+.{decimals}f}"
    return f"({i},{j}) {shift_str}"


# =============================================================================
# 2. ION SET BUILDER (with neutral-loss extension)
# =============================================================================

def _build_ions_for_line(pep, neutral_loss: float, iso_range: int) -> list:
    """
    Return theoretical b/y ions extended with neutral-loss shifted variants.

    For a line with shift = line.mass - parent_mass < 0, the neutral loss
    is the mass subtracted from one fragment.  We add `ion_mass - shifted_loss`
    variants (mirroring annotate_dataframe_loss) so that an FFC where
    fragment A lost ``neutral_loss`` Da can still be matched against the
    known b/y masses.

    Parameters
    ----------
    neutral_loss : float
        Positive value = mass lost by one fragment on this line.
        E.g. for a line at shift = -229.11,  neutral_loss = 229.11.
        If <= 1e-3 (parental / isotope line), no extension is added.
    iso_range : int
        Isotope variants already built into base_ions; also applied to
        the loss-shifted variants exactly as annotate_dataframe_loss does.
    """
    base_ions = _build_theoretical_ions(pep, iso_range)

    if neutral_loss <= 1e-3:
        return base_ions

    # Mirror annotate_dataframe_loss: for each base ion, add variants at
    # ion_mass - (neutral_loss + n * ISOTOPE_OFFSET) for n = 0..iso_range
    loss_variants = (
        [neutral_loss + n * ISOTOPE_OFFSET for n in range(iso_range + 1)]
        if iso_range > 0
        else [neutral_loss]
    )

    extended = list(base_ions)
    for ion in base_ions:
        for shifted_loss in loss_variants:
            extended.append({
                "base_name": ion["base_name"],
                "iso":       ion["iso"],
                "name":      f'({ion["name"]})-{round(shifted_loss, 3)}',
                "mass":      ion["mass"] - shifted_loss,
            })

    return extended


# =============================================================================
# 3. CORE COVERAGE TABLE
# =============================================================================

def coverage_table_greedy(
    ffc_df: pd.DataFrame,
    lines: Sequence[Line],
    pep,
    parent_mass: float,
    iso_range: int = 0,
    threshold: float = 0.05,
    shift_decimals: int = 2,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
) -> pd.DataFrame:
    """
    Build a bond-by-bond coverage table keyed by (i, j, shift).

    Parameters
    ----------
    ffc_df : DataFrame
        The **same** DataFrame passed to greedy_lines().
        Required columns: col_a, col_b, (optionally) ranking_col.
    lines : sequence of Line
        All detected lines — typically ``master + line_list`` from
        greedy_lines(return_master=True).  Lines with empty
        member_indices (satellites) are silently skipped.
    pep : Pep
        Peptide object with ion_mass() and AA_array.
    parent_mass : float
        Precursor neutral mass; used to compute shift = line.mass - parent_mass.
    iso_range : int
        Number of isotope variants to consider during b/y matching.
    threshold : float
        Mass tolerance (Da) for b/y ion matching.
    shift_decimals : int
        Rounding precision for shifts used in column labels.
    col_a, col_b : str
        Column names for the two m/z values in ffc_df.
    ranking_col : str
        Column name for FFC ranking (lower = better, -1 = unranked).

    Returns
    -------
    DataFrame with rows = backbone cleavage sites, columns = (i,j,shift)
    labels.  Same row/column structure as annotation.py's coverage_table().
    """
    row_labels, theo_masses = _build_row_labels(pep)
    length = len(pep.AA_array)
    has_ranking = ranking_col in ffc_df.columns

    # Cache extended ion sets keyed by rounded neutral_loss to avoid
    # rebuilding the same set for every FFC on the same loss line.
    _ions_cache: dict = {}

    # ── Build ordered, deduplicated column list ───────────────────────────
    col_order: list = []
    seen_cols: set = set()
    for ln in lines:
        if ln.member_indices.size == 0:
            continue
        shift = round(ln.mass - parent_mass, shift_decimals)
        key = _col_key(ln.i, ln.j, shift, shift_decimals)
        if key not in seen_cols:
            col_order.append(key)
            seen_cols.add(key)

    if not col_order:
        return pd.DataFrame(index=row_labels)

    # ── Initialise per-column dicts ───────────────────────────────────────
    annot_table  = {col: {r: None          for r in row_labels} for col in col_order}
    iso_tracker  = {col: {r: float("inf") for r in row_labels} for col in col_order}
    ffc_counts   = {col: 0 for col in col_order}
    fpc_both_unk = {col: 0 for col in col_order}
    fpc_one_unk  = {col: 0 for col in col_order}

    MAX_EX = 3
    ex_both   = {col: [] for col in col_order}
    ex_both_t = {col: [] for col in col_order}
    ex_one    = {col: [] for col in col_order}
    ex_one_t  = {col: [] for col in col_order}

    # ── Populate table ────────────────────────────────────────────────────
    for ln in lines:
        if ln.member_indices.size == 0:
            continue

        i, j = ln.i, ln.j
        actual_shift  = ln.mass - parent_mass
        shift         = round(actual_shift, shift_decimals)
        col           = _col_key(i, j, shift, shift_decimals)

        # Neutral loss for this line (positive = mass was lost by a fragment).
        # Use actual (unrounded) shift so the ion mass offset is accurate.
        neutral_loss  = max(0.0, -actual_shift)
        cache_key     = round(neutral_loss, 3)        # 1 mDa resolution for cache
        if cache_key not in _ions_cache:
            _ions_cache[cache_key] = _build_ions_for_line(pep, neutral_loss, iso_range)
        ions = _ions_cache[cache_key]

        for idx in ln.member_indices:
            if idx not in ffc_df.index:
                continue

            ffc_row = ffc_df.loc[idx]
            mz_a    = float(ffc_row[col_a])
            mz_b    = float(ffc_row[col_b])
            ranking = int(ffc_row[ranking_col]) if has_ranking else -1

            # Singly-charged masses: the (i,j) are known from the greedy line
            adj_a = i * mz_a - (i - 1) * MASS_H
            adj_b = j * mz_b - (j - 1) * MASS_H

            matches_a = _find_all_matches(adj_a, ions, threshold)
            matches_b = _find_all_matches(adj_b, ions, threshold)
            ffc_counts[col] += 1

            for ma in matches_a:
                for mb in matches_b:
                    both_known   = ma["base_name"] is not None and mb["base_name"] is not None
                    both_unknown = ma["base_name"] is None     and mb["base_name"] is None

                    # Build a row-dict matching what annotation.py helpers expect
                    row = pd.Series({
                        "m/z A":        mz_a,
                        "m/z B":        mz_b,
                        "charge_A":     i,
                        "charge_B":     j,
                        "adj_mass_A":   adj_a,
                        "adj_mass_B":   adj_b,
                        "base_name_A":  ma["base_name"],
                        "base_name_B":  mb["base_name"],
                        "explanation_A": ma["name"],
                        "explanation_B": mb["name"],
                        "deviation_A":  ma["dev"]  if ma["dev"]  is not None else float("inf"),
                        "deviation_B":  mb["dev"]  if mb["dev"]  is not None else float("inf"),
                        "iso_A":        ma["iso"]  if ma["iso"]  is not None else 999,
                        "iso_B":        mb["iso"]  if mb["iso"]  is not None else 999,
                        "Ranking":      ranking,
                    })

                    if both_known:
                        _record_full_match(row, col, annot_table, iso_tracker)
                    elif both_unknown:
                        fpc_both_unk[col] += 1
                        if len(ex_both[col]) < MAX_EX:
                            _record_example(row, ex_both[col], ex_both_t[col])
                    else:
                        fpc_one_unk[col] += 1
                        if len(ex_one[col]) < MAX_EX:
                            _record_partial_match(
                                row, col, pep, annot_table,
                                ex_one[col], ex_one_t[col], MAX_EX,
                            )

    # ── Assemble DataFrame ────────────────────────────────────────────────
    table = pd.DataFrame(annot_table, columns=col_order)
    table.index = row_labels

    # Sort columns: most-filled (fewest NaN) first
    table = table[table.isna().sum().sort_values().index]

    table["Row Count"]       = table.notna().sum(axis=1).astype("Int64")
    table["theoretical mass"] = theo_masses

    col_count = table.notna().sum(axis=0).astype("Int64")
    table = pd.concat([
        table,
        pd.DataFrame([col_count.values], columns=table.columns, index=["Col Count"]),
    ])
    table["AA"] = pep.AA_array

    # ── Summary rows (FFC count, false-positive counts, examples) ─────────
    summary = _build_summary_rows(
        table.columns, col_order,
        ffc_counts, fpc_both_unk, fpc_one_unk,
        ex_both, ex_both_t, ex_one, ex_one_t, MAX_EX,
    )
    table = pd.concat([table, summary])

    # ── Coverage column ───────────────────────────────────────────────────
    covered = (table["Row Count"] != 0).map({True: "+", False: "0"})
    n_covered = (covered == "+").sum()
    cov_pct = n_covered / length * 100
    table[f"Coverage={cov_pct:.1f}%"] = covered

    ordered = ["AA", f"Coverage={cov_pct:.1f}%"] + [
        c for c in table.columns if c not in ("AA", f"Coverage={cov_pct:.1f}%")
    ]
    return table[ordered]


def _build_summary_rows(
    all_columns,
    col_order: list,
    ffc_counts: dict,
    fpc_both: dict,
    fpc_one: dict,
    ex_both: dict,
    ex_both_t: dict,
    ex_one: dict,
    ex_one_t: dict,
    max_ex: int,
) -> pd.DataFrame:
    """Return a DataFrame of FFC count, FPR, and example rows."""

    def _make_row(data: dict, index_label: str) -> pd.DataFrame:
        row_dict = {col: data.get(col) for col in col_order}
        return pd.DataFrame([row_dict], index=[index_label]).reindex(columns=all_columns)

    def _expand(example_dict: dict, prefix: str) -> pd.DataFrame:
        rows = []
        for k in range(max_ex):
            row_data = {
                col: (example_dict[col][k] if col in example_dict and k < len(example_dict[col]) else None)
                for col in col_order
            }
            rows.append(row_data)
        return pd.DataFrame(
            rows, index=[f"{prefix}_{k+1}" for k in range(max_ex)]
        ).reindex(columns=all_columns)

    return pd.concat([
        _make_row(ffc_counts, "FFC"),
        _make_row(fpc_both,   "FPR0"),
        _make_row(fpc_one,    "FPR1"),
        _expand(ex_both,   "FPR0_LIST"),
        _expand(ex_both_t, "TFPR0_LIST"),
        _expand(ex_one,    "FPR1_LIST"),
        _expand(ex_one_t,  "TFPR1_LIST"),
    ])


# =============================================================================
# 3. PIPELINE RUNNER
# =============================================================================

def run_greedy_annotation(
    ffc_df: pd.DataFrame,
    pep,
    parent_charge: int,
    parent_mass: float,
    delta: float = 0.02,
    min_ffc_number: int = 3,
    line_tol: Optional[float] = None,
    iso_range: int = 0,
    threshold: float = 0.05,
    shift_decimals: int = 2,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
) -> pd.DataFrame:
    """
    Run greedy_lines + coverage_table_greedy in one call.

    Parameters
    ----------
    ffc_df : DataFrame
        FFC data (pre-filtered/sorted as needed).
    pep : Pep
        Peptide object.
    parent_charge, parent_mass : int, float
        Precursor charge state and neutral mass.
    delta : float
        Sort-and-Split gap threshold for line detection.
    min_ffc_number : int
        Minimum FFCs to form a valid line.
    line_tol : float, optional
        Tolerance for FFC-on-line predicate during greedy removal.
        Defaults to delta.
    iso_range, threshold, shift_decimals, col_a, col_b, ranking_col
        Passed to coverage_table_greedy.

    Returns
    -------
    Coverage table DataFrame with (i, j, shift) column keys.
    Also returns the detected lines as a second value so the caller
    can inspect them.  Use: ``table, lines = run_greedy_annotation(...)``
    """
    line_list, master = greedy_lines(
        ffc_df,
        parent_charge=parent_charge,
        delta=delta,
        min_ffc_number=min_ffc_number,
        col_a=col_a,
        col_b=col_b,
        ranking_col=ranking_col,
        line_tol=line_tol,
        return_master=True,
    )

    all_lines = master + line_list
    
    
    table = coverage_table_greedy(
        ffc_df,
        all_lines,
        pep,
        parent_mass,
        iso_range=iso_range,
        threshold=threshold,
        shift_decimals=shift_decimals,
        col_a=col_a,
        col_b=col_b,
        ranking_col=ranking_col,
    )

    return table, all_lines


# =============================================================================
# 4. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    import interpreter_modify  # noqa: F401
    import peptide                              # project-local module
    from line_finding import prepare_ffc_data  # reuse loader
    from merge_ffcs import merge_duplicate_ffcs

    # ── Configuration ─────────────────────────────────────────────────────
    #DATA_PATH    = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/VEA_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt"
    DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/HAD_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.GLP2_Z4_NCE15_200_ions"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/deiso/merge/KWK5_merged.tsv"
    
    #PEP_SEQ      = "VEADIAGHGQEVLIR"
    PEP_SEQ      = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
    #PEP_SEQ = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
    #CHARGE       = 3
    CHARGE       = 4
    #CHARGE = 5
    #PARENT_MASS  = 1608.87
    PARENT_MASS  = 941.96162 * 4
    TOP_N        = 1000
    DELTA        = 0.02
    MIN_FFC      = 3
    ISO_RANGE    = 1
    THRESHOLD    = 0.05
    OUTPUT_EXCEL = "result/Book4.xlsx"
    OUTPUT_SHEET = "greedy_annot_5+_deconv"

    # ── Build peptide & load data ──────────────────────────────────────────
    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
    #pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20="NH3")
    print(f"Precursor mass: {pep.pep_mass}")
    PARENT_MASS = pep.pep_mass

    
    #ffc_df = pd.read_csv(DATA_PATH, sep=r"\s+", skiprows=1, header=None, engine="python")
    #ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    ffc_df = pd.read_csv(DATA_PATH, sep="\t")
    ffc_df = prepare_ffc_data(ffc_df, top_n=TOP_N)
    ffc_df = merge_duplicate_ffcs(ffc_df)
    print(f"FFC rows after filter: {len(ffc_df)}")

    # ── Run pipeline ──────────────────────────────────────────────────────
    cov_table, all_lines = run_greedy_annotation(
        ffc_df,
        pep,
        parent_charge=CHARGE,
        parent_mass=PARENT_MASS,
        delta=DELTA,
        min_ffc_number=MIN_FFC,
        iso_range=ISO_RANGE,
        threshold=THRESHOLD,
    )

    # ── Inspect ───────────────────────────────────────────────────────────
    print(f"\nDetected lines: {len(all_lines)}")
    print(f"Coverage columns: {[c for c in cov_table.columns if c not in ('AA',) and 'Coverage' not in str(c)][:10]}")
    print("\nCoverage table (data rows only):")
    data_rows = cov_table.iloc[:pep.pep_len - 1]
    print(data_rows.to_string())
    

    # ── Export ────────────────────────────────────────────────────────────
    with pd.ExcelWriter(
        OUTPUT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="replace",
    ) as writer:
        cov_table.to_excel(writer, sheet_name=OUTPUT_SHEET, index_label="Bond")
    print(f"\nSaved to {OUTPUT_EXCEL} [{OUTPUT_SHEET}]")
    
    print(ffc_df[ffc_df["Ranking"] == 171])
    print(ffc_df[ffc_df["Ranking"] == 114])
    print(ffc_df[ffc_df["Ranking"] == 476])
