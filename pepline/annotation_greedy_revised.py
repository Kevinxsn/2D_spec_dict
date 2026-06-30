"""
Greedy-Line Annotation Pipeline — Revised Cell Format
======================================================
Identical to annotation_greedy.py except that the two leading tokens in each
coverage-table cell are the full ion names instead of bare isotope labels.

In annotation_greedy.py, _build_theoretical_ions stores the isotope-shift
label in the 'name' field ("0", "+1", "+2") rather than the ion identity.
So the current cell format is:

    "0, 0, [1, 3], (0.001, 0.002), [171]"       ← monoisotopic match
    "+1, 0, [1, 3], (0.001, 0.002), [171]"       ← isotope +1 on one side
    "(0)-229.11, 0, [1, 3], (0.001, 0.002), [171]"  ← loss line

This file replaces those leading tokens with the full ion name:

    "b5, y10, [1, 3], (0.001, 0.002), [171]"
    "b5+1, y10, [1, 3], (0.001, 0.002), [171]"
    "(b5)-229.11, y10, [1, 3], (0.001, 0.002), [171]"

Everything else (row labels, column keys, FFC/FPR counts, charges, deviations,
ranking) is identical to annotation_greedy.py.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ── Low-level helpers from annotation.py ─────────────────────────────────────
from annotation import (
    MASS_H,
    ISOTOPE_OFFSET,
    _build_row_labels,
    _build_theoretical_ions,
    _find_all_matches,
    _record_example,
    get_complementary_ion,
    process_ffc_annotations,
    _combine_coverage,
)

# ── Greedy-line detection ─────────────────────────────────────────────────────
from greedy_line import Line, greedy_lines

# ── Helpers shared with annotation_greedy.py (no changes needed) ─────────────
from annotation_greedy import (
    _col_key,
    _build_ions_for_line,
    _build_summary_rows,
)


# =============================================================================
# 1. ION-NAME FORMATTER
# =============================================================================

def _format_ion_name(base_name: str, label: str) -> str:
    """
    Combine a base ion name with its isotope/loss label into a readable string.

    ``label`` is the 'name' field produced by _build_theoretical_ions or
    _build_ions_for_line:
        "0"           → monoisotopic     → "b5"
        "+1"          → isotope +1       → "b5+1"
        "(0)-229.11"  → loss, mono       → "(b5)-229.11"
        "(+1)-229.11" → loss + isotope   → "(b5+1)-229.11"
    """
    if label is None:
        return base_name or "?"
    if label == "0":
        return base_name
    if label.startswith("+"):
        # pure isotope: "+1", "+2", ...
        return f"{base_name}{label}"
    if label.startswith("("):
        # loss expression: "(iso_label)-mass"
        close = label.index(")")
        inner = label[1:close]          # "0" or "+1"
        suffix = label[close:]          # ")-229.11"
        ion_label = base_name if inner == "0" else f"{base_name}{inner}"
        return f"({ion_label}{suffix}"
    # fallback (shouldn't happen with current ion builders)
    return f"{base_name}_{label}"


# =============================================================================
# 2. REVISED CELL-FORMAT RECORDERS
# =============================================================================

def _record_full_match(row, col, annot_table, iso_tracker):
    """
    Handle an FFC where both fragments are identified.

    Cell format (unchanged structure, improved leading tokens):
        "b5, y10, [1, 3], (0.001, 0.002), [171]"
    instead of the original:
        "0, 0, [1, 3], (0.001, 0.002), [171]"
    """
    bases = [
        (row["base_name_A"], row["explanation_A"], row["charge_A"],
         round(row["deviation_A"], 3)),
        (row["base_name_B"], row["explanation_B"], row["charge_B"],
         round(row["deviation_B"], 3)),
    ]
    bases.sort()
    pair_key = f"{bases[0][0]}, {bases[1][0]}"

    if pair_key not in annot_table[col]:
        return

    iso_sum = row["iso_A"] + row["iso_B"]
    if iso_sum < iso_tracker[col][pair_key]:
        display_0 = _format_ion_name(bases[0][0], bases[0][1])
        display_1 = _format_ion_name(bases[1][0], bases[1][1])
        annot_table[col][pair_key] = (
            f"{display_0}, {display_1}, [{bases[0][2]}, {bases[1][2]}], "
            f"{bases[0][3], bases[1][3]}, [{int(row['Ranking'])}]"
        )
        iso_tracker[col][pair_key] = iso_sum


def _record_partial_match(row, col, pep, annot_table,
                          example_list, example_t_list, max_ex):
    """
    Handle an FFC where only one fragment is identified.

    Cell format:
        "b5, y10???, [1, 3], (0.001, n/a), [171]"
    where the unknown side is the complementary ion name followed by "???".
    """
    if row["base_name_A"] is not None:
        known_base   = row["base_name_A"]
        known_label  = row["explanation_A"]
        known_charge = row["charge_A"]
        known_dev    = round(row["deviation_A"], 3)
        unk_charge   = row["charge_B"]
    else:
        known_base   = row["base_name_B"]
        known_label  = row["explanation_B"]
        known_charge = row["charge_B"]
        known_dev    = round(row["deviation_B"], 3)
        unk_charge   = row["charge_A"]

    complement = get_complementary_ion(pep, known_base)
    display_known = _format_ion_name(known_base, known_label)

    bases = [
        (known_base,  display_known,        known_charge, known_dev),
        (complement,  f"{complement}???",   unk_charge,   "n/a"),
    ]
    bases.sort()
    pair_key = f"{bases[0][0]}, {bases[1][0]}"

    annotation = (
        f"{bases[0][1]}, {bases[1][1]}, [{bases[0][2]}, {bases[1][2]}], "
        f"{bases[0][3], bases[1][3]}, [{int(row['Ranking'])}]"
    )

    if pair_key in annot_table[col] and annot_table[col][pair_key] is not None:
        annot_table[col][pair_key] = annotation

    if len(example_list) < max_ex:
        example_list.append(annotation + "|")
        example_t_list.append((
            round(row["m/z A"] * row["charge_A"], 4),
            round(row["m/z B"] * row["charge_B"], 4),
            round(row["m/z A"] * row["charge_A"] + row["m/z B"] * row["charge_B"], 4),
        ))


# =============================================================================
# 3. COVERAGE TABLE (identical logic, local recorders)
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

    Identical to annotation_greedy.coverage_table_greedy except cell content
    uses full ion names instead of bare isotope labels.
    """
    row_labels, theo_masses = _build_row_labels(pep)
    length = len(pep.AA_array)
    has_ranking = ranking_col in ffc_df.columns

    _ions_cache: dict = {}

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

    for ln in lines:
        if ln.member_indices.size == 0:
            continue

        i, j         = ln.i, ln.j
        actual_shift  = ln.mass - parent_mass
        shift         = round(actual_shift, shift_decimals)
        col           = _col_key(i, j, shift, shift_decimals)

        neutral_loss  = max(0.0, -actual_shift)
        cache_key     = round(neutral_loss, 3)
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

            adj_a = i * mz_a - (i - 1) * MASS_H
            adj_b = j * mz_b - (j - 1) * MASS_H

            matches_a = _find_all_matches(adj_a, ions, threshold)
            matches_b = _find_all_matches(adj_b, ions, threshold)
            ffc_counts[col] += 1

            for ma in matches_a:
                for mb in matches_b:
                    both_known   = ma["base_name"] is not None and mb["base_name"] is not None
                    both_unknown = ma["base_name"] is None     and mb["base_name"] is None

                    row = pd.Series({
                        "m/z A":         mz_a,
                        "m/z B":         mz_b,
                        "charge_A":      i,
                        "charge_B":      j,
                        "adj_mass_A":    adj_a,
                        "adj_mass_B":    adj_b,
                        "base_name_A":   ma["base_name"],
                        "base_name_B":   mb["base_name"],
                        "explanation_A": ma["name"],
                        "explanation_B": mb["name"],
                        "deviation_A":   ma["dev"]  if ma["dev"]  is not None else float("inf"),
                        "deviation_B":   mb["dev"]  if mb["dev"]  is not None else float("inf"),
                        "iso_A":         ma["iso"]  if ma["iso"]  is not None else 999,
                        "iso_B":         mb["iso"]  if mb["iso"]  is not None else 999,
                        "Ranking":       ranking,
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

    table = pd.DataFrame(annot_table, columns=col_order)
    table.index = row_labels

    table = table[table.isna().sum().sort_values().index]

    table["Row Count"]        = table.notna().sum(axis=1).astype("Int64")
    table["theoretical mass"] = theo_masses

    col_count = table.notna().sum(axis=0).astype("Int64")
    table = pd.concat([
        table,
        pd.DataFrame([col_count.values], columns=table.columns, index=["Col Count"]),
    ])
    table["AA"] = pep.AA_array

    summary = _build_summary_rows(
        table.columns, col_order,
        ffc_counts, fpc_both_unk, fpc_one_unk,
        ex_both, ex_both_t, ex_one, ex_one_t, MAX_EX,
    )
    table = pd.concat([table, summary])

    covered   = (table["Row Count"] != 0).map({True: "+", False: "0"})
    n_covered = (covered == "+").sum()
    cov_pct   = n_covered / length * 100
    table[f"Coverage={cov_pct:.1f}%"] = covered

    ordered = ["AA", f"Coverage={cov_pct:.1f}%"] + [
        c for c in table.columns if c not in ("AA", f"Coverage={cov_pct:.1f}%")
    ]
    return table[ordered]


# =============================================================================
# 4. PIPELINE RUNNER
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

    Returns (coverage_table, all_lines).
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
# 5. EXAMPLE USAGE
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
    #DATA_PATH    = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/VEA_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/pepline/result/HAD_merged.tsv"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.GLP2_Z4_NCE15_200_ions"
    #DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/deiso/merge/KWK5_merged.tsv"
    DATA_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/Covariances_MoreFragments/Covariance_Data.LL37_Z6.NCE30_130_ions"

    #PEP_SEQ      = "VEADIAGHGQEVLIR"
    #PEP_SEQ      = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
    #PEP_SEQ = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
    
    PEP_SEQ = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    #CHARGE       = 3
    #CHARGE       = 4
    CHARGE = 6
    TOP_N        = 1000
    DELTA        = 0.02
    MIN_FFC      = 3
    ISO_RANGE    = 3
    THRESHOLD    = 0.05
    OUTPUT_EXCEL = "result/Book4.xlsx"
    OUTPUT_SHEET = "greedy_annot_revised_LLG"

    # ── Build peptide & load data ─────────────────────────────────────────
    #pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20="NH3")
    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
    print(f"Precursor mass: {pep.pep_mass}")
    PARENT_MASS = pep.pep_mass

    #ffc_df = pd.read_csv(DATA_PATH, sep="\t")
    ffc_df = pd.read_csv(DATA_PATH, sep=r"\s+", skiprows=1, header=None, engine="python")

    ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    
    
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
    print("\nCoverage table (data rows only):")
    data_rows = cov_table.iloc[:pep.pep_len - 1]
    print(data_rows.to_string())

    # ── Export ────────────────────────────────────────────────────────────
    with pd.ExcelWriter(
        OUTPUT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="replace",
    ) as writer:
        cov_table.to_excel(writer, sheet_name=OUTPUT_SHEET, index_label="Bond")
    print(f"\nSaved to {OUTPUT_EXCEL} [{OUTPUT_SHEET}]")
