"""
Annotation Table Generator for Deconvolved FFC Maps
=====================================================
Given a charge-resolved FFC (X, Y) from a deconvolved FFC map
(output of ``deconv_combine.deconvolute_ffc_by_lines``), this module
generates the **Annotation Table** as specified by the PI.

Column mapping from deconv_combine
-----------------------------------
The "annotated" DataFrame provides everything we need:

    ``monoisotopic_mass_A/B``
        Neutral monoisotopic masses produced by the deconvolution
        pipeline.  These are the fragment neutral masses that we
        match against theoretical b/y ions.  Computed inside
        deconv_combine as::

            mono_mass = charge * (mono_mz_charge1 − proton_mass)

    ``charge_A/B`` (equivalently ``i/j``)
        Charge assignments from line geometry.

    ``deconvoluted_mz_A/B``
        Monoisotopic m/z on the **charge-1 axis** (singly protonated).

The "replaced" DataFrame back-converts to monoisotopic m/z at the
**original charge**::

    replaced["m/z A"] = monoisotopic_mass_A / charge_A + proton_mass

For the Annotation Table we report:

    * **X, Y (m/z)**: the deconvolved monoisotopic m/z at the original
      charge state, i.e. ``monoisotopic_mass / charge + proton``.
    * **m1, m2 (neutral)**: directly from ``monoisotopic_mass_A/B``.
    * **TFFC(m1, m2)**: same as m1, m2 (already neutral).
    * b/y matching is done on the neutral masses m1, m2.

Per-FFC row information
-----------------------
    X  (deconvolved m/z at original charge)
    Y  (deconvolved m/z at original charge)
    (i, j)         – estimated charges of X and Y
    TFFC (m1, m2)  – neutral monoisotopic masses from deconvolution
    m1-annotation  – (b(m1), shift_b(m1), y(m1), shift_y(m1))
    m2-annotation  – (b(m2), shift_b(m2), y(m2), shift_y(m2))
    b/y count      – number of b/y-masses among m1, m2 (0–2)

Ranks matrix
------------
    A 3 × (|P|−1) matrix *Ranks* where:
        Ranks(1, r) = min rank of FFC(X,Y) s.t. X or Y is a b_r mass
        Ranks(2, r) = min rank of FFC(X,Y) s.t. X or Y is a y_{|P|−r} mass
        Ranks(3, r) = min rank of FFC(X,Y) s.t. one of X,Y is b_r
                      AND the other is y_{|P|−r}

Usage
-----
    from annotation_table import build_annotation_table

    # pep: a peptide.Pep object
    # df:  the "annotated" DataFrame from deconv_combine output
    table, ranks = build_annotation_table(df, pep, threshold=0.5)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Physical Constants ──────────────────────────────────────────────────────
PROTON_MASS = 1.00728       # Da  (same as deconv_combine)


# =============================================================================
# 1. THEORETICAL ION TABLE
# =============================================================================

def _build_neutral_mass_table(pep) -> Dict[str, float]:
    """
    Build a dict of *neutral* (uncharged) monoisotopic masses for each
    b/y fragment.

    ``pep.ion_mass('bN')`` returns the singly-protonated m/z, i.e.
    M_neutral + proton_mass.  We subtract the proton to get neutral mass,
    which is directly comparable to ``monoisotopic_mass_A/B`` from
    deconv_combine.
    """
    ions = {}
    for i in range(1, pep.pep_len):
        for ion_type in ("b", "y"):
            name = f"{ion_type}{i}"
            ions[name] = pep.ion_mass(name) - PROTON_MASS
    return ions


# =============================================================================
# 2. FRAGMENT MATCHING
# =============================================================================

def _find_best_b_and_y(
    neutral_mass: float,
    neutral_table: Dict[str, float],
    threshold: float,
) -> dict:
    """
    For a given neutral mass, find the closest matching b-ion and y-ion
    (independently).

    Returns
    -------
    dict with keys:
        b_name   – e.g. 'b5' or None
        b_shift  – deviation in Da (observed − theoretical) or None
        y_name   – e.g. 'y7' or None
        y_shift  – deviation in Da or None
    """
    best_b_name, best_b_shift = None, None
    best_y_name, best_y_shift = None, None
    min_b_dev = float("inf")
    min_y_dev = float("inf")

    for name, theo_mass in neutral_table.items():
        dev = neutral_mass - theo_mass        # signed deviation
        abs_dev = abs(dev)
        if abs_dev > threshold:
            continue
        if name.startswith("b") and abs_dev < min_b_dev:
            min_b_dev = abs_dev
            best_b_name = name
            best_b_shift = round(dev, 4)
        elif name.startswith("y") and abs_dev < min_y_dev:
            min_y_dev = abs_dev
            best_y_name = name
            best_y_shift = round(dev, 4)

    return {
        "b_name": best_b_name,
        "b_shift": best_b_shift,
        "y_name": best_y_name,
        "y_shift": best_y_shift,
    }


# =============================================================================
# 3. PER-FFC ROW ANNOTATION
# =============================================================================

def _annotate_ffc_rows(
    df: pd.DataFrame,
    pep,
    threshold: float,
    mono_mass_col_a: str,
    mono_mass_col_b: str,
    charge_col_a: str,
    charge_col_b: str,
    ranking_col: str,
) -> pd.DataFrame:
    """
    For each FFC row, read deconvolved neutral masses and annotate.

    Uses ``monoisotopic_mass_A/B`` directly as the neutral fragment
    masses (m1, m2).  X and Y are reported as the deconvolved m/z at
    the original charge: ``mono_mass / charge + proton``.
    """
    neutral_table = _build_neutral_mass_table(pep)

    records = []
    for _, row in df.iterrows():
        m1 = float(row[mono_mass_col_a])
        m2 = float(row[mono_mass_col_b])
        i = int(row[charge_col_a])
        j = int(row[charge_col_b])
        ranking = int(row[ranking_col]) if pd.notna(row[ranking_col]) else -1

        # Deconvolved m/z at original charge (same formula as
        # deconv_combine's "replaced" output)
        x_mz = m1 / i + PROTON_MASS
        y_mz = m2 / j + PROTON_MASS

        # Annotate each neutral mass against theoretical b/y
        ann1 = _find_best_b_and_y(m1, neutral_table, threshold)
        ann2 = _find_best_b_and_y(m2, neutral_table, threshold)

        # b/y count: how many of m1, m2 are identified as a b or y mass
        by_count = 0
        if ann1["b_name"] is not None or ann1["y_name"] is not None:
            by_count += 1
        if ann2["b_name"] is not None or ann2["y_name"] is not None:
            by_count += 1

        records.append({
            "X (m/z)": round(x_mz, 4),
            "Y (m/z)": round(y_mz, 4),
            "i": i,
            "j": j,
            "m1 (neutral)": round(m1, 4),
            "m2 (neutral)": round(m2, 4),
            "m1_b": ann1["b_name"],
            "m1_shift_b": ann1["b_shift"],
            "m1_y": ann1["y_name"],
            "m1_shift_y": ann1["y_shift"],
            "m2_b": ann2["b_name"],
            "m2_shift_b": ann2["b_shift"],
            "m2_y": ann2["y_name"],
            "m2_shift_y": ann2["y_shift"],
            "b/y_count": by_count,
            "Ranking": ranking,
        })

    return pd.DataFrame(records)


# =============================================================================
# 4. RANKS MATRIX
# =============================================================================

def _compute_ranks_matrix(
    annotated_table: pd.DataFrame,
    pep,
    threshold: float,
) -> pd.DataFrame:
    """
    Build the 3 × (|P|−1) Ranks matrix.

    For each cleavage site r = 1, ..., |P|−1:
        Row "b_r":         min ranking of any FFC where m1 or m2
                           matches b_r neutral mass (within threshold)
        Row "y_{|P|-r}":   min ranking of any FFC where m1 or m2
                           matches y_{|P|−r} neutral mass
        Row "b_r & y_{|P|-r}": min ranking of any FFC where one of
                           (m1, m2) is b_r and the other is y_{|P|−r}
    """
    pep_len = pep.pep_len
    neutral_table = _build_neutral_mass_table(pep)

    # Initialize with infinity (no match found)
    b_ranks = np.full(pep_len - 1, np.inf)
    y_ranks = np.full(pep_len - 1, np.inf)
    by_ranks = np.full(pep_len - 1, np.inf)

    for _, row in annotated_table.iterrows():
        ranking = row["Ranking"]
        m1 = row["m1 (neutral)"]
        m2 = row["m2 (neutral)"]

        for r in range(1, pep_len):
            b_name = f"b{r}"
            y_name = f"y{pep_len - r}"

            b_theo = neutral_table.get(b_name)
            y_theo = neutral_table.get(y_name)
            if b_theo is None or y_theo is None:
                continue

            m1_is_b = abs(m1 - b_theo) <= threshold
            m1_is_y = abs(m1 - y_theo) <= threshold
            m2_is_b = abs(m2 - b_theo) <= threshold
            m2_is_y = abs(m2 - y_theo) <= threshold

            idx = r - 1

            # Row 1: either m1 or m2 matches b_r
            if m1_is_b or m2_is_b:
                b_ranks[idx] = min(b_ranks[idx], ranking)

            # Row 2: either m1 or m2 matches y_{|P|-r}
            if m1_is_y or m2_is_y:
                y_ranks[idx] = min(y_ranks[idx], ranking)

            # Row 3: one is b_r AND the other is y_{|P|-r}
            if (m1_is_b and m2_is_y) or (m1_is_y and m2_is_b):
                by_ranks[idx] = min(by_ranks[idx], ranking)

    # Replace inf with NaN
    b_ranks[np.isinf(b_ranks)] = np.nan
    y_ranks[np.isinf(y_ranks)] = np.nan
    by_ranks[np.isinf(by_ranks)] = np.nan

    # Convert to pandas nullable integer
    def _to_nullable_int(arr):
        return pd.array(
            [int(v) if not np.isnan(v) else pd.NA for v in arr],
            dtype=pd.Int64Dtype(),
        )

    row_labels = [f"b{r}" for r in range(1, pep_len)]

    ranks_df = pd.DataFrame(
        {
            "cleavage_site": row_labels,
            "complement": [f"y{pep_len - r}" for r in range(1, pep_len)],
            "b_r (min rank)": _to_nullable_int(b_ranks),
            "y_|P|-r (min rank)": _to_nullable_int(y_ranks),
            "b_r & y_|P|-r (min rank)": _to_nullable_int(by_ranks),
        },
    )

    return ranks_df


# =============================================================================
# 5. PUBLIC API
# =============================================================================

def build_annotation_table(
    df: pd.DataFrame,
    pep,
    threshold: float = 0.5,
    mono_mass_col_a: str = "monoisotopic_mass_A",
    mono_mass_col_b: str = "monoisotopic_mass_B",
    charge_col_a: str = "charge_A",
    charge_col_b: str = "charge_B",
    ranking_col: str = "Ranking",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the full Annotation Table from a deconvolved FFC DataFrame.

    Parameters
    ----------
    df : DataFrame
        The "annotated" DataFrame output from
        ``deconv_combine.deconvolute_ffc_by_lines``.  Must contain
        monoisotopic mass columns, charge columns, and a ranking column.
    pep : peptide.Pep
        Peptide object providing ``pep_len`` and ``ion_mass(name)``.
    threshold : float
        Mass tolerance (Da) for matching neutral masses to theoretical
        b/y ions.
    mono_mass_col_a, mono_mass_col_b : str
        Column names for the deconvolved neutral monoisotopic masses.
        Default: ``"monoisotopic_mass_A"`` / ``"monoisotopic_mass_B"``
        as produced by deconv_combine.
    charge_col_a, charge_col_b : str
        Column names for the charge assignments (i, j).
    ranking_col : str
        Column name for FFC ranking (lower = stronger correlation).

    Returns
    -------
    (annotation_table, ranks_matrix) : tuple of DataFrames

    annotation_table
        One row per FFC, with columns:
            X (m/z)       – deconvolved monoisotopic m/z at original charge
            Y (m/z)       – deconvolved monoisotopic m/z at original charge
            i, j          – charge assignments
            m1 (neutral)  – monoisotopic neutral mass from deconvolution
            m2 (neutral)  – monoisotopic neutral mass from deconvolution
            m1_b, m1_shift_b, m1_y, m1_shift_y  – b/y annotation for m1
            m2_b, m2_shift_b, m2_y, m2_shift_y  – b/y annotation for m2
            b/y_count     – number of matched masses (0, 1, or 2)
            Ranking       – FFC ranking

    ranks_matrix
        One row per cleavage site (|P|−1 rows), with columns:
            cleavage_site, complement,
            b_r (min rank), y_|P|-r (min rank),
            b_r & y_|P|-r (min rank)
    """
    # ── Validate columns ─────────────────────────────────────────────────
    for col, fallback in [
        (mono_mass_col_a, None),
        (mono_mass_col_b, None),
        (charge_col_a, "i"),
        (charge_col_b, "j"),
        (ranking_col, None),
    ]:
        if col not in df.columns:
            if fallback and fallback in df.columns:
                # Remap charge columns: deconv_combine has both "i"/"j"
                # and "charge_A"/"charge_B"
                if col == charge_col_a:
                    charge_col_a = fallback
                elif col == charge_col_b:
                    charge_col_b = fallback
            else:
                raise KeyError(f"Required column {col!r} not found in DataFrame")

    # ── Step 1: Per-row annotation ───────────────────────────────────────
    ann_table = _annotate_ffc_rows(
        df, pep, threshold,
        mono_mass_col_a, mono_mass_col_b,
        charge_col_a, charge_col_b,
        ranking_col,
    )

    # ── Step 2: Ranks matrix ─────────────────────────────────────────────
    ranks = _compute_ranks_matrix(ann_table, pep, threshold)

    return ann_table, ranks


def build_annotation_table_from_combine(
    combine_result: Dict[str, pd.DataFrame],
    pep,
    threshold: float = 0.5,
    ranking_col: str = "Ranking",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper that takes the dict output of
    ``deconv_combine.deconvolute_ffc_by_lines`` directly.

    Always uses the "annotated" DataFrame which contains
    ``monoisotopic_mass_A/B`` and ``charge_A/B``.

    Parameters
    ----------
    combine_result : dict
        Must contain key "annotated".
    pep : peptide.Pep
        Peptide object.
    threshold : float
        Mass tolerance (Da) for b/y matching.
    ranking_col : str
        Column name for FFC ranking.

    Returns
    -------
    (annotation_table, ranks_matrix)
    """
    df = combine_result["annotated"]

    if df.empty:
        empty_ann = pd.DataFrame(columns=[
            "X (m/z)", "Y (m/z)", "i", "j",
            "m1 (neutral)", "m2 (neutral)",
            "m1_b", "m1_shift_b", "m1_y", "m1_shift_y",
            "m2_b", "m2_shift_b", "m2_y", "m2_shift_y",
            "b/y_count", "Ranking",
        ])
        empty_ranks = pd.DataFrame(columns=[
            "cleavage_site", "complement",
            "b_r (min rank)", "y_|P|-r (min rank)",
            "b_r & y_|P|-r (min rank)",
        ])
        return empty_ann, empty_ranks

    return build_annotation_table(
        df, pep, threshold,
        mono_mass_col_a="monoisotopic_mass_A",
        mono_mass_col_b="monoisotopic_mass_B",
        charge_col_a="charge_A",
        charge_col_b="charge_B",
        ranking_col=ranking_col,
    )


# =============================================================================
# 6. PRETTY-PRINT / EXPORT HELPERS
# =============================================================================

def format_annotation_cell(row: pd.Series) -> str:
    """
    Format one row of the annotation table into a compact string
    matching the PI's specification:

        X | Y | (i,j) | TFFC(m1,m2) |
        (b(m1), shift_b(m1), y(m1), shift_y(m1)) |
        (b(m2), shift_b(m2), y(m2), shift_y(m2)) | b/y_count
    """
    parts = [
        f"X={row['X (m/z)']:.4f}",
        f"Y={row['Y (m/z)']:.4f}",
        f"({row['i']},{row['j']})",
        f"TFFC({row['m1 (neutral)']:.4f}, {row['m2 (neutral)']:.4f})",
        f"m1:({row['m1_b']}, {row['m1_shift_b']}, {row['m1_y']}, {row['m1_shift_y']})",
        f"m2:({row['m2_b']}, {row['m2_shift_b']}, {row['m2_y']}, {row['m2_shift_y']})",
        f"b/y={row['b/y_count']}",
    ]
    return " | ".join(parts)


def annotation_table_to_excel(
    ann_table: pd.DataFrame,
    ranks: pd.DataFrame,
    filepath: str,
    sheet_ann: str = "Annotation",
    sheet_ranks: str = "Ranks",
):
    """Write both tables to an Excel workbook."""
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        ann_table.to_excel(writer, sheet_name=sheet_ann, index=False)
        ranks.to_excel(writer, sheet_name=sheet_ranks, index=False)


# =============================================================================
# 7. STANDALONE DEMO
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # ── Project imports ───────────────────────────────────────────────────
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    import peptide  # noqa: F401

    # ── Configuration (example: KWK6) ────────────────────────────────────
    PEP_SEQ = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"
    CHARGE = 6
    THRESHOLD = 0.5

    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20="NH3")
    print(f"Peptide: {PEP_SEQ}")
    print(f"Precursor mass: {pep.pep_mass}")
    print(f"Peptide length: {pep.pep_len}")

    # ── Load deconv_combine output ────────────────────────────────────────
    ANNOTATED_PATH = (
        sys.argv[1] if len(sys.argv) > 1
        else "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/"
             "deconv/KWK6+NCE20_combine_annotated_test.txt"
    )

    df = pd.read_csv(ANNOTATED_PATH, sep="\t")
    print(f"Loaded {len(df)} rows from {ANNOTATED_PATH}")
    print(f"Columns: {list(df.columns)}")

    # ── Build annotation table ────────────────────────────────────────────
    ann_table, ranks = build_annotation_table(
        df, pep, threshold=THRESHOLD,
    )

    print("\n=== Annotation Table (first 20 rows) ===")
    print(ann_table.head(20).to_string(index=False))

    print(f"\n=== Ranks Matrix ({len(ranks)} cleavage sites) ===")
    print(ranks.to_string(index=False))

    # Summary stats
    total = len(ann_table)
    matched_2 = (ann_table["b/y_count"] == 2).sum()
    matched_1 = (ann_table["b/y_count"] == 1).sum()
    matched_0 = (ann_table["b/y_count"] == 0).sum()
    print(f"\nb/y match summary: 2={matched_2}, 1={matched_1}, 0={matched_0} (total={total})")

    covered_b = ranks["b_r (min rank)"].notna().sum()
    covered_y = ranks["y_|P|-r (min rank)"].notna().sum()
    covered_by = ranks["b_r & y_|P|-r (min rank)"].notna().sum()
    n_sites = len(ranks)
    print(f"Ranks coverage: b={covered_b}/{n_sites}, y={covered_y}/{n_sites}, b&y={covered_by}/{n_sites}")

    # ── Export ────────────────────────────────────────────────────────────
    OUTPUT = "annotation_table_output2.xlsx"
    annotation_table_to_excel(ann_table, ranks, OUTPUT)
    print(f"\nSaved to {OUTPUT}")