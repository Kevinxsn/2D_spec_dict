"""
FFC-Line-Based Deconvolution
============================
A charge-assignment + deconvolution pipeline that uses fragment-fragment
correlation (FFC) line structure as an external constraint, instead of
relying solely on isotope spacing.

Motivation
----------
Classic spacing-based deconvolution (see ``deconv_df_intensity.py``)
struggles when peaks from different charge states interleave.  Example:
a z=3 envelope at 100, 100.33, 100.66 sitting next to a z=2 envelope at
100, 100.5, 101 — the spacing search can group peaks across charges and
return wrong charges and wrong neutral masses.

This module fixes that by exploiting an *independent* constraint that
the spacing search ignores: in a 2D-PC-MS map, fragments born from the
same precursor satisfy

    i * (m/z A) + j * (m/z B)  ≈  M_parent  -  Δ      (i + j == z_parent)

Δ may be small (isotope envelope offset, neutral loss, ~0).  Each (i, j)
charge split therefore produces a *line* on the FFC map.  The line-finder
in ``line_finding.py`` already detects these.  Once we know which line an
FFC point lies on, we know its charge split — directly, without isotope
spacing.

Pipeline
--------
1. Run line finding (only lines with i + j == parent_charge are used).
2. For each FFC point that lies on at least one line, record (charge_A,
   charge_B).  Special case: if the point lies on the parental line
   (offset ≈ 0), it must lie on that line ONLY; otherwise multiple line
   assignments per point are allowed and each produces an output row.
3. Convert each charged m/z to its charge-1-equivalent
   (the singly-protonated m/z that the same fragment would have):

       mz1  =  z * mz  -  (z - 1) * proton

   This puts every fragment on a common axis where the spacing-based
   deconvoluter can group isotopologues unambiguously.
4. Run the existing isotope-envelope grouping + theo_patt monoisotopic
   refinement on the charge-1-equivalent values.
5. Return only the FFCs that were successfully assigned a charge.

Outputs
-------
A dict with three DataFrames:
    "annotated"  -- original FFC rows + charge_A, charge_B, line info,
                    deconvoluted m/z (charge-1 equivalent), neutral mass,
                    isotope similarity.  Multiple rows per original FFC
                    if it sits on multiple non-parental lines.
    "replaced"   -- same shape as annotated, but with m/z A and m/z B
                    replaced by deconvoluted neutral masses.
    "line_map"   -- one row per (FFC point, line) assignment; tells you
                    which line each FFC was attributed to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from line_finding import detect_line_clusters, compute_parent_offsets
from deconv_df_intensity import (
    PROTON_MASS,
    _load_lookup,
    _find_isotopic_envelopes,
    determine_monoisotopic_mz,
)


# =============================================================================
# 1. LINE-BASED CHARGE ASSIGNMENT
# =============================================================================

@dataclass
class LineAssignment:
    """One (FFC point) -> (line) attribution."""
    ffc_index: int          # row index in the input FFC DataFrame
    i: int                  # charge of fragment A
    j: int                  # charge of fragment B
    line_id: int            # index into the lines DataFrame
    line_offset: float      # Parent+X for this line (Da)
    line_n_points: int      # cluster size of this line
    is_parental: bool       # True iff this line's |offset| <= parental_tol


def find_charge_lines(
    ffc_df: pd.DataFrame,
    parent_charge: int,
    parent_mass: float,
    delta: float = 0.02,
    min_cluster_size: int = 3,
    max_offset: float = 4.05,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
) -> pd.DataFrame:
    """
    Detect FFC lines whose charge split sums to ``parent_charge``.

    Wraps ``line_finding.detect_line_clusters`` and keeps only the
    physically meaningful lines (i + j == parent_charge, offset within
    ``max_offset`` of the precursor mass).  Adds a stable ``line_id``
    column for downstream cross-referencing.

    Returns
    -------
    DataFrame with columns:
        line_id, i, j, n_points, center, min_v, max_v, Parent+X,
        point_indices
    sorted by n_points descending.
    """
    raw = detect_line_clusters(
        ffc_df,
        parent_charge=parent_charge,
        delta=delta,
        col_a=col_a,
        col_b=col_b,
        enforce_sum_leq_charge=True,
        min_cluster_size=min_cluster_size,
        return_point_indices=True,
    )
    if raw.empty:
        return raw

    lines = compute_parent_offsets(raw, parent_mass, max_offset=max_offset)
    if lines.empty:
        return lines

    # Restrict to lines whose total charge equals the precursor charge.
    lines = lines[(lines["i"] + lines["j"]) == parent_charge].copy()
    lines = (
        lines.sort_values("n_points", ascending=False)
        .reset_index(drop=True)
    )
    lines["line_id"] = np.arange(len(lines), dtype=int)
    return lines


def assign_points_to_lines(
    lines: pd.DataFrame,
    parental_tol: float = 0.05,
) -> List[LineAssignment]:
    """
    Convert a lines DataFrame into a flat list of point->line assignments.

    Rule:
        * Parental line: |Parent+X| <= parental_tol.  An FFC point
          attached to a parental line is "claimed" — it cannot also be
          attributed to any other line.
        * Non-parental line: an FFC point may be attributed to multiple
          non-parental lines simultaneously (different (i, j) splits are
          all kept).
    """
    if lines.empty:
        return []

    parental_mask = lines["Parent+X"].abs() <= parental_tol
    parental_lines = lines[parental_mask]
    other_lines = lines[~parental_mask]

    claimed: set[int] = set()
    assignments: List[LineAssignment] = []

    # Parental lines first — they own their points exclusively.
    for _, row in parental_lines.iterrows():
        for ffc_idx in row["point_indices"]:
            ffc_idx = int(ffc_idx)
            if ffc_idx in claimed:
                continue
            claimed.add(ffc_idx)
            assignments.append(
                LineAssignment(
                    ffc_index=ffc_idx,
                    i=int(row["i"]),
                    j=int(row["j"]),
                    line_id=int(row["line_id"]),
                    line_offset=float(row["Parent+X"]),
                    line_n_points=int(row["n_points"]),
                    is_parental=True,
                )
            )

    # Non-parental lines — multi-membership allowed, but skip points
    # already claimed by a parental line.
    for _, row in other_lines.iterrows():
        for ffc_idx in row["point_indices"]:
            ffc_idx = int(ffc_idx)
            if ffc_idx in claimed:
                continue
            assignments.append(
                LineAssignment(
                    ffc_index=ffc_idx,
                    i=int(row["i"]),
                    j=int(row["j"]),
                    line_id=int(row["line_id"]),
                    line_offset=float(row["Parent+X"]),
                    line_n_points=int(row["n_points"]),
                    is_parental=False,
                )
            )

    return assignments


# =============================================================================
# 2. CHARGE-1 EQUIVALENT  + ISOTOPE-ENVELOPE DECONVOLUTION
# =============================================================================

def _to_charge1_mz(mz: float, z: int) -> float:
    """Singly-protonated m/z that a fragment of charge z and m/z mz would have."""
    return z * mz - (z - 1) * PROTON_MASS


def _deconvolute_charge1_axis(
    mz1_vals: np.ndarray,
    intensities: np.ndarray,
    mono_lookup,
    env_lookup,
    mz_tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run isotope-envelope grouping + monoisotopic refinement on values
    that have ALREADY been transformed to the singly-charged axis.

    Crucially, we force every envelope to z = 1 *for spacing*, because
    the values are now charge-1-equivalent.  The "true" precursor charge
    of each peak is tracked separately by the caller.

    Returns three arrays of length len(mz1_vals):
        mono_mz1  -- monoisotopic m/z on the charge-1 axis
        mono_mass -- neutral monoisotopic mass
        sim       -- isotope-envelope cosine similarity (NaN if not scored)
    """
    n = len(mz1_vals)
    mono_mz1 = np.full(n, np.nan)
    mono_mass = np.full(n, np.nan)
    sim_arr = np.full(n, np.nan)

    if n == 0:
        return mono_mz1, mono_mass, sim_arr

    # _find_isotopic_envelopes scans z = 4..1; on a charge-1-equivalent
    # axis, only z=1 spacing (~1.00335) is physical.  We sidestep that
    # by directly building z=1 envelopes ourselves via single-linkage
    # gap clustering at the isotope spacing.
    order = np.argsort(mz1_vals)
    mz_s = mz1_vals[order]
    int_s = intensities[order]

    ISO = 1.003355
    tol = mz_tol

    envelopes: List[dict] = []
    start = 0
    for k in range(1, n):
        gap = mz_s[k] - mz_s[k - 1]
        # Continue the current envelope only if the gap is close to one
        # isotope step.  Otherwise, close it off and start a new one.
        if abs(gap - ISO) <= tol:
            continue
        # Close current envelope (single peak or multi-peak).
        envelopes.append(
            {
                "sorted_indices": list(range(start, k)),
                "original_indices": order[start:k].tolist(),
                "mz_values": mz_s[start:k].tolist(),
                "intensities": int_s[start:k].tolist(),
                "charge": 1,
            }
        )
        start = k
    envelopes.append(
        {
            "sorted_indices": list(range(start, n)),
            "original_indices": order[start:n].tolist(),
            "mz_values": mz_s[start:n].tolist(),
            "intensities": int_s[start:n].tolist(),
            "charge": 1,
        }
    )

    for env in envelopes:
        mono_mz_v, mono_mass_v, _z, sim = determine_monoisotopic_mz(
            env, mono_lookup, env_lookup
        )
        for orig_idx in env["original_indices"]:
            mono_mz1[orig_idx] = mono_mz_v
            mono_mass[orig_idx] = mono_mass_v
            sim_arr[orig_idx] = sim

    return mono_mz1, mono_mass, sim_arr


# =============================================================================
# 3. PUBLIC PIPELINE
# =============================================================================

def deconvolute_ffc_by_lines(
    df: pd.DataFrame,
    parent_charge: int,
    parent_mass: float,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = None,
    intensity_col_b: Optional[str] = None,
    theo_patt_path: Optional[str] = None,
    line_delta: float = 0.02,
    min_cluster_size: int = 3,
    max_offset: float = 4.05,
    parental_tol: float = 0.05,
    mz_tol: float = 0.02,
) -> Dict[str, pd.DataFrame]:
    """
    Full FFC-line-based deconvolution.

    Parameters
    ----------
    df : DataFrame
        Input FFC table.  Must contain ``mz_col_a`` and ``mz_col_b``.
    parent_charge : int
        Charge state of the precursor (only lines with i + j == this
        value are used).
    parent_mass : float
        Neutral monoisotopic mass of the precursor (Da).
    intensity_col_a, intensity_col_b : str, optional
        Per-peak intensity columns.  If absent, uniform intensities are
        used (envelope grouping still works; monoisotopic shape scoring
        becomes less informative).
    theo_patt_path : str, optional
        Path to TopPIC theo_patt.txt for monoisotopic refinement.
    line_delta : float
        Clustering gap for line detection (Da, on the i*x + j*y axis).
    min_cluster_size : int
        Minimum points per line.
    max_offset : float
        Discard lines whose |Parent+X| exceeds this.
    parental_tol : float
        Lines with |Parent+X| <= this are treated as the parental line
        (single-membership).
    mz_tol : float
        Isotope-spacing tolerance on the charge-1-equivalent axis.

    Returns
    -------
    dict with keys "annotated", "replaced", "line_map".
    """
    if mz_col_a not in df.columns or mz_col_b not in df.columns:
        raise KeyError(
            f"Input df must contain columns {mz_col_a!r} and {mz_col_b!r}"
        )

    work = df.reset_index(drop=True).copy()

    # ── Step 1: find lines ────────────────────────────────────────────────
    lines = find_charge_lines(
        work,
        parent_charge=parent_charge,
        parent_mass=parent_mass,
        delta=line_delta,
        min_cluster_size=min_cluster_size,
        max_offset=max_offset,
        col_a=mz_col_a,
        col_b=mz_col_b,
    )

    # ── Step 2: assign FFC points to lines ────────────────────────────────
    assignments = assign_points_to_lines(lines, parental_tol=parental_tol)

    if not assignments:
        empty = pd.DataFrame()
        return {"annotated": empty, "replaced": empty, "line_map": empty}

    # Build the per-assignment "long" table.  One row per (ffc_idx, line).
    line_map_rows = []
    for a in assignments:
        line_map_rows.append(
            {
                "ffc_index": a.ffc_index,
                "line_id": a.line_id,
                "i": a.i,
                "j": a.j,
                "line_offset": a.line_offset,
                "line_n_points": a.line_n_points,
                "is_parental": a.is_parental,
            }
        )
    line_map_df = pd.DataFrame(line_map_rows)

    # Join with the original FFC rows.
    base = work.iloc[line_map_df["ffc_index"].values].reset_index(drop=True)
    annotated = pd.concat(
        [base, line_map_df.reset_index(drop=True).drop(columns=["ffc_index"])],
        axis=1,
    )
    annotated["ffc_index"] = line_map_df["ffc_index"].values

    # ── Step 3: charge-1-equivalent transform ─────────────────────────────
    annotated["mz1_A"] = [
        _to_charge1_mz(mz, z)
        for mz, z in zip(annotated[mz_col_a].values, annotated["i"].values)
    ]
    annotated["mz1_B"] = [
        _to_charge1_mz(mz, z)
        for mz, z in zip(annotated[mz_col_b].values, annotated["j"].values)
    ]

    # ── Step 4: isotope-envelope deconvolution on the charge-1 axis ───────
    mono_lookup, env_lookup = _load_lookup(theo_patt_path)

    if intensity_col_a and intensity_col_a in annotated.columns:
        int_a = annotated[intensity_col_a].values.astype(float)
    else:
        int_a = np.ones(len(annotated), dtype=float)

    if intensity_col_b and intensity_col_b in annotated.columns:
        int_b = annotated[intensity_col_b].values.astype(float)
    else:
        int_b = np.ones(len(annotated), dtype=float)

    mono_mz1_a, mono_mass_a, sim_a = _deconvolute_charge1_axis(
        annotated["mz1_A"].values.astype(float), int_a,
        mono_lookup, env_lookup, mz_tol,
    )
    mono_mz1_b, mono_mass_b, sim_b = _deconvolute_charge1_axis(
        annotated["mz1_B"].values.astype(float), int_b,
        mono_lookup, env_lookup, mz_tol,
    )

    annotated["charge_A"] = annotated["i"]
    annotated["charge_B"] = annotated["j"]
    annotated["deconvoluted_mz_A"] = np.round(mono_mz1_a, 4)
    annotated["deconvoluted_mz_B"] = np.round(mono_mz1_b, 4)
    annotated["monoisotopic_mass_A"] = np.round(mono_mass_a, 4)
    annotated["monoisotopic_mass_B"] = np.round(mono_mass_b, 4)
    annotated["isotope_similarity_A"] = np.round(sim_a, 4)
    annotated["isotope_similarity_B"] = np.round(sim_b, 4)

    # ── Step 5: build the "replaced" view ─────────────────────────────────
    # Replace m/z A and m/z B with the deconvoluted *neutral mass* (the
    # quantity downstream b/y matching wants).  Same shape & columns as
    # the input df, plus charge columns appended for traceability.
    replaced = base.copy()
    replaced[mz_col_a] = annotated["monoisotopic_mass_A"].values / annotated["charge_A"] + PROTON_MASS
    replaced[mz_col_b] = annotated["monoisotopic_mass_B"].values / annotated["charge_B"] + PROTON_MASS
    #replaced["charge_A"] = annotated["charge_A"].values
    #replaced["charge_B"] = annotated["charge_B"].values
    replaced["line_id"] = annotated["line_id"].values

    return {
        "annotated": annotated.reset_index(drop=True),
        "replaced": replaced.reset_index(drop=True),
        "line_map": line_map_df.reset_index(drop=True),
    }


# =============================================================================
# 4. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Same data layout as your existing scripts.
    DATA_PATH = (
        "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/"
        "HAD4+_with_intensity"
    )
    SAVE_DIR = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/deconv/"

    PARENT_CHARGE = 4
    PARENT_MASS = 3767.8441  # <-- fill in the neutral precursor mass for HAD4+

    df = pd.read_csv(DATA_PATH, sep="\t", skiprows=1, header=None, engine="python")
    df.columns = [
        "m/z A", "m/z B", "Covariance", "Partial Cov.",
        "Score", "Ranking", "intensity A", "intensity B",
    ]
    df = df[df["Ranking"] != -1].sort_values("Ranking").head(2000)

    result = deconvolute_ffc_by_lines(
        df,
        parent_charge=PARENT_CHARGE,
        parent_mass=PARENT_MASS,
        intensity_col_a="intensity A",
        intensity_col_b="intensity B",
        theo_patt_path="theo_patt.txt",
        line_delta=0.02,
        min_cluster_size=3,
        max_offset=4.05,
        parental_tol=0.05,
        mz_tol=0.02,
    )

    result["annotated"].to_csv(SAVE_DIR + "HAD4_ffc_annotated.txt",
                               sep="\t", index=False)
    result["replaced"].to_csv(SAVE_DIR + "HAD4_ffc_replaced.txt",
                              sep="\t", index=False)
    result["line_map"].to_csv(SAVE_DIR + "HAD4_ffc_line_map.txt",
                              sep="\t", index=False)

    print(f"Lines used:        {result['line_map']['line_id'].nunique()}")
    print(f"FFC points kept:   {result['line_map']['ffc_index'].nunique()}")
    print(f"Total assignments: {len(result['annotated'])}")