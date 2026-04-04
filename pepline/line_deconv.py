"""
Line-Based FFC Deconvolution
=============================
Integrates the annotation pipeline's charge-partitioning logic with the
deconvolution pipeline's monoisotopic mass determination.

Core idea (Prof. Xiaowen Liu):
    FFCs on a mass-conservation line satisfy:
        i * (m/z A) + j * (m/z B) = ParentMass - deviation
    where i + j = precursor charge (or charge - 1 for charge-reduced lines).

    The line's slope = -i/j directly gives the charge pair (i, j).
    Once i, j are known for each FFC, deconvolution is straightforward:
        neutral_mass_A = i * (m/z_A - PROTON_MASS)
        neutral_mass_B = j * (m/z_B - PROTON_MASS)

    For ambiguous cases (e.g., precursor charge 4, slope -1 → could be (1,1)
    or (2,2)), we search for isotopic satellites at spacing 0.5 Da (charge 2)
    vs 1.0 Da (charge 1) among the FFCs on the same line.

Pipeline:
    1. partition_to_lines()  — assign each FFC to its best line & charge pair
    2. disambiguate_slope_minus_one() — resolve (1,1) vs (2,2) for charge-4
    3. deconvolve_by_charge() — compute monoisotopic masses using known charges
    4. line_deconvolute() — top-level function combining everything
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Physical Constants ──────────────────────────────────────────────────────
PROTON_MASS = 1.00728
ISOTOPE_OFFSET = 1.003355  # average isotopic spacing in Da
MASS_H = 1.00784           # hydrogen mass for singly-charged adjustment


# =============================================================================
# 1. THEORETICAL PATTERN LOOKUP (reused from deconv_dataframe)
# =============================================================================

def parse_theo_patt(filepath: str) -> list[dict]:
    """Parse TopPIC theoretical isotope pattern file."""
    envelopes = []
    current_peaks = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("formula:") or line.startswith("#"):
                if current_peaks:
                    intensities = [p[1] for p in current_peaks]
                    envelopes.append({
                        "mono_mass": current_peaks[0][0],
                        "peaks": current_peaks,
                        "max_peak_index": int(np.argmax(intensities)),
                    })
                current_peaks = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    current_peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

    if current_peaks:
        intensities = [p[1] for p in current_peaks]
        envelopes.append({
            "mono_mass": current_peaks[0][0],
            "peaks": current_peaks,
            "max_peak_index": int(np.argmax(intensities)),
        })
    return envelopes


def build_lookup_table(envelopes):
    envelopes_sorted = sorted(envelopes, key=lambda e: e["mono_mass"])
    mono_masses = np.array([e["mono_mass"] for e in envelopes_sorted])
    return mono_masses, envelopes_sorted


def find_closest_envelope(mono_masses, envelopes_sorted, target_mass):
    idx = np.searchsorted(mono_masses, target_mass)
    candidates = [i for i in [idx - 1, idx] if 0 <= i < len(mono_masses)]
    best = min(candidates, key=lambda i: abs(mono_masses[i] - target_mass))
    return envelopes_sorted[best]


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm else 0.0


def _load_lookup(theo_patt_path):
    """Load theoretical pattern lookup table if path is valid."""
    if theo_patt_path and Path(theo_patt_path).exists():
        raw_envs = parse_theo_patt(theo_patt_path)
        mono_lookup, env_lookup = build_lookup_table(raw_envs)
        print(f"  Loaded {len(env_lookup)} theoretical envelopes "
              f"from {theo_patt_path}")
        return mono_lookup, env_lookup
    elif theo_patt_path:
        print(f"  Warning: {theo_patt_path} not found. "
              "Monoisotopic peak assumed to be first observed peak.")
    return None, None


# =============================================================================
# 2. LINE-BASED CHARGE PARTITIONING
# =============================================================================

def _enumerate_charge_pairs(precursor_charge: int) -> List[Tuple[int, int]]:
    """
    Enumerate all valid (charge_A, charge_B) pairs where
    charge_A + charge_B = precursor_charge  OR  precursor_charge - 1
    (charge-reduced species).

    Returns list of (i, j) tuples.
    """
    pairs = []
    # Full precursor charge partitions
    for i in range(1, precursor_charge):
        j = precursor_charge - i
        pairs.append((i, j))
    # Charge-reduced partitions (charge - 1)
    if precursor_charge >= 3:
        for i in range(1, precursor_charge - 1):
            j = (precursor_charge - 1) - i
            pairs.append((i, j))
    return pairs


def partition_to_lines(
    df: pd.DataFrame,
    precursor_mass: float,
    precursor_charge: int,
    deviation_list: List[float],
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    mass_threshold: float = 0.1,
    iso_range: int = 0,
) -> pd.DataFrame:
    """
    Assign each FFC to its best mass-conservation line and charge pair.

    For each FFC, tests all (i, j) charge splits against all deviation lines:
        i * mz_A + j * mz_B  ≈  precursor_mass - deviation  (± isotope offsets)

    The best match (smallest residual within mass_threshold) wins.

    Parameters
    ----------
    df : DataFrame with mz_col_a, mz_col_b columns.
    precursor_mass : float
        Neutral mass of the precursor ion (including protons).
    precursor_charge : int
        Charge state of the precursor.
    deviation_list : list of float
        Deviations from parental mass for each line.
        E.g. [0, 1, 2, -229] means:
            line 0 (parent): i*A + j*B = M
            line 1:          i*A + j*B = M - 1
            line 2:          i*A + j*B = M - 2
            line -229:       i*A + j*B = M + 229
    mass_threshold : float
        Maximum allowed |residual| for a match.
    iso_range : int
        Number of parent-isotopic positions to consider (0 = monoisotopic only).

    Returns
    -------
    DataFrame with added columns:
        charge_A, charge_B, line_deviation, line_residual,
        parent_isotope_idx, component_A, component_B
    Unmatched FFCs are dropped.
    """
    mz_a = df[mz_col_a].values.astype(float)
    mz_b = df[mz_col_b].values.astype(float)

    charge_pairs = _enumerate_charge_pairs(precursor_charge)

    # Pre-compute all target masses:
    #   target = precursor_mass - deviation + k * ISOTOPE_OFFSET
    targets = []
    for dev in deviation_list:
        for k in range(iso_range + 1):
            target = precursor_mass - dev + k * ISOTOPE_OFFSET
            targets.append((dev, k, target))

    # For each FFC, find the best (charge_pair, line) combination
    n = len(df)
    best_charge_a = np.full(n, -1, dtype=int)
    best_charge_b = np.full(n, -1, dtype=int)
    best_deviation = np.full(n, np.nan)
    best_residual = np.full(n, np.inf)
    best_iso_idx = np.full(n, -1, dtype=int)
    best_comp_a = np.full(n, np.nan)
    best_comp_b = np.full(n, np.nan)

    for za, zb in charge_pairs:
        recon = za * mz_a + zb * mz_b

        for dev, iso_k, target in targets:
            residuals = np.abs(recon - target)
            mask = (residuals < best_residual) & (residuals <= mass_threshold)
            if mask.any():
                idx = np.where(mask)[0]
                best_charge_a[idx] = za
                best_charge_b[idx] = zb
                best_deviation[idx] = dev
                best_residual[idx] = residuals[idx]
                best_iso_idx[idx] = iso_k
                best_comp_a[idx] = za * mz_a[idx]
                best_comp_b[idx] = zb * mz_b[idx]

    # Filter to matched FFCs only
    matched = best_charge_a > 0
    result = df.loc[matched].copy()
    result["charge_A"] = best_charge_a[matched]
    result["charge_B"] = best_charge_b[matched]
    result["line_deviation"] = best_deviation[matched]
    result["line_residual"] = best_residual[matched]
    result["parent_isotope_idx"] = best_iso_idx[matched]
    result["component_A"] = best_comp_a[matched]
    result["component_B"] = best_comp_b[matched]

    return result.reset_index(drop=True)


# =============================================================================
# 3. DISAMBIGUATION FOR SLOPE -1 LINES (charge >= 4 precursor)
# =============================================================================

def disambiguate_slope_minus_one(
    df: pd.DataFrame,
    full_ffc_df: pd.DataFrame,
    precursor_charge: int,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    mz_tol: float = 0.02,
) -> pd.DataFrame:
    """
    For precursor charge >= 4, lines with slope -1 (charge_A == charge_B)
    are ambiguous: e.g., could be (1,1) or (2,2) for charge 4.

    Resolution strategy (Prof. Liu):
        Given FFCs (x_k, y_k) on the line, search the full FFC table for:
        - satellites at ±0.5 Da → evidence of charge 2
        - satellites at ±1.0 Da → evidence of charge 1

        If 0.5-Da satellites dominate → (2, 2)
        If 1.0-Da satellites dominate → (1, 1)

    Parameters
    ----------
    df : DataFrame from partition_to_lines (with charge_A, charge_B).
    full_ffc_df : Complete FFC table for satellite searching.
    precursor_charge : int
    mz_tol : tolerance for satellite matching.

    Returns
    -------
    DataFrame with charge_A, charge_B updated for disambiguated rows.
    """
    if precursor_charge < 4:
        return df

    result = df.copy()

    # Identify ambiguous rows: charge_A == charge_B
    ambiguous_mask = result["charge_A"] == result["charge_B"]
    if not ambiguous_mask.any():
        return result

    # Build fast lookup of all FFC pairs
    all_mz_a = full_ffc_df[mz_col_a].values.astype(float)
    all_mz_b = full_ffc_df[mz_col_b].values.astype(float)

    # Group ambiguous FFCs by line_deviation (same line)
    ambig_groups = result[ambiguous_mask].groupby("line_deviation")

    for line_dev, group in ambig_groups:
        x_vals = group[mz_col_a].values
        y_vals = group[mz_col_b].values
        current_z = group["charge_A"].iloc[0]

        # The two hypotheses for a slope-1 line in charge Z:
        # high_z = current_z (e.g. 2 for charge-4), spacing = 1/high_z = 0.5
        # low_z  = 1, spacing = 1.0
        high_z = current_z
        low_z = 1
        spacing_high = 1.0 / high_z  # 0.5 for z=2
        spacing_low = 1.0 / low_z    # 1.0 for z=1

        evidence_high = 0
        evidence_low = 0

        for x, y in zip(x_vals, y_vals):
            # Check A-side satellites at high-z spacing
            for delta in [spacing_high, -spacing_high]:
                if np.any(
                    (np.abs(all_mz_a - (x + delta)) < mz_tol) &
                    (np.abs(all_mz_b - y) < mz_tol)
                ):
                    evidence_high += 1

            # Check B-side satellites at high-z spacing
            for delta in [spacing_high, -spacing_high]:
                if np.any(
                    (np.abs(all_mz_a - x) < mz_tol) &
                    (np.abs(all_mz_b - (y + delta)) < mz_tol)
                ):
                    evidence_high += 1

            # Check A-side satellites at low-z spacing
            for delta in [spacing_low, -spacing_low]:
                if np.any(
                    (np.abs(all_mz_a - (x + delta)) < mz_tol) &
                    (np.abs(all_mz_b - y) < mz_tol)
                ):
                    evidence_low += 1

            # Check B-side satellites at low-z spacing
            for delta in [spacing_low, -spacing_low]:
                if np.any(
                    (np.abs(all_mz_a - x) < mz_tol) &
                    (np.abs(all_mz_b - (y + delta)) < mz_tol)
                ):
                    evidence_low += 1

        if evidence_high > evidence_low:
            result.loc[group.index, "charge_A"] = high_z
            result.loc[group.index, "charge_B"] = high_z
        elif evidence_low > evidence_high:
            result.loc[group.index, "charge_A"] = low_z
            result.loc[group.index, "charge_B"] = low_z
        # else: keep original (insufficient evidence)

    return result


# =============================================================================
# 4. MONOISOTOPIC MASS DETERMINATION (charge-aware)
# =============================================================================

def _determine_mono_mass_single_peak(
    mz_val: float,
    charge: int,
    mono_masses_lookup=None,
    envelopes_lookup=None,
) -> float:
    """
    Compute monoisotopic neutral mass for a single m/z with known charge.
    Without lookup: charge * (mz - PROTON_MASS).
    With lookup: corrects for the monoisotopic offset using closest
    theoretical envelope.
    """
    neutral_mass = charge * (mz_val - PROTON_MASS)

    if mono_masses_lookup is None or envelopes_lookup is None:
        return neutral_mass

    ref_env = find_closest_envelope(mono_masses_lookup, envelopes_lookup,
                                    neutral_mass)
    ref_offset = ref_env["max_peak_index"]
    mono_mass_corrected = neutral_mass - ref_offset * ISOTOPE_OFFSET

    return mono_mass_corrected


def _determine_mono_mass_envelope(
    mz_values: np.ndarray,
    intensities: np.ndarray,
    charge: int,
    mono_masses_lookup=None,
    envelopes_lookup=None,
    search_window: int = 3,
) -> float:
    """
    Determine monoisotopic mass from an isotopic envelope with known charge.
    Uses cosine similarity against theoretical patterns (same logic as
    deconv_dataframe.determine_monoisotopic_mass but accepts charge directly).
    """
    z = charge
    mz_arr = np.asarray(mz_values, dtype=float)
    int_arr = np.asarray(intensities, dtype=float)

    max_idx = int(np.argmax(int_arr))
    max_mz = mz_arr[max_idx]
    mass_of_max = z * (max_mz - PROTON_MASS)

    if mono_masses_lookup is None or envelopes_lookup is None:
        return z * (mz_arr[0] - PROTON_MASS)

    ref_env = find_closest_envelope(mono_masses_lookup, envelopes_lookup,
                                    mass_of_max)
    ref_offset = ref_env["max_peak_index"]
    ref_int = np.array([p[1] for p in ref_env["peaks"]])
    ref_int = ref_int / ref_int.max() * 100.0

    best_mass, best_sim = None, -1.0

    for delta in range(-search_window, search_window + 1):
        candidate_offset = ref_offset + delta
        candidate_mono_mass = mass_of_max - candidate_offset * ISOTOPE_OFFSET
        if candidate_mono_mass <= 0:
            continue

        first_obs_mass = z * (mz_arr[0] - PROTON_MASS)
        first_peak_offset = round(
            (first_obs_mass - candidate_mono_mass) / ISOTOPE_OFFSET
        )
        if first_peak_offset < 0:
            continue

        obs_aligned, theo_aligned = [], []
        for k, obs_int in enumerate(int_arr):
            theo_idx = first_peak_offset + k
            if 0 <= theo_idx < len(ref_int):
                obs_aligned.append(obs_int)
                theo_aligned.append(ref_int[theo_idx])

        if len(obs_aligned) < 2:
            continue

        obs_vec = np.array(obs_aligned)
        if obs_vec.max() > 0:
            obs_vec = obs_vec / obs_vec.max() * 100.0

        sim = cosine_similarity(obs_vec, np.array(theo_aligned))
        if sim > best_sim:
            best_sim = sim
            best_mass = candidate_mono_mass

    return best_mass if best_mass is not None else z * (mz_arr[0] - PROTON_MASS)


# =============================================================================
# 5. GROUP FFCs ON THE SAME LINE INTO ISOTOPIC ENVELOPES
# =============================================================================

def _group_line_envelopes(
    df_line: pd.DataFrame,
    mz_col: str,
    charge_col: str,
    mz_tol: float = 0.02,
) -> List[dict]:
    """
    Within a single line, group m/z values on one side into isotopic envelopes
    using the known charge.  Spacing = 1/charge between consecutive peaks.

    Returns list of envelope dicts with:
        mz_values, indices (into df_line), charge
    """
    if df_line.empty:
        return []

    charge = int(df_line[charge_col].iloc[0])
    spacing = 1.0 / charge
    mz_vals = df_line[mz_col].values.astype(float)
    indices = df_line.index.values

    order = np.argsort(mz_vals)
    mz_sorted = mz_vals[order]
    idx_sorted = indices[order]

    envelopes = []
    assigned = set()

    for i in range(len(mz_sorted)):
        if i in assigned:
            continue

        env_positions = [i]
        last_mz = mz_sorted[i]

        for j in range(i + 1, len(mz_sorted)):
            if j in assigned:
                continue
            expected = last_mz + spacing
            diff = mz_sorted[j] - expected
            if abs(diff) <= mz_tol:
                env_positions.append(j)
                last_mz = mz_sorted[j]
            elif diff > mz_tol:
                break

        for pos in env_positions:
            assigned.add(pos)

        envelopes.append({
            "mz_values": mz_sorted[env_positions].tolist(),
            "indices": idx_sorted[env_positions].tolist(),
            "charge": charge,
        })

    return envelopes


# =============================================================================
# 6. DECONVOLVE WITH KNOWN CHARGES
# =============================================================================

def deconvolve_by_charge(
    df: pd.DataFrame,
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = None,
    intensity_col_b: Optional[str] = None,
    theo_patt_path: Optional[str] = None,
    mz_tol: float = 0.02,
) -> pd.DataFrame:
    """
    Deconvolve FFCs that already have charge_A and charge_B assigned.

    For each side (A, B):
        1. Group FFCs on the same line into isotopic envelopes
        2. Determine monoisotopic neutral mass per envelope
        3. Compute deconvoluted m/z = (mono_mass + z * PROTON) / z

    Adds columns:
        monoisotopic_mass_{A,B}, deconvoluted_mz_{A,B}, envelope_id_{A,B},
        adj_mass_{A,B}  (singly-charged mass for b/y annotation)
    """
    result = df.copy()
    mono_lookup, env_lookup = _load_lookup(theo_patt_path)

    for side, mz_col, int_col, charge_col in [
        ("A", mz_col_a, intensity_col_a, "charge_A"),
        ("B", mz_col_b, intensity_col_b, "charge_B"),
    ]:
        n = len(result)
        mono_masses = np.full(n, np.nan)
        deconv_mz = np.full(n, np.nan)
        envelope_ids = np.full(n, -1, dtype=int)

        if int_col and int_col in result.columns:
            intensities = result[int_col].values.astype(float)
        else:
            intensities = np.ones(n)

        group_cols = ["line_deviation", "charge_A", "charge_B"]
        env_counter = 0

        for _, group_df in result.groupby(group_cols):
            envelopes = _group_line_envelopes(
                group_df, mz_col, charge_col, mz_tol
            )

            for env in envelopes:
                charge = env["charge"]
                env_mz = np.array(env["mz_values"])
                env_idx = env["indices"]

                # Gather intensities for envelope members
                env_int = np.array([
                    intensities[result.index.get_loc(i)]
                    if i in result.index else 1.0
                    for i in env_idx
                ])

                # Determine monoisotopic mass
                if len(env_mz) >= 2 and mono_lookup is not None:
                    mass = _determine_mono_mass_envelope(
                        env_mz, env_int, charge,
                        mono_lookup, env_lookup,
                    )
                else:
                    mass = _determine_mono_mass_single_peak(
                        env_mz[0], charge,
                        mono_lookup, env_lookup,
                    )

                # Deconvoluted m/z at native charge
                dmz = (mass + charge * PROTON_MASS) / charge

                for idx in env_idx:
                    pos = result.index.get_loc(idx)
                    mono_masses[pos] = round(mass, 4)
                    deconv_mz[pos] = round(dmz, 4)
                    envelope_ids[pos] = env_counter

                env_counter += 1

        result[f"monoisotopic_mass_{side}"] = mono_masses
        result[f"deconvoluted_mz_{side}"] = deconv_mz
        result[f"envelope_id_{side}"] = envelope_ids

    # Singly-charged adjusted mass for b/y annotation (same as annotation.py):
    #   adj_mass = charge * mz - (charge - 1) * MASS_H
    result["adj_mass_A"] = (
        result["charge_A"] * result[mz_col_a] - (result["charge_A"] - 1) * MASS_H
    )
    result["adj_mass_B"] = (
        result["charge_B"] * result[mz_col_b] - (result["charge_B"] - 1) * MASS_H
    )

    return result


# =============================================================================
# 7. TOP-LEVEL FUNCTION
# =============================================================================

def line_deconvolute(
    df: pd.DataFrame,
    precursor_mass: float,
    precursor_charge: int,
    deviation_list: List[float],
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = None,
    intensity_col_b: Optional[str] = None,
    theo_patt_path: Optional[str] = None,
    mass_threshold: float = 0.1,
    iso_range: int = 0,
    mz_tol: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full line-based deconvolution pipeline.

    Steps:
        1. Partition FFCs onto mass-conservation lines → charge pairs
        2. Disambiguate slope -1 lines for charge >= 4
        3. Deconvolve using known charges
        4. Produce replaced DataFrame (m/z swapped for deconvoluted values)

    Parameters
    ----------
    df : DataFrame with mz_col_a, mz_col_b columns.
    precursor_mass : Neutral mass of precursor (including protons).
    precursor_charge : Precursor charge state.
    deviation_list : Mass deviations for each line.
        [0, 1, 2, -229] → parent, +1 isotope loss, +2, neutral-loss 229.
    mz_col_a, mz_col_b : Column names for the two m/z axes.
    intensity_col_a, intensity_col_b : Optional intensity columns.
    theo_patt_path : Path to theo_patt.txt for monoisotopic correction.
    mass_threshold : Tolerance for line matching (Da).
    iso_range : Parent isotopic envelope positions for line matching.
    mz_tol : Tolerance for isotopic envelope grouping (Da).

    Returns
    -------
    df_annotated : DataFrame with all deconvolution columns.
    df_replaced : Same rows, but m/z columns replaced by deconvoluted values.
    """
    print(f"[line_deconv] Input: {len(df)} FFCs")
    print(f"  Precursor mass: {precursor_mass:.4f}, charge: {precursor_charge}")
    print(f"  Lines (deviations): {deviation_list}")

    # Step 1: Partition onto lines → get charge pairs
    df_partitioned = partition_to_lines(
        df, precursor_mass, precursor_charge,
        deviation_list, mz_col_a, mz_col_b,
        mass_threshold, iso_range,
    )
    print(f"  After partitioning: {len(df_partitioned)} FFCs on lines")

    if df_partitioned.empty:
        print("  Warning: No FFCs matched any line.")
        return pd.DataFrame(), pd.DataFrame()

    # Step 2: Disambiguate slope -1 for charge >= 4
    if precursor_charge >= 4:
        df_partitioned = disambiguate_slope_minus_one(
            df_partitioned, df, precursor_charge,
            mz_col_a, mz_col_b, mz_tol,
        )
        print("  Disambiguated slope-1 lines")

    # Step 3: Deconvolve with known charges
    df_deconv = deconvolve_by_charge(
        df_partitioned,
        mz_col_a, mz_col_b,
        intensity_col_a, intensity_col_b,
        theo_patt_path, mz_tol,
    )

    # Step 4: Build replaced DataFrame
    df_replaced = df_deconv.copy()
    df_replaced[mz_col_a] = df_deconv["deconvoluted_mz_A"]
    df_replaced[mz_col_b] = df_deconv["deconvoluted_mz_B"]

    # Summary
    for dev in deviation_list:
        line_mask = df_deconv["line_deviation"] == dev
        n_on_line = line_mask.sum()
        if n_on_line > 0:
            charges = df_deconv.loc[line_mask, ["charge_A", "charge_B"]].values
            unique_pairs = set(map(tuple, charges))
            label = "parent" if dev == 0 else f"dev={dev}"
            print(f"  Line {label}: {n_on_line} FFCs, "
                  f"charge pairs: {sorted(unique_pairs)}")

    return df_deconv, df_replaced


# =============================================================================
# 8. PER-LINE WRAPPER (compatible with annotation.py pipeline)
# =============================================================================

def line_deconvolute_per_line(
    df: pd.DataFrame,
    precursor_mass: float,
    precursor_charge: int,
    deviation_list: List[float],
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = None,
    intensity_col_b: Optional[str] = None,
    theo_patt_path: Optional[str] = None,
    mass_threshold: float = 0.1,
    iso_range: int = 0,
    mz_tol: float = 0.02,
) -> List[pd.DataFrame]:
    """
    Run line_deconvolute and split the result by deviation line,
    returning one DataFrame per line in the same order as deviation_list.

    This makes it easy to plug into the annotation.py pipeline where
    each loss line is annotated separately:

        frames = line_deconvolute_per_line(df, ...)
        for dev, frame in zip(deviation_list, frames):
            annotated = annotate_dataframe_loss(frame, pep, ...)
            ...

    Each returned DataFrame has the same columns as the output of
    select_best_partition() from annotation.py (charge_A, charge_B,
    adj_mass_A, adj_mass_B, etc.) plus the deconvolution columns.
    """
    df_deconv, _ = line_deconvolute(
        df, precursor_mass, precursor_charge, deviation_list,
        mz_col_a, mz_col_b,
        intensity_col_a, intensity_col_b,
        theo_patt_path, mass_threshold, iso_range, mz_tol,
    )

    if df_deconv.empty:
        return [pd.DataFrame() for _ in deviation_list]

    # Add columns expected by annotation.py
    df_deconv["source_column"] = df_deconv.apply(
        lambda r: f"{int(r['charge_A'])}*{mz_col_a} + {int(r['charge_B'])}*{mz_col_b}",
        axis=1,
    )
    df_deconv["deviation"] = df_deconv["line_residual"]
    df_deconv["selected_total"] = df_deconv["component_A"] + df_deconv["component_B"]

    frames = []
    for dev in deviation_list:
        mask = df_deconv["line_deviation"] == dev
        frames.append(df_deconv[mask].reset_index(drop=True))

    return frames


# =============================================================================
# main
# =============================================================================

if __name__ == "__main__":
    
    df_test = '/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt'
    save_path = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/deconv/"
    df_test = pd.read_csv(df_test, sep=r"\s+", skiprows=1, header=None, engine="python")
    df_test.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    PRECURSOR_MASS = 1608
    PRECURSOR_CHARGE = 3
    
    df_ann, df_rep = line_deconvolute(
        df_test,
        precursor_mass=PRECURSOR_MASS,
        precursor_charge=PRECURSOR_CHARGE,
        deviation_list=[0, 1, 2],
        mass_threshold=0.1,
        iso_range=0,
        mz_tol=0.02,
    )
    print()
    print("=" * 80)
    print("ANNOTATED")
    print("=" * 80)
    if not df_ann.empty:
        cols = [
            "m/z A", "m/z B", "charge_A", "charge_B",
            "line_deviation", "line_residual",
            "monoisotopic_mass_A", "monoisotopic_mass_B",
            "deconvoluted_mz_A", "deconvoluted_mz_B",
            "adj_mass_A", "adj_mass_B",
        ]
        print(df_ann[[c for c in cols if c in df_ann.columns]].to_string(index=False))

    print()
    print("=" * 80)
    print("REPLACED (deconvoluted m/z)")
    print("=" * 80)
    if not df_rep.empty:
        print(df_rep[["m/z A", "m/z B", "Ranking"]].to_string(index=False))
    df_ann.to_csv(save_path + 'VEA3+_charged_annotated.txt', sep = '\t', index = False)
    df_rep = df_rep[["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]]
    df_rep.to_csv(save_path + 'VEA3+_charged_replaced.txt', sep = '\t', index = False)

# =============================================================================
# 9. DEMO / TEST
# =============================================================================


'''
if __name__ == "__main__":

    # Synthetic test: charge-3 precursor, neutral mass ~ 1500 Da
    # precursor_mass = neutral_mass + charge * PROTON_MASS
    PRECURSOR_MASS = 1503.022
    PRECURSOR_CHARGE = 3

    np.random.seed(42)
    ffcs = []

    # Charge pair (1, 2) on parent line: 1*A + 2*B = 1503.022
    for mz_a in [300.0, 400.0, 500.0, 600.0]:
        mz_b = (PRECURSOR_MASS - 1 * mz_a) / 2
        ffcs.append((mz_a + np.random.normal(0, 0.005),
                      mz_b + np.random.normal(0, 0.005)))
        # Isotopic satellite (+1 Da on A side, -0.5 on B side)
        ffcs.append((mz_a + 1.0 + np.random.normal(0, 0.005),
                      mz_b - 0.5 + np.random.normal(0, 0.005)))

    # Charge pair (2, 1) on parent line: 2*A + 1*B = 1503.022
    for mz_a in [350.0, 450.0]:
        mz_b = PRECURSOR_MASS - 2 * mz_a
        ffcs.append((mz_a + np.random.normal(0, 0.005),
                      mz_b + np.random.normal(0, 0.005)))

    # Charge pair (1, 1) on deviation=1 line (charge-reduced):
    # 1*A + 1*B = 1503.022 - 1 = 1502.022
    for mz_a in [500.0, 600.0]:
        mz_b = (PRECURSOR_MASS - 1) - mz_a
        ffcs.append((mz_a + np.random.normal(0, 0.005),
                      mz_b + np.random.normal(0, 0.005)))

    # Noise FFCs
    for _ in range(5):
        ffcs.append((np.random.uniform(200, 800), np.random.uniform(200, 800)))

    df_test = pd.DataFrame(ffcs, columns=["m/z A", "m/z B"])
    df_test["Ranking"] = range(1, len(df_test) + 1)

    print("=" * 80)
    print("INPUT FFCs")
    print("=" * 80)
    print(df_test.to_string(index=False))
    print()

    # Run pipeline
    df_ann, df_rep = line_deconvolute(
        df_test,
        precursor_mass=PRECURSOR_MASS,
        precursor_charge=PRECURSOR_CHARGE,
        deviation_list=[0, 1, 2],
        mass_threshold=0.1,
        iso_range=0,
        mz_tol=0.02,
    )

    print()
    print("=" * 80)
    print("ANNOTATED")
    print("=" * 80)
    if not df_ann.empty:
        cols = [
            "m/z A", "m/z B", "charge_A", "charge_B",
            "line_deviation", "line_residual",
            "monoisotopic_mass_A", "monoisotopic_mass_B",
            "deconvoluted_mz_A", "deconvoluted_mz_B",
            "adj_mass_A", "adj_mass_B",
        ]
        print(df_ann[[c for c in cols if c in df_ann.columns]].to_string(index=False))

    print()
    print("=" * 80)
    print("REPLACED (deconvoluted m/z)")
    print("=" * 80)
    if not df_rep.empty:
        print(df_rep[["m/z A", "m/z B", "Ranking"]].to_string(index=False))
'''