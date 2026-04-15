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
   The caller supplies an explicit list of expected Parent+X offsets
   (parental, isotope satellites, neutral losses, ...) and only lines
   matching one of those offsets are kept.  This mirrors the way
   ``annotation.select_best_partition`` uses an explicit
   ``target_masses`` list.
2. For each FFC point that lies on at least one line, record (charge_A,
   charge_B).  Special case: if the point lies on the parental line
   (matched_offset == parental_offset), it must lie on that line ONLY;
   otherwise multiple line assignments per point are allowed and each
   produces an output row.
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
    ISOTOPE_SPACING,
    _load_lookup,
    find_closest_envelope,
    cosine_similarity,
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
    line_offset: float      # Parent+X for this line (Da, as detected)
    matched_offset: float   # which entry of expected_offsets this snapped to
    line_n_points: int      # cluster size of this line
    is_parental: bool       # True iff this line is the parental line


def find_charge_lines(
    ffc_df: pd.DataFrame,
    parent_charge: int,
    parent_mass: float,
    expected_offsets: List[float],
    offset_tol: float,
    delta: float = 0.02,
    min_cluster_size: int = 1,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
) -> pd.DataFrame:
    """
    Detect FFC lines whose Parent+X offset matches one of the
    user-supplied ``expected_offsets`` (within ``offset_tol``).

    The deviation list lets the caller specify exactly which physical
    lines are expected (e.g. parental at 0, isotope satellites at
    +1.003, +2.006, neutral-loss lines at -229 for a side-chain loss,
    etc.) and discards every other cluster the line-finder might pick
    up.  This mirrors the way ``annotation.select_best_partition`` uses
    an explicit ``target_masses`` list.

    Charge-conservation rule
    ------------------------
    The constraint on (i, j) charge splits depends on the *sign* of the
    matched offset:

    * **Non-negative offsets** (parental at 0, isotopic satellites at
      +1.003, +2.006, ...): the precursor's full charge is observed in
      the two charged fragments, so we require ``i + j == parent_charge``.

    * **Negative offsets** (e.g. -18 Da water loss, -229 Da side-chain
      loss): a fragment can carry away mass as a neutral, in which case
      the two charged fragments do not need to add up to the full
      precursor charge.  We allow any ``i + j <= parent_charge``.

    Parameters
    ----------
    expected_offsets : list of float
        Allowed Parent+X values (Da).  Detected lines outside this set
        are dropped.
    offset_tol : float
        Maximum |detected_offset - expected_offset| to accept (Da).

    Returns
    -------
    DataFrame with columns:
        line_id, i, j, n_points, center, min_v, max_v, Parent+X,
        matched_offset, point_indices
    sorted by n_points descending.  ``matched_offset`` records which
    entry from ``expected_offsets`` the line was attributed to.
    """
    if not expected_offsets:
        raise ValueError("expected_offsets must contain at least one value")

    # max_offset for the underlying compute_parent_offsets call has to be
    # wide enough to retain anything we might want -- pad the bounds of
    # the user's list by offset_tol on each side.
    max_offset = max(abs(o) for o in expected_offsets) + offset_tol + 1e-9

    # ── Symmetric (i, j) enumeration ──────────────────────────────────────
    # The underlying line_finding.detect_line_clusters only enumerates
    # (i, j) pairs with i <= j (to avoid duplicating symmetric splits).
    # For a 2D-PC-MS table where every fragment-pair (A, B) is mirrored
    # as (B, A) this is sufficient, but when a specific FFC appears in
    # only one ordering, splits with i > j are never tested on it and
    # the corresponding line can be missed.
    #
    # We fix this without modifying line_finding by running the line
    # finder a second time on a DataFrame with A and B swapped, then
    # remapping the detected (i, j) to (j, i) so every point index
    # still refers to the original DataFrame.  Concatenating both
    # result sets gives full coverage of all (i, j) splits with
    # i + j <= parent_charge.
    raw_forward = detect_line_clusters(
        ffc_df,
        parent_charge=parent_charge,
        delta=delta,
        col_a=col_a,
        col_b=col_b,
        enforce_sum_leq_charge=True,
        min_cluster_size=min_cluster_size,
        return_point_indices=True,
    )

    # Swap the roles of A and B without touching the row order, so
    # point_indices returned by the swapped call still index the
    # original ffc_df rows correctly.
    swapped_df = ffc_df.rename(columns={col_a: "__tmp_b__", col_b: col_a})
    swapped_df = swapped_df.rename(columns={"__tmp_b__": col_b})
    raw_reverse = detect_line_clusters(
        swapped_df,
        parent_charge=parent_charge,
        delta=delta,
        col_a=col_a,
        col_b=col_b,
        enforce_sum_leq_charge=True,
        min_cluster_size=min_cluster_size,
        return_point_indices=True,
    )
    if not raw_reverse.empty:
        # In the swapped frame a cluster at split (i, j) corresponds to
        # split (j, i) in the original frame.  Rename accordingly.
        raw_reverse = raw_reverse.rename(columns={"i": "j", "j": "i"})
        # Drop the (i == j) pairs from the reverse set: they are
        # already covered by raw_forward and would produce duplicate
        # clusters.
        raw_reverse = raw_reverse[raw_reverse["i"] != raw_reverse["j"]]

    if raw_forward.empty and (raw_reverse is None or raw_reverse.empty):
        return pd.DataFrame()

    raw = pd.concat([raw_forward, raw_reverse], ignore_index=True)
    if raw.empty:
        return raw

    lines = compute_parent_offsets(raw, parent_mass, max_offset=max_offset)
    if lines.empty:
        return lines

    # Charge filter rule (per offset sign):
    #   * Negative offsets (e.g. -229 Da neutral loss): a fragment can
    #     carry away mass as a neutral loss before re-ionization, so the
    #     two surviving charged fragments do not need to add up to the
    #     full precursor charge.  We allow any  i + j <= parent_charge.
    #   * Non-negative offsets (parental at 0, isotope satellites at
    #     +1.003, +2.006, ...): the parent's full charge is conserved
    #     in the two observed fragments, so we require  i + j == parent_charge.
    #
    # The offset for each detected line must already lie within
    # offset_tol of one of the user-supplied expected_offsets, so we
    # apply the charge rule based on the SIGN OF THE MATCHED expected
    # offset (not the raw Parent+X value, which might straddle 0 within
    # the tolerance).
    expected_arr = np.asarray(expected_offsets, dtype=float)
    parent_x = lines["Parent+X"].to_numpy(dtype=float)

    # For each line, find its closest expected offset and snap.
    dist = np.abs(parent_x[:, None] - expected_arr[None, :])
    nearest_idx = dist.argmin(axis=1)
    nearest_dist = dist[np.arange(len(parent_x)), nearest_idx]
    matched = expected_arr[nearest_idx]

    keep_offset = nearest_dist <= offset_tol

    # Apply the per-line charge rule based on the sign of the matched
    # offset.  Negative -> i+j <= parent_charge.  Non-negative -> i+j == parent_charge.
    line_sum = (lines["i"] + lines["j"]).to_numpy(dtype=int)
    is_negative_offset = matched < 0
    keep_charge = np.where(
        is_negative_offset,
        line_sum <= parent_charge,
        line_sum == parent_charge,
    )

    keep = keep_offset & keep_charge
    lines = lines.loc[keep].copy()
    lines["matched_offset"] = matched[keep]

    if lines.empty:
        return lines

    lines = (
        lines.sort_values("n_points", ascending=False)
        .reset_index(drop=True)
    )
    lines["line_id"] = np.arange(len(lines), dtype=int)
    return lines


def assign_points_to_lines(
    lines: pd.DataFrame,
    parental_offset: float = 0.0,
    parental_tol: float = 1e-6,
) -> List[LineAssignment]:
    """
    Convert a lines DataFrame into a flat list of point->line assignments.

    Rule:
        * Parental line: ``matched_offset`` equals ``parental_offset``
          (within ``parental_tol``).  An FFC point attached to the
          parental line is "claimed" — it cannot also be attributed to
          any other line.
        * Non-parental line: an FFC point may be attributed to multiple
          non-parental lines simultaneously (different (i, j) splits are
          all kept).
    """
    if lines.empty:
        return []

    parental_mask = (lines["matched_offset"] - parental_offset).abs() <= parental_tol
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
                    matched_offset=float(row["matched_offset"]),
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
                    matched_offset=float(row["matched_offset"]),
                    line_n_points=int(row["n_points"]),
                    is_parental=False,
                )
            )

    return assignments


# =============================================================================
# 2. CHARGE-1 EQUIVALENT  + ISOTOPE-ENVELOPE DECONVOLUTION
# =============================================================================

def _refine_monoisotopic_by_cosine(
    envelope: dict,
    mono_lookup,
    envelopes_lookup,
    max_shift: Optional[int] = None,
) -> Tuple[float, float, int, float]:
    """
    Pick the monoisotopic shift by **maximum cosine similarity**.

    Local override of ``deconv_df_intensity.determine_monoisotopic_mz``.
    The upstream version uses a "look up nearest theoretical envelope by
    mass-of-max, read its max_peak_index, apply that as the shift" rule.
    That heuristic only works when the lookup table densely covers the
    relevant mass range; for sparse lookups (e.g. a ~3 Da gap to the
    nearest theo entry) it returns the wrong shift while the cosine
    similarity it computes alongside is just used as a confidence
    score, not as the selector.

    This refiner instead enumerates every plausible shift k, looks up
    the theoretical envelope at the corresponding hypothetical mono
    mass, aligns observed peaks to theoretical positions, and picks
    the k with the highest cosine similarity.  This makes the choice
    *evidence-driven* rather than *table-driven*.

    Shift sweep range
    -----------------
    The number of shifts to try is bounded above by ``max_shift``.
    When ``max_shift`` is None (default), the bound is auto-scaled
    with the anchor's mass: ``floor(anchor_mass / 1500) + 2``.  The
    rationale is that for an averagine-like polymer the most intense
    isotope sits roughly at index ``mass / 1800``, so a fragment at
    ~3500 Da has its tallest peak around index 2-3.  Adding +2 of
    headroom guarantees we cover the true shift even when the
    observed envelope is heavily truncated to its top isotopes.  For
    a 1000 Da fragment this gives 2 shifts; for 5000 Da, 5 shifts.

    Note: the sweep is *not* limited by the number of observed peaks.
    A two-peak envelope of a 4 kDa molecule may legitimately need a
    shift of 3 or 4 (the two observed peaks being indices 3-4 of the
    true envelope).  Limiting the sweep to ``n_obs - 1`` would miss
    such cases.

    Returns ``(mono_mz, mono_neutral_mass, charge, cosine_similarity)``.
    """
    z = envelope["charge"]
    mz_arr = np.asarray(envelope["mz_values"], dtype=float)
    int_arr = np.asarray(envelope["intensities"], dtype=float)
    spacing_mz = ISOTOPE_SPACING / z
    n_obs = len(mz_arr)

    # Sort by m/z so peak indices are stable.
    order = np.argsort(mz_arr)
    mz_sorted = mz_arr[order]
    int_sorted = int_arr[order]

    # Anchor = most intense observed peak (used only for the shift
    # search, not as the final mono).
    max_pos = int(np.argmax(int_sorted))
    max_mz = float(mz_sorted[max_pos])

    # Fallback: no lookup table, or single-peak envelope.  Take the
    # lowest observed m/z as the monoisotopic peak (this is the same
    # behaviour as the upstream fallback).
    if mono_lookup is None or envelopes_lookup is None or n_obs < 2:
        mono_mz = float(mz_sorted[0])
        mono_mass = z * (mono_mz - PROTON_MASS)
        return mono_mz, mono_mass, z, float("nan")

    # Determine sweep bound.
    if max_shift is None:
        anchor_mass = z * (max_mz - PROTON_MASS)
        max_shift = int(anchor_mass // 1500) + 2
    # Make sure we always try at least k = 0, 1.
    max_shift = max(max_shift, 1)

    obs_norm = int_sorted / int_sorted.max() * 100.0
    obs_norm_v = np.asarray(obs_norm, dtype=float)

    best_score = -1.0
    best_cos = float("nan")
    best_mono_mz = float(mz_sorted[0])

    # Sweep all hypotheses: "the tallest observed peak is the k-th
    # isotope of the true envelope".  k = 0 means the tallest observed
    # peak IS the monoisotopic peak.
    #
    # Selection metric
    # ----------------
    # We rank hypotheses by  cosine_similarity(obs, theo) * (max_theo / 100).
    # The cosine factor measures shape agreement; the max_theo factor is
    # a magnitude correction that prevents tail-of-envelope hypotheses
    # from winning by accident.
    #
    # Why bare cosine fails: cosine is scale-invariant, so a hypothesis
    # that places the observed peaks in the deep tail of a theoretical
    # envelope (all theo intensities ~1-2%) can score nearly 1.0 just
    # because the tail decays in roughly the same ratio as the observed
    # peaks.  Multiplying by max(theo)/100 forces the winning hypothesis
    # to put at least one observed peak near the *bulk* of the
    # theoretical envelope, which is where real fragments live.  When
    # the winning hypothesis includes the theoretical envelope's
    # tallest peak (intensity 100), the multiplier is 1.0 and the
    # reported similarity is identical to the bare cosine.
    for k in range(0, max_shift + 1):
        hyp_mono_mz = max_mz - k * spacing_mz
        hyp_mono_mass = z * (hyp_mono_mz - PROTON_MASS)

        # Look up the theoretical envelope nearest the hypothesised
        # mono mass.
        ref_env = find_closest_envelope(
            mono_lookup, envelopes_lookup, hyp_mono_mass
        )
        ref_int = np.asarray(
            [p[1] for p in ref_env["peaks"]], dtype=float
        )
        if ref_int.size == 0 or ref_int.max() <= 0:
            continue
        ref_int = ref_int / ref_int.max() * 100.0

        # Align each observed peak to its theoretical isotope index
        # under this hypothesis.  Peaks more than 0.3 isotope steps
        # away from any theo position are dropped from the score.
        theo_aligned = np.zeros(n_obs, dtype=float)
        ok = True
        for idx_obs, m in enumerate(mz_sorted):
            delta_steps = (m - hyp_mono_mz) / spacing_mz
            theo_idx = int(round(delta_steps))
            if abs(delta_steps - theo_idx) > 0.3:
                # This shift can't even align the observed peaks to
                # integer isotope positions -- skip.
                ok = False
                break
            if 0 <= theo_idx < len(ref_int):
                theo_aligned[idx_obs] = ref_int[theo_idx]
            else:
                theo_aligned[idx_obs] = 0.0
        if not ok:
            continue

        if theo_aligned.max() <= 0:
            continue

        cos = cosine_similarity(obs_norm_v, theo_aligned)
        magnitude_factor = theo_aligned.max() / 100.0
        score = cos * magnitude_factor

        if score > best_score:
            best_score = score
            best_cos = cos
            best_mono_mz = hyp_mono_mz

    if best_score < 0:
        # No shift hypothesis was scoreable -- fall back to the lowest
        # observed peak.
        best_mono_mz = float(mz_sorted[0])
        best_cos = float("nan")

    mono_mass = z * (best_mono_mz - PROTON_MASS)
    return float(best_mono_mz), float(mono_mass), z, float(best_cos)


def _to_charge1_mz(mz: float, z: int) -> float:
    """Singly-protonated m/z that a fragment of charge z and m/z mz would have."""
    return z * mz - (z - 1) * PROTON_MASS


def _deconvolute_charge1_axis(
    mz1_vals: np.ndarray,
    intensities: np.ndarray,
    mono_lookup,
    env_lookup,
    mz_tol: float = 0.02,
    max_shift: Optional[int] = None,
    charges: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deconvolute a pool of charge-1-equivalent m/z values.

    Strategy
    --------
    Forget which side of the FFC each value came from and pool every
    unique (charge-1-equivalent m/z, precursor charge) pair together,
    run the standard 1D isotope-envelope deconvoluter on each charge
    class separately, then map the monoisotopic result back to every
    input row that referenced each unique value.  This mirrors the way
    one-dimensional deconvolution works on a flat peak list, with two
    additions:

    1.  **Deduplication.**  The same physical peak appears many times
        in the long FFC table (once per FFC it participates in).
    2.  **Charge-class partitioning.**  Values that collapse to the
        same charge-1-equivalent m/z but came from fragments with
        different precursor charges are kept in separate envelopes.
        A peak at m/z 229.66 measured as 2+ and a peak at m/z 458.31
        measured as 1+ both land at ~458.31 on the charge-1 axis, but
        they are different observations (possibly of the same species,
        but in different charge states) and should not be pooled into
        one envelope for isotope-spacing purposes.  Isotope spacing on
        the charge-1 axis is 1.003 Da regardless of the underlying
        precursor charge; what differs is which other peaks are
        physically part of the same envelope, and that is determined
        by the precursor charge, not by the transformed m/z.

    Why pool A and B together (within a charge class)?  Because a
    fragment that appears as ``m/z A`` in one FFC and as ``m/z B`` in
    another (at the same charge) is the same physical peak after the
    charge-1 transform.  Treating the two sides separately splits its
    isotope envelope and produces inconsistent monoisotopic assignments.

    Parameters
    ----------
    charges : np.ndarray, optional
        Per-row precursor charge.  When provided, envelope grouping is
        done independently per charge class.  When None, all rows are
        treated as a single pool (legacy behaviour).

    Returns four arrays of length len(mz1_vals):
        mono_mz1    -- monoisotopic m/z on the charge-1 axis
        mono_mass   -- neutral monoisotopic mass
        sim         -- isotope-envelope cosine similarity (NaN if not scored)
        envelope_id -- 0-indexed envelope label, -1 if the peak was not
                       placed in any envelope
    """
    n = len(mz1_vals)
    mono_mz1 = np.full(n, np.nan)
    mono_mass = np.full(n, np.nan)
    sim_arr = np.full(n, np.nan)
    env_id_arr = np.full(n, -1, dtype=int)

    if n == 0:
        return mono_mz1, mono_mass, sim_arr, env_id_arr

    # Legacy path: no charges supplied -- treat the whole pool as one
    # class.  Preserves previous behaviour for callers that don't pass
    # the charges argument.
    if charges is None:
        charge_classes = [np.arange(n, dtype=int)]
    else:
        charges = np.asarray(charges, dtype=int)
        # Build one index list per distinct charge.
        unique_charges = np.unique(charges)
        charge_classes = [np.where(charges == c)[0] for c in unique_charges]

    # Envelope IDs are globally unique across charge classes.
    next_env_id = 0

    for class_rows in charge_classes:
        if class_rows.size == 0:
            continue

        class_mz = mz1_vals[class_rows]
        class_int = intensities[class_rows]

        # ── Step 1: collapse near-duplicates within this charge class ────
        # The same physical peak can arrive with slightly different
        # charge-1-equivalent values when reconstructed via different
        # (i, j) splits (typical disagreement <0.01 Da).  Simple
        # rounding misses borderline cases; instead we do single-
        # linkage gap-based merging: sort, scan for gaps < merge_tol,
        # and group consecutive values that are within the tolerance
        # into one representative (weighted-mean m/z, max intensity).
        merge_tol = mz_tol  # reuse the isotope-spacing tolerance

        order = np.argsort(class_mz)
        mz_sorted = class_mz[order]
        int_sorted = class_int[order]

        # Single-linkage: consecutive values within merge_tol are in
        # the same group.
        groups: List[List[int]] = []
        start = 0
        for k in range(1, len(mz_sorted)):
            if mz_sorted[k] - mz_sorted[k - 1] > merge_tol:
                groups.append(list(range(start, k)))
                start = k
        groups.append(list(range(start, len(mz_sorted))))

        n_unique = len(groups)
        unique_keys = np.empty(n_unique, dtype=float)
        unique_int = np.empty(n_unique, dtype=float)
        # inverse: for each element in class_mz, which unique group
        # does it belong to?
        inverse = np.empty(len(class_mz), dtype=int)

        for g_idx, members in enumerate(groups):
            member_mz = mz_sorted[members]
            member_int = int_sorted[members]
            # Representative m/z = intensity-weighted mean of the group
            unique_keys[g_idx] = np.average(member_mz, weights=member_int)
            # Representative intensity = max in the group (most
            # informative for envelope shape scoring downstream)
            unique_int[g_idx] = member_int.max()
            # Map original (sorted) positions back
            for m in members:
                inverse[order[m]] = g_idx

        # ── Step 2: group into isotope envelopes ────────────────────────────
        # On the charge-1 axis the only physical spacing is ~1.003 Da.
        # Real envelopes can have missing isotopes (the middle peak
        # wasn't captured in the FFC data), so we accept gaps that are
        # integer multiples of the isotope spacing, up to
        # ``max_missing + 1`` steps.  This is more tolerant than the
        # standard ``_find_isotopic_envelopes`` (which requires strict
        # consecutive 1/z spacing and would split an envelope at a
        # 2-step gap).
        ISO = 1.003355
        max_missing = 3  # allow up to 2 missing isotopes

        env_order = np.argsort(unique_keys)
        mz_env = unique_keys[env_order]
        int_env = unique_int[env_order]

        envelopes: List[dict] = []
        start = 0
        for k in range(1, len(mz_env)):
            gap = mz_env[k] - mz_env[k - 1]
            attached = False
            for steps in range(1, max_missing + 2):
                if abs(gap - steps * ISO) <= mz_tol:
                    attached = True
                    break
            if not attached:
                envelopes.append({
                    "sorted_indices": list(range(start, k)),
                    "original_indices": env_order[start:k].tolist(),
                    "mz_values": mz_env[start:k].tolist(),
                    "intensities": int_env[start:k].tolist(),
                    "charge": 1,
                })
                start = k
        envelopes.append({
            "sorted_indices": list(range(start, len(mz_env))),
            "original_indices": env_order[start:].tolist(),
            "mz_values": mz_env[start:].tolist(),
            "intensities": int_env[start:].tolist(),
            "charge": 1,
        })

        # ── Step 3: refine and broadcast back to input rows ──────────────
        for env in envelopes:
            mono_mz_v, mono_mass_v, _z, sim = _refine_monoisotopic_by_cosine(
                env, mono_lookup, env_lookup, max_shift=max_shift,
            )
            for unique_idx in env["original_indices"]:
                # Rows (within this charge class) that map to this peak
                local_mask = inverse == unique_idx
                global_rows = class_rows[local_mask]
                mono_mz1[global_rows] = mono_mz_v
                mono_mass[global_rows] = mono_mass_v
                sim_arr[global_rows] = sim
                env_id_arr[global_rows] = next_env_id
            next_env_id += 1

    return mono_mz1, mono_mass, sim_arr, env_id_arr


def _deconvolute_pooled(
    mz1_a: np.ndarray,
    mz1_b: np.ndarray,
    int_a: np.ndarray,
    int_b: np.ndarray,
    mono_lookup,
    env_lookup,
    mz_tol: float = 0.02,
    max_shift: Optional[int] = None,
    charge_a: Optional[np.ndarray] = None,
    charge_b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Pool the charge-1-equivalent values from both sides of the FFC table
    and deconvolute them as a single 1D peak list, partitioned by
    precursor charge.

    The pooled approach is essential because a fragment can appear as
    ``m/z A`` in one FFC and as ``m/z B`` in another -- on the charge-1
    axis these are the same physical peak, and they need to share an
    envelope so they get one consistent monoisotopic assignment.

    The charge partitioning prevents mixing values that happen to
    collide on the charge-1 axis but come from different precursor
    charges (e.g., a 2+ peak at m/z 229.66 and a 1+ peak at m/z 458.31
    both land near 458.31 after the transform, but they are separate
    observations).

    Returns eight arrays (one set of four per side):
        mono_mz1_a, mono_mass_a, sim_a, env_id_a,
        mono_mz1_b, mono_mass_b, sim_b, env_id_b
    """
    n_a = len(mz1_a)

    pooled_mz = np.concatenate([mz1_a, mz1_b])
    pooled_int = np.concatenate([int_a, int_b])

    if charge_a is not None and charge_b is not None:
        pooled_charge = np.concatenate([
            np.asarray(charge_a, dtype=int),
            np.asarray(charge_b, dtype=int),
        ])
    else:
        pooled_charge = None

    pooled_mono_mz, pooled_mono_mass, pooled_sim, pooled_env = (
        _deconvolute_charge1_axis(
            pooled_mz, pooled_int, mono_lookup, env_lookup,
            mz_tol=mz_tol, max_shift=max_shift,
            charges=pooled_charge,
        )
    )

    return (
        pooled_mono_mz[:n_a], pooled_mono_mass[:n_a],
        pooled_sim[:n_a], pooled_env[:n_a],
        pooled_mono_mz[n_a:], pooled_mono_mass[n_a:],
        pooled_sim[n_a:], pooled_env[n_a:],
    )


# =============================================================================
# 3. PUBLIC PIPELINE
# =============================================================================

def deconvolute_ffc_by_lines(
    df: pd.DataFrame,
    parent_charge: int,
    parent_mass: float,
    expected_offsets: List[float],
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
    intensity_col_a: Optional[str] = None,
    intensity_col_b: Optional[str] = None,
    theo_patt_path: Optional[str] = None,
    line_delta: float = 0.02,
    min_cluster_size: int = 3,
    offset_tol: float = 0.05,
    parental_offset: float = 0.0,
    parental_tol: float = 1e-6,
    mz_tol: float = 0.02,
    max_shift: Optional[int] = None,
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
    expected_offsets : list of float
        Allowed Parent+X offsets to look for, in Da.  Detected lines
        whose Parent+X is not within ``offset_tol`` of one of these
        values are discarded.  Mirrors the ``target_masses`` list in
        ``annotation.select_best_partition``.
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
    offset_tol : float
        Tolerance for matching a detected line's Parent+X against the
        ``expected_offsets`` list.
    parental_offset : float
        Which entry of ``expected_offsets`` is the parental line.
        Defaults to 0.0 (the precursor itself).
    parental_tol : float
        Tolerance for identifying a line as parental within the kept
        set.  Defaults to a tight value because ``matched_offset`` has
        already been snapped to one of the discrete expected values.
    mz_tol : float
        Isotope-spacing tolerance on the charge-1-equivalent axis.
    max_shift : int, optional
        Maximum number of isotope steps to consider when picking the
        monoisotopic shift via cosine similarity.  If None (default),
        the bound is auto-scaled with anchor mass as
        ``floor(anchor_mass / 1500) + 2``, which gives 2 for ~1 kDa
        fragments and ~5 for ~5 kDa fragments.  Override if your
        molecules have unusual envelope shapes.

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
        expected_offsets=expected_offsets,
        offset_tol=offset_tol,
        delta=line_delta,
        min_cluster_size=min_cluster_size,
        col_a=mz_col_a,
        col_b=mz_col_b,
    )

    # ── Step 2: assign FFC points to lines ────────────────────────────────
    assignments = assign_points_to_lines(
        lines,
        parental_offset=parental_offset,
        parental_tol=parental_tol,
    )

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
                "matched_offset": a.matched_offset,
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

    (mono_mz1_a, mono_mass_a, sim_a, env_id_a,
     mono_mz1_b, mono_mass_b, sim_b, env_id_b) = _deconvolute_pooled(
        annotated["mz1_A"].values.astype(float),
        annotated["mz1_B"].values.astype(float),
        int_a, int_b,
        mono_lookup, env_lookup, mz_tol,
        max_shift=max_shift,
        charge_a=annotated["i"].values.astype(int),
        charge_b=annotated["j"].values.astype(int),
    )

    annotated["charge_A"] = annotated["i"]
    annotated["charge_B"] = annotated["j"]
    annotated["envelope_id_A"] = env_id_a
    annotated["envelope_id_B"] = env_id_b
    annotated["deconvoluted_mz_A"] = np.round(mono_mz1_a, 4)
    annotated["deconvoluted_mz_B"] = np.round(mono_mz1_b, 4)
    annotated["monoisotopic_mass_A"] = np.round(mono_mass_a, 4)
    annotated["monoisotopic_mass_B"] = np.round(mono_mass_b, 4)
    annotated["isotope_similarity_A"] = np.round(sim_a, 4)
    annotated["isotope_similarity_B"] = np.round(sim_b, 4)

    # ── Step 5: build the "replaced" view ─────────────────────────────────
    # Replace m/z A and m/z B with the monoisotopic m/z at the ORIGINAL
    # charge state (mass/z + proton).  Same shape & columns as the input
    # df, plus line_id appended for traceability.
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
        "KWK6+NCE20_with_intensity_top_8000"
    )
    SAVE_DIR = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/deconv/"

    PARENT_CHARGE = 6
    #PARENT_MASS = 3767.8441  # neutral precursor mass for HAD4+
    #PARENT_MASS = 1608.86917  # neutral precursor mass for HAD4+
    PARENT_MASS = 667.90419 * 6

    # Lines we expect to find on the FFC map: parental + a few isotope
    # satellites on each side.  Mirrors annotation.py's iso_range idea
    # but as an explicit list so unusual offsets (e.g. -18 for water
    # loss) can be added without code changes.
    EXPECTED_OFFSETS = [
        0.0,
        +1 * 1.00335,
        +2 * 1.00335,
        +3 * 1.00335,
        +4 * 1.00335,
        +5 * 1.00335,
        +6 * 1.00335,
    ]

    df = pd.read_csv(DATA_PATH, sep="\t", skiprows=1, header=None, engine="python")
    df.columns = [
        "m/z A", "m/z B", "Covariance", "Partial Cov.",
        "Score", "Ranking", "intensity A", "intensity B",
    ]
    df = df[df["Ranking"] != -1].sort_values("Ranking").head(8000)
    print(df[df["Ranking"] == 194])
    df = df.dropna(subset=["intensity A", "intensity B"])
    print(df[df["Ranking"] == 194])

    result = deconvolute_ffc_by_lines(
        df,
        parent_charge=PARENT_CHARGE,
        parent_mass=PARENT_MASS,
        expected_offsets=EXPECTED_OFFSETS,
        intensity_col_a="intensity A",
        intensity_col_b="intensity B",
        theo_patt_path="theo_patt.txt",
        line_delta=0.02,
        min_cluster_size=2,
        offset_tol=0.05,
        parental_offset=0.0,
        mz_tol=0.02,
    )

    result["annotated"].to_csv(SAVE_DIR + "KWK6+NCE20_ffc_loss_annotated.txt",
                               sep="\t", index=False)
    result["replaced"].to_csv(SAVE_DIR + "KWK6+NCE20_ffc_loss_replaced.txt",
                              sep="\t", index=False)
    result["line_map"].to_csv(SAVE_DIR + "KWK6+NCE20_ffc_loss_line_map.txt",
                              sep="\t", index=False)

    print(f"Lines used:        {result['line_map']['line_id'].nunique()}")
    print(f"FFC points kept:   {result['line_map']['ffc_index'].nunique()}")
    print(f"Total assignments: {len(result['annotated'])}")