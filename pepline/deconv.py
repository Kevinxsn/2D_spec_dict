"""
Mass spectrum deconvolution script.

Method (per Prof. Xiaowen Liu):
1. Group peaks into isotopic envelopes by searching for consistent
   spacing of 1/z (z = 1..4).
2. Determine charge state z from the spacing.
3. Deconvolute: M_neutral = z * (m/z - 1.00728)  (proton mass).
4. Use theoretical isotope pattern lookup table (theo_patt.txt from
   TopPIC) to identify the monoisotopic mass, handling cases where
   the monoisotopic peak is missing.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Parse the TopPIC theoretical-pattern file
# ---------------------------------------------------------------------------

def parse_theo_patt(filepath: str) -> list[dict]:
    """
    Parse theo_patt.txt into a list of envelope dicts:
      {
        'mono_mass': float,          # monoisotopic mass
        'peaks': [(mass, intensity), ...],  # all isotope peaks
        'max_peak_index': int        # index of most abundant peak
      }
    """
    envelopes = []
    current_peaks = []
    current_mono = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header line like:
            # formula: C223H353N61O66S2 charge: 0 limit: 0.000000 mass: 5005.558258
            if line.startswith("formula:") or line.startswith("#"):
                # Save previous envelope
                if current_peaks:
                    intensities = [p[1] for p in current_peaks]
                    envelopes.append({
                        "mono_mass": current_peaks[0][0],
                        "peaks": current_peaks,
                        "max_peak_index": int(np.argmax(intensities)),
                    })
                current_peaks = []
                # Extract mass from header if present
                m = re.search(r"mass:\s*([\d.]+)", line)
                if m:
                    current_mono = float(m.group(1))
                continue

            # Data line: mass  intensity
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mass = float(parts[0])
                    intensity = float(parts[1])
                    current_peaks.append((mass, intensity))
                except ValueError:
                    continue

    # Don't forget last envelope
    if current_peaks:
        intensities = [p[1] for p in current_peaks]
        envelopes.append({
            "mono_mass": current_peaks[0][0],
            "peaks": current_peaks,
            "max_peak_index": int(np.argmax(intensities)),
        })

    return envelopes


def build_lookup_table(envelopes: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Return sorted array of monoisotopic masses and corresponding envelopes
    for fast binary-search lookup.
    """
    envelopes_sorted = sorted(envelopes, key=lambda e: e["mono_mass"])
    mono_masses = np.array([e["mono_mass"] for e in envelopes_sorted])
    return mono_masses, envelopes_sorted


def find_closest_envelope(mono_masses: np.ndarray,
                          envelopes_sorted: list[dict],
                          target_mass: float) -> dict:
    """Binary search for the envelope whose monoisotopic mass is closest."""
    idx = np.searchsorted(mono_masses, target_mass)
    candidates = []
    for i in [idx - 1, idx]:
        if 0 <= i < len(mono_masses):
            candidates.append(i)
    best = min(candidates, key=lambda i: abs(mono_masses[i] - target_mass))
    return envelopes_sorted[best]


# ---------------------------------------------------------------------------
# 2. Cosine similarity for comparing isotope distributions
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return dot / norm


# ---------------------------------------------------------------------------
# 3. Core deconvolution logic
# ---------------------------------------------------------------------------

PROTON_MASS = 1.00728


def _find_isotopic_envelopes(mz_vals: np.ndarray,
                              intensities: np.ndarray,
                              mz_tol: float = 0.02) -> list[dict]:
    """
    Group peaks into isotopic envelopes.

    Strategy:
      For each unassigned peak, try charge states z = 4, 3, 2, 1.
      Look for a series of peaks spaced by ~1/z within mz_tol.
      Accept the highest charge that gives >= 2 peaks.
    """
    n = len(mz_vals)
    assigned = np.zeros(n, dtype=bool)
    envelopes = []

    # Sort by m/z (should already be, but ensure)
    order = np.argsort(mz_vals)
    mz_sorted = mz_vals[order]
    int_sorted = intensities[order]
    orig_idx = order  # maps back to original indices

    for i in range(n):
        if assigned[i]:
            continue

        best_envelope = None

        for z in [4, 3, 2, 1]:
            spacing = 1.0 / z
            members = [i]
            current_mz = mz_sorted[i]

            # Search forward for next isotope peaks
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                expected_mz = current_mz + spacing
                diff = abs(mz_sorted[j] - expected_mz)
                if diff <= mz_tol:
                    members.append(j)
                    current_mz = mz_sorted[j]
                elif mz_sorted[j] > expected_mz + mz_tol:
                    break  # too far ahead

            if len(members) >= 2:
                best_envelope = {
                    "indices": [orig_idx[m] for m in members],
                    "mz_values": mz_sorted[members].tolist(),
                    "intensities": int_sorted[members].tolist(),
                    "charge": z,
                }
                break  # take highest charge that works

        if best_envelope is not None:
            for m in best_envelope["indices"]:
                # Mark the sorted-index as assigned
                pass
            # We need to mark by sorted position; let's redo
            # Actually members are sorted-indices
            # Re-extract members from the envelope search
            # Mark assigned by original index
            for oi in best_envelope["indices"]:
                assigned[list(orig_idx).index(oi)] = True
            envelopes.append(best_envelope)
        else:
            # Single peak, assign charge 1 by default
            envelopes.append({
                "indices": [orig_idx[i]],
                "mz_values": [mz_sorted[i]],
                "intensities": [int_sorted[i]],
                "charge": 1,
            })
            assigned[i] = True

    return envelopes


def _find_isotopic_envelopes_v2(mz_vals: np.ndarray,
                                 intensities: np.ndarray,
                                 mz_tol: float = 0.02) -> list[dict]:
    """
    Improved envelope detection.

    For each unassigned peak, try charge states z = 4, 3, 2, 1.
    A charge state is accepted only if consecutive peaks are spaced
    at exactly 1/z (within tolerance). No gap-skipping during charge
    determination to avoid false high-charge assignments.

    Among valid charge states, pick the one that explains the most peaks.
    Ties broken by higher charge.
    """
    n = len(mz_vals)
    order = np.argsort(mz_vals)
    mz_s = mz_vals[order]
    int_s = intensities[order]
    assigned = [False] * n
    envelopes = []

    for i in range(n):
        if assigned[i]:
            continue

        best_env = None
        best_z = 0
        best_count = 0

        for z in [4, 3, 2, 1]:
            spacing = 1.0 / z
            members = [i]
            last_mz = mz_s[i]

            j = i + 1
            while j < n:
                if assigned[j]:
                    j += 1
                    continue
                expected = last_mz + spacing
                diff = mz_s[j] - expected
                if diff < -mz_tol:
                    j += 1
                    continue
                if diff > mz_tol:
                    break  # no gap skipping
                # Within tolerance
                members.append(j)
                last_mz = mz_s[j]
                j += 1

            # Prefer the charge giving the most peaks; break ties with higher z
            if len(members) >= 2:
                if (len(members) > best_count or
                        (len(members) == best_count and z > best_z)):
                    best_env = list(members)
                    best_z = z
                    best_count = len(members)

        if best_env is not None:
            for idx in best_env:
                assigned[idx] = True
            envelopes.append({
                "sorted_indices": best_env,
                "original_indices": order[best_env].tolist(),
                "mz_values": mz_s[best_env].tolist(),
                "intensities": int_s[best_env].tolist(),
                "charge": best_z,
            })
        else:
            assigned[i] = True
            envelopes.append({
                "sorted_indices": [i],
                "original_indices": [order[i]],
                "mz_values": [mz_s[i]],
                "intensities": [int_s[i]],
                "charge": 1,
            })

    return envelopes


def determine_monoisotopic_mass(envelope: dict,
                                 mono_masses_lookup: np.ndarray | None,
                                 envelopes_lookup: list[dict] | None,
                                 search_window: int = 3) -> float:
    """
    Given an isotopic envelope (with charge, m/z values, intensities),
    determine the monoisotopic neutral mass.

    Steps:
      1. Find the most abundant peak in the envelope.
      2. Compute its neutral mass: M = z * (mz - PROTON_MASS).
      3. Look up the closest theoretical envelope in the lookup table.
      4. The theoretical envelope tells us how many peaks before the
         max-abundance peak the monoisotopic peak should be (offset).
      5. Test candidate offsets around that value using cosine similarity.
      6. Return the best monoisotopic mass.
    """
    z = envelope["charge"]
    mz_arr = np.array(envelope["mz_values"])
    int_arr = np.array(envelope["intensities"])

    # Index of highest-intensity peak within this envelope
    max_idx = int(np.argmax(int_arr))
    max_mz = mz_arr[max_idx]

    # Neutral mass of the most abundant peak
    mass_of_max = z * (max_mz - PROTON_MASS)

    if mono_masses_lookup is None or envelopes_lookup is None:
        # No lookup table: assume the first observed peak is monoisotopic
        mono_mz = mz_arr[0]
        return z * (mono_mz - PROTON_MASS)

    # Look up closest theoretical envelope
    ref_env = find_closest_envelope(mono_masses_lookup, envelopes_lookup, mass_of_max)
    ref_offset = ref_env["max_peak_index"]  # how many peaks the max is from mono

    # The observed max peak is at index max_idx in our envelope.
    # In the theoretical pattern, the max peak is at index ref_offset from mono.
    # So the monoisotopic peak should be at index (max_idx - ref_offset) in our
    # observed envelope — but it might be missing (negative index).

    # Candidate monoisotopic masses: try offsets around ref_offset
    best_mass = None
    best_sim = -1.0

    ref_peaks = ref_env["peaks"]
    ref_int = np.array([p[1] for p in ref_peaks])
    # Normalize
    ref_int = ref_int / ref_int.max() * 100.0

    for delta in range(-search_window, search_window + 1):
        candidate_offset = ref_offset + delta
        # candidate_offset = number of peaks from monoisotopic to max peak
        # So monoisotopic mass = mass_of_max - candidate_offset * ~1.003355
        candidate_mono_mass = mass_of_max - candidate_offset * 1.003355

        if candidate_mono_mass <= 0:
            continue

        # Build expected intensity distribution for this candidate
        # The observed peaks start at some offset from the candidate mono
        # First observed peak offset from candidate mono:
        first_obs_mass = z * (mz_arr[0] - PROTON_MASS)
        first_peak_offset = round((first_obs_mass - candidate_mono_mass) / 1.003355)

        if first_peak_offset < 0:
            continue

        # Align observed intensities with theoretical
        obs_aligned = []
        theo_aligned = []
        for k, obs_int in enumerate(int_arr):
            theo_idx = first_peak_offset + k
            if 0 <= theo_idx < len(ref_int):
                obs_aligned.append(obs_int)
                theo_aligned.append(ref_int[theo_idx])

        if len(obs_aligned) < 2:
            continue

        obs_vec = np.array(obs_aligned)
        theo_vec = np.array(theo_aligned)

        # Normalize observed
        if obs_vec.max() > 0:
            obs_vec = obs_vec / obs_vec.max() * 100.0

        sim = cosine_similarity(obs_vec, theo_vec)
        if sim > best_sim:
            best_sim = sim
            best_mass = candidate_mono_mass

    if best_mass is None:
        # Fallback: use first peak as monoisotopic
        best_mass = z * (mz_arr[0] - PROTON_MASS)

    return best_mass


# ---------------------------------------------------------------------------
# 4. Main function: deconvolute a DataFrame
# ---------------------------------------------------------------------------

def deconvolute(df: pd.DataFrame,
                mz_col: str = "mz",
                intensity_col: str = "intensity",
                theo_patt_path: str | None = None,
                mz_tol: float = 0.02) -> pd.DataFrame:
    """
    Deconvolute a peak list.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for m/z and intensity.
    mz_col : str
        Name of the m/z column.
    intensity_col : str
        Name of the intensity column.
    theo_patt_path : str or None
        Path to TopPIC's theo_patt.txt. If None, the monoisotopic peak is
        assumed to be the first observed peak in each envelope.
    mz_tol : float
        Tolerance in m/z for grouping isotopic peaks (default 0.02 Da).

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
          - 'charge'             : determined charge state
          - 'envelope_id'        : integer grouping peaks in the same envelope
          - 'monoisotopic_mass'  : deconvoluted monoisotopic neutral mass
    """
    df = df.copy()
    mz_vals = df[mz_col].values.astype(float)
    intensities = df[intensity_col].values.astype(float)

    # Load lookup table if provided
    mono_masses_lookup = None
    envelopes_lookup = None
    if theo_patt_path and Path(theo_patt_path).exists():
        raw_envs = parse_theo_patt(theo_patt_path)
        mono_masses_lookup, envelopes_lookup = build_lookup_table(raw_envs)
        print(f"Loaded {len(envelopes_lookup)} theoretical envelopes from {theo_patt_path}")
    elif theo_patt_path:
        print(f"Warning: {theo_patt_path} not found. "
              "Monoisotopic peak assumed to be first observed peak.")

    # Find isotopic envelopes
    envelopes = _find_isotopic_envelopes_v2(mz_vals, intensities, mz_tol=mz_tol)

    # Initialize new columns
    df["charge"] = 0
    df["envelope_id"] = -1
    df["monoisotopic_mass"] = np.nan

    for env_id, env in enumerate(envelopes):
        mono_mass = determine_monoisotopic_mass(
            env, mono_masses_lookup, envelopes_lookup
        )
        for orig_idx in env["original_indices"]:
            df.loc[orig_idx, "charge"] = env["charge"]
            df.loc[orig_idx, "envelope_id"] = env_id
            df.loc[orig_idx, "monoisotopic_mass"] = round(mono_mass, 4)

    return df

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Demo / test
# ---------------------------------------------------------------------------
'''
if __name__ == "__main__":
    # Example 1: charge-2 envelope (from Prof. Liu's email)
    print("=" * 60)
    print("Example 1: Simple charge-2 envelope")
    print("=" * 60)
    df1 = pd.DataFrame({
        "mz": [500.20, 500.70, 501.20, 501.70],
        "intensity": [100, 80, 40, 15],
    })
    result1 = deconvolute(df1, theo_patt_path=None)
    print(result1.to_string(index=False))
    print()

    # Example 2: mixed charge states
    print("=" * 60)
    print("Example 2: Mixed charge states")
    print("=" * 60)
    df2 = pd.DataFrame({
        "mz": [
            # Charge 3 envelope (~1500 Da neutral mass)
            500.68, 501.01, 501.35, 501.68,
            # Charge 1 envelope
            800.40, 801.40, 802.40,
            # Charge 2 envelope
            600.30, 600.80, 601.30,
        ],
        "intensity": [
            60, 100, 70, 30,
            100, 55, 18,
            100, 65, 25,
        ],
    })
    result2 = deconvolute(df2, theo_patt_path=None)
    print(result2.to_string(index=False))
    print()

    # Example 3: with theo_patt.txt (if available)
    theo_path = "theo_patt.txt"
    if Path(theo_path).exists():
        print("=" * 60)
        print("Example 3: Using theoretical pattern lookup")
        print("=" * 60)
        result3 = deconvolute(df2, theo_patt_path=theo_path)
        print(result3.to_string(index=False))
    else:
        print(f"To use monoisotopic correction, download theo_patt.txt from:")
        print(f"  https://github.com/toppic-suite/toppic-suite/blob/main/resources/base_data/theo_patt.txt")
        print(f"  and place it in the working directory.")
'''