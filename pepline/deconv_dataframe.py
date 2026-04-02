"""
Mass spectrum deconvolution script for paired m/z columns.

Method (per Prof. Xiaowen Liu):
1. Group peaks into isotopic envelopes by searching for consistent
   spacing of 1/z (z = 1..4).
2. Determine charge state z from the spacing.
3. Deconvolute: M_neutral = z * (m/z - 1.00728).
4. Use theoretical isotope pattern lookup table (theo_patt.txt from
   TopPIC) to identify the monoisotopic mass, handling cases where
   the monoisotopic peak is missing.
5. Report deconvoluted m/z = M_neutral + 1.00728 (singly protonated).
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Parse the TopPIC theoretical-pattern file
# ---------------------------------------------------------------------------

def parse_theo_patt(filepath: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# 2. Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm else 0.0


# ---------------------------------------------------------------------------
# 3. Core deconvolution logic
# ---------------------------------------------------------------------------

PROTON_MASS = 1.00728


def _find_isotopic_envelopes(mz_vals, intensities, mz_tol=0.02):
    """
    Group peaks into isotopic envelopes by spacing.

    For each unassigned peak, try z = 4, 3, 2, 1.
    Accept charge that explains the most peaks (ties → higher z).
    No gap-skipping to avoid false high-charge assignments.
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
                diff = mz_s[j] - (last_mz + spacing)
                if diff < -mz_tol:
                    j += 1
                    continue
                if diff > mz_tol:
                    break
                members.append(j)
                last_mz = mz_s[j]
                j += 1

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


def determine_monoisotopic_mass(envelope, mono_masses_lookup=None,
                                 envelopes_lookup=None, search_window=3):
    """
    Determine the monoisotopic neutral mass for an envelope.
    Without lookup table: assumes first observed peak is monoisotopic.
    With lookup table: uses cosine similarity to find best offset.
    """
    z = envelope["charge"]
    mz_arr = np.array(envelope["mz_values"])
    int_arr = np.array(envelope["intensities"])

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
        candidate_mono_mass = mass_of_max - candidate_offset * 1.003355
        if candidate_mono_mass <= 0:
            continue

        first_obs_mass = z * (mz_arr[0] - PROTON_MASS)
        first_peak_offset = round(
            (first_obs_mass - candidate_mono_mass) / 1.003355
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


# ---------------------------------------------------------------------------
# 4. Internal helpers
# ---------------------------------------------------------------------------

def _load_lookup(theo_patt_path):
    """Load theoretical pattern lookup table if path is valid."""
    if theo_patt_path and Path(theo_patt_path).exists():
        raw_envs = parse_theo_patt(theo_patt_path)
        mono_lookup, env_lookup = build_lookup_table(raw_envs)
        print(f"Loaded {len(env_lookup)} theoretical envelopes "
              f"from {theo_patt_path}")
        return mono_lookup, env_lookup
    elif theo_patt_path:
        print(f"Warning: {theo_patt_path} not found. "
              "Monoisotopic peak assumed to be first observed peak.")
    return None, None


def _deconvolute_single_column(mz_vals, intensities,
                                mono_lookup, env_lookup, mz_tol=0.02):
    """
    Deconvolute a single array of m/z values.

    Returns
    -------
    charges      : int array
    envelope_ids : int array
    mono_masses  : float array  (neutral monoisotopic mass)
    deconv_mz    : float array  (monoisotopic mass + proton = [M+H]+)
    """
    n = len(mz_vals)
    charges = np.zeros(n, dtype=int)
    envelope_ids = np.full(n, -1, dtype=int)
    mono_masses = np.full(n, np.nan)
    deconv_mz = np.full(n, np.nan)

    envelopes = _find_isotopic_envelopes(mz_vals, intensities, mz_tol)

    for env_id, env in enumerate(envelopes):
        mass = determine_monoisotopic_mass(env, mono_lookup, env_lookup)
        mz_out = mass + PROTON_MASS  # [M+H]+ singly protonated
        for orig_idx in env["original_indices"]:
            charges[orig_idx] = env["charge"]
            envelope_ids[orig_idx] = env_id
            mono_masses[orig_idx] = round(mass, 4)
            deconv_mz[orig_idx] = round(mz_out, 4)

    return charges, envelope_ids, mono_masses, deconv_mz


# ---------------------------------------------------------------------------
# 5. Main public function
# ---------------------------------------------------------------------------

def deconvolute_pairs(df: pd.DataFrame,
                      mz_col_a: str = "m/z A",
                      mz_col_b: str = "m/z B",
                      intensity_col_a: str | None = None,
                      intensity_col_b: str | None = None,
                      theo_patt_path: str | None = None,
                      mz_tol: float = 0.02):
    """
    Deconvolute two m/z columns (A and B) in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input with at least `mz_col_a` and `mz_col_b`.
    mz_col_a, mz_col_b : str
        Column names for the two m/z columns.
    intensity_col_a, intensity_col_b : str or None
        Intensity column names.  If None → uniform intensities.
    theo_patt_path : str or None
        Path to TopPIC's theo_patt.txt for monoisotopic correction.
    mz_tol : float
        Tolerance for grouping isotopic peaks (default 0.02 Da).

    Returns
    -------
    df_annotated : pd.DataFrame
        Original columns + for each of A and B:
          charge_{A/B}, envelope_id_{A/B},
          monoisotopic_mass_{A/B}, deconvoluted_mz_{A/B}
    df_replaced : pd.DataFrame
        Same shape & columns as `df`, but mz_col_a and mz_col_b values
        are replaced by their deconvoluted m/z ([M+H]+).
    """
    df_ann = df.copy()

    # Load lookup table once
    mono_lookup, env_lookup = _load_lookup(theo_patt_path)

    # ---- Column A ----
    mz_a = df[mz_col_a].values.astype(float)
    int_a = (df[intensity_col_a].values.astype(float)
             if intensity_col_a and intensity_col_a in df.columns
             else np.ones(len(mz_a)))

    ch_a, env_a, mass_a, dmz_a = _deconvolute_single_column(
        mz_a, int_a, mono_lookup, env_lookup, mz_tol
    )
    df_ann["charge_A"] = ch_a
    df_ann["envelope_id_A"] = env_a
    df_ann["monoisotopic_mass_A"] = mass_a
    df_ann["deconvoluted_mz_A"] = dmz_a

    # ---- Column B ----
    mz_b = df[mz_col_b].values.astype(float)
    int_b = (df[intensity_col_b].values.astype(float)
             if intensity_col_b and intensity_col_b in df.columns
             else np.ones(len(mz_b)))

    ch_b, env_b, mass_b, dmz_b = _deconvolute_single_column(
        mz_b, int_b, mono_lookup, env_lookup, mz_tol
    )
    df_ann["charge_B"] = ch_b
    df_ann["envelope_id_B"] = env_b
    df_ann["monoisotopic_mass_B"] = mass_b
    df_ann["deconvoluted_mz_B"] = dmz_b

    # ---- Replaced DataFrame ----
    df_rep = df.copy()
    df_rep[mz_col_a] = dmz_a
    df_rep[mz_col_b] = dmz_b

    return df_ann, df_rep


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
data_path = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt"
save_path = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/deconv/"
df = pd.read_csv(data_path, sep=r"\s+", skiprows=1, header=None, engine="python")
df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
df_annotated, df_replaced = deconvolute_pairs(
    df,
    mz_col_a="m/z A",
    mz_col_b="m/z B",
    intensity_col_a="intensity A",
    intensity_col_b="intensity B",
    theo_patt_path='theo_patt.txt',
)

df_annotated.to_csv(save_path + 'VEA3+_annotated.txt', sep = '\t', index = False)
df_replaced.to_csv(save_path + 'VEA3+_replaced.txt', sep = '\t', index = False)


# ---------------------------------------------------------------------------
# 6. Demo / test
# ---------------------------------------------------------------------------

'''
if __name__ == "__main__":

    df = pd.DataFrame({
        "m/z A": [
            # Charge-3 envelope
            500.68, 501.01, 501.35,
            # Charge-2 envelope
            600.30, 600.80,
            # Charge-1 envelope
            800.40, 801.40,
        ],
        "m/z B": [
            # Charge-2 envelope
            700.50, 701.00, 701.50,
            # Charge-1 envelope
            900.45, 901.45,
            # Charge-3 envelope
            400.35, 400.68,
        ],
        "intensity A": [60, 100, 70, 100, 65, 100, 55],
        "intensity B": [80, 100, 50, 100, 60, 50, 100],
        "correlation_score": [0.95, 0.93, 0.90, 0.88, 0.85, 0.82, 0.78],
    })

    print("=" * 80)
    print("INPUT")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    df_annotated, df_replaced = deconvolute_pairs(
        df,
        mz_col_a="m/z A",
        mz_col_b="m/z B",
        intensity_col_a="intensity A",
        intensity_col_b="intensity B",
        theo_patt_path='theo_patt.txt',
    )

    print("=" * 80)
    print("ANNOTATED  (original + new columns)")
    print("=" * 80)
    print(df_annotated.to_string(index=False))
    print()

    print("=" * 80)
    print("REPLACED  (same shape, m/z swapped for deconvoluted m/z)")
    print("=" * 80)
    print(df_replaced.to_string(index=False))
'''