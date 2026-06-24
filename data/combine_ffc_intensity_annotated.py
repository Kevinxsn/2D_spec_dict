"""
Combine FFC covariance data with intensity-sum / annotation data.

For each FFC row, the m/z A and m/z B values are matched independently to
the nearest entry in the intensity-sum file (within `TOLERANCE` Da).  The
matched intensity and full annotation string are added as new columns.

Output columns:
    m/z A, m/z B, Covariance, Partial Cov., Score, Ranking,
    intensity A, annotation A, intensity B, annotation B
"""

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

INTENSITY_PATH = (
    "/Users/kevinmbp/Desktop/2D_spec_dict/data/virtual_MSMS/LL37_Z6_NCE33_150_ions_Intensity_Sum"
)
FFC_PATH = (
     "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.LL37_Z6_NCE33_150_ions"
)
OUTPUT_PATH = (
    "/Users/kevinmbp/Desktop/2D_spec_dict/data/"
    "LLG_annote"
)
TOLERANCE = 0.01   # Da, nearest-match window for m/z lookup


# ── Core function ─────────────────────────────────────────────────────────────

def combine_ffc_intensity(intensity_path, ffc_path, tolerance=0.01):
    # ── Load intensity/annotation file ────────────────────────────────────
    # The annotation field contains spaces, so read all tokens and re-join
    # everything from column index 3 onwards as a single annotation string.
    # Use enough named columns to absorb any annotation length; extras fill as NaN
    raw = pd.read_csv(
        intensity_path, sep=r"\s+", header=None,
        names=range(40), engine="python",
    )
    intensity_df = pd.DataFrame({
        "intensity": pd.to_numeric(raw.iloc[:, 0], errors="coerce"),
        "mz":        pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
        "error":     pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
        "annotation": raw.iloc[:, 3:].apply(
            lambda row: " ".join(row.dropna().astype(str)), axis=1
        ),
    })
    intensity_df = (
        intensity_df
        .dropna(subset=["mz", "intensity"])
        .sort_values("mz")
        .reset_index(drop=True)
    )

    # ── Load FFC file (first row is a count header, skip it) ──────────────
    ffc_df = pd.read_csv(ffc_path, sep=r"\s+", skiprows=1, header=None, engine="python")
    ffc_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    ffc_df["m/z A"] = pd.to_numeric(ffc_df["m/z A"], errors="coerce")
    ffc_df["m/z B"] = pd.to_numeric(ffc_df["m/z B"], errors="coerce")

    lookup = intensity_df[["mz", "intensity", "annotation"]].copy()

    # ── Nearest-match intensity + annotation for m/z A ───────────────────
    a_df = ffc_df[["m/z A"]].copy().sort_values("m/z A").reset_index()
    a_matched = pd.merge_asof(
        a_df,
        lookup.rename(columns={"mz": "m/z A"}),
        on="m/z A",
        direction="nearest",
        tolerance=tolerance,
    ).rename(columns={"intensity": "intensity A", "annotation": "annotation A"})
    a_matched = a_matched.set_index("index")

    # ── Nearest-match intensity + annotation for m/z B ───────────────────
    b_df = ffc_df[["m/z B"]].copy().sort_values("m/z B").reset_index()
    b_matched = pd.merge_asof(
        b_df,
        lookup.rename(columns={"mz": "m/z B"}),
        on="m/z B",
        direction="nearest",
        tolerance=tolerance,
    ).rename(columns={"intensity": "intensity B", "annotation": "annotation B"})
    b_matched = b_matched.set_index("index")

    ffc_df["intensity A"]   = a_matched["intensity A"]
    ffc_df["annotation A"]  = a_matched["annotation A"]
    ffc_df["intensity B"]   = b_matched["intensity B"]
    ffc_df["annotation B"]  = b_matched["annotation B"]

    return ffc_df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = combine_ffc_intensity(INTENSITY_PATH, FFC_PATH, tolerance=TOLERANCE)
    result.to_csv(OUTPUT_PATH, sep="\t", index=False)

    matched_a = result["intensity A"].notna().sum()
    matched_b = result["intensity B"].notna().sum()
    print(f"Rows: {len(result)}")
    print(f"m/z A matched: {matched_a} / {len(result)}")
    print(f"m/z B matched: {matched_b} / {len(result)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print(result.head(10).to_string(index=False))
