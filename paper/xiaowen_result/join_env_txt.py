"""
Combine deconvolved envelope data (deconv_ms2.env) with the pairwise
covariance table (VEA3+_with_intensity.txt).

For each row in the txt file, m/z A and m/z B are independently matched
to the nearest ORIG_MZ in the env file (within TOLERANCE Da).  All env
columns are added twice — once for the A-side match, once for the B-side
match — using column-name suffixes _A and _B.

When a given m/z has multiple env rows (same peak assigned to different
isotope envelopes), the one with the highest ENV_CNN_SCORE is kept for
the lookup so the result stays one-row-per-txt-row.

Output is a full outer join: all txt rows are kept; txt rows whose m/z A
or m/z B is absent from the env will have NaN in the corresponding env
columns.
"""

import pandas as pd

ENV_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/paper/xiaowen_result/deconv_ms2.env"
TXT_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+_with_intensity.txt"
OUT_PATH = "/Users/kevinmbp/Desktop/2D_spec_dict/paper/xiaowen_result/deconv_ms2_combined.csv"

TOLERANCE = 0.001  # Da, nearest-match window for ORIG_MZ lookup

# ── Load ──────────────────────────────────────────────────────────────────────

env = pd.read_csv(ENV_PATH, sep="\t")
txt = pd.read_csv(TXT_PATH, sep="\t")

# When the same ORIG_MZ appears in multiple envelopes, keep the best scoring one
env_lookup = (
    env.sort_values("ENV_CNN_SCORE", ascending=False)
       .drop_duplicates(subset="ORIG_MZ")
       .sort_values("ORIG_MZ")
       .reset_index(drop=True)
)

env_cols = [c for c in env_lookup.columns if c != "ORIG_MZ"]

# ── Nearest-match env data for m/z A ─────────────────────────────────────────

a_df = txt[["m/z A"]].copy().sort_values("m/z A").reset_index()
a_matched = pd.merge_asof(
    a_df,
    env_lookup.rename(columns={"ORIG_MZ": "m/z A"}),
    on="m/z A",
    direction="nearest",
    tolerance=TOLERANCE,
).rename(columns={c: f"{c}_A" for c in env_cols}).set_index("index")

# ── Nearest-match env data for m/z B ─────────────────────────────────────────

b_df = txt[["m/z B"]].copy().sort_values("m/z B").reset_index()
b_matched = pd.merge_asof(
    b_df,
    env_lookup.rename(columns={"ORIG_MZ": "m/z B"}),
    on="m/z B",
    direction="nearest",
    tolerance=TOLERANCE,
).rename(columns={c: f"{c}_B" for c in env_cols}).set_index("index")

# ── Assemble result ───────────────────────────────────────────────────────────

for col in [f"{c}_A" for c in env_cols]:
    txt[col] = a_matched[col]
for col in [f"{c}_B" for c in env_cols]:
    txt[col] = b_matched[col]

txt.to_csv(OUT_PATH, index=False)

matched_a = txt["ENV_CNN_SCORE_A"].notna().sum()
matched_b = txt["ENV_CNN_SCORE_B"].notna().sum()
print(f"Total rows       : {len(txt)}")
print(f"m/z A matched    : {matched_a} / {len(txt)}")
print(f"m/z B matched    : {matched_b} / {len(txt)}")
print(f"Saved to: {OUT_PATH}")
print()
print(txt.head(5).to_string(index=False))
