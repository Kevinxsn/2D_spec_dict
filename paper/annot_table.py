import pandas as pd

path = '/Users/kevinmbp/Desktop/2D_spec_dict/data/virtual_MSMS/VEADIAGHGQEVLIR-mz536-z3_Intensity_Sum'

# 1. Read the spectral file with the custom split to keep annotations intact
with open(path, 'r') as f:
    data = [line.strip().split(None, 3) for line in f if line.strip()]

# 2. Create the spectral DataFrame
df_msms = pd.DataFrame(data, columns=["intensity", "mz", "error", "annotation"])

# 3. Convert columns to numeric types
df_msms["mz"] = pd.to_numeric(df_msms["mz"], errors="coerce")
df_msms["intensity"] = pd.to_numeric(df_msms["intensity"], errors="coerce")
df_msms["error"] = pd.to_numeric(df_msms["error"], errors="coerce")

# 4. Clean up NaNs and sort by m/z (required for merge_asof)
df_msms = df_msms.dropna(subset=["mz", "intensity"]).sort_values("mz").reset_index(drop=True)

# --- Process MMS File ---
mss_input_file = "/Users/kevinmbp/Desktop/2D_spec_dict/data/short_peptide/VEA3+.txt"
mms_df = pd.read_csv(
    mss_input_file,
    sep=r"\s+",
    skiprows=1,
    header=None,
    engine="python"
)

mms_df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
mms_df["m/z A"] = pd.to_numeric(mms_df["m/z A"], errors="coerce")
mms_df["m/z B"] = pd.to_numeric(mms_df["m/z B"], errors="coerce")
tolerance = 0.001

# -------- Match for m/z A --------
a_df = mms_df[["m/z A"]].copy().sort_values("m/z A").reset_index()
a_matched = pd.merge_asof(
    a_df,
    df_msms,
    left_on="m/z A",
    right_on="mz",
    direction="nearest",
    tolerance=tolerance
)
# Rename intensity, annotation, and error for A
a_matched = a_matched.rename(columns={
    "intensity": "intensity A", 
    "annotation": "annotation A",
    "error": "error A"
})
a_matched = a_matched.set_index("index")

# -------- Match for m/z B --------
b_df = mms_df[["m/z B"]].copy().sort_values("m/z B").reset_index()
b_matched = pd.merge_asof(
    b_df,
    df_msms,
    left_on="m/z B",
    right_on="mz",
    direction="nearest",
    tolerance=tolerance
)
# Rename intensity, annotation, and error for B
b_matched = b_matched.rename(columns={
    "intensity": "intensity B", 
    "annotation": "annotation B",
    "error": "error B"
})
b_matched = b_matched.set_index("index")

# -------- Final Join --------
# Add all matched columns back to the original MMS dataframe
mms_df["intensity A"] = a_matched["intensity A"]
mms_df["annotation A"] = a_matched["annotation A"]
mms_df["error A"] = a_matched["error A"]

mms_df["intensity B"] = b_matched["intensity B"]
mms_df["annotation B"] = b_matched["annotation B"]
mms_df["error B"] = b_matched["error B"]

mms_df.head()

mms_df = mms_df[mms_df['Ranking'] > 0]
mms_df = mms_df.sort_values('Ranking')

mms_df["pair_key"] = mms_df.apply(
    lambda row: tuple(sorted([row["m/z A"], row["m/z B"]])),
    axis=1
)
mms_df = mms_df.drop_duplicates(subset="pair_key").drop(columns="pair_key")

mms_df = mms_df.rename(columns={"annotation A": 'Interpretation A', 'annotation B': 'Interpretation B'})
mms_df[['Ranking','m/z A', 'Interpretation A', 'error A', 'm/z B', 'Interpretation B', 'error B']].to_csv('VEA_Annot_table.csv')