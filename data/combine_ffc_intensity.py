import argparse
import pandas as pd


def add_intensity_columns(msms_input_file, mss_input_file, tolerance=0.01):
    # Read MS/MS file
    df_msms = pd.read_csv(
        msms_input_file,
        sep=r"\s+",
        header=None,
        names=["intensity", "mz", "error", "annotation"],
        engine="python",
        usecols=[0, 1, 2, 3]
    )

    df_msms = df_msms[["mz", "intensity"]].copy()
    df_msms["mz"] = pd.to_numeric(df_msms["mz"], errors="coerce")
    df_msms["intensity"] = pd.to_numeric(df_msms["intensity"], errors="coerce")
    df_msms = df_msms.dropna().sort_values("mz").reset_index(drop=True)

    # Read MMS file
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

    # -------- Match intensity for m/z A --------
    a_df = mms_df[["m/z A"]].copy().sort_values("m/z A").reset_index()
    a_matched = pd.merge_asof(
        a_df,
        df_msms,
        left_on="m/z A",
        right_on="mz",
        direction="nearest",
        tolerance=tolerance
    )
    a_matched = a_matched.rename(columns={"intensity": "intensity A"})
    a_matched = a_matched.set_index("index")

    # -------- Match intensity for m/z B --------
    b_df = mms_df[["m/z B"]].copy().sort_values("m/z B").reset_index()
    b_matched = pd.merge_asof(
        b_df,
        df_msms,
        left_on="m/z B",
        right_on="mz",
        direction="nearest",
        tolerance=tolerance
    )
    b_matched = b_matched.rename(columns={"intensity": "intensity B"})
    b_matched = b_matched.set_index("index")

    # Add matched intensities back to original MMS dataframe
    mms_df["intensity A"] = a_matched["intensity A"]
    mms_df["intensity B"] = b_matched["intensity B"]

    return mms_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msms_input_file", required=True, help="Path to MS/MS peak file")
    parser.add_argument("--mss_input_file", required=True, help="Path to MMS/FFC file")
    parser.add_argument("--output_file", required=True, help="Path to output file")
    parser.add_argument("--tolerance", type=float, default=0.001, help="m/z matching tolerance")

    args = parser.parse_args()

    result_df = add_intensity_columns(
        args.msms_input_file,
        args.mss_input_file,
        tolerance=args.tolerance
    )

    result_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Saved result to {args.output_file}")
