#!/usr/bin/env python3
"""
Annotate myoglobin_mms rows with deconv_dedup_ms2.env information.

For each row in myoglobin_mms, look up col1 and col2 (ORIG_MZ values)
in deconv_dedup_ms2.env and append THEO_MONO_MZ, THEO_MONO_MASS,
THEO_INTE_SUM, THEO_CHARGE for each match (suffixed _1 and _2).
"""

import sys
import argparse
import pandas as pd

ANNO_COLS = ["THEO_MONO_MZ", "THEO_MONO_MASS", "THEO_INTE_SUM", "THEO_CHARGE"]
MZ_TOL = 1e-4  # matching tolerance for ORIG_MZ


def lookup(env_indexed: pd.DataFrame, mz: float) -> dict:
    """Return annotation dict for a given ORIG_MZ value, or NaNs if not found."""
    # round to 6 dp to handle floating-point representation differences
    key = round(mz, 6)
    if key in env_indexed.index:
        row = env_indexed.loc[key]
        return {col: row[col] for col in ANNO_COLS}
    # fallback: tolerance-based search
    candidates = env_indexed[abs(env_indexed["ORIG_MZ"] - mz) < MZ_TOL]
    if not candidates.empty:
        row = candidates.iloc[0]
        return {col: row[col] for col in ANNO_COLS}
    return {col: float("nan") for col in ANNO_COLS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mms_file", help="Path to myoglobin_mms (space-separated)")
    parser.add_argument(
        "--env",
        default="deconv_dedup_ms2.env",
        help="Path to deconv_dedup_ms2.env (TSV, default: deconv_dedup_ms2.env)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: <mms_file>_annotated.tsv)",
    )
    args = parser.parse_args()

    output_path = args.output or args.mms_file + "_annotated.tsv"

    # --- load env ---
    env = pd.read_csv(args.env, sep="\t")
    env["_mz_key"] = env["ORIG_MZ"].round(6)
    env_indexed = env.set_index("_mz_key")

    # --- read mms ---
    mms_rows = []
    with open(args.mms_file) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            mms_rows.append(parts)

    if not mms_rows:
        sys.exit("No data rows found in mms file.")

    # build column names: col0..colN-1
    n_cols = max(len(r) for r in mms_rows)
    col_names = [f"col{i}" for i in range(n_cols)]
    mms_df = pd.DataFrame(mms_rows, columns=col_names[:n_cols])

    # convert first two columns to float for lookup
    mms_df["col0"] = pd.to_numeric(mms_df["col0"])
    mms_df["col1"] = pd.to_numeric(mms_df["col1"])

    # --- annotate from col0 (suffix _1) ---
    anno1 = []
    for i, mz in enumerate(mms_df["col0"], 1):
        anno1.append(lookup(env_indexed, mz))
        print(f"\rAnnotating col0: {i}/{len(mms_df)}", end="", flush=True)
    print()
    anno1_df = pd.DataFrame(anno1).rename(columns={c: c + "_1" for c in ANNO_COLS})

    # --- annotate from col1 (suffix _2) ---
    anno2 = []
    for i, mz in enumerate(mms_df["col1"], 1):
        anno2.append(lookup(env_indexed, mz))
        print(f"\rAnnotating col1: {i}/{len(mms_df)}", end="", flush=True)
    print()
    anno2_df = pd.DataFrame(anno2).rename(columns={c: c + "_2" for c in ANNO_COLS})

    # --- combine and filter ---
    result = pd.concat([mms_df, anno1_df, anno2_df], axis=1)
    before = len(result)
    result = result[
        result["THEO_MONO_MZ_1"].notna() & result["THEO_MONO_MZ_2"].notna()
    ]
    result.to_csv(output_path, sep="\t", index=False)

    print(f"Rows processed : {before}")
    print(f"Rows kept      : {len(result)} (both col0 and col1 matched)")
    print(f"Rows removed   : {before - len(result)}")
    print(f"Output written : {output_path}")


if __name__ == "__main__":
    main()
