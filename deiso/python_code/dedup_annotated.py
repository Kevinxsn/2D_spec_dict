#!/usr/bin/env python3
"""
Deduplicate annotated MMS TSV by (THEO_MONO_MASS_1, THEO_MONO_MASS_2) pairs.
Keeps one representative row per unique pair and writes to output TSV.
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input annotated TSV file")
    parser.add_argument("output", help="Output deduplicated TSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    before = len(df)

    df_dedup = df.drop_duplicates(subset=["THEO_MONO_MASS_1", "THEO_MONO_MASS_2"], keep="first")

    df_dedup.to_csv(args.output, sep="\t", index=False)

    print(f"Rows before : {before}")
    print(f"Rows after  : {len(df_dedup)}")
    print(f"Removed     : {before - len(df_dedup)}")
    print(f"Output      : {args.output}")


if __name__ == "__main__":
    main()
