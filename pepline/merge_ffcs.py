"""
Duplicate-FFC merger
====================
After deconvolution the same raw FFC point can appear multiple times
(different charge-state hypotheses, different line passes, etc.).
``merge_duplicate_ffcs`` collapses those duplicates, keeping the single
row that has the best (lowest) Ranking for each unique (m/z A, m/z B) pair.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def merge_duplicate_ffcs(
    df: pd.DataFrame,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    ranking_col: str = "Ranking",
    tolerance: float = 0.0,
    symmetric: bool = False,
    unranked_value: int = -1,
    reset_ranking: bool = True,
) -> pd.DataFrame:
    """
    Collapse duplicate FFC rows, keeping the best-ranked row per unique pair.

    Parameters
    ----------
    df : DataFrame
        FFC data that may contain duplicate (col_a, col_b) pairs.
    col_a, col_b : str
        Column names for the two m/z values used to identify duplicates.
    ranking_col : str
        Column used to pick the winner within each duplicate group.
        Lower value = better.  Rows whose ranking equals ``unranked_value``
        are treated as worst and only kept if no ranked row exists for that pair.
    tolerance : float
        If > 0, m/z values are rounded to the nearest multiple of this value
        before grouping, so pairs within ±tolerance/2 of each other collapse.
        If 0 (default), exact float equality is used.
    symmetric : bool
        If True, treat (A, B) and (B, A) as the same FFC pair by
        canonicalising so the smaller value is always in col_a.
    unranked_value : int
        Sentinel value that marks an unranked row (default -1).
    reset_ranking : bool
        If True, reassign rankings to consecutive integers (1, 2, 3, …) based
        on the relative order of rows after merging.  FFC(X, Y) and FFC(Y, X)
        are treated as the same pair for this purpose and receive the same new
        rank number.  Unranked rows (ranking == unranked_value) keep
        ``unranked_value`` and remain at the end.

    Returns
    -------
    DataFrame with one row per unique pair, sorted by ranking_col ascending.
    All original columns are preserved; only the best row's values are kept.
    """
    if df.empty:
        return df.copy()

    result = df.copy()

    # ── Build group keys ──────────────────────────────────────────────────
    a = result[col_a].to_numpy(dtype=float)
    b = result[col_b].to_numpy(dtype=float)

    if tolerance > 0:
        a = np.round(a / tolerance) * tolerance
        b = np.round(b / tolerance) * tolerance

    if symmetric:
        # Canonicalise so the smaller value is always in key_a
        key_a = np.minimum(a, b)
        key_b = np.maximum(a, b)
    else:
        key_a, key_b = a, b

    result["_key_a"] = key_a
    result["_key_b"] = key_b

    # ── Sort so best-ranked rows come first ───────────────────────────────
    # Replace unranked sentinel with inf so it sorts last
    sort_rank = result[ranking_col].replace(unranked_value, float("inf"))
    result["_sort_rank"] = sort_rank

    result = (
        result
        .sort_values("_sort_rank")
        .drop_duplicates(subset=["_key_a", "_key_b"], keep="first")
        .drop(columns=["_key_a", "_key_b", "_sort_rank"])
    )

    # Restore a clean index, sorted by ranking (unranked last)
    ranked = result[result[ranking_col] != unranked_value].sort_values(ranking_col)
    unranked = result[result[ranking_col] == unranked_value]
    result = pd.concat([ranked, unranked], ignore_index=True)

    if reset_ranking:
        # Build canonical (min, max) keys so FFC(X,Y) and FFC(Y,X) land in the
        # same group regardless of the symmetric merge setting.
        a_vals = result[col_a].to_numpy(dtype=float)
        b_vals = result[col_b].to_numpy(dtype=float)
        if tolerance > 0:
            a_vals = np.round(a_vals / tolerance) * tolerance
            b_vals = np.round(b_vals / tolerance) * tolerance

        result["_sym_a"] = np.minimum(a_vals, b_vals)
        result["_sym_b"] = np.maximum(a_vals, b_vals)

        # Representative rank for each canonical group = min of non-unranked values.
        def _group_min(x: pd.Series) -> float:
            valid = x[x != unranked_value]
            return float(valid.min()) if len(valid) > 0 else float("inf")

        result["_group_rank"] = result.groupby(["_sym_a", "_sym_b"])[ranking_col].transform(_group_min)

        # Assign sequential integers to ranked groups; preserve unranked_value for unranked groups.
        unique_groups = (
            result[["_sym_a", "_sym_b", "_group_rank"]]
            .drop_duplicates(subset=["_sym_a", "_sym_b"])
            .sort_values("_group_rank")
        )
        ranked_groups = unique_groups[unique_groups["_group_rank"] != float("inf")].copy()
        unranked_groups = unique_groups[unique_groups["_group_rank"] == float("inf")].copy()
        ranked_groups["_new_rank"] = range(1, len(ranked_groups) + 1)
        unranked_groups["_new_rank"] = unranked_value

        group_map = pd.concat([ranked_groups, unranked_groups])[["_sym_a", "_sym_b", "_new_rank"]]
        result = result.merge(group_map, on=["_sym_a", "_sym_b"], how="left")
        result[ranking_col] = result["_new_rank"].astype(result[ranking_col].dtype)
        result = result.drop(columns=["_sym_a", "_sym_b", "_group_rank", "_new_rank"])

    return result


# ── Example / quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = pd.DataFrame({
        "m/z A":    [159.092, 159.092, 181.109, 181.109, 215.140],
        "m/z B":    [181.109, 181.109, 159.092, 159.092, 159.092],
        "Covariance": [5556.2, 5556.2, 5556.2, 5556.2, -2294.1],
        "Ranking":  [1183, 500, 1183, 750, 138541],
        "charge_A": [1, 2, 2, 1, 1],
        "charge_B": [2, 1, 1, 2, 1],
        "deconv_method": ["ffc_line", "ffc_line", "ffc_line", "ffc_line", "ffc_line"],
    })

    print("=== Input ===")
    print(sample.to_string(index=False))

    print("\n=== Merged (exact, non-symmetric) ===")
    out = merge_duplicate_ffcs(sample)
    print(out.to_string(index=False))

    print("\n=== Merged (exact, symmetric) ===")
    out_sym = merge_duplicate_ffcs(sample, symmetric=True)
    print(out_sym.to_string(index=False))

    print("\n=== Merged (non-symmetric, reset_ranking=True) ===")
    out_reset = merge_duplicate_ffcs(sample, reset_ranking=True)
    print(out_reset.to_string(index=False))

    print("\n=== Merged (symmetric, reset_ranking=True) ===")
    out_sym_reset = merge_duplicate_ffcs(sample, symmetric=True, reset_ranking=True)
    print(out_sym_reset.to_string(index=False))
