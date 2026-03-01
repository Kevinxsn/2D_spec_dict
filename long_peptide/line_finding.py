from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import re

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Add parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import peptide
import math
import interpreter_modify


from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class LineCluster:
    i: int
    j: int
    n_points: int
    diameter: float
    center: float
    min_v: float
    max_v: float
    # indices in the original dataframe that support this cluster
    point_indices: np.ndarray


def _cluster_sorted_values(vals_sorted: np.ndarray,
                           idx_sorted: np.ndarray,
                           delta: float,
                           min_cluster_size: int) -> List[Tuple[np.ndarray, float, float]]:
    """
    Single-linkage-by-adjacent-gap clustering on pre-sorted values.
    Returns list of (cluster_indices, min_v, max_v) for clusters with size >= min_cluster_size.
    """
    clusters = []
    if vals_sorted.size == 0:
        return clusters

    start = 0
    for k in range(1, vals_sorted.size):
        if (vals_sorted[k] - vals_sorted[k - 1]) > delta:
            # close cluster [start, k)
            if (k - start) >= min_cluster_size:
                c_idx = idx_sorted[start:k]
                min_v = vals_sorted[start]
                max_v = vals_sorted[k - 1]
                clusters.append((c_idx, min_v, max_v))
            start = k

    # last cluster
    if (vals_sorted.size - start) >= min_cluster_size:
        c_idx = idx_sorted[start:]
        min_v = vals_sorted[start]
        max_v = vals_sorted[-1]
        clusters.append((c_idx, min_v, max_v))

    return clusters


def detect_ffc_line_clusters(
    df: pd.DataFrame,
    parent_charge: int,
    delta: float,
    col_a: str = "m/z A",
    col_b: str = "m/z B",
    enforce_sum_leq_charge: bool = True,
    min_cluster_size: int = 3,
    return_point_indices: bool = True,
) -> pd.DataFrame:
    """
      For each (i,j), compute v = i*x + j*y for all points (x,y) in the FFC map.
      Cluster v values by sorting and grouping consecutive values whose gap <= delta.
      Output clusters with size >= min_cluster_size, with diameter=max-min and center=(max+min)/2.

    Parameters
    ----------
    df : DataFrame with columns col_a, col_b
    parent_charge : int
    delta : float
        Absolute threshold on |v1 - v2| for adjacent-gap clustering.
        Note: v has units of m/z (since i,j are unitless multipliers).
    enforce_sum_leq_charge : bool
        If True, only consider (i,j) such that i + j <= parent_charge.
    min_cluster_size : int
        Default 3 (as PI requested "more than 2 elements")
    return_point_indices : bool
        If True, include a column containing numpy arrays of original df indices in each cluster.

    Returns
    -------
    result_df : DataFrame
        Columns: i, j, n_points, diameter, center, min_v, max_v, (optional) point_indices
    """
    if parent_charge < 1:
        raise ValueError("parent_charge must be >= 1")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size should be >= 2")

    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"DataFrame must contain columns '{col_a}' and '{col_b}'")

    # Extract numeric arrays (ignore rows with NaNs)
    sub = df[[col_a, col_b]].copy()
    sub = sub.dropna()
    x = sub[col_a].to_numpy(dtype=float)
    y = sub[col_b].to_numpy(dtype=float)
    original_indices = sub.index.to_numpy()

    results: List[Dict[str, Any]] = []

    # loop over i <= j
    for i in range(1, parent_charge + 1):
        for j in range(i, parent_charge + 1):
            if enforce_sum_leq_charge and (i + j > parent_charge):
                continue

            v = i * x + j * y  # shape (N,)

            # sort and keep mapping back to original indices
            order = np.argsort(v)
            v_sorted = v[order]
            idx_sorted = original_indices[order]

            clusters = _cluster_sorted_values(v_sorted, idx_sorted, delta, min_cluster_size)

            for c_idx, min_v, max_v in clusters:
                diameter = float(max_v - min_v)
                center = float((max_v + min_v) / 2.0)
                row = {
                    "i": i,
                    "j": j,
                    "n_points": int(c_idx.size),
                    "diameter": diameter,
                    "center": center,
                    "min_v": float(min_v),
                    "max_v": float(max_v),
                }
                if return_point_indices:
                    row["point_indices"] = c_idx
                results.append(row)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # convenient ordering: strongest evidence first
        result_df = result_df.sort_values(
            by=["n_points", "diameter"], ascending=[False, True]
        ).reset_index(drop=True)

    return result_df


def group_by_from_center(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Merge rows where from_center values are close:
    consecutive rows belong to same group if the gap <= thresh.

    Aggregation:
      - min_v: min
      - max_v: max
      - center: mean
    Also keeps:
      - from_center: mean (you can change to median if preferred)
      - n_points: sum (common choice; change if you prefer max/mean)
      - i, j: keeps the mode if present; otherwise drops them
      - point_indices: concatenates into a single list (if it's list-like or string)
    """
    df2 = df.copy()

    # Ensure numeric
    df2["Parent+X"] = pd.to_numeric(df2["Parent+X"])

    # Sort so "close" means consecutive proximity
    df2 = df2.sort_values("Parent+X").reset_index(drop=True)

    # New group whenever gap > threshold
    gap = df2["Parent+X"].diff().abs()
    df2["_grp"] = (gap.gt(thresh)).cumsum()

    # Helper to merge point_indices robustly
    def merge_point_indices(s):
        # If already lists, flatten; if strings like "[1,2,...]" keep as concatenated strings
        out = []
        for x in s.dropna():
            if isinstance(x, (list, tuple, np.ndarray)):
                out.extend(list(x))
            else:
                # keep as string fragment
                out.append(str(x))
        return out

    agg_dict = {
        "min_v": "min",
        "max_v": "max",
        "center": "mean",
        "Parent+X": "mean",
    }

    # Optional columns if they exist
    if "n_points" in df2.columns:
        agg_dict["n_points"] = "sum"  # or "max" / "mean"
    if "diameter" in df2.columns:
        agg_dict["diameter"] = "mean"
    if "i" in df2.columns:
        agg_dict["i"] = lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]
    if "j" in df2.columns:
        agg_dict["j"] = lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]
    if "point_indices" in df2.columns:
        agg_dict["point_indices"] = merge_point_indices

    grouped = (
        df2.groupby("_grp", as_index=False)
           .agg(agg_dict)
           .sort_values("Parent+X")
           .reset_index(drop=True)
    )

    return grouped


if __name__ == "__main__":
    
    #ffc_df = pd.read_excel('/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/data/Covariance Scoring Tables 10000 Scans.xlsx', sheet_name='HGTVVLTALGGILK-mz460-3_cov')
    #ffc_df = ffc_df[['m/z fragment 1', 'm/z fragment 2', 'Covariance', 'Partial Cov.', 'Score', 'Ranking']]
    
    parent_charge = 19
    
    
    ffc_df = pd.read_csv(
        "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/Covariance_Data_Myoglobin_Z19_NCE35_250_ions_2000Fragments",
        sep=r"\s+",          # any whitespace
        skiprows=1,          
        header=None,
        engine="python"
    )
    ffc_df.columns = ['m/z A', 'm/z B', 'Covariance', 'Partial Cov.', 'Score', 'Ranking'] 
    
    
    
    ffc_df = ffc_df.sort_values('Ranking', ascending=True)
    
    
    '''
    ffc_df["pair_key"] = ffc_df.apply(
        lambda row: tuple(sorted([row["m/z fragment 1"], row["m/z fragment 2"]])),
        axis=1
    )

    # drop duplicated pairs, keep first occurrence
    ffc_df = ffc_df.drop_duplicates(subset="pair_key").drop(columns="pair_key")
    '''
    
    ffc_df = ffc_df[ffc_df['Ranking'] != -1]
    #ffc_df = ffc_df[ffc_df['Ranking'] <= 5000]
    ffc_df.columns = ['m/z A', 'm/z B', 'Covariance', 'Partial Cov.', 'Score', 'Ranking']
    
    clusters_df = detect_ffc_line_clusters(
        df=ffc_df,
        parent_charge=parent_charge,
        delta=0.05,                 # example absolute threshold in "v" space
        enforce_sum_leq_charge=True,
        min_cluster_size=3
    )

    #print(clusters_df.head(30))
    
    
    ## bewlow are some further analysis
    
    
    #parent = 3767.844130000001
    #parent = 1887.0362399999997
    #parent = 1380.85609
    parent = 892.6370 * 19
    clusters_df['Parent+X'] =  clusters_df['center'] - parent
    clusters_df = clusters_df.sort_values('Parent+X', ascending=False)
    clusters_df = clusters_df[clusters_df['Parent+X'] < 4.05]
    clusters_df['Parent+X'] = clusters_df['Parent+X'].round(3)
    clusters_df = clusters_df.sort_values('n_points', ascending=False)
    
    clusters_df['i+j'] = clusters_df['i'] + clusters_df['j']
    clusters_chosen_df = clusters_df[clusters_df['i+j'] > int(parent_charge - 2)]
    
    print(clusters_df.head(50))
    print(clusters_df[(clusters_df['i+j'] > (parent_charge - 2)) & (clusters_df['Parent+X'] >= -1)].head(50))
    print(clusters_chosen_df.head(50))
    
    
    
    '''
    df_grouped = group_by_from_center(clusters_df, 0.5)
    df_grouped = df_grouped.sort_values('Parent+X', ascending=False)
    df_grouped = df_grouped[['min_v', 'max_v', 'center', 'n_points',
       'diameter', 'Parent+X']]
    df_grouped = df_grouped.sort_values('n_points', ascending=False)
    
    
    
    
    print(df_grouped.head(50))
    
    
    with pd.ExcelWriter("lines4+.xlsx", engine="openpyxl") as writer:
        clusters_df.to_excel(writer, sheet_name="Sheet1", index=False)
        df_grouped.to_excel(writer, sheet_name="Sheet2", index=False)
    
    '''