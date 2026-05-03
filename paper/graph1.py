import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.patches as patches
from adjustText import adjust_text

current_dir = Path.cwd()
parent_dir = current_dir.parent
vis_dir = parent_dir / "vis"
connected_graphs_dir = parent_dir / "vis_connect"
long_dir = parent_dir / "long_peptide"

sys.path.insert(0, str(vis_dir))
sys.path.insert(0, str(connected_graphs_dir))
sys.path.insert(0, str(long_dir))

import data_parse
import util
import peptide
import connected_graph
from bisect import bisect_left

path = '/Users/kevinmbp/Desktop/2D_spec_dict/data/virtual_MSMS/VEADIAGHGQEVLIR-mz536-z3_Intensity_Sum'

with open(path, 'r') as f:
    data = [line.strip().split(None, 3) for line in f if line.strip()]

df_msms = pd.DataFrame(data, columns=["intensity", "mz", "error", "annotation"])
df_msms["mz"] = pd.to_numeric(df_msms["mz"], errors="coerce")
df_msms["intensity"] = pd.to_numeric(df_msms["intensity"], errors="coerce")
df_msms["error"] = pd.to_numeric(df_msms["error"], errors="coerce")
df_msms = df_msms.dropna(subset=["mz", "intensity"]).sort_values("mz").reset_index(drop=True)

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

a_df = mms_df[["m/z A"]].copy().sort_values("m/z A").reset_index()
a_matched = pd.merge_asof(
    a_df, df_msms,
    left_on="m/z A", right_on="mz",
    direction="nearest", tolerance=tolerance
)
a_matched = a_matched.rename(columns={
    "intensity": "intensity A",
    "annotation": "annotation A",
    "error": "error A"
})
a_matched = a_matched.set_index("index")

b_df = mms_df[["m/z B"]].copy().sort_values("m/z B").reset_index()
b_matched = pd.merge_asof(
    b_df, df_msms,
    left_on="m/z B", right_on="mz",
    direction="nearest", tolerance=tolerance
)
b_matched = b_matched.rename(columns={
    "intensity": "intensity B",
    "annotation": "annotation B",
    "error": "error B"
})
b_matched = b_matched.set_index("index")

mms_df["intensity A"] = a_matched["intensity A"]
mms_df["annotation A"] = a_matched["annotation A"]
mms_df["error A"] = a_matched["error A"]
mms_df["intensity B"] = b_matched["intensity B"]
mms_df["annotation B"] = b_matched["annotation B"]
mms_df["error B"] = b_matched["error B"]

mms_df = mms_df[['m/z A', 'm/z B', 'Ranking', 'annotation A', 'annotation B']]
mms_df = mms_df[mms_df['Ranking'] != -1]
mms_df = mms_df[mms_df['Ranking'] <= 50]

# --- Diagnostic: check for NaNs that will cause point loss ---
nan_mz_a = mms_df['m/z A'].isna().sum()
nan_mz_b = mms_df['m/z B'].isna().sum()
print(f"Total rows in mms_df before plotting: {len(mms_df)}")
print(f"Rows with NaN in 'm/z A': {nan_mz_a}  <- these will be dropped in the plot")
print(f"Rows with NaN in 'm/z B': {nan_mz_b}  <- these will be dropped in the plot")
print(f"Expected points on plot: {len(mms_df.dropna(subset=['m/z A', 'm/z B', 'Ranking']))}")

PEP_SEQ = "VEADIAGHGQEVLIR"
CHARGE = 3
pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)

b_ions = {f'b{i}': pep.ion_mass(f'b{i}') for i in range(1, len(PEP_SEQ))}
y_ions = {f'y{i}': pep.ion_mass(f'y{i}') for i in range(1, len(PEP_SEQ))}


def plot_ffc_map(
    df,
    mz_a_col="mz_A",
    mz_b_col="mz_B",
    ranking_col="ranking",
    lines=None,
    b_ions=None,
    y_ions=None,
    num_random_lines=4,
    xlim=None,
    ylim=None,
    figsize=(8, 7),
    point_size=80,
    line_alpha=0.45,
    grid_alpha=0.12,
    grid_linewidth=0.8,
    random_seed=42,
    annotate_ranking=True,       # toggle ranking labels on/off
    annotation_fontsize=7,       # font size for ranking labels
):
    data = df.copy()

    data[mz_a_col] = pd.to_numeric(data[mz_a_col], errors="coerce")
    data[mz_b_col] = pd.to_numeric(data[mz_b_col], errors="coerce")
    data[ranking_col] = pd.to_numeric(data[ranking_col], errors="coerce")

    before = len(data)
    data = data.dropna(subset=[mz_a_col, mz_b_col, ranking_col])
    after = len(data)

    # Diagnostic print inside plot function
    if before != after:
        print(f"[plot_ffc_map] WARNING: {before - after} row(s) dropped due to NaN in "
              f"'{mz_a_col}', '{mz_b_col}', or '{ranking_col}'. "
              f"Plotting {after} of {before} points.")
    else:
        print(f"[plot_ffc_map] All {after} points will be plotted.")

    if xlim is None:
        x_min = data[mz_a_col].min() - 50
        x_max = data[mz_a_col].max() + 50
        xlim = (x_min, x_max)

    if ylim is None:
        y_min = data[mz_b_col].min() - 50
        y_max = data[mz_b_col].max() + 50
        ylim = (y_min, y_max)

    fig, ax = plt.subplots(figsize=figsize)

    # b/y ion grid lines
    if b_ions is not None:
        for ion_name, mz_value in b_ions.items():
            mz_value = float(mz_value)
            if xlim[0] <= mz_value <= xlim[1]:
                ax.axvline(x=mz_value, color="gray", alpha=grid_alpha,
                           linewidth=grid_linewidth, zorder=0)

    if y_ions is not None:
        for ion_name, mz_value in y_ions.items():
            mz_value = float(mz_value)
            if ylim[0] <= mz_value <= ylim[1]:
                ax.axhline(y=mz_value, color="gray", alpha=grid_alpha,
                           linewidth=grid_linewidth, zorder=0)

    # Scatter plot — low alpha so overlapping points visually accumulate
    scatter = ax.scatter(
        data[mz_a_col],
        data[mz_b_col],
        c=data[ranking_col],
        cmap="plasma",
        s=point_size,
        edgecolors="cyan",
        linewidths=0.8,
        alpha=0.35,   # transparent: stacked points appear brighter/denser
        zorder=3
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("FFC ranking", fontsize=14)

    # --- Annotate each point with its ranking, using repulsion to avoid overlaps ---
    if annotate_ranking:
        texts = []
        for _, row in data.iterrows():
            t = ax.text(
                row[mz_a_col],
                row[mz_b_col],
                str(int(row[ranking_col])),
                fontsize=annotation_fontsize,
                color="black",
                fontweight="bold",
                zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec="gray",
                    alpha=0.35,   # transparent: stacked points appear brighter/denser
                    lw=0.5,
                )
            )
            texts.append(t)

        # adjustText pushes labels apart and draws a thin line from label to its point
        adjust_text(
            texts,
            x=data[mz_a_col].values,
            y=data[mz_b_col].values,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            expand=(1.3, 1.5),      # how much extra space to keep around labels
            force_points=(0.3, 0.3),  # repulsion strength from points
            force_text=(0.5, 0.5),    # repulsion strength between labels
            iter_lim=300,
        )

    # Draw lines
    if lines is None:
        np.random.seed(random_seed)
        lines = []
        for i in range(num_random_lines):
            slope = np.random.uniform(-2.0, -0.3)
            intercept = np.random.uniform(800, 1700)
            color = np.random.choice(["blue", "green", "purple", "orange", "red"])
            lines.append({
                "slope": slope,
                "intercept": intercept,
                "color": color,
                "label": f"random line {i + 1}"
            })

    x_values = np.linspace(xlim[0], xlim[1], 500)
    for line in lines:
        slope = line.get("slope")
        intercept = line.get("intercept")
        color = line.get("color", "black")
        label = line.get("label", None)
        y_values = slope * x_values + intercept
        ax.plot(x_values, y_values, color=color, alpha=line_alpha,
                linewidth=2, label=label, zorder=2)

    ax.set_xlabel("m/z A", fontsize=14)
    ax.set_ylabel("m/z B", fontsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis="both", labelsize=12)

    if any(line.get("label") is not None for line in lines):
        ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('graph/fig1.1_transparent_annotated.png')

    return fig, ax


exact_lines = [
    {"slope": -1, "intercept": 1608.869 - 100.069, "color": "blue",   "label": "internal line (V)"},
    {"slope": -1, "intercept": 1608.869 - 229.112, "color": "purple", "label": "internal line (VE)"},
    {"slope": -0.5, "intercept": 1608.869 / 2,     "color": "green",  "label": "parental line"},
    {"slope": -2,   "intercept": 1608.869,          "color": "green",  "label": "parental line"}
]

print(mms_df.shape)

plot_ffc_map(
    mms_df,
    mz_a_col="m/z A",
    mz_b_col="m/z B",
    ranking_col="Ranking",
    b_ions=b_ions,
    y_ions=y_ions,
    xlim=(0, 1500),
    ylim=(0, 1500),
    grid_alpha=0.10,
    lines=exact_lines,
    annotate_ranking=True,       # show ranking labels next to each point
    annotation_fontsize=7,       # adjust size as needed
)