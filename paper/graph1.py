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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
except Exception:  # adjustText optional at import time
    adjust_text = None


def _arrange_labels(texts, xs, ys, ax):
    """Run adjustText with version-tolerant kwargs (falls back gracefully)."""
    if adjust_text is None or not texts:
        return
    arrow = dict(arrowstyle="-", color="#9aa3ad", lw=0.5)
    # Newer adjustText API first, then fall back to the older one.
    try:
        adjust_text(
            texts, x=xs, y=ys, ax=ax,
            arrowprops=arrow,
            expand=(1.25, 1.4),
            force_text=(0.4, 0.5),
            max_move=None,
        )
    except TypeError:
        try:
            adjust_text(
                texts, x=xs, y=ys, ax=ax,
                arrowprops=arrow,
                expand=(1.3, 1.5),
                force_points=(0.3, 0.3),
                force_text=(0.5, 0.5),
                iter_lim=300,
            )
        except TypeError:
            adjust_text(texts, x=xs, y=ys, ax=ax, arrowprops=arrow)


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
    figsize=(9, 7.5),
    point_size=80,
    cmap="plasma",
    point_alpha=0.85,             # crisp points (was 0.35); each is labelled anyway
    point_edgecolor="white",      # white halo separates overlapping points (was "cyan")
    point_linewidth=0.5,
    line_alpha=0.55,
    b_ion_color="#2f6690",        # b-ions: muted blue
    y_ion_color="#c0504d",        # y-ions: muted brick-red
    grid_alpha=0.30,              # solid (primary-axis) ladder lines
    dashed_grid_alpha=0.22,       # dashed (complementary-axis) ladder lines
    grid_linewidth=0.8,
    grid_label_alpha=0.75,
    grid_label_fontsize=7,
    show_grid_labels=True,
    annotate_ranking=True,
    annotation_fontsize=7,
    spine_color="#cfd4d9",
    tick_color="#5a5f66",
    equal_aspect=True,            # force a square plotting box (needs equal xlim/ylim)
    random_seed=42,
    save_path="graph/fig1.1_transparent_annotated.png",
):
    """
    FFC map coloured by ranking, with a four-sided b/y ion ladder.

    Ion grid (b = blue, y = red):
      - b-ions: solid vertical   (labels bottom)  + dashed horizontal (labels right)
      - y-ions: solid horizontal  (labels left)   + dashed vertical   (labels top)

    Points are coloured by ``ranking_col`` and individually labelled with their
    ranking; labels are de-overlapped with adjustText.
    """
    data = df.copy()

    data[mz_a_col] = pd.to_numeric(data[mz_a_col], errors="coerce")
    data[mz_b_col] = pd.to_numeric(data[mz_b_col], errors="coerce")
    data[ranking_col] = pd.to_numeric(data[ranking_col], errors="coerce")

    before = len(data)
    data = data.dropna(subset=[mz_a_col, mz_b_col, ranking_col])
    after = len(data)
    if before != after:
        print(f"[plot_ffc_map] WARNING: {before - after} row(s) dropped due to NaN in "
              f"'{mz_a_col}', '{mz_b_col}', or '{ranking_col}'. "
              f"Plotting {after} of {before} points.")
    else:
        print(f"[plot_ffc_map] All {after} points will be plotted.")

    if data.empty:
        raise ValueError("No valid data points remain after cleaning the dataframe.")

    if xlim is None:
        xlim = (data[mz_a_col].min() - 50, data[mz_a_col].max() + 50)
    if ylim is None:
        ylim = (data[mz_b_col].min() - 50, data[mz_b_col].max() + 50)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    # --- label offsets: every ion label sits OUTSIDE the plotting area ---
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    bottom_label_y = ylim[0] - 0.02 * y_span   # b solid vertical  -> below
    left_label_x   = xlim[0] - 0.01 * x_span   # y solid horizontal -> left
    top_label_y    = ylim[1] + 0.01 * y_span   # y dashed vertical  -> above
    right_label_x  = xlim[1] + 0.012 * x_span  # b dashed horizontal-> right

    # ---------------- b-ions: solid vertical + dashed horizontal ----------------
    if b_ions is not None:
        for ion_name, mz_value in b_ions.items():
            mz_value = float(mz_value)
            if xlim[0] <= mz_value <= xlim[1]:
                ax.axvline(mz_value, color=b_ion_color, alpha=grid_alpha,
                           linewidth=grid_linewidth, linestyle="-", zorder=0)
                if show_grid_labels:
                    ax.text(mz_value, bottom_label_y, ion_name,
                            color=b_ion_color, alpha=grid_label_alpha,
                            fontsize=grid_label_fontsize, rotation=90,
                            ha="center", va="top", zorder=1, clip_on=False)
            if ylim[0] <= mz_value <= ylim[1]:
                ax.axhline(mz_value, color=b_ion_color, alpha=dashed_grid_alpha,
                           linewidth=grid_linewidth, linestyle="--", zorder=0)
                if show_grid_labels:
                    ax.text(right_label_x, mz_value, ion_name,
                            color=b_ion_color, alpha=grid_label_alpha,
                            fontsize=grid_label_fontsize,
                            ha="left", va="center", zorder=1, clip_on=False)

    # ---------------- y-ions: solid horizontal + dashed vertical ----------------
    if y_ions is not None:
        for ion_name, mz_value in y_ions.items():
            mz_value = float(mz_value)
            if ylim[0] <= mz_value <= ylim[1]:
                ax.axhline(mz_value, color=y_ion_color, alpha=grid_alpha,
                           linewidth=grid_linewidth, linestyle="-", zorder=0)
                if show_grid_labels:
                    ax.text(left_label_x, mz_value, ion_name,
                            color=y_ion_color, alpha=grid_label_alpha,
                            fontsize=grid_label_fontsize,
                            ha="right", va="center", zorder=1, clip_on=False)
            if xlim[0] <= mz_value <= xlim[1]:
                ax.axvline(mz_value, color=y_ion_color, alpha=dashed_grid_alpha,
                           linewidth=grid_linewidth, linestyle="--", zorder=0)
                if show_grid_labels:
                    ax.text(mz_value, top_label_y, ion_name,
                            color=y_ion_color, alpha=grid_label_alpha,
                            fontsize=grid_label_fontsize, rotation=90,
                            ha="center", va="bottom", zorder=1, clip_on=False)

    # ---------------- scatter coloured by ranking ----------------
    scatter = ax.scatter(
        data[mz_a_col], data[mz_b_col],
        c=data[ranking_col], cmap=cmap,
        s=point_size,
        edgecolors=point_edgecolor, linewidths=point_linewidth,
        alpha=point_alpha, zorder=3,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.10, fraction=0.046)
    cbar.set_label("FFC ranking", fontsize=13, color="#2b2f33")
    cbar.outline.set_edgecolor(spine_color)
    cbar.ax.tick_params(labelsize=10, colors=tick_color)

    # ---------------- ranking annotations (arranged) ----------------
    if annotate_ranking:
        texts = [
            ax.text(
                row[mz_a_col], row[mz_b_col], str(int(row[ranking_col])),
                fontsize=annotation_fontsize, color="#1b1f23", zorder=5,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="#b9bfc6", alpha=0.88, lw=0.5),
            )
            for _, row in data.iterrows()
        ]
        _arrange_labels(texts, data[mz_a_col].values, data[mz_b_col].values, ax)

    # ---------------- diagonal / charge lines ----------------
    if lines is None:
        rng = np.random.default_rng(random_seed)
        palette = ["#2f6690", "#5a9367", "#7d5ba6", "#d08c34", "#c45b4c"]
        lines = [{
            "slope": rng.uniform(-2.0, -0.3),
            "intercept": rng.uniform(800, 1700),
            "color": palette[i % len(palette)],
            "label": f"random line {i + 1}",
        } for i in range(num_random_lines)]

    x_values = np.linspace(xlim[0], xlim[1], 500)
    for line in lines:
        y_values = line.get("slope") * x_values + line.get("intercept")
        ax.plot(x_values, y_values, color=line.get("color", "#c45b4c"),
                alpha=line_alpha, linewidth=2, label=line.get("label"), zorder=2)

    # ---------------- axes cosmetics ----------------
    ax.set_xlabel("m/z A", fontsize=14, labelpad=18, color="#2b2f33")
    ax.set_ylabel("m/z B", fontsize=14, labelpad=18, color="#2b2f33")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if equal_aspect:
        # force the data box to be square (independent of figsize / colorbar)
        ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.9)
    ax.tick_params(axis="x", labelsize=12, pad=22, colors=tick_color)
    ax.tick_params(axis="y", labelsize=12, pad=22, colors=tick_color)

    handles, labels = ax.get_legend_handles_labels()
    if any(lbl is not None for lbl in labels):
        # collapse duplicate labels (e.g. two "parental line") to a single entry
        unique = {}
        for h, lbl in zip(handles, labels):
            if lbl is not None and lbl not in unique:
                unique[lbl] = h
        ax.legend(unique.values(), unique.keys(),
                  loc="lower left", bbox_to_anchor=(0.0, 1.05),
                  fontsize=9, framealpha=0.95, edgecolor=spine_color, ncol=2)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.show()

    return fig, ax


tffc_table = pd.read_excel('/Users/kevinmbp/Desktop/2D_spec_dict/paper/VEA_annot_with_charges.xlsx')
tffc_table['tffc_A'] = (tffc_table['m/z A'] - 1.0072) * tffc_table['charge A']
tffc_table['tffc_B'] = (tffc_table['m/z B'] - 1.0072) * tffc_table['charge B']

# Suppose your original table is called tffc_table

df = tffc_table.copy()

# Create the swapped version

swapped = df.copy()

# Swap A-side and B-side columns

swap_pairs = [

    ("m/z A", "m/z B"),

    ("Interpretation A", "Interpretation B"),

    ("error A", "error B"),

    ("charge A", "charge B"),

    ("tffc_A", "tffc_B"),

]

for col_a, col_b in swap_pairs:

    swapped[col_a], swapped[col_b] = df[col_b], df[col_a]

# Concatenate original + swapped

tffc_table_doubled = pd.concat([df, swapped], ignore_index=True)

# Optional: reset Ranking from 1 to length

#tffc_table_doubled["Ranking"] = range(1, len(tffc_table_doubled) + 1)

tffc_table_doubled


exact_lines = [
    {"slope": -1, "intercept": 1608.869 - 100.069, "color": "blue",   "label": "internal line (V)"},
    {"slope": -1, "intercept": 1608.869 - 229.112, "color": "purple", "label": "internal line (VE)"},
    {"slope": -0.5, "intercept": 1608.869 / 2,     "color": "green",  "label": "parental line"},
    {"slope": -2,   "intercept": 1608.869,          "color": "green",  "label": "parental line"}
]



exact_lines2 = [
    {"slope": -1, "intercept": 1608.869-100.069, "color": "blue", "label": "internal line (V)"},
    {"slope": -1, "intercept": 1608.869 - 229.112, "color": "purple", "label": "internal line (VE)"},
    {"slope": -1, "intercept": 1608.869, "color": "green", "label": "parental line"},
]


print(mms_df.shape)

plot_ffc_map(
    tffc_table_doubled,
    mz_a_col="m/z A",
    mz_b_col="m/z B",
    ranking_col="Ranking",
    b_ions=b_ions,
    y_ions=y_ions,
    xlim=(0, 1600),
    ylim=(0, 1600),
    grid_alpha=0.80,
    lines=exact_lines,
    annotate_ranking=True,       # show ranking labels next to each point
    annotation_fontsize=7,       # adjust size as needed
    show_grid_labels=True,       # show b/y ion labels on grid lines
    grid_label_fontsize=7,       # adjust size as needed
    grid_label_alpha=0.45,       # match the subtle gray style
)