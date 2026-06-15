import re
import os
import pandas as pd
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
except Exception:  # adjustText optional at import
    adjust_text = None
    
    
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


def _arrange_labels(texts, xs, ys, ax, expand, force_points, force_text, iter_lim):
    """adjustText call that works across old and new API versions."""
    if adjust_text is None or not texts:
        return
    arrow = dict(arrowstyle="-", color="#9aa3ad", lw=0.5)
    try:  # older adjustText API (force_points / iter_lim)
        adjust_text(texts, x=xs, y=ys, ax=ax, arrowprops=arrow,
                    expand=expand, force_points=force_points,
                    force_text=force_text, iter_lim=iter_lim)
    except TypeError:
        try:  # newer adjustText API
            adjust_text(texts, x=xs, y=ys, ax=ax, arrowprops=arrow,
                        expand=expand, force_text=force_text, max_move=None)
        except TypeError:
            adjust_text(texts, x=xs, y=ys, ax=ax, arrowprops=arrow)


def plot_ffc_map_with_bottom_annotation_table_new(
    df,
    mz_a_col="m/z A",
    mz_b_col="m/z B",
    ranking_col="ranking",
    mz_a_anno_col="mz_A_annotation",
    mz_b_anno_col="mz_B_annotation",
    b_ions=None,
    y_ions=None,
    xlim=None,
    ylim=None,
    start_from_zero=True,
    figsize=(9, 14),
    plot_height_ratio=5.0,
    table_height_ratio=1.2,
    point_size=80,
    point_alpha=0.85,
    point_edgecolor="white",
    point_linewidth=0.5,
    cmap="plasma",
    b_ion_color="#2f6690",
    y_ion_color="#c0504d",
    grid_alpha=0.30,
    dashed_grid_alpha=0.22,
    grid_linewidth=0.8,
    grid_label_alpha=0.75,
    grid_label_fontsize=7,
    show_grid_labels=True,
    background_grid=False,
    force_equal_axis_limits=False,
    equal_aspect=True,
    spine_color="#cfd4d9",
    tick_color="#5a5f66",
    number_fontsize=7,
    number_alpha=1.0,
    table_fontsize=9,
    table_scale_y=1.35,
    max_table_rows=None,
    sort_by_ranking=True,
    table_unique_ranking=True,
    save_path=None,
    # ---- NEW: layout / table-appearance controls -------------------------
    table_gap=0.012,            # fig-fraction gap between plot bottom & table
    table_row_height=0.026,     # fig-fraction height per table row (incl header)
    table_align="plot",         # "plot" -> table width matches plot; "full" -> original full width
    table_header_color="#34699a",
    table_stripe_color="#eef2f7",
    table_text_color="#2b2f33",
    table_edge_color="#d7dde5",
    pull_plot_to_top=True,      # remove the big whitespace above the plot too
    # adjustText tuning parameters
    adjust_expand=(1.3, 1.5),
    adjust_force_points=(0.3, 0.3),
    adjust_force_text=(0.5, 0.5),
    adjust_iter_lim=300,
):
    """
    Plot an FFC map with ranking labels on the points and an annotation table below.

    Layout note:
    Because the map is drawn with equal aspect, the square plot box shrinks inside
    its gridspec cell, which used to leave a large gap before the table. After the
    figure is laid out we measure the plot's *actual* box and dock the table right
    beneath it (controlled by `table_gap`), so the two always sit close together
    regardless of figsize.
    """

    def clean_annotation(x):
        if pd.isna(x):
            return ""
        x = str(x).strip()
        x = x.replace("internal", "int")
        return x

    def starts_with_b(anno):
        return bool(re.match(r"^\s*b\d+", anno))

    def starts_with_y(anno):
        return bool(re.match(r"^\s*y\d+", anno))

    def reorder_annotations(anno_a, anno_b):
        anno_a = clean_annotation(anno_a)
        anno_b = clean_annotation(anno_b)
        a_is_y = starts_with_y(anno_a)
        b_is_y = starts_with_y(anno_b)
        a_is_b = starts_with_b(anno_a)
        b_is_b = starts_with_b(anno_b)
        if a_is_y and not b_is_y:
            return anno_b, anno_a
        if b_is_y:
            return anno_a, anno_b
        if a_is_b:
            return anno_a, anno_b
        if b_is_b and not a_is_y:
            return anno_b, anno_a
        return anno_a, anno_b

    def format_ranking_value(value):
        if pd.isna(value):
            return ""
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)

    # ------------------------------------------------------------------ clean
    data = df.copy()
    required_cols = [mz_a_col, mz_b_col, ranking_col, mz_a_anno_col, mz_b_anno_col]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data[mz_a_col] = pd.to_numeric(data[mz_a_col], errors="coerce")
    data[mz_b_col] = pd.to_numeric(data[mz_b_col], errors="coerce")
    data[ranking_col] = pd.to_numeric(data[ranking_col], errors="coerce")
    data[mz_a_anno_col] = data[mz_a_anno_col].apply(clean_annotation)
    data[mz_b_anno_col] = data[mz_b_anno_col].apply(clean_annotation)

    before = len(data)
    data = data.dropna(subset=[mz_a_col, mz_b_col, ranking_col])
    after = len(data)
    if before != after:
        print(f"[plot] WARNING: {before - after} row(s) dropped due to NaN. "
              f"Plotting {after} of {before} points.")
    else:
        print(f"[plot] All {after} points will be plotted.")
    if data.empty:
        raise ValueError("No valid data points remain after cleaning the dataframe.")

    ordered_pairs = data.apply(
        lambda row: reorder_annotations(row[mz_a_anno_col], row[mz_b_anno_col]),
        axis=1,
    )
    data["annotation_1"] = [p[0] for p in ordered_pairs]
    data["annotation_2"] = [p[1] for p in ordered_pairs]

    if sort_by_ranking:
        data = data.sort_values(ranking_col).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    # ----------------------------------------------------------------- ranges
    if xlim is None:
        xlim = (0, data[mz_a_col].max() + 50) if start_from_zero \
            else (data[mz_a_col].min() - 50, data[mz_a_col].max() + 50)
    if ylim is None:
        ylim = (0, data[mz_b_col].max() + 50) if start_from_zero \
            else (data[mz_b_col].min() - 50, data[mz_b_col].max() + 50)
    if force_equal_axis_limits:
        min_limit = min(xlim[0], ylim[0])
        max_limit = max(xlim[1], ylim[1])
        xlim = (min_limit, max_limit)
        ylim = (min_limit, max_limit)

    # ----------------------------------------------------------------- layout
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[plot_height_ratio, table_height_ratio], hspace=0.10,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    if pull_plot_to_top:
        ax.set_anchor("N")  # square plot sits at the top of its cell

    # ----------------------------------------------------- b/y ion grid lines
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    bottom_label_y = ylim[0] - 0.02 * y_span
    left_label_x = xlim[0] - 0.01 * x_span
    top_label_y = ylim[1] + 0.01 * y_span
    right_label_x = xlim[1] + 0.012 * x_span

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

    # --------------------------------------------------- scatter + colorbar
    scatter = ax.scatter(
        data[mz_a_col], data[mz_b_col],
        c=data[ranking_col], cmap=cmap, s=point_size,
        edgecolors=point_edgecolor, linewidths=point_linewidth,
        alpha=point_alpha, zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.10)
    cbar.set_label("FFC ranking", fontsize=13, color="#2b2f33")
    cbar.outline.set_edgecolor(spine_color)
    cbar.ax.tick_params(labelsize=10, colors=tick_color)

    # ----------------------------------------------------- per-point labels
    texts = []
    for _, row in data.iterrows():
        t = ax.text(
            row[mz_a_col], row[mz_b_col], format_ranking_value(row[ranking_col]),
            fontsize=number_fontsize, alpha=number_alpha, color="#1b1f23",
            fontweight="bold", ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#b9bfc6", alpha=0.88, lw=0.5),
        )
        texts.append(t)
    _arrange_labels(
        texts, data[mz_a_col].values, data[mz_b_col].values, ax,
        adjust_expand, adjust_force_points, adjust_force_text, adjust_iter_lim,
    )

    # ------------------------------------------------------- axis formatting
    ax.set_xlabel("m/z A", fontsize=14, labelpad=18, color="#2b2f33")
    ax.set_ylabel("m/z B", fontsize=14, labelpad=18, color="#2b2f33")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.9)
    ax.tick_params(axis="x", labelsize=12, pad=22, colors=tick_color)
    ax.tick_params(axis="y", labelsize=12, pad=22, colors=tick_color)
    if background_grid:
        ax.grid(True, alpha=0.12, linewidth=0.5)

    # ============================================================ TABLE BUILD
    ax_table.axis("off")

    table_data = data.copy()
    if table_unique_ranking:
        table_data = (
            table_data.sort_values(ranking_col)
            .drop_duplicates(subset=[ranking_col], keep="first")
            .reset_index(drop=True)
        )
    if max_table_rows is not None:
        table_data = table_data.head(max_table_rows)

    table_rows = []
    for _, row in table_data.iterrows():
        table_rows.append([
            format_ranking_value(row[ranking_col]),
            f"{row[mz_a_col]:.3f}",
            row["annotation_1"],
            f"{row[mz_b_col]:.3f}",
            row["annotation_2"],
        ])
    col_labels = ["Ranking", "m/z X", "Annotation A", "m/z Y", "Annotation B"]

    table = ax_table.table(
        cellText=table_rows, colLabels=col_labels,
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.10, 0.14, 0.30, 0.14, 0.30],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)

    # column alignment: ranking centred, m/z right, annotations left.
    # NOTE: matplotlib positions cell text from the Text object's ha, so we
    # must call set_text_props(ha=...).  Also: a cell only fills when its path
    # is *closed*, so we keep visible_edges="closed" with linewidth 0 (no
    # visible box) and draw the horizontal rules ourselves afterwards.
    align_by_col = {0: "center", 1: "right", 2: "left", 3: "right", 4: "left"}
    n_body = len(table_rows)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.0)                  # no per-cell borders
        cell.PAD = 0.05                           # inner breathing room
        ha = align_by_col.get(c, "center")
        if r == 0:                               # header band
            cell.set_facecolor(table_header_color)
            cell.set_text_props(ha=ha, color="white", weight="bold")
        else:
            cell.set_facecolor(table_stripe_color if r % 2 else "white")
            family = "monospace" if c in (1, 3) else None
            if family:
                cell.set_text_props(ha=ha, color=table_text_color, family=family)
            else:
                cell.set_text_props(ha=ha, color=table_text_color)

    # ============================================ DOCK TABLE BELOW THE PLOT
    # Measure the plot's real extent *including* tick labels and the x-axis
    # title, then place the table flush beneath it -> no dead space, and no
    # overlap with the "m/z A" label.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    plot_pos = ax.get_position()

    n_rows_total = n_body + 1                     # + header
    desired_h = table_row_height * n_rows_total
    avail_h = (ax_tight.y0 - table_gap) - 0.015   # down to a small bottom margin
    tbl_height = min(desired_h, max(avail_h, 0.04))
    if tbl_height / n_rows_total < 0.013:         # rows would be genuinely cramped
        print(f"[plot] NOTE: {n_rows_total} table rows are tight below the map. "
              f"Increase figsize height (e.g. (9, 17)) or reduce rows for more room.")

    if table_align == "plot":
        tbl_x0, tbl_w = plot_pos.x0, plot_pos.width
    else:  # "full" -> keep the original gridspec-cell width
        orig = ax_table.get_position()
        tbl_x0, tbl_w = orig.x0, orig.width

    tbl_top = ax_tight.y0 - table_gap             # below ticklabels + xlabel
    tbl_y0 = tbl_top - tbl_height
    ax_table.set_position([tbl_x0, tbl_y0, tbl_w, tbl_height])

    # ----- clean horizontal rules (drawn separately so cell fills survive) ---
    from matplotlib.lines import Line2D
    n_total = n_body + 1
    # y boundaries in axes coords: row 0 (header) occupies the TOP slice
    def _row_top(k):   # top edge of row k (0 = header)
        return 1.0 - k / n_total

    def _hline(y, color, lw, z=4):
        ax_table.add_line(Line2D([0, 1], [y, y], transform=ax_table.transAxes,
                                 color=color, linewidth=lw, zorder=z,
                                 solid_capstyle="butt", clip_on=False))

    _hline(1.0, table_header_color, 1.2)                  # top frame (header top)
    _hline(_row_top(1), table_header_color, 1.4)          # accent under header
    for k in range(2, n_total):                           # thin inter-row rules
        _hline(_row_top(k), table_edge_color, 0.6)
    _hline(0.0, table_edge_color, 0.8)                    # bottom frame

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax, ax_table, data

PEP_SEQ = "VEADIAGHGQEVLIR"
CHARGE = 3
pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
b_ions = {f'b{i}': pep.ion_mass(f'b{i}') for i in range(1, len(PEP_SEQ))}
y_ions = {f'y{i}': pep.ion_mass(f'y{i}') for i in range(1, len(PEP_SEQ))}

ranking_to_drop = [3, 9, 10, 12, 18, 24, 27, 34, 36, 45, 6, 22, 32, 35, 49, 1, 1, 2, 2, 4, 4, 5, 5, 7, 7, 11, 11, 13, 13, 14, 14, 19, 19, 8, 8, 17, 17, 31, 31, 37, 37, 38, 38, 40, 40, 47, 47, 23, 23, 29, 29, 46, 46]
df_annot = pd.read_csv('point_not_line_VEA.csv')
df_annot.head()

df_annot = df_annot[~df_annot["Ranking"].isin(ranking_to_drop)].copy()
df_annot_fixed = df_annot.copy()

# Specify replacements as:
# (row_index, column_name): new_value
no_match_replacements = {
    (19, "annotation A"): "486.221",       # example
    (25, "annotation B"): "486.221"
}

for (idx, col), new_value in no_match_replacements.items():
    df_annot_fixed.loc[idx, col] = new_value

fig, ax, ax_table, annotated_df = plot_ffc_map_with_bottom_annotation_table_new(
    df_annot_fixed,
    mz_a_col="m/z A",
    mz_b_col="m/z B",
    ranking_col="Ranking",
    mz_a_anno_col="annotation A",
    mz_b_anno_col="annotation B",
    b_ions=b_ions,
    y_ions=y_ions,
    xlim=(0, 1600),
    ylim=(0, 1600),
    figsize=(9, 15),
    number_fontsize=8,
    table_fontsize=8,
)