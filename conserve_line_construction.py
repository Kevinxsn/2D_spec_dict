import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import peptide
import matplotlib.patheffects as pe
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Iterable, Optional
from math import sqrt




original_sequence = ''

df = pd.read_csv('data/data_table/data_sheet7.csv')
the_pep = peptide.Pep(original_sequence)

Mass = the_pep.pep_mass

conserve_line_df = df[(df['m1+m2']< Mass+1) & (df['m1+m2']> Mass-1)]
df['ion'] = df['ion1'] +df['ion2']

df_current = df.iloc[[31, 2, 1, 0, 3, 5, 13, 35, 10, 12,24,27,25,11,15,23]]


def plot_points_on_sum_line(
    df,
    xcol="correct_mass1",
    ycol="correct_mass2",
    label_col="ion",
    c: float | None = None,
    figsize=(8, 8),
    point_size=18,
    point_alpha=0.85,
    annotate=True,
    use_adjusttext=True,
    adjust_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    text_kwargs: dict | None = None,
    ax=None,
    show=True,
    save_path: str | None = None,
):
    """
    Plot points (x, y) that lie near a line x + y = c, label them,
    and auto-resolve annotation overlaps (if adjustText is installed).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing coordinates and labels.
    xcol, ycol : str
        Column names for x and y.
    label_col : str or None
        Column name for text labels. If None, no labels are drawn.
    c : float or None
        Constant for the line x + y = c. If None, uses median(x+y).
    figsize : tuple
        Figure size if a new axes is created.
    annotate : bool
        Whether to draw point labels.
    use_adjusttext : bool
        Try to use adjustText to avoid label overlaps.
    adjust_kwargs : dict
        Extra kwargs passed to adjust_text (if available).
    line_kwargs, scatter_kwargs, text_kwargs : dict
        Extra kwargs passed to ax.axline, ax.scatter, and ax.text respectively.
    ax : matplotlib.axes.Axes or None
        If provided, plot on this axes; otherwise create a new one.
    show : bool
        Whether to call plt.show() at the end.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    fig, ax, texts
        Figure, Axes, and list of text artists (possibly empty).
    """
    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute c if not provided
    x_vals = df[xcol].to_numpy()
    y_vals = df[ycol].to_numpy()
    if c is None:
        c = float(np.median(x_vals + y_vals))

    # Scatter defaults
    s_kwargs = dict(s=point_size, alpha=point_alpha, label="points", zorder=1)
    if scatter_kwargs:
        s_kwargs.update(scatter_kwargs)

    ax.scatter(x_vals, y_vals, **s_kwargs)

    # Line defaults
    l_kwargs = dict(linestyle="--", linewidth=1, alpha=0.9, label=f"x + y = {c:g}", zorder=0)
    if line_kwargs:
        l_kwargs.update(line_kwargs)

    ax.axline((0, c), (c, 0), **l_kwargs)

    texts = []
    if annotate and (label_col is not None):
        t_kwargs = dict(
            fontsize=7, ha="left", va="bottom",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            zorder=3
        )
        if text_kwargs:
            t_kwargs.update(text_kwargs)

        for x, y, lab in df[[xcol, ycol, label_col]].itertuples(index=False, name=None):
            t = ax.text(x, y, str(lab), **t_kwargs)
            texts.append(t)

        if use_adjusttext and texts:
            try:
                from adjustText import adjust_text
                adj_default = dict(
                    ax=ax,
                    only_move={"points": "y", "texts": "xy"},
                    expand_points=(1.2, 1.6),
                    expand_text=(1.1, 1.3),
                    force_points=(0.2, 0.5),
                    force_text=(0.1, 0.3),
                    arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.6),
                )
                if adjust_kwargs:
                    adj_default.update(adjust_kwargs)
                adjust_text(texts, **adj_default)
            except ImportError:
                print("Tip: install adjustText to avoid overlaps: pip install adjustText")

    # Cosmetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.margins(x=0.05, y=0.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, texts


plot_points_on_sum_line(df_current, c=Mass, point_size=30, annotate=True, label_col="ion")





starting_point = (182.08, 1071.42)


amino_acid_masses = {
    "A": 71.03711,   # Alanine
    "R": 156.10111,  # Arginine
    "N": 114.04293,  # Asparagine
    "D": 115.02694,  # Aspartic acid
    "C": 103.00919,  # Cysteine
    "E": 129.04259,  # Glutamic acid
    "Q": 128.05858,  # Glutamine
    "G": 57.02146,   # Glycine
    "H": 137.05891,  # Histidine
    "(I/L)": 113.08406,  # Isoleucine
    #"L": 113.08406,  # Leucine
    "K": 128.09496,  # Lysine
    "M": 131.04049,  # Methionine
    "F": 147.06841,  # Phenylalanine
    "P": 97.05276,   # Proline
    "S": 87.03203,   # Serine
    "T": 101.04768,  # Threonine
    "W": 186.07931,  # Tryptophan
    "Y": 163.06333,  # Tyrosine
    "V": 99.06841,    # Valine
    "R(Me)": 156.10111 + 14.01565,  # Methylated Arginine
    "T(p)": 101.04768 + 79.96633,  # Phosphorylated Threonine
}

amino_acid_masses_value = {value: key for key, value in amino_acid_masses.items()}

ptm_masses = {
    "Me": 14.01565,  
    'Me2': 28.0313,
    }

the_pep = peptide.Pep("[GGNFSGR(Me)GGFGGSR+2H]2+")

df_current.loc[:,'correct_sum'] = df_current['correct_mass1'] + df_current['correct_mass2']

df_current.loc[:,'conserve'] = df_current['correct_sum'].apply(lambda x: abs(x - Mass) < 1)

def _q1(v: float, eps: float) -> int:
    """Quantize a float to an integer key with tolerance eps."""
    return int(round(v / eps))

def _q2(x: float, y: float, eps: float) -> Tuple[int, int]:
    return _q1(x, eps), _q1(y, eps)

def build_adj_2d(points_xy: Iterable[Tuple[float, float]],
                 steps: Iterable[float],
                 line_sum: Optional[float] = None,
                 eps: float = 1e-6,
                 forward_only: bool = True):
    """
    Build adjacency on points lying on x+y ~= a using Euclidean step lengths.
    Returns (pts, q2idx, out_edges, in_edges, a_est).
    """
    pts = sorted(set((float(x), float(y)) for x, y in points_xy), key=lambda p: (p[0], -p[1]))
    if not pts:
        raise ValueError("No points given.")
    # Estimate 'a' if not provided, and sanity-check colinearity
    a_vals = [x + y for x, y in pts]
    a_est = sum(a_vals) / len(a_vals) if line_sum is None else float(line_sum)
    for (x, y) in pts:
        if abs((x + y) - a_est) > 5 * eps:
            raise ValueError(f"Point {(x, y)} is not on x+y≈{a_est} within tolerance.")
    q2idx = {_q2(x, y, eps): i for i, (x, y) in enumerate(pts)}

    # Convert step lengths (Euclidean) to Δx along the line (Δy = -Δx)
    # s >= 0 means a geometric distance; negative steps (if supplied) are respected as direction.
    dxs_raw = []
    for s in steps:
        s = float(s)
        if s == 0:
            continue
        dx = s / sqrt(2.0)
        dxs_raw.append(dx)

    out_edges = [[] for _ in pts]
    in_edges  = [[] for _ in pts]

    # Helper: add edge if destination exists
    def try_edge(i: int, dx: float):
        x, y = pts[i]
        if forward_only and dx < -eps:
            return
        x2, y2 = x + dx, y - dx  # stays on x+y=a
        j = q2idx.get(_q2(x2, y2, eps))
        if j is not None:
            out_edges[i].append(j)
            in_edges[j].append(i)

    for i in range(len(pts)):
        if forward_only:
            # Only use the given dxs (assume caller supplies positive s for forward),
            # but allow tiny negative from rounding
            for dx in dxs_raw:
                try_edge(i, dx)
        else:
            # Both directions for each geometric step length
            for dx in dxs_raw:
                try_edge(i, dx)
                try_edge(i, -dx)

    return pts, q2idx, out_edges, in_edges, a_est

def enumerate_paths_2d(points_xy: Iterable[Tuple[float, float]],
                       steps: Iterable[float],
                       start: Tuple[float, float],
                       target: Optional[Tuple[float, float]] = None,
                       line_sum: Optional[float] = None,
                       eps: float = 1e-6,
                       forward_only: bool = True,
                       max_paths: Optional[int] = None,
                       return_steps: bool = False) -> List[List[Tuple[float, float]]]:
    """
    Enumerate all paths over 2D points lying on x+y=a, taking Euclidean step lengths in `steps`.
    If `target` is provided: start -> target. Else: start -> all sinks (no outgoing edges).

    Parameters
    ----------
    points_xy   : iterable of (x, y) floats (must satisfy x+y≈a)
    steps       : iterable of allowed Euclidean step lengths (floats)
    start       : (x, y) start point (must be in points within eps)
    target      : optional (x, y) target point (must be in points within eps if given)
    line_sum    : optional 'a' in x+y=a; if None, estimated from data
    eps         : coordinate tolerance for matching points (also used in line check)
    forward_only: if True, only allow edges with increasing x (DAG, fast DP)
    max_paths   : cap on number of paths to reconstruct (None = no cap)
    return_steps: if True, return sequences of Euclidean step lengths instead of points

    Returns
    -------
    List of paths; each path is a list of (x, y) points (or list of step lengths if return_steps=True).
    """
    pts, q2idx, out_edges, in_edges, a_est = build_adj_2d(
        points_xy, steps, line_sum=line_sum, eps=eps, forward_only=forward_only
    )

    def must_have_xy(p: Tuple[float, float], name: str) -> int:
        k = q2idx.get(_q2(p[0], p[1], eps))
        if k is None:
            raise ValueError(f"{name}={p} not found in points (within eps={eps}).")
        return k

    s_idx = must_have_xy(start, "start")
    t_idx = must_have_xy(target, "target") if target is not None else None

    # If forward_only, we can DP (topo order by increasing x)
    if forward_only:
        n = len(pts)
        order = list(range(n))  # already sorted by x asc
        ways = [0] * n
        ways[s_idx] = 1
        parents = [[] for _ in range(n)]

        for i in order:
            if ways[i] == 0:
                continue
            for j in out_edges[i]:
                parents[j].append(i)
                ways[j] += ways[i]

        targets = [t_idx] if t_idx is not None else [i for i in range(n) if not out_edges[i]]
        results: List = []

        def backtrack(cur: int, buf: List[int]):
            if max_paths is not None and len(results) >= max_paths:
                return
            if cur == s_idx:
                idx_path = buf + [cur]
                idx_path.reverse()
                if return_steps:
                    # Euclidean step lengths between successive points
                    step_seq = []
                    for k in range(len(idx_path) - 1):
                        x1, y1 = pts[idx_path[k]]
                        x2, y2 = pts[idx_path[k + 1]]
                        # On the line, distance is sqrt(2)*|Δx|
                        step_seq.append(abs(x2 - x1) * sqrt(2.0))
                    results.append(step_seq)
                else:
                    results.append([pts[k] for k in idx_path])
                return
            for p in parents[cur]:
                backtrack(p, buf + [cur])

        for tgt in targets:
            if tgt is None or ways[tgt] == 0:
                continue
            backtrack(tgt, [])
        return results

    # General (possibly cyclic) case: enumerate simple paths with DFS
    results: List = []
    visited = set()

    def dfs(u: int, path: List[int]):
        if max_paths is not None and len(results) >= max_paths:
            return
        visited.add(u)
        path.append(u)

        is_target = (t_idx is not None and u == t_idx)
        is_sink = (t_idx is None and len(out_edges[u]) == 0)
        if is_target or is_sink:
            if return_steps:
                step_seq = []
                for k in range(len(path) - 1):
                    x1, y1 = pts[path[k]]
                    x2, y2 = pts[path[k + 1]]
                    step_seq.append(abs(x2 - x1) * sqrt(2.0))
                results.append(step_seq)
            else:
                results.append([pts[i] for i in path])

        for v in out_edges[u]:
            if v not in visited:
                dfs(v, path)

        path.pop()
        visited.remove(u)

    dfs(s_idx, [])
    return results

print(df_current[['correct_mass1', 'correct_mass2', 'conserve']])

points_xy = []
for index, row in df_current.iterrows():
    #print(row['ion'])
    if row['conserve']:
        print(row['ion'])
        points_xy.append((row['correct_mass1'], row['correct_mass2']))
        print((row['correct_mass1'], row['correct_mass2']))
    else:
        print(row['ion'])
        points_xy.append((row['correct_mass1'] + 17.031, row['correct_mass2']))
        points_xy.append((row['correct_mass1'], row['correct_mass2'] + 17.031))
        print((row['correct_mass1'] + 17.031, row['correct_mass2']))
        print((row['correct_mass1'], row['correct_mass2'] + 17.031))
        
all_values = np.array(list(amino_acid_masses.values()))
all_values = all_values * np.sqrt(2)


paths = enumerate_paths_2d(points_xy, all_values, starting_point, return_steps=False, eps=0.2)

def track_path(the_path):
    result = []
    for i in the_path:
        each_result1 = ''
        each_result2 = ''
        for j in range(1, len(i)):
            mass = i[j][0] - i[j-1][0]
            for aa, aa_mass in amino_acid_masses.items():
                if abs(mass - aa_mass) < 0.01:
                    each_result1 += aa
                    break
        
        for j in range(len(i) - 1, 0, -1):
            mass = i[j - 1][1] - i[j][1]
            for aa, aa_mass in amino_acid_masses.items():
                if abs(mass - aa_mass) < 0.01:
                    each_result2 += aa
                    break
        each_result1 = str(i[0][0]), each_result1, str(i[-1][0])
        each_result2 = str(i[-1][1]), each_result2, str(i[0][1])
        result.append(each_result1)
        result.append(each_result2)
    
    return result
            
path_combined = track_path(paths)

print(paths)

print(path_combined)







