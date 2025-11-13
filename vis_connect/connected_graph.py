import pandas as pd
import numpy as np
import re
import peptide
import dataframe_image as dfi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import HTML
from html2image import Html2Image
import util
import os
import data_parse
import matplotlib.patches as mpatches
import neutral_loss_mass
from matplotlib.patches import Arc
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple, Literal


proton = 1.007276
RepMethod = Literal["first", "last", "mean", "median"]

def merge_close_values(sorted_array, threshold, method="first"):
    """
    Merge consecutive float values in a sorted list if their differences
    are smaller than the threshold.

    Parameters
    ----------
    sorted_array : list[float]
        Sorted list of float values.
    threshold : float
        Maximum allowed difference between values to be merged.
    method : str, optional
        How to choose the representative value of each merged group:
        - "first" : keep the first value in the group
        - "last"  : keep the last value in the group
        - "mean"  : keep the average of all values in the group
        - "median": keep the median of all values in the group

    Returns
    -------
    list[float]
        List of merged float values.
    """
    if not sorted_array:
        return []

    merged = []
    group = [sorted_array[0]]

    for x in sorted_array[1:]:
        if abs(x - group[-1]) <= threshold:
            group.append(x)
        else:
            # finalize previous group
            if method == "mean":
                merged.append(sum(group) / len(group))
            elif method == "median":
                n = len(group)
                mid = n // 2
                merged.append((group[mid] if n % 2 else (group[mid-1] + group[mid]) / 2))
            elif method == "last":
                merged.append(group[-1])
            else:  # "first"
                merged.append(group[0])
            group = [x]

    # finalize last group
    if method == "mean":
        merged.append(sum(group) / len(group))
    elif method == "median":
        n = len(group)
        mid = n // 2
        merged.append((group[mid] if n % 2 else (group[mid-1] + group[mid]) / 2))
    elif method == "last":
        merged.append(group[-1])
    else:
        merged.append(group[0])

    return merged
 

AA_MASSES = {
    'G': 57.021464, 'A': 71.037114, 'S': 87.032028, 'P': 97.052764,
    'V': 99.068414, 'T': 101.047679, 'C': 103.009185, '(I/L)': 113.084064,
    'N': 114.042927, 'D': 115.026943, 'Q': 128.058578,
    'K': 128.094963, 'E': 129.042593, 'M': 131.040485, 'H': 137.058912,
    'F': 147.068414, 'R': 156.101111, 'Y': 163.063329, 'W': 186.079313,
    'Me': 14.01565, 'Me2':28.03130, 'nitro': 44.98508, "Ac": 42.01056,
    'E(nitro)': 129.042593 + 44.98508
}
DOUBLE_AA_MASSES = {}
for aa1, aa2 in combinations_with_replacement(AA_MASSES.keys(), 2):
    mass_sum = AA_MASSES[aa1] + AA_MASSES[aa2]
    # Store with a combined label, e.g., "A+G"
    DOUBLE_AA_MASSES[f"{aa1}+{aa2}"] = mass_sum
    
def build_triple_aa_masses(AA_MASSES: dict) -> dict:
    """
    Return a dict like {"A+G+S": mass_sum, ...} for all 3-AA combos (with replacement).
    """
    triple = {}
    # sort keys for stable, deterministic labels
    keys = sorted(AA_MASSES.keys())
    for aa1, aa2, aa3 in combinations_with_replacement(keys, 3):
        label = f"{aa1}+{aa2}+{aa3}"
        triple[label] = AA_MASSES[aa1] + AA_MASSES[aa2] + AA_MASSES[aa3]
    return triple
TRIPLE_AA_MASSES = build_triple_aa_masses(AA_MASSES)

def build_quadra_aa_masses(AA_MASSES: dict) -> dict:
    """
    Return a dict like {"A+G+S+P": mass_sum, ...}
    for all 4-AA combinations (with replacement).
    """
    quad = {}
    keys = sorted(AA_MASSES.keys())   # deterministic order
    
    for aa1, aa2, aa3, aa4 in combinations_with_replacement(keys, 4):
        label = f"{aa1}+{aa2}+{aa3}+{aa4}"
        quad[label] = (
            AA_MASSES[aa1] +
            AA_MASSES[aa2] +
            AA_MASSES[aa3] +
            AA_MASSES[aa4]
        )
    return quad

QUADRA_AA_MASSES = build_quadra_aa_masses(AA_MASSES)


def cluster_mass_dict(
    name_to_mass: Dict[str, float],
    threshold: float,
    method: RepMethod = "mean",
) -> Tuple[Dict[str, float], Dict[str, str], List[List[Tuple[str, float]]]]:
    """
    Merge keys whose masses are within `threshold` (Da) of each other.
    Returns:
      merged_dict: { "A+B/C+D": representative_mass, ... }
      inverse_map: { "A+B": "A+B/C+D", "C+D": "A+B/C+D", ... }
      groups:      [[(name, mass), ...], ...]  # raw groups for inspection

    Notes
    -----
    - Input order doesn’t matter; clustering uses sorted-by-mass order.
    - Groups are formed by chaining: consecutive items whose DIFFERENCE
      from the previous in the sorted order is ≤ threshold end up in the same group.
    - `method` selects the representative mass for each merged key.
    """
    if not name_to_mass:
        return {}, {}, []

    # sort by mass, then by name for stability
    items = sorted(name_to_mass.items(), key=lambda kv: (kv[1], kv[0]))

    groups: List[List[Tuple[str, float]]] = []
    cur: List[Tuple[str, float]] = [items[0]]

    for name, mass in items[1:]:
        # Compare to the last element in the current group (consecutive proximity)
        if abs(mass - cur[-1][1]) <= threshold:
            cur.append((name, mass))
        else:
            groups.append(cur)
            cur = [(name, mass)]
    groups.append(cur)

    def representative(vals: List[float]) -> float:
        if method == "first":
            return vals[0]
        if method == "last":
            return vals[-1]
        if method == "median":
            n = len(vals)
            mid = n // 2
            return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
        # default: mean
        return sum(vals) / len(vals)

    merged_dict: Dict[str, float] = {}
    inverse_map: Dict[str, str] = {}

    for group in groups:
        names = [n for n, _ in group]
        masses = [m for _, m in group]
        merged_key = "/".join(names)   # e.g., "A+B/C+D/..."
        rep_mass = representative(masses)
        merged_dict[merged_key] = rep_mass
        for n in names:
            inverse_map[n] = merged_key

    return merged_dict, inverse_map, groups

    
def find_all_connections(peaks, tolerance=0.005, merge_double = False):
    """
    Identifies pairs of peaks separated by single AA mass OR double AA mass.
    Returns two separate lists of connections: (single_conns, double_conns)
    """
    peaks = sorted(peaks)
    single_conns = []
    double_conns = []
    triple_conns = []
    quodra_conns = []
    if merge_double:
        double_dict, inv_map, groups = cluster_mass_dict(DOUBLE_AA_MASSES, threshold=tolerance, method="mean")
        triple_dict, inv_map, groups = cluster_mass_dict(TRIPLE_AA_MASSES, threshold=tolerance, method="mean")
        quodra_dict, inv_map, groups = cluster_mass_dict(QUADRA_AA_MASSES, threshold=tolerance, method="mean")
    else:
        double_dict = DOUBLE_AA_MASSES
        triple_dict = TRIPLE_AA_MASSES
        quodra_dict = QUADRA_AA_MASSES
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            mass_diff = peaks[j] - peaks[i]

            # Optimization: Stop if diff is too large for even the largest double (Trp+Trp ~ 372)
            if mass_diff > 400:
                break

            # 1. Check for Single AA matches
            for aa, aa_mass in AA_MASSES.items():
                if abs(mass_diff - aa_mass) <= tolerance:
                    single_conns.append((peaks[i], peaks[j], aa))

            
            # 2. Check for Double AA matches
            for label, double_mass in double_dict.items():
                 if abs(mass_diff - double_mass) <= tolerance:
                    double_conns.append((peaks[i], peaks[j], label))
            
            for label, triple_mass in triple_dict.items():
                 if abs(mass_diff - triple_mass) <= tolerance:
                    triple_conns.append((peaks[i], peaks[j], label))
                    
            ## the quadra connection only exist from 0 or to entire_pepmass
            if i == 0 or j == len(peaks) - 1:
                for label, quodra_mass in quodra_dict.items():
                    if abs(mass_diff - quodra_mass) <= tolerance:
                        quodra_conns.append((peaks[i], peaks[j], label))

    return single_conns, double_conns, triple_conns, quodra_conns




def plot_complex_arc_graph(peaks, single_conns, double_conns, highlight_sequence=None, seq = '', show_graph = True, save_path = False):
    """
    Visualizes peaks with single AA arcs (above axis) and double AA arcs (below axis).
    Includes mass labels for each peak.
    
    Can optionally highlight a specific sequence of arcs (path) based on the order
    of amino acids in 'highlight_sequence', assuming a left-to-right path.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    default_color = 'steelblue'
    highlight_color = 'crimson'

    # --- NEW / MODIFIED: Pre-process highlight_sequence to identify specific arcs ---
    highlight_arc_set = set() # This set will store (start, end, label) for arcs to highlight
    if highlight_sequence is not None and len(highlight_sequence) > 0:
        # Create a mapping from (start, end, label) to arc type (single/double)
        # This is important because the labels can be the same but the connections different
        all_arcs_with_types = []
        for s, e, l in single_conns:
            all_arcs_with_types.append(((s, e, l), 'single'))
        for s, e, l in double_conns:
            all_arcs_with_types.append(((s, e, l), 'double'))
        
        # We need to link the 'peaks' (masses) to their positions in the graph
        # This will allow us to follow the path "from left to right"
        # Assuming 'peaks' are already sorted by mass (left to right)
        # If not, you might need to sort them or use their original indices.
        
        current_peak_index = 0
        # Iterate through possible starting peaks to find the full path
        for start_idx in range(len(peaks)):
            current_path_matched = []
            current_mass = peaks[start_idx]
            path_found = True

            for aa_label in highlight_sequence:
                # Look for an arc (single or double) that starts at current_mass
                # and has the amino acid label, and connects to the next peak.
                found_next_step = False
                for arc_tuple, arc_type in all_arcs_with_types:
                    arc_start, arc_end, arc_label = arc_tuple
                    # Check if the arc starts at our current mass, matches the label,
                    # and moves to a *greater* mass (left to right)
                    if np.isclose(arc_start, current_mass) and arc_label == aa_label and arc_end > current_mass:
                        current_path_matched.append(arc_tuple)
                        current_mass = arc_end
                        found_next_step = True
                        break # Found this step in the path, move to next AA
                
                if not found_next_step:
                    path_found = False
                    break # This path failed
            
            if path_found and len(current_path_matched) == len(highlight_sequence):
                # If a complete path is found, add all its arcs to the highlight set
                for arc_tuple in current_path_matched:
                    highlight_arc_set.add(arc_tuple)
                # We can choose to find only the first instance of the path, or all.
                # For "the path will always start from left to the right", finding the first
                # full path is usually sufficient unless there are multiple identical paths.
                # If multiple valid paths exist, this will highlight all of them.
    # --- End of NEW / MODIFIED section ---

    # Plot Nodes on the centerline (y=0)
    ax.scatter(peaks, np.zeros_like(peaks), color='black', s=40, zorder=5)
    
    num_peaks = len(peaks)
    for i, peak_mass in enumerate(peaks):
        if i < 15 or i >= (num_peaks - 3):
            label = f"{peak_mass:.2f}"
            ax.text(peak_mass, 0, label,
                    ha='right', va='top', fontsize=8,
                    color='#404040', rotation=45, zorder=6)
    
    max_upper_height = 1.0
    max_lower_height = 1.0

    # --- Plot SINGLE connections (MODIFIED to use highlight_arc_set) ---
    for start, end, label in single_conns:
        midpoint = (start + end) / 2
        width = end - start
        height = width * 0.5
        max_upper_height = max(max_upper_height, height)

        # Check if THIS specific arc (start, end, label) is in our highlight_arc_set
        is_highlighted = ((start, end, label) in highlight_arc_set)
        
        if is_highlighted:
            arc_color = highlight_color
            arc_lw = 2.2
            arc_zorder = 10
            text_color = highlight_color
            text_weight = 'extra bold'
        else:
            arc_color = default_color
            arc_lw = 1.5
            arc_zorder = 5
            text_color = 'darkblue'
            text_weight = 'bold'

        arc = Arc(xy=(midpoint, 0), width=width, height=height,
                  theta1=0, theta2=180, color=arc_color, alpha=0.7, 
                  lw=arc_lw, zorder=arc_zorder)
        ax.add_patch(arc)

        ax.text(midpoint, (height / 2) + (width * 0.02), label,
                ha='center', va='bottom', fontsize=9, color=text_color, weight=text_weight,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

    # --- Plot DOUBLE connections (MODIFIED to use highlight_arc_set) ---
    for start, end, label in double_conns:
        midpoint = (start + end) / 2
        width = end - start
        height = width * 0.5
        max_lower_height = max(max_lower_height, height)

        # Check if THIS specific arc (start, end, label) is in our highlight_arc_set
        is_highlighted = ((start, end, label) in highlight_arc_set)
        
        if is_highlighted:
            arc_color = highlight_color
            arc_lw = 2.2
            arc_zorder = 10
            text_color = highlight_color
            text_weight = 'extra bold'
        else:
            arc_color = default_color
            arc_lw = 1.5
            arc_zorder = 5
            text_color = 'darkred'
            text_weight = 'normal'

        arc = Arc(xy=(midpoint, 0), width=width, height=height,
                  theta1=180, theta2=360, color=arc_color, alpha=0.5, 
                  lw=arc_lw, linestyle='--', zorder=arc_zorder)
        ax.add_patch(arc)

        ax.text(midpoint, -(height / 2) - (width * 0.02), label,
                ha='center', va='top', fontsize=8, color=text_color, weight=text_weight,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

    # --- Formatting (Unchanged) ---
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Mass (m/z)", fontsize=12)
    ax.set_yticks([]) 
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    ax.set_ylim(-(max_lower_height * 1.2), (max_upper_height * 1.2))
    pad = (max(peaks) - min(peaks)) * 0.05
    ax.set_xlim(min(peaks) - pad, max(peaks) + pad)

    plt.title(f"Mass Spec Connectivity\nTop: Single AA ({len(single_conns)}) | Bottom: Double AA ({len(double_conns)}) \n {seq}", fontsize=14)
    plt.tight_layout()
    if show_graph:
        plt.show()
    if save_path:
        # Use bbox_inches='tight' to ensure the legend is included when saving.
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    


def build_mass_list(
    data: str,
    *,
    base_dir: str = None,
    head_n: int = 50,
    conserve_rows = ('Parent','(NH3)','(H2O)','(NH3)-(H2O)','(H2O)-(NH3)','a','2(H2O)','2(NH3)', 'H4PO3')
    ) -> list:
    """
    Given a dataset name like 'ME4_2+', read its annotated CSV, compute classifications,
    and return the unique sorted masses on the Parent conservation line.

    Parameters
    ----------
    data : str
        Base name without extension (e.g., 'ME4_2+').
    base_dir : str, optional
        Directory to resolve the CSV path from. Defaults to the directory of this file.
    head_n : int, optional
        Use only the first N rows of the CSV for processing (default 50).
    conserve_rows : iterable[str]
        Conservation-line labels to build the mass dictionary.

    Returns
    -------
    list[float]
        Sorted unique masses from 'correct_mass1' and 'correct_mass2' on the Parent line.
    """
    # Resolve file path
    csv_name = f"{data}.csv"
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    file_path = os.path.abspath(
        os.path.join(
            base_dir,
            "../data/Top_Correlations_At_Full_Num_Scans_PCov/annotated",
            csv_name,
        )
    )

    # Build peptide from sequence name parsed out of the csv_name
    sequence = util.name_ouput(csv_name)  # keep your existing helper name
    pep = peptide.Pep(sequence)
    pep_mass= pep.pep_mass

    # Read and pre-filter
    df = pd.read_csv(file_path)
    df = df[df["Index"].notna()]
    if head_n:
        df = df.head(head_n)

    # Domain processing + classification
    results = data_parse.process_ion_dataframe(df, pep)
    results["classification"] = results.apply(data_parse.data_classify, args=(pep,), axis=1)

    # Normalize loss columns
    results["loss1"] = results["loss1"].replace({None: np.nan})
    results["loss2"] = results["loss2"].replace({None: np.nan})

    # Build conservation-line mass dictionary
    conserve_line_mass_dict = {
        "Parent": pep.pep_mass,
        "a": pep.pep_mass - 28.0106,
    }
    for label in conserve_rows:
        if label not in conserve_line_mass_dict:
            conserve_line_mass_dict[label] = pep.pep_mass - neutral_loss_mass.mass_of_loss(label)

    # Tag rows by closest conservation line (±1 Da window)
    def classify_conserve_line(row):
        m = row["chosen_sum"]
        for label, target in conserve_line_mass_dict.items():
            if (target - 1) < m < (target + 1):
                return label
        return None

    results["conserve_line"] = results.apply(classify_conserve_line, axis=1)

    # Collect masses from the Parent line
    df_parent = results[results["conserve_line"] == "Parent"]
    # Suppose your DataFrame is named df
    df_parent = df_parent.dropna(subset=['charge1'])
    df_parent = df_parent.dropna(subset=['charge2'])
    print(df_parent)
    
    def mass_mul_charge(row):
        mass1_charge = None
        mass2_charge = None
        if row['charge1'][0] == '1':
            mass1_charge = row['correct_mass1']
        elif row['charge1'][0] == '2':
            mass1_charge = row['correct_mass1'] * 2 - proton

        if row['charge2'][0] == '1':
            mass2_charge = row['correct_mass2']
        elif row['charge2'][0] == '2':
            mass2_charge = row['correct_mass2'] * 2 - proton

        return mass1_charge, mass2_charge
        
            
    df_parent[['mass1_charge', 'mass2_charge']] = df_parent.apply(mass_mul_charge, axis=1, result_type='expand')
    print(df_parent[['ion1', 'ion2', 'mass1_charge', 'mass2_charge']])
    mass_list = list(set(df_parent["mass1_charge"].tolist() + df_parent["mass2_charge"].tolist()))
    mass_list.sort()
    return mass_list, f'{data}: {sequence}', pep_mass


# --- NEW: Pathfinding Function ---

def find_all_paths(start_peak, all_connections):
    """
    Finds all possible paths (sequences of labels) starting from a given
    start_peak, using the provided connections.

    Args:
        start_peak (float): The exact mass of the peak to start from.
                            Must be one of the peaks used to generate connections.
        all_connections (list): A *combined* list of all connection tuples
                                [(start, end, label), ...] to be considered.

    Returns:
        list of lists: A list where each inner list is a path (sequence of labels).
                       e.g., [['A', 'B'], ['A', 'C', 'D']]
    """
    
    # --- Step 1: Build the adjacency list (graph_map) ---
    # This map makes it fast to look up all outgoing edges from any peak.
    graph_map = {}
    for start, end, label in all_connections:
        if start not in graph_map:
            graph_map[start] = []
        # Store the destination peak and the label of the edge
        graph_map[start].append((end, label))

    # --- Step 2: Define the recursive DFS helper function ---
    def dfs_recursive(current_peak, current_path):
        """
        Recursively explores paths from the current_peak.
        'current_path' holds the labels taken to get here.
        'found_paths' is populated by the outer function.
        """
        
        # Find all outgoing connections from the current peak
        outgoing_edges = graph_map.get(current_peak, [])

        # --- Base Case ---
        # If there are no outgoing edges, this is the end of a path.
        if not outgoing_edges:
            # If the path is not empty, save a *copy* of it.
            if current_path:
                found_paths.append(list(current_path))
            return # Stop recursion for this branch

        # --- Recursive Step ---
        # Explore each outgoing edge
        for next_peak, label in outgoing_edges:
            # 1. "Take" this path: Add the label
            current_path.append(label)
            
            # 2. Recurse: Explore from the next peak
            dfs_recursive(next_peak, current_path)
            
            # 3. "Backtrack": Remove the label so the loop can
            #    explore the *next* outgoing edge from 'current_peak'.
            current_path.pop()

    # --- Step 3: Start the search ---
    found_paths = []
    # Call the recursive helper starting from the 'start_peak' with an empty path
    dfs_recursive(start_peak, [])
    
    return found_paths





if __name__ == "__main__":
    data = 'ME4_2+'
    my_peaks, sequence, pep_mass = build_mass_list(data)
    
    print(my_peaks)
    
    my_peaks = [0] + my_peaks + [pep_mass]
    my_peaks = merge_close_values(my_peaks, threshold=0.001)
    print(my_peaks)
    s_conns, d_conns, t_conns, q_conns = find_all_connections(my_peaks, tolerance=0.001, merge_double=True)
    
    print(q_conns)

    d_conns = d_conns + t_conns + q_conns

    # Define the sequence you want to find
    seq_to_find = ['A', '(I/L)', 'Q', '(I/L)', 'D', 'K+Ac/A+V/G+(I/L)', 'P', '(I/L)+M'] 
    # Call WITH the new argument
    plot_complex_arc_graph(my_peaks, s_conns, d_conns, highlight_sequence=seq_to_find, seq=sequence, show_graph=False, save_path=f'vis_connect/connected_graph/{data}.png')
    
    all_conns = s_conns + d_conns + t_conns
    start_from_peak = my_peaks[0]
    all_paths = find_all_paths(start_from_peak, all_conns)
    
    '''
    print(f"\n--- Paths found starting from {start_from_peak} ---")
    if all_paths:
        for i, path in enumerate(all_paths):
            print(f"Path {i+1}: {' -> '.join(path)}")
    else:
        print("No paths found from this starting peak.")
    '''
    