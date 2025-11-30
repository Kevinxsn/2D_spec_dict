import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# Directory where *this script* is located
current_dir = Path(__file__).resolve().parent

# Parent directory of the script
parent_dir = current_dir.parent

# Directories containing your modules
vis_dir = parent_dir / "vis"
connected_graphs_dir = parent_dir / "vis_connect"

# Put them at the front of sys.path so they are found first
sys.path.insert(0, str(vis_dir))
sys.path.insert(0, str(connected_graphs_dir))

import data_parse
import util
import peptide
import pandas as pd
import numpy as np
import connected_graph




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
        "I": 113.08406,  # Isoleucine
        "L": 113.08406,  # Leucine
        "K": 128.09496,  # Lysine
        "M": 131.04049,  # Methionine
        "F": 147.06841,  # Phenylalanine
        "P": 97.05276,   # Proline
        "S": 87.03203,   # Serine
        "T": 101.04768,  # Threonine
        "W": 186.07931,  # Tryptophan
        "Y": 163.06333,  # Tyrosine
        "V": 99.06841    # Valine
    }


amino_acid_masses_merge = {
        "A": 71.03711,   # Alanine
        "R": 156.10111,  # Arginine
        "N": 114.04293,  # Asparagine
        "D": 115.02694,  # Aspartic acid
        "C": 103.00919,  # Cysteine
        "E": 129.04259,  # Glutamic acid
        "Q": 128.05858,  # Glutamine
        "G": 57.02146,   # Glycine
        "H": 137.05891,  # Histidine
        "113(I/L)": 113.08406,  # Isoleucine
        #"L": 113.08406,  # Leucine
        "K": 128.09496,  # Lysine
        "M": 131.04049,  # Methionine
        "F": 147.06841,  # Phenylalanine
        "P": 97.05276,   # Proline
        "S": 87.03203,   # Serine
        "T": 101.04768,  # Threonine
        "W": 186.07931,  # Tryptophan
        "Y": 163.06333,  # Tyrosine
        "V": 99.06841    # Valine
    }

def create_fake_pairs(the_peptide):
    the_pep = peptide.Pep(the_peptide)
    pair_result = []
    result = []
    for i in range(1, len(the_pep.AA_array)):
        frag1 = the_pep.AA_array[:i]
        frag2 = the_pep.AA_array[i:]
        mass1 = sum([i.get_mass() for i in frag1])
        mass2 = sum([i.get_mass() for i in frag2])
        mass2 += 18.01056  # Adding H2O mass to the second fragment
        print(f"Fragment 1: {frag1}, Mass: {mass1:.4f}")
        print(f"Fragment 2: {frag2}, Mass: {mass2:.4f}")
        pair_result.append([mass1, mass2]) 
        result.append(mass1)
        result.append(mass2)
    result.sort()
    
    return result, pair_result


def get_pep_mass(seq):
    mass= 0
    for i in seq:
        mass += amino_acid_masses.get(i, 0)
    mass += 18.01056  # Adding H2O mass
    return mass

def find_peptide_paths(spectrum, allowed_masses=None, tolerance=0.02, start_point=(0.0, 18.01056)):
    """
    Finds all valid paths in the PSP graph, usually starting at (0, Water_Mass).
    
    Args:
        spectrum: A list or set of masses (e.g., {0.0, 18.01, 75.03...})
        allowed_masses: A list of valid jump sizes (e.g., amino acid masses). 
        tolerance: The allowable difference (delta) to consider a match valid.
        start_point: Tuple (x1, x2) to start search. Default is (0, Mass_H2O).
    """
    
    # 1. Setup
    S = sorted(list(set(spectrum)))
    
    # Helper: Find values in S that are within 'tolerance' of 'target'
    def get_matches_in_spectrum(target_val):
        return [s for s in S if abs(s - target_val) <= tolerance]

    # --- Start Node Logic ---
    target_x1, target_x2 = start_point
    
    matches_x1 = get_matches_in_spectrum(target_x1)
    matches_x2 = get_matches_in_spectrum(target_x2)
    
    if not matches_x1:
        print(f"Warning: Start value x1={target_x1} not found in spectrum (within tol={tolerance}).")
        return []
    if not matches_x2:
        print(f"Warning: Start value x2={target_x2} not found in spectrum (within tol={tolerance}).")
        return []
        
    # Generate all valid start combinations from the fuzzy matches
    start_nodes = [(m1, m2) for m1 in matches_x1 for m2 in matches_x2]

    # If no masses provided, use float versions of your example
    if allowed_masses is None:
        allowed_masses = [57.021, 71.037, 87.032, 97.053] # Gly, Ala, Ser, Pro (examples)

    all_paths = []

    # 2. Recursive DFS Function
    def dfs(current_path):
        current_node = current_path[-1] # (x1, x2)
        x1, x2 = current_node
        current_max = max(x1, x2)
        
        found_extension = False
        
        # Try all possible mass jumps
        for m in allowed_masses:
            
            # --- Check "Down ↓" (Increase x1) ---
            target_x1_next = x1 + m
            matches_x1_next = get_matches_in_spectrum(target_x1_next)
            
            for s_next in matches_x1_next:
                next_node = (s_next, x2)
                
                # Growth Condition: max(new) must be > max(old)
                if max(next_node) <= current_max:
                    continue
                    
                found_extension = True
                dfs(current_path + [next_node])

            # --- Check "Right →" (Increase x2) ---
            target_x2_next = x2 + m
            matches_x2_next = get_matches_in_spectrum(target_x2_next)
            
            for s_next in matches_x2_next:
                next_node = (x1, s_next)
                
                if max(next_node) <= current_max:
                    continue

                found_extension = True
                dfs(current_path + [next_node])

        # If we can't extend further, this path is complete (or dead end)
        if not found_extension:
            all_paths.append(current_path)

    # Launch search from all valid start nodes
    for start_node in start_nodes:
        dfs([start_node])
    
    return all_paths

def format_path_string(path, with_aa= False):
    """
    Helper to turn a list of nodes [(0,0), (0,2)...] into the arrow string format
    Rounds numbers for cleaner display.
    """
    if not path: return ""
    
    def fmt_node(n):
        return f"({round(n[0], 3)}, {round(n[1], 3)})"
    
    output = fmt_node(path[0])
    
    for i in range(len(path) - 1):
        curr = path[i]
        next_n = path[i+1]
        
        # Determine direction
        # We use a small epsilon for direction check due to float precision,
        # though standard inequality usually works fine.
        if next_n[0] > curr[0]:
            direction = "Down ↓"
        else:
            direction = "Right →"
        if with_aa:
            if direction == "Down ↓":
                mass = round(next_n[0] - curr[0], 3)
            else:
                mass = round(next_n[1] - curr[1], 3)
            aa = None
            for key, value in amino_acid_masses.items():
                if abs(value - mass) <= 0.01:  # Allow small tolerance for matching
                    aa = key
                    break
            output += f" {direction}({aa}) {fmt_node(next_n)}"
        else:
        
            output += f" {direction} {fmt_node(next_n)}"
        
    return output

def format_path_string_no_aa(path, with_aa= False):
    """
    Helper to turn a list of nodes [(0,0), (0,2)...] into the arrow string format
    Rounds numbers for cleaner display.
    """
    if not path: return ""
    
    def fmt_node(n):
        return f"({round(n[0], 3)}, {round(n[1], 3)})"
    
    output = fmt_node(path[0])
    
    for i in range(len(path) - 1):
        curr = path[i]
        next_n = path[i+1]
        
        # Determine direction
        # We use a small epsilon for direction check due to float precision,
        # though standard inequality usually works fine.
        if next_n[0] > curr[0]:
            direction = "Down ↓"
        else:
            direction = "Right →"
        if with_aa:
            if direction == "Down ↓":
                mass = round(next_n[0] - curr[0], 3)
            else:
                mass = round(next_n[1] - curr[1], 3)
            aa = None
            for key, value in amino_acid_masses.items():
                if abs(value - mass) <= 0.05:  # Allow small tolerance for matching
                    aa = key
                    break
            output += f" {direction}({aa}) {fmt_node(next_n)}"
        else:
        
            output += f" {direction} {fmt_node(next_n)}"
        
    return output


AA_MASSES = connected_graph.AA_MASSES
DOUBLE_AA_MASSES = connected_graph.DOUBLE_AA_MASSES
TRIPLE_AA_MASSES = connected_graph.TRIPLE_AA_MASSES
QUADRA_AA_MASSES = connected_graph.QUADRA_AA_MASSES


MASTER_MASS_MAP = {
    **AA_MASSES,
    **DOUBLE_AA_MASSES,
    **TRIPLE_AA_MASSES,
    **QUADRA_AA_MASSES
}

def find_possible_labels(target_mass: float, tol: float = 0.01):
    """
    Given a mass, return all keys whose values fall within ±tol.

    Returns:
        List of (label, mass) pairs.
    """
    matches = []
    for label, mass in MASTER_MASS_MAP.items():
        if abs(mass - target_mass) <= tol:
            matches.append((label, mass))
    if len(matches) == 0:
        #return ['?']
        return tuple([target_mass])
    else:
        return tuple(matches)
    
    
def path_to_seq_multiple_edges(path, seq_mass, thresh = 0.1):
    """
    Helper to turn a list of nodes [(0,0), (0,2)...] into the arrow string format
    Rounds numbers for cleaner display.
    """
    if not path: return ""
    forward = []
    backward = []
    middle = None
    def fmt_node(n):
        return f"({round(n[0], 3)}, {round(n[1], 3)})"
    
    output = fmt_node(path[0])
    
    for i in range(len(path) - 1):
        curr = path[i]
        next_n = path[i+1]
        
        # Determine direction
        # We use a small epsilon for direction check due to float precision,
        # though standard inequality usually works fine.
        if next_n[0] > curr[0]:
            direction = "Down ↓"
    
        else:
            direction = "Right →"

        if direction == "Down ↓":
            mass = round(next_n[0] - curr[0], 3)
        else:
            mass = round(next_n[1] - curr[1], 3)
        
        possible_labels = find_possible_labels(mass, tol=thresh)
        
        output += f" {direction}({possible_labels}) {fmt_node(next_n)}"
        if direction == "Down ↓":
            forward.append(possible_labels)
        else:
            backward.append(possible_labels)
    backward.reverse()
    
    the_middle_diff = seq_mass - (path[-1][0] + path[-1][1])
    
    
    possible_middle_labels = find_possible_labels(the_middle_diff, tol=thresh)
    
    

    #full_seq = "".join(forward)
    #full_seq += f"({possible_middle_labels})"
    #full_seq += "".join(backward)
    
    full_seq = []
    full_seq.extend(forward)
    full_seq.append(possible_middle_labels)
    full_seq.extend(backward)
    
    return tuple(full_seq)

def format_path_to_seq(the_set):
    result = ''
    for ind, each_set in enumerate(the_set):
        result += f'Path {ind+1}:\n'
        for i, each in enumerate(each_set):
            #print(each)
            result += f'edge{i+1}' + ': ' + ' \n '.join([str(x) for x in each]) + '\n'
    return result


## visulization 

def visualize_complete_graph(spectrum, spurious_masses=None, 
                             candidate_paths=None, correct_path=None, 
                             tolerance=0.02, aa_map=None):
    
    # 1. Setup & Classification
    S = sorted(list(set(spectrum)))
    S_arr = np.array(S)
    
    spurious_set = set(spurious_masses) if spurious_masses else set()
    def is_spurious(val): return val in spurious_set

    fig, ax = plt.subplots(figsize=(16, 16))

    # 2. Draw Grid Lines
    for mass in S:
        if is_spurious(mass):
            c, a, ls = 'salmon', 0.25, '--'
        else:
            c, a, ls = 'gray', 0.2, ':'
        ax.axvline(x=mass, color=c, linestyle=ls, linewidth=0.5, alpha=a)
        ax.axhline(y=mass, color=c, linestyle=ls, linewidth=0.5, alpha=a)

    # 3. Draw Nodes
    X, Y = np.meshgrid(S, S)
    is_noise_mask = np.array([is_spurious(m) for m in S])
    X_noise, Y_noise = np.meshgrid(is_noise_mask, is_noise_mask)
    node_is_noise = X_noise | Y_noise
    
    ax.scatter(X[node_is_noise], Y[node_is_noise], s=15, c='pink', alpha=0.4, zorder=1)
    ax.scatter(X[~node_is_noise], Y[~node_is_noise], s=25, c='lightblue', alpha=0.5, zorder=2)

    # 4. Helpers (Snapping & Annotation)
    def get_closest_in_spectrum(val):
        idx = (np.abs(S_arr - val)).argmin()
        closest = S_arr[idx]
        return closest if abs(closest - val) <= tolerance else val

    def get_snapped_path(raw_path):
        return [(get_closest_in_spectrum(px), get_closest_in_spectrum(py)) for px, py in raw_path]

    def get_anno_text(mass_diff):
        txt = f"{mass_diff:.2f}"
        if aa_map:
            min_diff = tolerance + 1e-9
            best_name = None
            for aa_mass, name in aa_map.items():
                diff = abs(mass_diff - aa_mass)
                if diff <= tolerance and diff < min_diff:
                    min_diff = diff
                    best_name = name
            if best_name: txt = f"{best_name}\n({mass_diff:.2f})"
        return txt

    # 5. Draw Candidate Paths
    if candidate_paths:
        for path in candidate_paths:
            if not path or len(path) < 2: continue
            snapped_path = get_snapped_path(path)
            p_xs, p_ys = zip(*snapped_path)
            ax.plot(p_xs, p_ys, color='purple', linewidth=1.5, alpha=0.3, linestyle='--', zorder=10)
            ax.scatter(p_xs, p_ys, s=20, color='purple', alpha=0.3, zorder=10)

    # 6. Draw Correct Path
    if correct_path and len(correct_path) > 1:
        snapped_correct = get_snapped_path(correct_path)
        start = snapped_correct[0]
        ax.scatter([start[0]], [start[1]], color='red', s=300, edgecolors='black', zorder=50)

        for i in range(len(snapped_correct) - 1):
            p_start = snapped_correct[i]
            p_end = snapped_correct[i+1]
            
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                    color='lime', linewidth=3.5, alpha=0.9, linestyle='-', 
                    marker='o', markersize=9, markerfacecolor='white', markeredgecolor='green', markeredgewidth=2,
                    zorder=30)

            dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
            mass_jump = max(abs(dx), abs(dy))
            mid_x, mid_y = (p_start[0] + p_end[0]) / 2, (p_start[1] + p_end[1]) / 2
            
            ax.text(mid_x, mid_y, get_anno_text(mass_jump), 
                    ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='green', alpha=0.9, lw=1),
                    zorder=40)
        
        end = snapped_correct[-1]
        ax.scatter([end[0]], [end[1]], color='gold', marker='*', s=400, edgecolors='black', linewidth=2, zorder=55)

    # --- 7. AXIS FORMATTING & COLORING FIX ---
    
    # Set tick locations explicitly
    ax.set_xticks(S)
    ax.set_yticks(S)
    
    # Set text labels
    x_labels_text = [f"{val:.2f}" for val in S]
    y_labels_text = [f"{val:.2f}" for val in S]
    ax.set_xticklabels(x_labels_text, rotation=90, fontsize=9)
    ax.set_yticklabels(y_labels_text, fontsize=9)
    
    # Move X Axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # --- THE FIX IS HERE ---
    # We iterate over the 'major ticks' directly to access label2 (top label)
    
    # X-Axis Coloring
    x_ticks = ax.xaxis.get_major_ticks()
    for tick, val in zip(x_ticks, S):
        if is_spurious(val):
            # Because tick_top() is on, the visible label is 'label2'
            tick.label2.set_color('salmon')
            tick.label2.set_fontweight('bold')
        else:
            tick.label2.set_color('black')

    # Y-Axis Coloring
    y_ticks = ax.yaxis.get_major_ticks()
    for tick, val in zip(y_ticks, S):
        if is_spurious(val):
            # Y-axis is standard (left), so visible label is 'label1'
            tick.label1.set_color('salmon')
            tick.label1.set_fontweight('bold')
        else:
            tick.label1.set_color('black')

    # Standard Axis Config
    pad = 5
    ax.set_xlim(min(S) - pad, max(S) + pad)
    ax.set_ylim(min(S) - pad, max(S) + pad)
    ax.invert_yaxis()
    ax.set_xlabel("Mass 1 (x1)", labelpad=10)
    ax.set_ylabel("Mass 2 (x2)")

    # Legend
    custom_lines = [
        Line2D([0], [0], color='lime', lw=3.5, label='Correct Path'),
        Line2D([0], [0], color='purple', lw=1.5, linestyle='--', alpha=0.5, label='Candidate Paths'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', label='Real Signal Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', label='Spurious Noise Node'),
        Line2D([0], [0], color='salmon', lw=2, label='Spurious Axis Label')
    ]
    ax.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title("Mass Spectrometry Graph Search\n(Color Coded Axes & Nodes)", pad=40, fontsize=16)
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    data = 'ME9_2+'
    csv_data = f"{data}.csv"
    file_path = f"/Users/kevinmbp/Desktop/2D_spec_dict/data/Top_Correlations_At_Full_Num_Scans_PCov/annotated/{csv_data}"
    file_path = os.path.abspath(file_path) 
    sequence = util.name_ouput(csv_data)
    pep = peptide.Pep(sequence)
    the_length = len(pep.AA_array)
    csv_data = file_path
    df = pd.read_csv(csv_data)
    df = df[df['Index'].notna()]
    results = data_parse.process_ion_dataframe(df.head(50), pep)
    results['classification'] = results.apply(data_parse.data_classify, args=(pep,), axis=1)
    the_list = []
    the_y_list = []

    results['loss1'] = results['loss1'].replace({None: np.nan})
    results['loss2'] = results['loss2'].replace({None: np.nan})
    df = results
    df['ranking'] = df['Index']
    
    LETTER_ORDER = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    rows = ['Parent','(NH3)','(H2O)', '(NH3)-(H2O)','(H2O)-(NH3)', 'a', '2(H2O)', '2(NH3)', '(H3PO4)']
    conserve_line_mass_dict = {'Parent': pep.pep_mass, 'a': pep.pep_mass - 28.0106}

    def classify_conserve_line(row):
        the_mass = row['chosen_sum']
        for i in conserve_line_mass_dict:
            if the_mass < conserve_line_mass_dict[i] + 1 and the_mass > conserve_line_mass_dict[i] - 1:
                return i
        else:
            return None

    df['conserve_line'] = df.apply(classify_conserve_line, axis = 1)
    my_peaks, sequence, pep = connected_graph.build_mass_list(data)
    
    sorted_array = [0.0] + my_peaks + [pep.seq_mass]
    
    #print("Sorted Array:", sorted_array)
    #print(df[['chosen_sum', 'ranking', 'classification', 'conserve_line', 'y_mz', 'b_mz']])
    
    mid_point = len(sorted_array) // 2
    lower_half = sorted_array[:mid_point]
    upper_half = sorted_array[mid_point:]
    lower_half_modified = lower_half + [18.01056]
    lower_half_modified.sort()
    
    allowed_mass_list = list(AA_MASSES.values()) + list(DOUBLE_AA_MASSES.values()) + list(TRIPLE_AA_MASSES.values()) #+ list(QUADRA_AA_MASSES.values())
    merge_close_values = connected_graph.merge_close_values
    allowed_mass_list = merge_close_values(allowed_mass_list, 0.1)
    
    paths = find_peptide_paths(
        lower_half_modified, 
        allowed_masses=allowed_mass_list, 
        tolerance=0.02,
        start_point=(0.0, 18.01056)
    )
    
    #for p in paths:
    #    print(format_path_string_no_aa(p, with_aa=True))
    
    the_max = max([len(p) for p in paths])
    the_max_length_num = sum([1 for p in paths if len(p) == the_max])
    the_max_length_paths = set([tuple(p) for p in paths if len(p) == the_max])

    the_max_length_pep = set([format_path_string_no_aa(p) for p in paths if len(p) == the_max])
    
    for p in the_max_length_paths:
        print(p)


    print(len(paths), "paths found.", "Max length:", the_max, 'There are', len(the_max_length_pep), "paths of max length.")
    print("Max length paths:")
    the_max_length_pep = set([path_to_seq_multiple_edges(p, pep.seq_mass + 18.01056) for p in paths if len(p) == the_max])
    
    print(format_path_to_seq(the_max_length_pep))
    
    real_spectrum = lower_half_modified
    noise = []