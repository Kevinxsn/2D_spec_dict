import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.patches as patches
import anti_symmetric_util as ay_util
import img2pdf

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

def visualize_all_paths(spectrum, spurious_masses=None, 
                              candidate_paths=None, correct_path=None, 
                              tolerance=0.02, aa_map=None, pep_mass=None,
                              title="Peptide Spectrum Graph",
                              save_path=None):
    """
    Visualizes paths with Edge Annotations AND End-Point Residual Annotations.
    
    Args:
        pep_mass (float): The total mass of the peptide. Used to calculate 
                          the residual at the end point (PepMass - x1 - x2).
    """
    
    # --- 1. Setup & Classification ---
    S = sorted(list(set(spectrum)))
    S_arr = np.array(S)
    
    spurious_set = set(spurious_masses) if spurious_masses else set()
    def is_spurious(val): return val in spurious_set

    # Create Figure
    fig, ax = plt.subplots(figsize=(18, 18))

    # --- 2. Draw Grid Lines ---
    for mass in S:
        if is_spurious(mass):
            c, a, ls = 'salmon', 0.25, '--'
        else:
            c, a, ls = 'gray', 0.2, ':'
        ax.axvline(x=mass, color=c, linestyle=ls, linewidth=0.5, alpha=a)
        ax.axhline(y=mass, color=c, linestyle=ls, linewidth=0.5, alpha=a)

    # --- 3. Draw Nodes ---
    X, Y = np.meshgrid(S, S)
    is_noise_mask = np.array([is_spurious(m) for m in S])
    X_noise, Y_noise = np.meshgrid(is_noise_mask, is_noise_mask)
    node_is_noise = X_noise | Y_noise
    
    ax.scatter(X[node_is_noise], Y[node_is_noise], s=15, c='pink', alpha=0.4, zorder=1)
    ax.scatter(X[~node_is_noise], Y[~node_is_noise], s=25, c='lightblue', alpha=0.5, zorder=2)

    # --- 4. Helpers ---
    def get_closest_in_spectrum(val):
        idx = (np.abs(S_arr - val)).argmin()
        closest = S_arr[idx]
        return closest if abs(closest - val) <= tolerance else val

    def get_snapped_path(raw_path):
        return [(get_closest_in_spectrum(px), get_closest_in_spectrum(py)) for px, py in raw_path]

    def get_anno_text(mass_diff):
        # Allow negative residuals for the end point check
        abs_diff = abs(mass_diff)
        txt = f"{mass_diff:.2f}"
        
        if aa_map:
            min_diff = tolerance + 1e-9
            best_name = None
            for aa_mass, name in aa_map.items():
                diff = abs(abs_diff - aa_mass)
                if diff <= tolerance and diff < min_diff:
                    min_diff = diff
                    best_name = name
            if best_name: 
                txt = f"{best_name}\n({mass_diff:.2f})"
        return txt

    # --- 5. Unified Drawing Function ---
    def draw_path_logic(path_data, line_color, box_color, text_color, z_order, is_hero=False):
        if not path_data or len(path_data) < 2: return

        snapped = get_snapped_path(path_data)
        
        # Style settings
        lw = 3.5 if is_hero else 1.5
        ls = '-' if is_hero else '--'
        alpha = 0.9 if is_hero else 0.6
        marker_size = 9 if is_hero else 6
        font_size = 9 if is_hero else 7
        
        # Draw Start
        start = snapped[0]
        s_size = 300 if is_hero else 100
        ax.scatter([start[0]], [start[1]], color='red', s=s_size, edgecolors='black', zorder=z_order+5)

        # Draw Edges
        for i in range(len(snapped) - 1):
            p_start = snapped[i]
            p_end = snapped[i+1]
            
            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                    color=line_color, linewidth=lw, alpha=alpha, linestyle=ls, 
                    marker='o', markersize=marker_size, markerfacecolor='white', 
                    markeredgecolor=line_color, markeredgewidth=1.5,
                    zorder=z_order)

            dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
            mass_jump = max(abs(dx), abs(dy))
            mid_x, mid_y = (p_start[0] + p_end[0]) / 2, (p_start[1] + p_end[1]) / 2
            
            ax.text(mid_x, mid_y, get_anno_text(mass_jump), 
                    ha='center', va='center', fontsize=font_size, 
                    color=text_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc=box_color, ec=line_color, alpha=0.85, lw=1),
                    zorder=z_order+10)

        # Draw End Node
        end = snapped[-1]
        e_size = 400 if is_hero else 150
        ax.scatter([end[0]], [end[1]], color='gold', marker='*', s=e_size, edgecolors='black', linewidth=1.5, zorder=z_order+15)
        
        # --- NEW: End Point Annotation (Residual) ---
        if pep_mass is not None:
            # Calculate what is missing: PepMass - (x1 + x2)
            residual = pep_mass - (end[0] + end[1])
            
            # Format the text (check if the missing part looks like an AA)
            res_text = get_anno_text(residual)
            label = f"Rem:\n{res_text}"
            
            # Position: Shift visually UP (va='bottom') so it sits on top of the star
            # We use a Gold/Orange box to distinguish it from edge annotations
            ax.text(end[0], end[1], label, 
                    ha='center', va='bottom', fontsize=font_size, 
                    color='darkgoldenrod', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gold', alpha=0.9, lw=2),
                    zorder=z_order+20)

    # --- 6. Draw Paths ---
    if candidate_paths:
        for path in candidate_paths:
            draw_path_logic(path, 
                            line_color='purple', box_color='lavender', text_color='indigo',
                            z_order=20, is_hero=False)

    if correct_path:
        draw_path_logic(correct_path, 
                        line_color='lime', box_color='white', text_color='darkgreen',
                        z_order=50, is_hero=True)

    # --- 7. Axis Formatting & Colors ---
    ax.set_xticks(S)
    ax.set_yticks(S)
    ax.set_xticklabels([f"{val:.2f}" for val in S], rotation=90, fontsize=9)
    ax.set_yticklabels([f"{val:.2f}" for val in S], fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    x_ticks = ax.xaxis.get_major_ticks()
    for tick, val in zip(x_ticks, S):
        if is_spurious(val):
            tick.label2.set_color('salmon')
            tick.label2.set_fontweight('bold')
        else:
            tick.label2.set_color('black')

    y_ticks = ax.yaxis.get_major_ticks()
    for tick, val in zip(y_ticks, S):
        if is_spurious(val):
            tick.label1.set_color('salmon')
            tick.label1.set_fontweight('bold')
        else:
            tick.label1.set_color('black')

    pad = 5
    ax.set_xlim(min(S) - pad, max(S) + pad)
    ax.set_ylim(min(S) - pad, max(S) + pad)
    ax.invert_yaxis()
    ax.set_xlabel("Mass 1 (x1)", labelpad=10)
    ax.set_ylabel("Mass 2 (x2)")

    # Legend
    custom_lines = [
        Line2D([0], [0], color='lime', lw=3.5, label='Correct Path'),
        Line2D([0], [0], color='purple', lw=1.5, linestyle='--', label='Candidate Path'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=10, label='End (w/ Residual)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', label='Spurious Node'),
        Line2D([0], [0], color='salmon', lw=2, label='Spurious Axis Label')
    ]
    ax.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.set_title(title, pad=40, fontsize=16)
    
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Graph saved successfully to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()










if __name__ == "__main__":
    data = 'test4_3+'
    csv_data = f"{data}.csv"
    file_path = f"/Users/kevinmbp/Desktop/2D_spec_dict/data/Top_Correlations_At_Full_Num_Scans_PCov/annotated/{csv_data}"
    file_path = os.path.abspath(file_path) 
    sequence = util.name_ouput(csv_data)
    pep = peptide.Pep(sequence)
    print(pep.AA_array)
    the_length = len(pep.AA_array)
    csv_data = file_path
    df = pd.read_csv(csv_data)
    df = df[df['Index'].notna()]
    #results = data_parse.process_ion_dataframe(df.head(50), pep)
    results = data_parse.process_ion_dataframe(df.head(62), pep)
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
            if the_mass < conserve_line_mass_dict[i] + 1.1 and the_mass > conserve_line_mass_dict[i] - 1.1:
                return i
        else:
            return None

    df['conserve_line'] = df.apply(classify_conserve_line, axis = 1)
    print(df[['conserve_line', 'chosen_sum']].head(30))
    
    my_peaks, sequence, pep, paired_peaks, paired_mass_dict = connected_graph.build_mass_list_with_ion(data)
    
    
        #print('b_ion_list:', b_ion_list)

    #mass - 18 is also a mass of b ion
    paired_peaks = paired_peaks + [(pep.seq_mass, f'b{len(pep.AA_array)}')]
    b_ion_list = [i[1] for i in paired_peaks if i[1].startswith('b') and i[1][1] != 'i' and len(i[1]) < 4]

    
    sorted_array = [0.0] + my_peaks + [pep.seq_mass - 18.01056] + [pep.seq_mass]
    #print("Sorted Array:", sorted_array)
    #print(df[['chosen_sum', 'ranking', 'classification', 'conserve_line', 'y_mz', 'b_mz']])
    
    mid_point = len(sorted_array) // 2 + 2
    lower_half = sorted_array[:mid_point]
    upper_half = sorted_array[mid_point:]
    lower_half_modified = lower_half + [18.01056]
    lower_half_modified.sort()

    print('lower half:', lower_half_modified)
    
    allowed_mass_list = list(AA_MASSES.values()) + list(DOUBLE_AA_MASSES.values()) + list(TRIPLE_AA_MASSES.values()) #+ list(QUADRA_AA_MASSES.values())
    merge_close_values = connected_graph.merge_close_values
    allowed_mass_list = merge_close_values(allowed_mass_list, 0.01)
    
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
    print(lower_half_modified)
    noise = [246.161, 247.107, 353.240, 392.122]
    #noise = []
    full_spec = sorted(real_spectrum + noise)
    
    amino_acid_masses_switch = {v: k for k, v in amino_acid_masses_merge.items()}
    

    #correct = [(0.0, 18.01056), (0.0, 332.184804), (0.0, 419.216834), (484.254614, 419.216834), (484.254614, 566.285244), (613.297204, 566.285244), (613.297204, 679.369304), (742.339794, 679.369304)]
    #correct = [(0.0, 18.01056), (243.100744, 18.01056), (300.122204, 18.01056), (300.122204, 332.184804), (300.122204, 566.285244), (613.2971779999999, 566.285244), (613.2971779999999, 679.369304)]
    #candidates = [[(0.0, 18.01056), (0.0, 332.184804), (0.0, 419.216834), (484.254614, 419.216834), (484.254614, 566.285244), (613.297204, 566.285244), (613.297204, 679.369304), (742.339794, 679.369304)]]
    
    
    correct = [(0.0, 18.01056), (0.0, 259.189554), (0.0, 372.27361399999995), (394.196434, 372.27361399999995), (394.196434, 429.29507399999994), (394.196434, 486.31653399999993), (493.264844, 486.31653399999993), (493.264844, 599.4005679999999), (606.348904, 599.4005679999999)]
    #candidates = [list(p) for p in the_max_length_paths]
    candidates = [correct]
    
    #candidates = [correct,[(0.0, 18.01056), (243.100744, 18.01056), (300.122204, 18.01056), (300.122204, 332.184804), (557.2572246963999, 332.184804), (557.2572246963999, 566.285244), (557.2572246963999, 679.369304)]]
    #candidates = [correct,[(0.0, 18.01056), (243.100744, 18.01056), (300.122204, 18.01056), (300.122204, 332.184804), (557.2572246963999, 332.184804), (557.2572246963999, 566.285244), (557.2572246963999, 679.369304)]]

    #correct = [(0.0, 18.01056), (0.0, 174.111644), (344.133164, 174.111644), (344.133164, 386.264114), (344.133164, 499.348148), (344.133164, 570.385284), (344.133164, 684.4281879999999), (711.2863639999999, 684.4281879999999)]
    #candidates = [[(0.0, 18.01056), (0.0, 174.111644), (344.133164, 174.111644), (344.133164, 386.264114), (344.133164, 499.348148), (546.2073633159699, 499.348148), (546.2073633159699, 570.3852579999999), (546.2073633159699, 684.4281879999999)]]

    visualize_all_paths(full_spec, spurious_masses=noise, 
                         candidate_paths=candidates, 
                         correct_path=correct, 
                         aa_map=amino_acid_masses_switch,
                         title = sequence,
                         pep_mass=pep.seq_mass + 18.01056,
                         save_path=f'/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}.png'
                         )
    
    ay_util.visualize_array_range(pep.AA_array, [6,7], save_path=f'/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_colored_array.png')
    print(lower_half_modified)
    
    #old conserved number code
    #universal_set = lower_half_modified
    #print(candidates)
    #cons, non_cons = ay_util.find_conserved_numbers(candidates, universal_set)
    #print(cons, non_cons)
    #ay_util.visualize_sets(lower_half_modified, cons, non_cons, save_path=f'/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_conserved_numbers.png')
    
    def get_corresponded_aa(the_ion_name, the_peptide = pep):
        # Given a peptide sequence and an ion name (e.g., 'b3', 'y5'), return the corresponding amino acid.
        
        if the_ion_name is None or len(the_ion_name) < 2:
            return None
        ion_type = the_ion_name[0]  # 'b' or 'y'
        ion_index = int(the_ion_name[1:])
        if ion_type == 'b':
            if 1 <= ion_index <= len(the_peptide.AA_array):
                return the_peptide.AA_array[ion_index - 1]  # b-ions are 1-indexed
        elif ion_type == 'y':
            if 1 <= ion_index <= len(the_peptide.AA_array):
                return the_peptide.AA_array[-ion_index]  # y-ions count from the end
    
    
    
    def get_ion_segment(target_ion, ion_list = b_ion_list, peptide = pep.AA_array):
        #Returns the amino acid sequence segment between the previous ion 
        #in the list and the target ion.
        peptide = [str(i) for i in peptide]
        # 1. Extract the numeric positions from the ion strings (e.g., 'b2' -> 2)
        # We use a set to remove duplicates, then sort them to find order
        # Format: dictionary mapping {position: 'ion_name'}
        ion_map = {int(ion[1:]): ion for ion in ion_list}
        
        # Get the sorted list of positions (e.g., [2, 4])
        sorted_positions = sorted(ion_map.keys())
        
        # 2. Parse the target position
        target_pos = int(target_ion[1:])
        
        # Check if target is actually in the list provided
        if target_pos not in sorted_positions:
            return f"Error: {target_ion} is not in the ion list."

        # 3. Find the index of the target in our sorted list
        current_index_in_list = sorted_positions.index(target_pos)

        # 4. Determine Start and End indices for slicing the peptide
        # The end of the slice is always the target ion's position
        end_slice = target_pos
        
        if current_index_in_list == 0:
            # If it's the first ion (e.g., b2), start from the beginning (0)
            start_slice = 0
        else:
            # If there is a previous ion (e.g., b2 comes before b4), 
            # start where the previous ion ended.
            previous_pos = sorted_positions[current_index_in_list - 1]
            start_slice = previous_pos

        # 5. Extract the segment from the peptide list
        # Python slicing is [start:end], where start is inclusive and end is exclusive
        segment_list = peptide[start_slice : end_slice]
        
        return "".join(segment_list)
    
    
    ground_truth = paired_peaks
    
    print(paired_mass_dict)


    candidates = [ay_util.mass_b_y_indentification(i, paired_dict=paired_mass_dict) for i in candidates]
    #ay_util.draw_aligned_comparison(ground_truth, candidates, aa_converter = get_ion_segment,save_path=f"/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_colored_peak.png")
    
    ay_util.draw_aligned_comparison_b_only(ground_truth, candidates, aa_converter = get_ion_segment,save_path=f"/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_colored_peak.png")

    ay_util.draw_sequence_with_middle_points(ay_util.mass_b_y_indentification_with_middle(correct), save_path=f"/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_colored_peak_with_middle.png")
    
    
    images = [f"/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_colored_peak.png",
              f'/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}.png', 
              f"/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_parent_table.png",         
              f'/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_colored_array.png',
              f"/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/graph/{data}_colored_peak_with_middle.png"]

    # Create the PDF
    with open(f"/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/{data}.pdf", "wb") as f:
        f.write(img2pdf.convert(images))
    