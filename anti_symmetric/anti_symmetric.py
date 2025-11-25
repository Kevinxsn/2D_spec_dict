import sys
import os

# Get current notebook directory
current_dir = os.getcwd()

# Add parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import peptide
import math


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
                if abs(value - mass) <= 0.05:  # Allow small tolerance for matching
                    aa = key
                    break
            output += f" {direction}({aa}) {fmt_node(next_n)}"
        else:
        
            output += f" {direction} {fmt_node(next_n)}"
        
    return output


def path_to_seq(path, seq_mass):
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
        aa = None
        for key, value in amino_acid_masses.items():
            if abs(value - mass) <= 0.05:  # Allow small tolerance for matching
                aa = key
                break
        output += f" {direction}({aa}) {fmt_node(next_n)}"
        if direction == "Down ↓":
            forward.append(aa)
        else:
            backward.append(aa)
    backward.reverse()
    
    the_middle_diff = seq_mass - (path[-1][0] + path[-1][1])
    for key, value in amino_acid_masses.items():
        if abs(value - the_middle_diff) <= 0.05:
            middle = key
            break

    full_seq = "".join(forward)
    if middle:
        full_seq += middle
    else:
        full_seq += "?"
    full_seq += "".join(backward)
    return full_seq