import sys
import os
import parent_annot
import bisect
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import pandas as pd
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import peptide

def cluster_isotopic_points(points, tolerance=0.01, mass_shift=1.002, max_degree=4):
    """
    Groups 2D points (FFCs) that are separated by integer multiples of a mass_shift.
    
    Args:
        points (list or np.array): A list/array of (x, y) coordinates.
        tolerance (float): The allowed deviation for the shift (e.g., 0.01).
        mass_shift (float): The isotopic shift value (default 1.002).
        max_degree (int): The maximum integer multiplier for m and n (default 4).
        
    Returns:
        list of np.array: A list where each element is a numpy array of points belonging to a cluster.
    """
    
    data = np.array(points)
    n_points = len(data)
    
    if n_points < 2:
        return []

    # 1. Build Spatial Index
    tree = cKDTree(data)
    
    # Calculate the search radius.
    # The maximum distance occurs when m=max_degree and n=max_degree.
    # d = sqrt((4*1.002)^2 + (4*1.002)^2)
    max_dist = np.sqrt(2 * (max_degree * mass_shift) ** 2) + tolerance

    # 2. Query Pairs efficiently
    # This finds all pairs (i, j) where distance(i, j) <= max_dist
    # It returns a set of pairs (i, j) where i < j
    pairs = tree.query_pairs(r=max_dist)
    
    # 3. Build Adjacency Graph based on strict grid validation
    adjacency = {i: [] for i in range(n_points)}
    
    for i, j in pairs:
        p1 = data[i]
        p2 = data[j]
        
        diff = np.abs(p1 - p2)
        dx, dy = diff[0], diff[1]
        
        # Calculate nearest integer multipliers
        m = round(dx / mass_shift)
        n = round(dy / mass_shift)
        
        # Check constraints:
        # 1. m and n must be within range [0, max_degree]
        # 2. At least one of them must be > 0 (points are not identical)
        if not (0 <= m <= max_degree and 0 <= n <= max_degree):
            continue
        if m == 0 and n == 0:
            continue
            
        # 3. Check if the actual distance is close enough to the theoretical shift
        expected_dx = m * mass_shift
        expected_dy = n * mass_shift
        
        if (abs(dx - expected_dx) <= tolerance) and (abs(dy - expected_dy) <= tolerance):
            # Valid isotopic link found
            adjacency[i].append(j)
            adjacency[j].append(i)

    # 4. Find Connected Components (Clustering)
    visited = np.zeros(n_points, dtype=bool)
    clusters = []
    
    for i in range(n_points):
        if not visited[i] and adjacency[i]: # Only start if point has neighbors
            # Start BFS for this component
            component = []
            queue = deque([i])
            visited[i] = True
            
            while queue:
                node = queue.popleft()
                component.append(data[node])
                
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            # 5. Filter for minimum cluster size
            if len(component) >= 2:
                clusters.append(np.array(component))
                
    return clusters



def extract_monoisotopic_features(clusters):
    """
    Takes a list of point clusters and returns the representative (min_x, min_y) for each.
    
    Args:
        clusters (list of np.array): The output from cluster_isotopic_points.
        
    Returns:
        np.array: An array of shape (N, 2) containing the representative points.
    """
    representatives = []
    
    for cluster in clusters:
        # Sort by x (primary) and y (secondary)
        # np.lexsort sorts by keys in reverse order, so we pass (y, x) to sort by x then y
        # However, a simpler way for standard sorting is simply using python's sort or numpy arguments
        
        # Method 1: Using lexsort (very efficient for large arrays)
        # indices = np.lexsort((cluster[:, 1], cluster[:, 0]))
        # best_point = cluster[indices[0]]
        
        # Method 2: Structuring as void type for simple sorting (often cleaner to read)
        # or simply sorting a standard list since clusters are usually small (2-20 points).
        
        # Let's stick to a robust sorting method:
        # Sort by column 0 (x), then column 1 (y)
        #sorted_indices = np.lexsort((cluster[:, 1], cluster[:, 0]))
        #min_point = cluster[sorted_indices[0]]
        min_point_x = min([i[0] for i in cluster])
        min_point_y = min([i[1] for i in cluster])
        min_point = [min_point_x, min_point_y]
        
        representatives.append(min_point)
        
    return np.array(representatives)

def project_monoisotopic_candidates(representatives, pep_mass, mass_shift=1.002, tolerance=0.1):
    """
    Projects observed features back to their theoretical monoisotopic coordinates based on peptide mass.
    
    Args:
        representatives (np.array): Array of (x_min, y_min) points from the previous step.
        pep_mass (float): The target theoretical peptide mass (x + y).
        mass_shift (float): The mass difference per isotope (default 1.002).
        tolerance (float): Allowable error margin for mass grouping.
        
    Returns:
        np.array: A list of all potential projected points (x, y).
    """
    projected_points = []
    
    for point in representatives:
        x, y = point
        observed_mass = x + y
        
        # Calculate the difference between observed mass and target peptide mass
        diff = observed_mass - pep_mass
        
        # Determine how many isotopic shifts (k) this difference corresponds to
        k = int(round(diff / mass_shift))
        
        # Check if the mass matches the grid (k must be non-negative)
        # We also check if the remainder is within tolerance to ensure it's a valid isotope
        remainder = abs(diff - (k * mass_shift))
        
        if remainder > tolerance or k < 0:
            # If it doesn't fit the mass pattern, we can choose to discard it or keep it as is.
            # Based on the prompt "if point doesn't belong... discard", we skip.
            # However, if k=0 (it is the pep_mass), we keep it.
            if k == 0 and remainder <= tolerance:
                 projected_points.append([x, y])
            continue

        # Case 1: Monoisotopic match (k=0)
        if k == 0:
            projected_points.append([x, y])
            
        # Case 2 & 3: Isotopic match (k > 0)
        # We need to generate all combinations of (i, j) such that i + j = k
        # where i is shift in x, and j is shift in y.
        else:
            # Loop i from 0 to k
            for i in range(k + 1):
                j = k - i
                
                # Calculate the projected coordinates
                # We subtract the shift because we are projecting BACK to the monoisotopic state
                new_x = x - (i * mass_shift)
                new_y = y - (j * mass_shift)
                
                projected_points.append([new_x, new_y])

    return np.array(projected_points)





def calculate_coverage_binary(pep_obj, peptide_length, spectrum, tolerance=0.05):
    """
    Calculates the sequence coverage binary map.
    
    Args:
        pep_obj (object): Your custom object with method .ion_mass(type).
        peptide_length (int): Length of the peptide sequence.
        spectrum (list or np.array): List of experimental features. 
                                     Can be list of floats (masses) or list of [x, y] points.
        tolerance (float): Mass tolerance for matching (default 0.05).
        
    Returns:
        list[int]: A binary list (e.g., [1, 0, 1]) representing bond coverage.
                   Length will be peptide_length - 1.
    """
    
    # 1. Preprocess Spectrum
    # If the input is a list of [x, y] points, sum them to get mass.
    # If it's already a list of floats, use as is.
    processed_spectrum = []
    if len(spectrum) > 0:
        # Check type of first element to decide how to process
        first_elem = spectrum[0]
        if hasattr(first_elem, '__len__') and len(first_elem) == 2:
            # It's a point [x, y], so mass = x + y
            processed_spectrum = sorted([x + y for x, y in spectrum])
        else:
            # It's already a mass
            processed_spectrum = sorted(spectrum)
    
    n_peaks = len(processed_spectrum)
    coverage_map = []
    
    # Helper function for efficient binary search matching
    def find_match(target_mass):
        if n_peaks == 0:
            return False
            
        # Find insertion point
        idx = bisect.bisect_left(processed_spectrum, target_mass - tolerance)
        
        # Check if the item at idx or subsequent items are within tolerance
        if idx < n_peaks and abs(processed_spectrum[idx] - target_mass) <= tolerance:
            return True
        return False

    # 2. Iterate through bond positions (1 to L-1)
    # A peptide of length 5 has 4 bonds. 
    # Bond 1 corresponds to b1 and y(L-1)
    for i in range(1, peptide_length):
        is_covered = 0
        
        # Check b-ion (b1, b2, ... b_L-1)
        b_ion_label = f"b{i}"
        try:
            b_mass = pep_obj.ion_mass(b_ion_label)
            if find_match(b_mass):
                is_covered = 1
        except Exception:
            pass # Handle cases where ion generation might fail

        # If not already found, check y-ion pair
        # The bond at index i corresponds to y_{Length - i}
        # e.g., Length 5, bond 1 (after 1st AA) is b1 and y4
        if is_covered == 0:
            y_ion_label = f"y{peptide_length - i}"
            try:
                y_mass = pep_obj.ion_mass(y_ion_label)
                if find_match(y_mass):
                    is_covered = 1
            except Exception:
                pass

        coverage_map.append(is_covered)

    return coverage_map


# --- Example Usage ---
if __name__ == "__main__":
    '''
    # Generate some dummy data
    # Cluster 1: Base (100, 200) + isotopic shifts
    c1 = [
        [100.0, 200.0],
        [100.0 + 1.002, 200.0],           # m=1, n=0
        [100.0 + 2.004, 200.0 + 1.002],   # m=2, n=1
    ]
    
    # Cluster 2: Base (500, 500) + shifts
    c2 = [
        [500.0, 500.0],
        [500.0, 500.0 + 1.002],           # m=0, n=1
    ]
    
    # Noise (points that shouldn't be grouped)
    noise = [
        [150.0, 150.0],
        [100.0 + 0.5, 200.0], # Too far from 1.002 grid
    ]
    
    all_points = np.vstack([c1, c2, noise])
    
    print(f"Total points: {len(all_points)}")
    
    result_clusters = cluster_isotopic_points(all_points, tolerance=0.01)
    
    print(f"Found {len(result_clusters)} clusters:")
    for idx, cluster in enumerate(result_clusters):
        print(f"Cluster {idx+1} size: {len(cluster)}")
        print(cluster)
        print("-" * 20)
    '''
    
    #pep_seq = 'KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK'
    pep_seq = 'HADGSFSDEMNTILDNLAARDFINWLIQTKITD'
    MASS_H = 1.00784
    charge = 4
    iso = 4
    #pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+', end_h20='NH3')
    pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+', end_h20=True)
    pep_mass = pep.pep_mass
    df = pd.read_excel('/Users/kevinmbp/Desktop/isotope.xlsx', sheet_name = '+4')
    
    parent_df = df[abs(df['selected_total'] - pep_mass) <= 0.1]
    parent_xs = list(parent_df['m/z A'] * parent_df['charge_A']-(parent_df['charge_A'] - 1) * MASS_H)
    parent_ys = list(parent_df['m/z B'] * parent_df['charge_B'] - (parent_df['charge_B'] - 1) * MASS_H)
    parent_spec = parent_xs + parent_ys
    
    df = df[abs(df['selected_total'] - pep_mass) > 0.1]
    xs = list(df['m/z A'] * df['charge_A']-(df['charge_A'] - 1) * MASS_H)
    ys = list(df['m/z B'] * df['charge_B'] - (df['charge_B'] - 1) * MASS_H)
    ffcs = [[xs[i], ys[i]] for i in range(len(xs))]
    [i.sort() for i in ffcs]
    result_clusters = cluster_isotopic_points(ffcs, tolerance=0.01)
    
    
    print(f"Found {len(result_clusters)} clusters:")
    for idx, cluster in enumerate(result_clusters):
        print(f"Cluster {idx+1} size: {len(cluster)}")
        print(cluster)
        print("-" * 20)
    
    
    representatives = extract_monoisotopic_features(result_clusters)
    print(representatives)
    print([sum(i) for i in representatives])
    projected = project_monoisotopic_candidates(representatives, pep_mass - (charge - 2)*1.002)
    print(projected)
    
    parent_cov = calculate_coverage_binary(pep, len(pep.AA_array), parent_spec, tolerance = 0.5)
    
    non_project_spec = [i[0] for i in representatives] + [i[1] for i in representatives] + parent_spec
    project_spec = [i[0] for i in projected] + [i[1] for i in projected] + parent_spec
    non_project_cov = calculate_coverage_binary(pep, len(pep.AA_array), non_project_spec, tolerance = 0.5)
    project_cov = calculate_coverage_binary(pep, len(pep.AA_array), project_spec, tolerance = 0.5)
    
    project_spec.sort()
    parent_spec.sort()
    print(parent_spec)
    print(project_spec)
    
    print(parent_cov)
    #print(non_project_cov)
    print(project_cov)
    print(pep.ion_mass('b7'))