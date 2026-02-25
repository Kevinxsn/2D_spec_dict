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


def partition_dataframe_by_charge(df, charge):
    result_df = df.copy()
    x1_col, x2_col = 'm/z A', 'm/z B'
    partitioned_names = []
    
    for w1 in range(1, charge):
        w2 = charge - w1
        
        sum_name = f"{w1}*{x1_col} + {w2}*{x2_col}"
        partitioned_names.append(sum_name)
        
        # Pre-calculate components for easy retrieval later
        result_df[f"comp_{w1}_{x1_col}"] = w1 * df[x1_col]
        result_df[f"comp_{w2}_{x2_col}"] = w2 * df[x2_col]
        
        # Calculate the total sum
        result_df[sum_name] = result_df[f"comp_{w1}_{x1_col}"] + result_df[f"comp_{w2}_{x2_col}"]
        
    return result_df, partitioned_names

def select_best_partition(df, original_cols, target_number, threshold, partitioned_names, iso_range=0):
    """
    Finds combinations matching the parental isotopic envelope and 
    calculates adjusted masses for downstream fragment annotation.
    """
    ISOTOPE_OFFSET = 1.00335
    MASS_H = 1.00784  # Hydrogen mass for charge adjustment
    
    # 1. Generate target masses for the isotopic envelope
    target_masses = [target_number + (i * ISOTOPE_OFFSET) for i in range(iso_range + 1)]
    
    all_results = []

    for t_mass in target_masses:
        # Calculate absolute deviations for this specific parental isotope
        deviations = df[partitioned_names].sub(t_mass).abs()
        
        min_deviations = deviations.min(axis=1)
        best_col_names = deviations.idxmin(axis=1)
        
        mask = min_deviations <= threshold
        if not mask.any():
            continue
            
        result_subset = df.loc[mask, original_cols].copy()
        row_indices = np.where(mask)[0]
        
        # Lists to store the new calculated values
        selected_totals = []
        comp_x1_vals = []
        comp_x2_vals = []
        charge_A_list = []
        charge_B_list = []
        adj_mass_A_list = []
        adj_mass_B_list = []
        
        for idx, col_name in zip(row_indices, best_col_names[mask]):
            # Extract weights (charges) n and m
            weights = re.findall(r'(\d+)\*', col_name)
            w1, w2 = int(weights[0]), int(weights[1])
            
            # Component values (w * m/z)
            val_A = df.iloc[idx][f"comp_{w1}_m/z A"]
            val_B = df.iloc[idx][f"comp_{w2}_m/z B"]
            
            selected_totals.append(df.iloc[idx][col_name])
            comp_x1_vals.append(val_A)
            comp_x2_vals.append(val_B)
            
            # --- CALCULATE ADJUSTED MASSES ---
            charge_A_list.append(w1)
            charge_B_list.append(w2)
            
            # Mass - (Charge - 1) * H (Standardizing to z=1 for b/y matching)
            adj_mass_A_list.append(val_A - (w1 - 1) * MASS_H)
            adj_mass_B_list.append(val_B - (w2 - 1) * MASS_H)

        # Populate the result subset
        result_subset['selected_total'] = selected_totals
        result_subset['component_x1'] = comp_x1_vals
        result_subset['component_x2'] = comp_x2_vals
        result_subset['charge_A'] = charge_A_list
        result_subset['charge_B'] = charge_B_list
        result_subset['adj_mass_A'] = adj_mass_A_list
        result_subset['adj_mass_B'] = adj_mass_B_list
        result_subset['source_column'] = best_col_names[mask].values
        result_subset['deviation'] = min_deviations[mask].values
        result_subset['parent_isotope_idx'] = (t_mass - target_number) / ISOTOPE_OFFSET
        
        all_results.append(result_subset)

    if not all_results:
        return pd.DataFrame()

    # Combine all found isotopes and keep the best match for each original row
    final_df = pd.concat(all_results).sort_values('deviation').drop_duplicates(subset=original_cols)
    return final_df.reset_index(drop=True)


def annotate_dataframe(df, pep, threshold):
    """
    Annotates the dataframe with:
      - explanation (ion name)
      - deviation (abs difference)
      - theoretical_mass
    for both component A and B.
    """
    
    # --- 1. Pre-calculate all theoretical ions ---
    theoretical_ions = {}
    
    # Range is 1 to pep_len - 1 for standard b/y ions
    for i in range(1, pep.pep_len):
        b_name = f"b{i}"
        y_name = f"y{i}"
        theoretical_ions[b_name] = pep.ion_mass(b_name)
        theoretical_ions[y_name] = pep.ion_mass(y_name)

    # --- 2. Define the matching logic ---
    def find_best_match(observed_mass):
        best_name = None
        best_dev = None
        best_theo = None
        min_diff = float('inf')
        
        for name, theoretical_mass in theoretical_ions.items():
            diff = abs(observed_mass - theoretical_mass)
            
            # Check if within threshold AND closer than any previous match
            if diff <= threshold and diff < min_diff:
                min_diff = diff
                best_name = name
                best_dev = diff
                best_theo = theoretical_mass
        
        return best_name, best_dev, best_theo

    # --- 3. Apply logic to create the 6 new columns ---
    
    # Apply to A
    results_A = df['adj_mass_A'].apply(find_best_match)
    # Extract tuples into separate columns
    df['explanation_A'] = [x[0] for x in results_A]
    df['deviation_A'] = [x[1] for x in results_A]
    df['theoretical_mass_A'] = [x[2] for x in results_A]
    
    # Apply to B
    results_B = df['adj_mass_B'].apply(find_best_match)
    # Extract tuples into separate columns
    df['explanation_B'] = [x[0] for x in results_B]
    df['deviation_B'] = [x[1] for x in results_B]
    df['theoretical_mass_B'] = [x[2] for x in results_B]
    
    return df


def annotate_dataframe_iso(df, pep, threshold, iso_range=0):
    """
    Annotates the dataframe with theoretical ion matches.
    
    Args:
        df: The partitioned dataframe.
        pep: The peptide object.
        threshold: Mass error tolerance.
        iso_range: Number of isotopes to consider (0 = monoisotopic only).
    """
    ISOTOPE_OFFSET = 1.00335
    
    # 1. Pre-calculate theoretical ions with a dynamic isotope range
    theoretical_ions = []
    for i in range(1, pep.pep_len):
        for ion_type in ['b', 'y']:
            base_name = f"{ion_type}{i}"
            base_mass = pep.ion_mass(base_name)
            
            # Loop from 0 to iso_range inclusive
            for iso in range(iso_range + 1): 
                name = base_name if iso == 0 else f"{base_name}+{iso}"
                mass = base_mass + (iso * ISOTOPE_OFFSET)
                theoretical_ions.append({'name': name, 'mass': mass})

    # 2. Helper to find ALL matches for a specific mass
    def get_all_matches(observed_mass):
        matches = []
        for ion in theoretical_ions:
            diff = abs(observed_mass - ion['mass'])
            if diff <= threshold:
                matches.append({
                    'name': ion['name'],
                    'dev': diff,
                    'theo': ion['mass']
                })
                
                
        # If no match is found, return a placeholder to maintain the row data
        return matches if matches else [{'name': None, 'dev': None, 'theo': None}]

    # 3. Expand the Dataframe using a cross-product of A and B matches
    new_rows = []
    for _, row in df.iterrows():
        matches_A = get_all_matches(row['adj_mass_A'])
        matches_B = get_all_matches(row['adj_mass_B'])
        
        for mA in matches_A:
            for mB in matches_B:
                new_row = row.copy()
                
                # Assign A results
                new_row['explanation_A'] = mA['name']
                new_row['deviation_A'] = mA['dev']
                new_row['theoretical_mass_A'] = mA['theo']
                
                # Assign B results
                new_row['explanation_B'] = mB['name']
                new_row['deviation_B'] = mB['dev']
                new_row['theoretical_mass_B'] = mB['theo']
                
                new_rows.append(new_row)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def classify_isotopes(df):
    """
    Final refined classifier:
    1. Extracts b+n or y+n from explanation columns.
    2. Combines them into a sorted string.
    3. If no isotopes are found and parent_isotope_idx is 0, returns 'Parent'.
    """
    def get_suffix(name):
        if not name or not isinstance(name, str):
            return None
        # Use regex to find the ion type and the +n suffix
        # Handles 'b12+1' -> 'b+1'
        match = re.search(r'([by])\d*\+(\d+)', name)
        if match:
            return f"{match.group(1)}+{match.group(2)}"
        return None

    def logic(row):
        # 1. Extract potential suffixes from both annotations
        iso_a = get_suffix(row['explanation_A'])
        iso_b = get_suffix(row['explanation_B'])
        
        # 2. Collect and clean the list
        found_isos = [i for i in [iso_a, iso_b] if i is not None]
        
        # 3. If we found isotopic fragments, join them (e.g., "b+1, y+1")
        if found_isos:
            found_isos.sort() # Keeps output consistent
            return ", ".join(found_isos)
        
        # 4. If no isotopic suffixes were found:
        # Check if it belongs on the monoisotopic parental line
        if float(row['parent_isotope_idx']) == 0:
            return "Parent"
        
        # 5. Fallback for isotopic parent lines that haven't been annotated yet
        return f"Parent+{int(row['parent_isotope_idx'])} (Unannotated)"

    df['isotopic_classification'] = df.apply(logic, axis=1)
    return df





if __name__ == "__main__":
    pep_seq = 'HADGSFSDEMNTILDNLAARDFINWLIQTKITD'
    #pep_seq = 'VEADIAGHGQEVLIR'
    charge = 4
    iso = 4
    pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+', end_h20=True)
    
    print(pep.pep_mass)
    #pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+')
    #df = pd.read_excel('/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/data/Covariance Scoring Tables 10000 Scans.xlsx', sheet_name='VEADIAGHGQEVLIR-mz536-3_cov')
    #df = df[['m/z fragment 1', 'm/z fragment 2', 'Covariance', 'Partial Cov.', 'Score', 'Ranking']]
    
    df = pd.read_csv(
        "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.GLP2_Z4_NCE15_200_ions",
        sep=r"\s+",          # any whitespace
        skiprows=1,          
        header=None,
        engine="python"
    )
    

    df.columns = ['m/z A', 'm/z B', 'Covariance', 'Partial Cov.', 'Score', 'Ranking']  # rename as you like
    df = df[df['Score'] > 0]
    data = df
    data = data.sort_values('Ranking', ascending=True)
    data = data[data['Ranking'] != -1]
    
    
    
    data["pair_key"] = data.apply(
        lambda row: tuple(sorted([row["m/z A"], row["m/z B"]])),
        axis=1
    )
    data = data.drop_duplicates(subset="pair_key").drop(columns="pair_key")
    
    print(data.head())
    
    data = data[['m/z A', 'm/z B', 'Ranking']]
    data = data.head(300)
    data, partitioned_names = partition_dataframe_by_charge(data, charge)
    data = select_best_partition(data, ['m/z A', 'm/z B', 'Ranking'], pep.pep_mass, 0.1,partitioned_names, iso_range=iso)

    data = data.sort_values('Ranking', ascending=True)
    data = annotate_dataframe_iso(data, pep, 0.1, iso_range=iso)
    data = classify_isotopes(data)
    data = data.sort_values(by = ['parent_isotope_idx', 'isotopic_classification'])
    print(data.head(30))
    data = data[['m/z A', 'm/z B', 'Ranking', 'selected_total', 'charge_A', 'charge_B', 'explanation_A', 'explanation_B', 'parent_isotope_idx', 'isotopic_classification', 'deviation_A', 'deviation_B', 'deviation', 'theoretical_mass_A', 'theoretical_mass_B']]
    #data.to_csv('+19.csv')
    print(data.head(50))
    
    print(pep.pep_mass)
    print(pep.ion_mass('y16'))
    print(pep.ion_mass('b16'))
    print(pep.ion_mass('y18') + pep.ion_mass('b15'))