import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import re
from typing import Iterable, Tuple, List

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Add parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import peptide
import math
import interpreter_modify




'''
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
'''


def partition_dataframe_by_charge(
    df: pd.DataFrame,
    charge_list: Iterable[int],
    x1_col: str = "m/z A",
    x2_col: str = "m/z B",
) -> Tuple[pd.DataFrame, List[str]]:
    result_df = df.copy()
    partitioned_names: List[str] = []

    # optional: de-duplicate charges while preserving order
    seen = set()
    charges = []
    for c in charge_list:
        if c is None:
            continue
        c = int(c)
        if c not in seen:
            seen.add(c)
            charges.append(c)

    for charge in charges:
        if charge < 2:
            continue

        for w1 in range(1, charge):
            w2 = charge - w1

            sum_name = f"{w1}*{x1_col} + {w2}*{x2_col}"
            partitioned_names.append(sum_name)

            comp1 = f"comp_{w1}_{x1_col}"
            comp2 = f"comp_{w2}_{x2_col}"

            # Only create component cols if they don't exist yet
            if comp1 not in result_df.columns:
                result_df[comp1] = w1 * result_df[x1_col]
            if comp2 not in result_df.columns:
                result_df[comp2] = w2 * result_df[x2_col]

            result_df[sum_name] = result_df[comp1] + result_df[comp2]

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


def annotate_dataframe_loss(df, pep, threshold, diffs, charge, iso_range = 0):
    ISOTOPE_OFFSET = 1.00335
    theoretical_ions = []
    for i in range(1, pep.pep_len):
        for ion_type in ['b', 'y']:
            base_name = f"{ion_type}{i}"
            base_mass = pep.ion_mass(base_name)
            
            # Loop from 0 to iso_range inclusive
            for iso in range(iso_range + 1): 
                #name = base_name if iso == 0 else f"{base_name}+{iso}"
                name = '0' if iso == 0 else f"+{iso}"
                mass = base_mass + (iso * ISOTOPE_OFFSET)
                theoretical_ions.append({'name': name, 'mass': mass, 'iso':iso, 'base_name':base_name})
    thi = theoretical_ions.copy()
    for i in thi:
        for the_diff in diffs:
            
            ## for each diff we want to try, list the iso we also need to accept
            if the_diff > 0:
                diff_range = [the_diff + n*ISOTOPE_OFFSET for n in range(0, iso_range+1)] if iso_range != 0 else [the_diff]
            else:
                diff_range = [0]
            for the_index, diff in enumerate(diff_range):
                #if diff != 0 and the_index!=i['iso'] :
                if diff != 0: 
                    name = i['name']
                    name = f'({name})-{round(diff, 3)}' 
                    mass = i['mass'] - diff
                    theoretical_ions.append({'name': name, 'mass': mass, 'iso':i['iso'], 'base_name':i['base_name']})
    def get_all_matches(observed_mass):
        matches = []
        for ion in theoretical_ions:
            diff = abs(observed_mass - ion['mass'])
                
            if diffs[0] == 0:
                if diff<=threshold and ion['iso'] == 0:
                    matches.append({
                    'name': ion['name'],
                    'dev': diff,
                    'theo': ion['mass'],
                    'iso': ion['iso'],
                    'base_name': ion['base_name']
                    })
            
            elif diff <= threshold:
                matches.append({
                    'name': ion['name'],
                    'dev': diff,
                    'theo': ion['mass'],
                    'iso': ion['iso'],
                    'base_name': ion['base_name']
                })
                
                
        # If no match is found, return a placeholder to maintain the row data
        return matches if matches else [{'name': None, 'dev': None, 'theo': None, 'iso':None, 'base_name':None}]

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
                new_row['iso_A'] = mA['iso']
                new_row['base_name_A'] = mA['base_name']
                
                # Assign B results
                new_row['explanation_B'] = mB['name']
                new_row['deviation_B'] = mB['dev']
                new_row['theoretical_mass_B'] = mB['theo']
                new_row['iso_B'] = mB['iso']
                new_row['base_name_B'] = mB['base_name']
                
                if diffs[0] >= 0 and diffs[0] < 71.037:
                    if row['charge_A'] + row['charge_B'] == charge:
                        new_rows.append(new_row)    
                else:
                    new_rows.append(new_row)

    
    #print([i for i in theoretical_ions if i['base_name'] == 'y12'])
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


def get_complementray(pep, base):
    length = len(pep.AA_array)
    the_type = base[0]
    the_index = int(base[1:])
    if the_type == 'y':
        return f'b{length - the_index}'
    else:
        return f'y{length - the_index}'
        
    



def coverage_table(df, lines, pep, iso, tresh = 0.05):
    
    ISOTOPE_OFFSET = 1.00335
    
    length = len(pep.AA_array)
    rows = [f'b{i}, y{length - i}' for i in range(1, length)]
    theo_mass = [(round(pep.ion_mass(f'b{i}'), 3), round(pep.ion_mass(f'y{length-i}'), 3)) for i in range(1, length)]
    
    columns = []
    for i in lines:
        if i == 0:
            columns.append('parent')
        else:
            columns.append(-i)
    the_table = {}
    iso_table = {} # tracking the lowest iso 
    for i in columns:
        the_table[i] = {j:None for j in rows}
        iso_table[i] = {j:float('inf') for j in rows}
    
    fcc, fpr1, fpr2 = {i:0 for i in columns}, {i:0 for i in columns}, {i:0 for i in columns}
    fpr_list = {i:[] for i in columns}
    
    
            
    pair_set = set(rows)
    for i in lines:
        if i == 0:
            i = 'parent'
        else:
            i = -i
        the_df = df[df['line'] == i]
        for idx, row in the_df.iterrows():
            if row['base_name_A'] is not None and row['base_name_B'] is not None:
                charge_A, charge_B = row['charge_A'], row['charge_B']
                base_A, base_B = row['base_name_A'], row['base_name_B']
                exp_A, exp_B = row['explanation_A'], row['explanation_B']
                dev_A, dev_B = round(row['deviation_A'],3), round(row['deviation_B'],3)
                iso_A, iso_B = row['iso_A'], row['iso_B']
                
                bases = [(base_A, exp_A, charge_A, dev_A), (base_B, exp_B, charge_B, dev_B)]
                bases.sort()
                bases_name = f'{bases[0][0]}, {bases[1][0]}'
                if bases_name in the_table[i]:
                    #the_table[i][bases_name] = f"{bases[0][1]}, {bases[1][1]}, [{bases[0][2]}, {bases[1][2]}], {bases[0][3], bases[1][3]}, [{row['Ranking']}]"
                    if iso_A + iso_B < iso_table[i][bases_name]:
                        the_table[i][bases_name] = f"{bases[0][1]}, {bases[1][1]}, [{bases[0][2]}, {bases[1][2]}], {bases[0][3], bases[1][3]}, [{row['Ranking']}]"
                        iso_table[i][bases_name] = iso_A + iso_B
                
            elif row['base_name_A'] is None and row['base_name_B'] is None:
                fpr1[i] += 1
                if len(fpr_list[i]) < 3:
                    fpr_list[i].append((row['m/z A'], row['m/z B'], [row['charge_A'], row['charge_B']], row['Ranking']))
                    
            
            elif row['base_name_A'] is None or row['base_name_B'] is None:
                fpr2[i] += 1
                

                if row['base_name_A'] is not None:
                    charge_A, charge_B = row['charge_A'], row['charge_B']
                    base_A, base_B = row['base_name_A'], get_complementray(pep, row['base_name_A'])
                    exp_A, exp_B = row['explanation_A'], '???'
                    dev_A, dev_B = round(row['deviation_A'],3), 'n/a'
                    bases = [(base_A, exp_A, charge_A, dev_A), (base_B, exp_B, charge_B, dev_B)]
                    bases.sort()
                    bases_name = f'{bases[0][0]}, {bases[1][0]}'
                    
                    the_name = f"{bases[0][1]}, {bases[1][1]}, [{bases[0][2]}, {bases[1][2]}], {bases[0][3], bases[1][3]}, [{row['Ranking']}]"
                    if (bases_name in the_table[i]) and (the_table[i][bases_name] is not None):
                        the_table[i][bases_name] = the_name
                    if len(fpr_list[i]) < 3:
                        fpr_list[i].append(the_name+'|') 
                else:
                    charge_A, charge_B = row['charge_A'], row['charge_B']
                    base_A, base_B = get_complementray(pep, row['base_name_B']), row['base_name_B']
                    exp_A, exp_B = '???', row['explanation_B']
                    dev_A, dev_B = 'n/a', round(row['deviation_B'],3)
                    bases = [(base_A, exp_A, charge_A, dev_A), (base_B, exp_B, charge_B, dev_B)]
                    bases.sort()
                    bases_name = f'{bases[0][0]}, {bases[1][0]}'
                    
                    the_name = f"{bases[0][1]}, {bases[1][1]}, [{bases[0][2]}, {bases[1][2]}], {bases[0][3], bases[1][3]}, [{row['Ranking']}]"
                    if (bases_name in the_table[i]) and (the_table[i][bases_name] is not None):
                        the_table[i][bases_name] = the_name
                    if len(fpr_list[i]) < 3:
                        fpr_list[i].append(the_name+'|') 
                        
                        
                        
        
            fcc[i] = row['num_ffc']
        
    the_table =  pd.DataFrame(the_table)
    
        
    
    ## Adding offset FFCs
    
    
    offset = {}
    
    for i in columns:
        if i != 'parent' and i < 0:
            vals = abs(i)
            offset[-vals] = None
    
    b_ions = {f'b{i}':pep.ion_mass(f'b{i}') for i in range(0, length)}
    for i in range(1, length):
        b_ions[f'y{i}'] = pep.ion_mass(f'y{i}')
    b_ions_with_iso = b_ions.copy()
    for i in range(1, iso + 1):
        for j in b_ions:
            b_ions_with_iso[j+'-'+str(i)] = b_ions[j] - ISOTOPE_OFFSET * i
            b_ions_with_iso[j+'+'+str(i)] = b_ions[j] + ISOTOPE_OFFSET * i
    
    #print(b_ions_with_iso)
    
    
    for i in offset:
        for key, val in b_ions_with_iso.items():
            diff = abs(float(-i)-float(val))
            if diff < tresh:
                offset[i] = key
    
    offset_cov = {i:'0' for i in rows} 
    for i in offset:
        if offset[i] is not None:
            for j in offset_cov:
                if offset[i] == j.split(',')[0]:
                    offset_cov[j] = '+*'

                    
            
    
    
    rest = [c for c in the_table.columns]
    none_counts_rest = the_table[rest].isna().sum()
    rest_sorted = none_counts_rest.sort_values().index
    the_table = the_table[list(rest_sorted)]
    
    
    the_table["Row Count"] = the_table.notna().sum(axis=1).astype("Int64")
    the_table['Offset Cov'] = offset_cov
    the_table['theoretical mass'] = theo_mass
    the_table.loc["Col Count"] = the_table.notna().sum(axis=0).astype("Int64")
    the_table['AA'] = pep.AA_array
    
    
    
    the_table = pd.concat([the_table, pd.DataFrame([fcc], index=['FFC'])])
    the_table = pd.concat([the_table, pd.DataFrame([fpr1], index=['FPR0'])])
    the_table = pd.concat([the_table, pd.DataFrame([fpr2], index=['FPR1'])])
    #the_table = pd.concat([the_table, pd.DataFrame([fpr_list], index=['FPR_LIST'])])
    the_table = pd.concat([the_table, pd.DataFrame([offset], index=['offset'])])
    max_k = 3
    
    fpr_rows = []
    for i in range(max_k):
        row_i = {col: (vals[i] if i < len(vals) else None) for col, vals in fpr_list.items()}
        fpr_rows.append(row_i)

    fpr_df = pd.DataFrame(fpr_rows, index=[f"FPR_LIST_{i+1}" for i in range(max_k)])

    the_table = pd.concat([the_table, fpr_df])
    
    
    the_table["Covered"] = (the_table["Row Count"] != 0).map({True: "+", False: "0"})
    cov_pct = sum(the_table["Covered"] == "+") / (length) * 100
    new_col = f"Coverage={cov_pct:.1f}%"
    the_table = the_table.rename(columns={"Covered": new_col})
    

    
    cols = ['Offset Cov'] + [c for c in the_table.columns if c != 'Offset Cov']
    the_table = the_table[cols]
    
    cols = [new_col] + [c for c in the_table.columns if c != new_col]
    the_table = the_table[cols]
    
    col_a = the_table[new_col]   # or whatever your first coverage column is named
    col_b = the_table["Offset Cov"]
    combined = np.select(
        [
            (col_a == "+") | (col_b == "+"),                         # if any is "+"
            ((col_a == "+*") | (col_b == "+*")) & ~((col_a == "+") | (col_b == "+")),  # else if any is "+*"
            (col_a == "0") & (col_b == "0")                          # both 0
        ],
        ["+", "+*", "0"],
        default=None  # keep NaN if both are NaN / not in {0,+,+*}
    )
    the_table["Covered"] = combined
    valid = the_table["Covered"].isin(["0", "+", "+*"])
    covered = the_table["Covered"].isin(["+", "+*"])

    cov_pct = covered[valid].mean() * 100  # mean of True/False
    new_col_comb = f"All Coverage={cov_pct:.1f}%"

    # 3) rename + reorder (put Offset Cov first, then the new coverage col, then the rest)
    the_table = the_table.rename(columns={"Covered": new_col_comb})

    cols = (
        ["Offset Cov", new_col_comb]
        + [c for c in the_table.columns if c not in ["Offset Cov", new_col_comb]]
    )
    the_table = the_table[cols]
    
    the_table = the_table.drop(columns=[ "Offset Cov", new_col])
    
    
    
    cols = ["AA"] + [c for c in the_table.columns if c != "AA"]
    the_table = the_table[cols]
    
    return the_table
    

'''
    
def isocolumns(df, mass_list):
    df = df.rename(columns={"parent": 0})
    MASS = min(mass_list)
    mass_list.sort()
    all_columns = df[mass_list]
    return all_columns
'''

def isocolumns(df, mass_list, base_is_parent=None, keep_multiple=False):
    """
    Build single 'isocolumn' from selected mass columns using min_b / min_y logic.

    Core idea:
    - For each non-empty cell, parse all candidate isotope pairs (db, dy)
    - Compute dx = min(db), dy = min(dy) across ALL candidates in that cell
    - Add (dx, dy) to isoline (subject to dominance rule)
    - This intentionally allows mixing minima from different candidate lines to recover parental-line points

    Parameters
    ----------
    df : DataFrame
    mass_list : list
        Selected columns in left-to-right order (or sortable), e.g. [0,1,2] or [-347.163,-346.151,-345.133]
    base_is_parent : bool or None
        True: first selected column is true parent line, so any non-empty cell there => (0,0)
        False: do NOT force that
        None: auto infer (True only if first col is 0 and cols look like integer offsets)
    keep_multiple : bool
        False -> final output returns one best pair (smallest x+y; tie by x then y)
        True  -> keep all non-dominated pairs in list form
    """
    df = df.copy()

    if "parent" in df.columns:
        df = df.rename(columns={"parent": 0})

    mass_list = sorted(mass_list)
    used_cols = [c for c in mass_list if c in df.columns]
    if not used_cols:
        raise ValueError("None of the requested mass_list columns are present in df.")

    # Auto infer whether first column is truly parent (only for offset-style columns like 0,1,2)
    if base_is_parent is None:
        def _is_near_int(x):
            try:
                return abs(float(x) - round(float(x))) < 1e-9
            except Exception:
                return False
        base_is_parent = (abs(float(used_cols[0])) < 1e-9) and all(_is_near_int(c) for c in used_cols)

    all_columns = df[used_cols].copy()

    # ---------------- helpers ----------------
    def is_empty_cell(x):
        if x is None:
            return True
        if isinstance(x, float) and np.isnan(x):
            return True
        s = str(x).strip()
        return s == "" or s.lower() in {"none", "nan"}

    def parse_row_base_from_index(idx):
        # row labels like "b10, y23"
        s = str(idx)
        m = re.search(r"b\s*(\d+)\s*,\s*y\s*(\d+)", s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    def split_annotations(cell_text):
        """
        Split a cell string into annotation chunks.
        Heuristic but works for your examples.
        """
        text = str(cell_text).strip()
        if not text:
            return []
        parts = re.split(r'(?=(?:\(\s*[+-]?\d+\s*\)|[+-]?\d+)\s*,)', text)
        parts = [p.strip(" ,") for p in parts if p.strip(" ,")]
        return parts if parts else [text]

    def parse_first_two_iso_shifts_from_chunk(chunk):
        """
        Extract (db, dy) from one annotation chunk.
        Supports both:
          '0, +1, [..]'
          '(+1)-345.133, 0, [..]'
          '+1, (0)-346.136, [..]'
        """
        s = chunk.strip()
        if not s:
            return None

        head = s.split('[', 1)[0].strip().rstrip(',')
        tokens = [t.strip() for t in head.split(',')]
        if len(tokens) < 2:
            return None
        t1, t2 = tokens[0], tokens[1]

        def parse_iso_token(tok):
            tok = tok.strip()

            # token starts with "(+1)" / "(0)"
            m = re.match(r'^\(\s*([+-]?\d+)\s*\)', tok)
            if m:
                return int(m.group(1))

            # token starts with "+1" / "0" / "-1"
            m = re.match(r'^([+-]?\d+)\b', tok)
            if m:
                return int(m.group(1))

            # fallback: find any "(+1)" inside token
            m = re.search(r'\(\s*([+-]?\d+)\s*\)', tok)
            if m:
                return int(m.group(1))

            return None

        a = parse_iso_token(t1)
        b = parse_iso_token(t2)
        if a is None or b is None:
            return None
        return (a, b)

    def extract_offset_pairs(cell):
        """
        Strictly extract isotope pairs (db, dy) from a cell.

        Supports both styles:
        old: '0, +1, [2.0, 3.0], (0.001, 0.018), [244.0]'
        new: '(+1)-345.133, 0, [1.0, 2.0], ...'
            '+1, (0)-346.136, [2.0, 2.0], ...'

        IMPORTANT:
        - Only capture the FIRST TWO fields of each annotation (the isotope shifts)
        - Do not parse later numeric tuples like (0.001, 0.018)
        """
        if cell is None:
            return []
        s = str(cell).strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return []

        # A token can be:
        #   (+1)-345.133
        #   (0)-346.151
        #   (+1)
        #   +1
        #   0
        token = r'(?:\(\s*[+-]?\d+\s*\)(?:\s*-\s*\d+(?:\.\d+)?)?|[+-]?\d+)'

        # Match: <token> , <token> , [
        # The trailing "\[" is key: it ensures we are matching annotation starts,
        # not decimal tuples like (0.001, 0.018)
        pattern = re.compile(
            rf'({token})\s*,\s*({token})\s*,\s*\['
        )

        def parse_iso_token(tok):
            tok = tok.strip()

            # Case 1: starts with "(+1)" / "(0)" / "(-1)"
            m = re.match(r'^\(\s*([+-]?\d+)\s*\)', tok)
            if m:
                return int(m.group(1))

            # Case 2: plain integer token "+1", "0", "-2"
            m = re.match(r'^([+-]?\d+)$', tok)
            if m:
                return int(m.group(1))

            # Case 3 (fallback): integer prefix
            m = re.match(r'^([+-]?\d+)\b', tok)
            if m:
                return int(m.group(1))

            return None

        pairs = []
        for m in pattern.finditer(s):
            t1, t2 = m.group(1), m.group(2)
            a = parse_iso_token(t1)
            b = parse_iso_token(t2)
            if a is not None and b is not None:
                pairs.append((a, b))

        # de-duplicate preserve order
        seen = set()
        out = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def pair_from_mixed_minima(pairs):
        """
        INTENTIONALLY mix minima across candidate lines:
          dx = min(db), dy = min(dy)
        This is your desired behavior for reconstructing missing parental-line points.
        """
        if not pairs:
            return None
        dx = min(p[0] for p in pairs)
        dy = min(p[1] for p in pairs)
        return (int(round(dx)), int(round(dy)))

    def add_pair_with_dominance(isolist, new_pair):
        """
        Add (x,y) unless an existing pair (a,b) has a+b <= x+y.
        """
        x, y = new_pair
        s_new = x + y
        for a, b in isolist:
            if (a + b) <= s_new:
                return isolist
        isolist.append((x, y))
        return isolist

    def final_prune(pairs):
        """
        Remove dominated pairs globally.
        Optionally reduce to one final pair unless keep_multiple=True.
        """
        if not pairs:
            return []

        # dedupe
        pairs = list(dict.fromkeys(pairs))

        # keep only non-dominated by sum
        kept = []
        for p in sorted(pairs, key=lambda t: (t[0] + t[1], t[0], t[1])):
            if not any((q[0] + q[1]) <= (p[0] + p[1]) for q in kept):
                kept.append(p)

        # The above usually leaves at most one by your rule, but keep safe:
        if keep_multiple:
            return kept
        else:
            return [min(kept, key=lambda t: (t[0] + t[1], t[0], t[1]))]

    def format_isocolumn_value(pairs):
        if not pairs:
            return None
        pairs = final_prune(pairs)
        if not pairs:
            return None
        if len(pairs) == 1:
            return str(pairs[0])
        return str(pairs)

    # ---------------- main ----------------
    iso_values = []

    for idx, row in all_columns.iterrows():
        # Skip summary rows (Col Count, FFC, FPR0...)
        b_base, y_base = parse_row_base_from_index(idx)
        if b_base is None or y_base is None:
            iso_values.append(None)
            continue

        isoline = []

        for j, col in enumerate(used_cols):
            cell = row[col]
            if is_empty_cell(cell):
                continue

            # If first selected column is truly parent line, any annotation means (0,0)
            if j == 0 and base_is_parent:
                isoline = [(0, 0)]
                break

            offset_pairs = extract_offset_pairs(cell)
            if not offset_pairs:
                continue

            # >>> your intended logic <<<
            pair_to_add = pair_from_mixed_minima(offset_pairs)
            dx, dy = pair_to_add
            #print(row, col, dx, dy)
            # If perfect reconstruction appears, keep (0,0) and stop (best possible)
            if dx == 0 and dy == 0:
                isoline = [(0, 0)]
                break

            isoline = add_pair_with_dominance(isoline, (dx, dy))

        iso_values.append(format_isocolumn_value(isoline))

    return pd.DataFrame({f"isocolumn:{min(mass_list)}": iso_values}, index=df.index)

def group_consecutive_floats(nums, threshold=0.5):
    if not nums:
        return []

    groups = []
    individuals = []
    current_group = [nums[0]]

    # Iterate starting from the second element
    for i in range(1, len(nums)):
        diff = nums[i] - current_group[-1]
        
        # Check if the difference is 1 (give or take the threshold)
        if abs(diff - 1.0) <= threshold:
            current_group.append(nums[i])
        else:
            # Route to the appropriate list based on length
            if len(current_group) > 1:
                groups.append(current_group)
            else:
                individuals.append(current_group[0])
            
            # Start a new group
            current_group = [nums[i]]

    # Process the final group after the loop finishes
    if len(current_group) > 1:
        groups.append(current_group)
    else:
        individuals.append(current_group[0])

    # Combine them, putting the grouped lists first
    return groups + individuals

def prioritize_zero(mixed_list):
    has_zero = []
    others = []

    for item in mixed_list:
        # 1. If it is a list, check if 0 (or 0.0) is inside it
        if isinstance(item, list) and 0 in item:
            has_zero.append(item)
        # 2. If it is an individual number, check if the number itself is 0
        elif not isinstance(item, list) and item == 0:
            has_zero.append(item)
        # 3. Otherwise, it goes into the 'others' pile
        else:
            others.append(item)

    # Combine them, putting the zero-containing items first
    return has_zero + others


if __name__ == "__main__":
    #pep_seq = 'KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK'
    pep_seq = 'VEADIAGHGQEVLIR'
    #pep_seq = 'HADGSFSDEMNTILDNLAARDFINWLIQTKITD'
    #pep_seq = 'YLEFISDAIIHVLHSK'
    charge =3
    iso = 4
    #pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+', end_h20='NH3')
    pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+', end_h20=True)
    
    print(pep.pep_mass)
    #pep = peptide.Pep(f'[{pep_seq}+{charge}H]{charge}+')
    df = pd.read_excel('/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/data/Covariance Scoring Tables 10000 Scans.xlsx', sheet_name='VEADIAGHGQEVLIR-mz536-3_cov')
    df = df[['m/z fragment 1', 'm/z fragment 2', 'Covariance', 'Partial Cov.', 'Score', 'Ranking']]
    
    
    '''
    df = pd.read_csv(
        "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/CovarianceData.GLP2_Z4_NCE15_200_ions",
        sep=r"\s+",          # any whitespace
        skiprows=1,          
        header=None,
        engine="python"
    )
    '''
    
    
    #ffc_df = pd.read_excel('/Users/kevinmbp/Desktop/2D_spec_dict/anti_symmetric/data/Covariance Scoring Tables 10000 Scans.xlsx', sheet_name='YLEFISDAIIHVLHSK-mz629-3_cov')
    #ffc_df = ffc_df[['m/z fragment 1', 'm/z fragment 2', 'Covariance', 'Partial Cov.', 'Score', 'Ranking']]
    #df = ffc_df
    
    
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
    
    
    
    data = data[['m/z A', 'm/z B', 'Ranking']]
    #loss_list = [-1, -2, -3, -4 , 0, 14.993, 347.163, 346.151, 345.133, 348.173, 15.992, 12.965, 537.187, 162.016, 41.986, 26.982, 16.998, 943.469, 249.379, 111.040, 126.033, 132.035, 145.030, 238.052]
    #loss_list = [-1, 0, 228.242, 227.239, 98.196, 652.483, 215.288, 651.981, 602.459, 716.999, 99.199, 298.277, 97.173]
    #loss_list = [-1, -2, -3, -4, 0, 15.002, 16.005, 98.081]
    loss_list = [229.111, -1, 0, 228.109, 99.065, 653.352, 216.157, 652.851, 100.068, 299.146, 98.042, -2]
    #loss_list = [-1,-2, 0, 112.080, 113.083, 17.003, 18.006, 318.194, 618.430, 487.301, 474.274, 471.310, 442.297, 430.299, 398.217, 331.224, 339.235, 26.991, 15.990, 99.084, 114.086]
    #loss_list = [-1, 0, 2, 276.144, 277.146, 406.188, 666.362, 831.940, 295.157, 275.141, 405.187, 26.988, 112.091, 113.094, 25.970, 390.231, 739.397, 389.229, 722.441, 665.359, 113.080, 552.257]
    
    
    data = data.head(800
                     )
    partitioned_data, partitioned_names = partition_dataframe_by_charge(data, [charge, charge - 1])
    #data = select_best_partition(data, ['m/z A', 'm/z B', 'Ranking'], pep.pep_mass, 0.1,partitioned_names, iso_range=iso)
    print(partitioned_data.head())

    df_all = []
    for i in loss_list:
        each_data = select_best_partition(partitioned_data, ['m/z A', 'm/z B', 'Ranking'], pep.pep_mass - i, 0.1,partitioned_names, iso_range=0)
        #each_data = each_data.sort_values('Ranking', ascending=True)
        num_ffc = each_data.shape[0]
        
        each_data = annotate_dataframe_loss(each_data, pep, 0.1, diffs=[i], charge=charge,iso_range=iso)
        each_data['line'] = 'parent' if i == 0 else -i
        each_data['num_ffc'] = num_ffc
        df_all.append(each_data)
    
    
    df_all = pd.concat(df_all, ignore_index=True)
    df_all = df_all.sort_values('Ranking', ascending = False)
    print(df_all)
    cov_table = coverage_table(df_all, loss_list, pep, iso)
    print(cov_table)
    
    
    
    '''
    #print(cov_table[[-347.163, -346.151, -345.133]])
    #print(isocolumns(cov_table, [-347.163, -346.151, -345.133]))
    the_list = [i for i in cov_table.columns if type(i) != str]
    the_list.append(0)
    the_list.sort()
    the_list = prioritize_zero(group_consecutive_floats(the_list))
    #df_list = [cov_table[:, 0], cov_table[:, 1]]
    df_list = [cov_table[cov_table.columns[0]], cov_table[cov_table.columns[1]]]
    for i in range(len(the_list)):
        if type(the_list[i]) is list:

            df_list.append(isocolumns(cov_table, the_list[i]))
            the_list[i] = ['parent' if x == 0 else x for x in the_list[i]]
            df_list.append(cov_table[the_list[i]])
        else:

            df_list.append(cov_table[the_list[i:]])
            break
    final_df = pd.concat(df_list, axis=1)
    final_df = final_df.join(cov_table[cov_table.columns[-2]])
    final_df = final_df.join(cov_table[cov_table.columns[-1]])
    print(final_df)
            
    
    path = "isocolumn.xlsx"
    sheet = "3+"
    
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        final_df.to_excel(writer, sheet_name=sheet, index_label=f'N={800}')
    '''