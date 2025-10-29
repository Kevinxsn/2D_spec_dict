import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import peptide
import os
import json
import util

def parse_csv_ion_string(s):
    """
    Parses an ion string from the CSV format.
    
    Examples:
        'y4(1+)' -> ('y4', None, None, '1+')
        'b8-(H2O)(1+)' -> ('b8', 'H2O', '-', '1+')
        'b5-H2O(1+)' -> ('b5', 'H2O', '-', '1+')

    Args:
        s (str): The ion string (e.g., 'y4(1+)' or 'b8-(H2O)(1+)').

    Returns:
        A tuple of (ion, loss, loss_sign, charge).
        Any part can be None if not present.
    """
    if not s or s == '???' or pd.isna(s): # Added check for pandas NA/NaN
        return '???', None, None, None

    # Normalize fancy dashes to ASCII '-'
    s_normalized = str(s).strip().replace('–', '-').replace('—', '-').replace('−', '-')

    # 1. Extract charge from the end, e.g., '(1+)'
    charge = None
    core_string = s_normalized
    charge_match = re.search(r'\(([\d\+-]+)\)$', s_normalized)
    
    if charge_match:
        charge = charge_match.group(1)
        core_string = s_normalized[:charge_match.start()].strip() # Get everything before the charge

    # 2. Define a pattern for the core string (ion + optional loss)
    # This will match 'y4' or 'b8-(H2O)' or 'b8-H2O'
    CORE_PATTERN = re.compile(
        r"""
        ^
        (?P<ion>
            [a-z]+(\d+(?:-\d+)?|\(\d+-\d+\))?  # ion + index e.g. 'y4', 'b8'
        )
        (?P<loss_part>
            [+\-].* # sign + the rest of the loss string e.g. '-(H2O)', '-NH3'
        )?
        $
        """,
        re.VERBOSE | re.IGNORECASE
    )

    # 3. Parse the core string
    match = CORE_PATTERN.fullmatch(core_string)
    
    if not match:
        # Fallback if it's just an ion string with no loss, but also no charge
        # (e.g., if the input was just 'y4')
        if re.fullmatch(r'[a-z]+(\d+(?:-\d+)?|\(\d+-\d+\))?', core_string, re.IGNORECASE):
            return core_string, None, None, charge
        return None, None, None, charge

    ion = match.group('ion')
    loss_part = match.group('loss_part')
    loss = None
    loss_sign = None
    
    if loss_part:
        loss_sign = loss_part[0]  # The '+' or '-'
        loss = loss_part[1:]      # Everything after the sign
        
        # Clean up loss: remove surrounding parentheses if they exist, e.g., '(H2O)' -> 'H2O'
        if loss.startswith('(') and loss.endswith(')'):
            loss = loss[1:-1]

    return ion, loss, loss_sign, charge

def process_ion_dataframe(df):
    """
    Reads a DataFrame and parses the interpretation columns.

    Args:
        df (pd.DataFrame): DataFrame containing the ion data.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              a parsed row from the DataFrame.
    """
    parsed_results = []
    
    try:
        for index, row in df.iterrows():
            # --- Parse Interpretation A ---
            interp_a_str = row.get('Interpretation A')
            mass_a_str = row.get('m/z A')
            ion_a, loss_a, sign_a, charge_a = parse_csv_ion_string(interp_a_str)
            
            try:
                # pd.to_numeric is safer for various inputs, handles None/NaN
                mass_a = pd.to_numeric(mass_a_str, errors='coerce')
                if pd.isna(mass_a):
                    mass_a = None
            except (ValueError, TypeError):
                mass_a = None

            # --- Parse Interpretation B ---
            interp_b_str = row.get('Interpretation B')
            mass_b_str = row.get('m/z B')
            ion_b, loss_b, sign_b, charge_b = parse_csv_ion_string(interp_b_str)

            try:
                mass_b = pd.to_numeric(mass_b_str, errors='coerce')
                if pd.isna(mass_b):
                    mass_b = None
            except (ValueError, TypeError):
                mass_b = None
            
            # --- Store results ---
            result_row = {
                # Use row.get('Index') or index
                'Index': row.get('Index', index), 
                'Interpretation_A': {
                    'raw': interp_a_str,
                    'ion': ion_a,
                    'loss': loss_a,
                    'loss_sign': sign_a,
                    'charge': charge_a,
                    'mass': mass_a
                },
                'Interpretation_B': {
                    'raw': interp_b_str,
                    'ion': ion_b,
                    'loss': loss_b,
                    'loss_sign': sign_b,
                    'charge': charge_b,
                    'mass': mass_b
                }
            }
            parsed_results.append(result_row)
                
    except Exception as e:
        print(f"An error occurred while processing the DataFrame: {e}")
        return []
        
    return parsed_results





if __name__ == "__main__":
    # 1. Define the CSV data as a string
    csv_data = "ME4_2+.csv"
    file_path = os.path.join(
        os.path.dirname(__file__),
        f"../data/Top_Correlations_At_Full_Num_Scans_PCov/annotated/{csv_data}"
    )
    file_path = os.path.abspath(file_path) 
    
    ## Store sequence into peptide class
    sequence = util.name_ouput(csv_data)
    csv_data = file_path
    pep = peptide.Pep(sequence)
    df = pd.read_csv(csv_data)
    
    
    ## Choose the first interpertation
    df = df[df['Index'].notna()]
    results = process_ion_dataframe(df.head(5))
    #print(results)
    if results:
        print("--- Parsed Results ---")
        print(json.dumps(results, indent=2))
        print("\n--- End of Results ---")