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
import neutral_loss_mass
import b_y_graph


def choose_sum(row, entire_pep_seq_mass):
    diff1 = abs(row['m1+m2']  - entire_pep_seq_mass)
    diff2 = abs(row['2m1+m2'] - entire_pep_seq_mass)
    diff3 = abs(row['m1+2m2'] - entire_pep_seq_mass)
    min_diff = min(diff1, diff2, diff3)
    if min_diff == diff1:
        return 'm1+m2', row['m1+m2']
    elif min_diff == diff2:
        return '2m1+m2', row['2m1+m2']
    else:
        return 'm1+2m2', row['m1+2m2']

def classify_ion(ion):
    if isinstance(ion, str):
        if ion.lower().startswith('y'):
            return 'y'
        elif ion.lower().startswith(('b', 'a')):
            return 'b'
    return None  # for NaN or unrecognized types
    


def correct_mass_calc1(row):
    proton = 1.00725
    charge = int(row['charge1'][0]) if row['charge1'] != None else None
    result = None
    if charge is not None:
        if row['loss_sign1'] is not None:
            if row['loss_sign1'] == '+':
                result = (row['ion_mass1'] - proton + row['addition_mass1'] + charge * proton) / charge
            else:
                result = (row['ion_mass1'] - proton - row['addition_mass1'] + charge * proton) / charge
        else:
            result = (row['ion_mass1'] - proton + charge * proton) / charge
    elif charge is None and row['ion1'] is not None:
        if row['loss_sign1'] == '+':
            result = (row['ion_mass1'] - proton + row['addition_mass1'] + 1 * proton) / 1
        else:
            result = (row['ion_mass1'] - proton - row['addition_mass1'] + 1 * proton) / 1
    return result


def correct_mass_calc2(row):
    proton = 1.00725
    charge = int(row['charge2'][0]) if row['charge2'] != None else None
    result = None
    if charge is not None:
        if row['loss_sign2'] is not None:
            if row['loss_sign2'] == '+':
                result = (row['ion_mass2'] - proton + row['addition_mass2'] + charge * proton) / charge
            else:
                result = (row['ion_mass2'] - proton - row['addition_mass2'] + charge * proton) / charge
        else:
            result = (row['ion_mass2'] - proton + charge * proton) / charge
    elif charge is None and row['ion2'] is not None:
        if row['loss_sign2'] == '+':
            result = (row['ion_mass2'] - proton + row['addition_mass2'] + 1 * proton) / 1
        else:
            result = (row['ion_mass2'] - proton - row['addition_mass2'] + 1 * proton) / 1
    return result

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
        #if loss.startswith('(') and loss.endswith(')'):
        #    loss = loss[1:-1]

    return ion, loss, loss_sign, charge

def process_ion_dataframe(df, the_pep):
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
        
    df_current = pd.json_normalize(parsed_results)
    
    df_current.columns = ['Index', 'A_raw', 'ion1', 'loss1', 'loss_sign1', 'charge1', 'mass1', 'B_raw', 'ion2', 'loss2', 'loss_sign2', 'charge2', 'mass2']
    
    H2O_decider = the_pep.AA_array[-1].attach is None
    
    
    
    #df_current['loss1']= df_current['loss1'].apply(lambda x: '(' + x + ')' if pd.notna(x) else x)
    #df_current['loss2']= df_current['loss2'].apply(lambda x: '(' + x + ')' if pd.notna(x) else x)
    #print(df_current)
    
    
    
    df_current['addition_mass1']= df_current['loss1'].apply(neutral_loss_mass.mass_of_loss)
    df_current['addition_mass2']= df_current['loss2'].apply(neutral_loss_mass.mass_of_loss)
    
    
    
    df_current['ion_mass1'] = df_current['ion1'].apply(the_pep.ion_mass, defult_H2O = H2O_decider)
    df_current['ion_mass2'] = df_current['ion2'].apply(the_pep.ion_mass, defult_H2O = H2O_decider)
    proton = 1.00725
    entire_pep_seq_mass = the_pep.pep_mass
    
    
    df_current['correct_mass1'] = df_current.apply(correct_mass_calc1, axis=1)
    df_current['correct_mass2'] = df_current.apply(correct_mass_calc2, axis=1)
    df_current['mass_difference1'] = df_current['mass1'] - df_current['correct_mass1']
    df_current['mass_difference2'] = df_current['mass2'] - df_current['correct_mass2']

    df_current['m1+m2'] = df_current['mass1'] + df_current['mass2']
    df_current['2m1+m2'] = 2 * df_current['mass1'] + df_current['mass2']
    df_current['m1+2m2'] = df_current['mass1'] + 2 * df_current['mass2']
    
    df_current[['chosen_sum_from', 'chosen_sum']] = df_current.apply(
        choose_sum, args=(entire_pep_seq_mass,), axis=1, result_type='expand'
    )
    
    df_current['type1'] = df_current['ion1'].apply(classify_ion)
    df_current['type2'] = df_current['ion2'].apply(classify_ion)
    df_current['y_ion'] = np.nan
    df_current['y_mz'] = np.nan
    df_current['b_ion'] = np.nan
    df_current['b_mz'] = np.nan
    df_current.loc[df_current['type1'] == 'y', ['y_ion', 'y_mz']] = df_current.loc[df_current['type1'] == 'y', ['ion1', 'mass1']].values
    df_current.loc[df_current['type1'] == 'b', ['b_ion', 'b_mz']] = df_current.loc[df_current['type1'] == 'b', ['ion1', 'mass1']].values

    df_current.loc[df_current['type2'] == 'y', ['y_ion', 'y_mz']] = df_current.loc[df_current['type2'] == 'y', ['ion2', 'mass2']].values
    df_current.loc[df_current['type2'] == 'b', ['b_ion', 'b_mz']] = df_current.loc[df_current['type2'] == 'b', ['ion2', 'mass2']].values


    
    return df_current

common_neutrol_loss_group = {'(NH3)', '(H2O)', '2(H2O)', '2(NH3)'}

def data_classify(row, the_pep):
    ion1 = row['ion1']
    ion2 = row['ion2']
    loss1 = row['loss1']
    loss2 = row['loss2']
    
    if (ion1 == '???') or (ion2 == '???'):
        return 'unclear'
        
    elif (ion1[:2] == 'ai' or ion1[:2] =='bi') or (ion2[:2] == 'ai' or ion2[:2] =='bi'):
        return 'internal_acid'
    
    elif ('/' in ion1) or ('/' in ion2):
        return 'ambiguous'
    elif (loss1 is not None and loss1 not in common_neutrol_loss_group) or (loss2 is not None and loss2 not in common_neutrol_loss_group):
        return 'rare_mode'
    elif (type(ion1) is str) and (type(ion2) is str):
        ion1_num = ion1[1:]
        ion2_num = ion2[1:]
        if int(ion1_num) + int(ion2_num) != len(the_pep.AA_array):
            return 'non_complementary'
    
    return 'usable'
    







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
    print(sequence)
    csv_data = file_path
    pep = peptide.Pep(sequence)
    df = pd.read_csv(csv_data)
    
    
    ## Choose the first interpertation
    df = df[df['Index'].notna()]
    results = process_ion_dataframe(df.head(50), pep)
    
    results['classification'] = results.apply(data_classify, args=(pep,), axis=1)
    #print(results)
    
    the_list = []
    the_y_list = []
    
    results['loss1'] = results['loss1'].replace({None: np.nan})
    results['loss2'] = results['loss2'].replace({None: np.nan})
    
    
    df_y = b_y_graph.ion_data_organizer_y(results, sequence)
    df_x = b_y_graph.ion_data_organizer_d(results, sequence)
    
    print(df_y)
    print(df_x)
    
    the_length = len(pep.AA_array)
    b_list = b_y_graph.create_annotation_list_from_df(df_x, the_length, b_y_graph.neutral_loss_colors)
    y_list = b_y_graph.create_annotation_list_from_df(df_y, the_length, b_y_graph.neutral_loss_colors)
    b_y_graph.plot_peptide_fragmentation(pep.seq, annotations=b_list, y_line_annotations = y_list, color_map=b_y_graph.neutral_loss_colors, show=True)
    

    print("--- Parsed Results ---")
    print(results[['ion1', 'loss1', 'ion2', 'loss2', 'classification', 'y_ion', 'b_ion', 'y_mz', 'mass1', 'mass_difference1']])
    print("\n--- End of Results ---")