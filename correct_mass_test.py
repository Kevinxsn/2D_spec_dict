#This code try to caculate the correct mass for a given ion and a peptide, calculating different combinations(1m1+2m2 ect.)

import numpy as np
import pandas as pd
import re
import peptide
import neutral_loss_mass
import matplotlib.pyplot as plt
import seaborn as sns
#from clustering import find_mass_clusters
from clustering import find_mass_clusters_with_labels

data_number = 1

with open(f'data/data{data_number}.txt', 'r', encoding='utf-8') as file:
    content = file.read()


def parse_ion_with_mass(s):
    """
    Parses a single ion string to extract the core ion, neutral loss, charge, and mass.
    
    Example: '[y9 - CH3NH2] (1+) @ 816.84' -> ('y9', 'CH3NH2', '1+', 816.84)
    
    Args:
        s (str): The ion string to parse.

    Returns:
        A tuple of (ion, loss, charge, mass). 
        - Returns ('???', None, None, mass) for '???' input.
        - Any part can be None if not present.
    """
    s = s.strip()
    s = s.replace(' ','')

    # Define the main parsing pattern
    _TOKEN = r'(?:\d*\([^\)]*\)|[^\]\s\(\)\+\-]+)'
    PATTERN = re.compile(
        rf"""
        ^
        (?P<ion>
            [a-z]+                                          # ion type like 'b', 'y', 'bi'
            (?:\d+(?:-\d+)?|\(\d+-\d+\))?                    # optional index or range e.g., '9', '(2-4)'
            (?:/[a-z]+(?:\d+(?:-\d+)?|\(\d+-\d+\))?)?        # optional /other ion e.g., '/y4'
        )
        (?P<loss>
            (?:[+\-]{_TOKEN})+                               # one or more +/- segments e.g., '-NH3-H2O'
        )?
        $
        """,
        re.VERBOSE | re.IGNORECASE
    )

    # 1. Extract mass (m/z value) from the end of the string
    mass = None
    s_main = s
    mass_match = re.search(r'@\s*([\d\.]+)$', s)
    if mass_match:
        try:
            mass = float(mass_match.group(1))
        except ValueError:
            mass = None  # In case of malformed number
        s_main = s[:mass_match.start()].strip()

    # 2. Handle '???' case
    if s_main.startswith('???'):
        return '???', None, None, None, mass

    # Normalize fancy dashes to ASCII '-'
    s_normalized = s_main.replace('–', '-').replace('—', '-').replace('−', '-')

    # 3. Extract charge from the end of the string, e.g., '(1+)'
    charge = None
    main_part = s_normalized
    charge_match = re.search(r'\s*\(([\d\+-]+)\)$', s_normalized)
    if charge_match:
        charge = charge_match.group(1)
        main_part = s_normalized[:charge_match.start()].strip()

    # 4. Handle square brackets, extracting the content inside
    in_brackets_match = re.match(r'\[([^\]]+)\]$', main_part)
    if in_brackets_match:
        core_string = in_brackets_match.group(1)
    else:
        core_string = main_part
        
    # 5. Remove all whitespace from the core string (e.g., 'y9 - NH3' -> 'y9-NH3')
    core_string = re.sub(r'\s+', '', core_string)

    # 6. Use the main regex to parse the ion and loss from the core string
    match = PATTERN.fullmatch(core_string)
    if not match:
        # Fallback for simple ions
        if re.fullmatch(r'[a-z]+(?:\d+(?:-\d+)?|\(\d+-\d+\))?(?:/[a-z]+(?:\d+(?:-\d+)?|\(\d+-\d+\))?)?', core_string, re.IGNORECASE):
             return core_string,None, None, charge, mass
        return None, None, None, charge, mass

    ion = match.group('ion')
    loss = match.group('loss')
    loss_sign = None
    
    if loss:
        # Remove the leading sign from the loss string (e.g., '-NH3' -> 'NH3')
        loss_sign = loss[0] if loss[0] in '+-' else '-' ## the defult is -, but could be +
        loss = loss[1:] if loss[0] in '+-' else loss
    
    return ion, loss, loss_sign, charge, mass



def is_float(s: str) -> bool:
    s = s.replace(" ", "") 
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False




data = content
lines = data.strip().split('\n')[1:] # Skip the first header line
parsed_data = []
peptide_seq = data.strip().split('\n')[0]

pep = peptide.Pep(peptide_seq)
print('the AA array with modification of this peptide is ', pep.AA_array)
peptide_seq = pep.seq

for line in lines:
    # Clean up line number and split into two ion pairs
    line_content = re.sub(r'^\d+\.\s*', '', line.strip())
    parts = line_content.split('&')

    if len(parts) == 2:
        ion1_str, ion2_str = parts
        #print(ion1_str)
        ion1, loss1,loss_sign1, charge1, mass1 = parse_ion_with_mass(ion1_str)
        ion2, loss2, loss_sign2, charge2, mass2 = parse_ion_with_mass(ion2_str)
        
        ## at least assing the mess is there is no anotation, only numbers, like 286 & 294
        if ion1 is None and ion2 is None and is_float(ion1_str) and is_float(ion2_str):
            ion1, loss1,loss_sign1, charge1, mass1 = None, None, None, None, float(ion1_str.replace(" ", ""))
            ion2, loss2,loss_sign2, charge2, mass2 = None, None, None, None, float(ion2_str.replace(" ", ""))
            
        
        parsed_data.append([line_content, ion1, loss1, loss_sign1, charge1, mass1, ion2, loss2, loss_sign2,charge2, mass2])
df = pd.DataFrame(parsed_data, columns=['each_original_data', 'ion1', 'loss1','loss_sign1','charge1','mass1', 'ion2', 'loss2','loss_sign2', 'charge2', 'mass2'])
df.index = df.index + 1


## see if the last peptide having annotated attachment to decide if we need a water molecule. 
H2O_decider = pep.AA_array[-1].attach is None


df['addition_mass1']= df['loss1'].apply(neutral_loss_mass.mass_of_loss)
df['addition_mass2']= df['loss2'].apply(neutral_loss_mass.mass_of_loss)
df['ion_mass1'] = df['ion1'].apply(pep.ion_mass, defult_H2O = H2O_decider)
df['ion_mass2'] = df['ion2'].apply(pep.ion_mass, defult_H2O = H2O_decider)
#df['correct_mass1'] = df['ion_mass1'] - df['addition_mass1']
#df['correct_mass2'] = df['ion_mass2'] - df['addition_mass2']

proton = 1.00725

def correct_mass_calc1(row):
    charge = int(row['charge1'][0]) if row['charge1'] != None else None
    result = None
    if charge is not None:
        if row['loss_sign1'] is not None:
            if row['loss_sign1'] == '+':
                result = (row['ion_mass1'] + row['addition_mass1'] + charge * proton) / charge
            else:
                result = (row['ion_mass1'] - row['addition_mass1'] + charge * proton) / charge
        else:
            result = (row['ion_mass1'] + charge * proton) / charge
    elif charge is None and row['ion1'] is not None:
        if row['loss_sign1'] == '+':
            result = (row['ion_mass1'] + row['addition_mass1'] + 1 * proton) / 1
        else:
            result = (row['ion_mass1'] - row['addition_mass1'] + 1 * proton) / 1
    return result


def correct_mass_calc2(row):
    charge = int(row['charge2'][0]) if row['charge2'] != None else None
    result = None
    if charge is not None:
        if row['loss_sign2'] is not None:
            if row['loss_sign2'] == '+':
                result = (row['ion_mass2'] + row['addition_mass2'] + charge * proton) / charge
            else:
                result = (row['ion_mass2'] - row['addition_mass2'] + charge * proton) / charge
        else:
            result = (row['ion_mass2'] + charge * proton) / charge
    elif charge is None and row['ion2'] is not None:
        if row['loss_sign2'] == '+':
            result = (row['ion_mass2'] + row['addition_mass2'] + 1 * proton) / 1
        else:
            result = (row['ion_mass2'] - row['addition_mass2'] + 1 * proton) / 1
    return result



entire_pep_seq_mass = sum([i.get_mass() for i in pep.AA_array]) + 18.01056 if H2O_decider else sum([i.get_mass() for i in pep.AA_array])
entire_pep_seq_mass += 2


df['correct_mass1'] = df.apply(correct_mass_calc1, axis=1)
df['correct_mass2'] = df.apply(correct_mass_calc2, axis=1)
df['mass_difference1'] = df['mass1'] - df['correct_mass1']
df['mass_difference2'] = df['mass2'] - df['correct_mass2']

df['m1+m2'] = df['mass1'] + df['mass2']
df['2m1+m2'] = 2 * df['mass1'] + df['mass2']
df['m1+2m2'] = df['mass1'] + 2 * df['mass2']

'''
def choose_sum(row):
    ## from m1+m2, 2m1+m2, m1+2m2, choose the one closest to the peptide mass
    diff1, diff2, diff3 = abs(row['m1+m2'] - entire_pep_seq_mass), abs(row['2m1+m2'] - entire_pep_seq_mass), abs(row['m1+2m2'] - entire_pep_seq_mass)
    min_diff = min([diff1, diff2, diff3])
    if min_diff == diff1:
        return 'm1+m2', min_diff 
    elif min_diff == diff2:
        return '2m1+m2',min_diff
    else:
        return 'm1+2m2', min_diff


df[['chosen_sum', 'chosen_sum_from']] = df.apply(choose_sum, axis=1)
'''

def choose_sum(row):
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

# note the double brackets and the axis/result_type
df[['chosen_sum_from', 'chosen_sum']] = df.apply(
    choose_sum, axis=1, result_type='expand'
)



cluster_data = np.array(df['chosen_sum'])
cluster_data = cluster_data[np.isfinite(cluster_data)]

eps = 0.8
'''
cluster_df = find_mass_clusters(
    
    mass_sums=cluster_data,
    actual_peptide_mass=entire_pep_seq_mass,
    eps=eps
)
'''
cluster_summary, df_with_labels = find_mass_clusters_with_labels(
    df,
    actual_peptide_mass=entire_pep_seq_mass,
    eps=eps
)

print(df_with_labels)
df_with_labels.to_csv(f'data/data_table/data_sheet{data_number}.csv')

print(cluster_summary)
cluster_summary.to_csv(f'data/cluster/data_cluster{data_number}.csv')