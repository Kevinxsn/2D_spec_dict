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


data = 'ME9_2+'
csv_data = f"{data}.csv"
file_path = os.path.join(
    os.path.dirname(__file__),
    f"../data/Top_Correlations_At_Full_Num_Scans_PCov/annotated/{csv_data}"
)
file_path = os.path.abspath(file_path) 

## Store sequence into peptide class
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
rows = ['Parent','(NH3)','(H2O)', '(NH3)-(H2O)','(H2O)-(NH3)', 'a', '2(H2O)', '2(NH3)']
conserve_line_mass_dict = {'Parent': pep.pep_mass, 'a': pep.pep_mass - 28.0106}
for i in rows:
    if i not in conserve_line_mass_dict:
        conserve_line_mass_dict[i] = pep.pep_mass - neutral_loss_mass.mass_of_loss(i)
print(conserve_line_mass_dict)

def classify_conserve_line(row):
    the_mass = row['chosen_sum']
    for i in conserve_line_mass_dict:
        if the_mass < conserve_line_mass_dict[i] + 1 and the_mass > conserve_line_mass_dict[i] - 1:
            return i
    else:
        return None

df['conserve_line'] = df.apply(classify_conserve_line, axis = 1)
    

def combine_rows(row1, row2):
    return [
        r1 if r1 != '--' else r2
        if r2 != '--' else '--'
        for r1, r2 in zip(row1, row2)
    ]

# Optional: define a custom letter order; default is alphabetical a<...<z
LETTER_ORDER = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}

def first_letter(s):
    """Return the first alphabetic letter (lowercased) in s, or None if not found/NaN."""
    if pd.isna(s):
        return None
    m = re.search(r'[A-Za-z]', str(s))
    return m.group(0).lower() if m else None

def ion_key_by_first_letter(s):
    """Sort key based only on first alphabetic letter; unknowns go to the end."""
    lt = first_letter(s)
    return LETTER_ORDER.get(lt, 10_000)  # big number pushes unknowns to the end

def canonicalize_row_by_first_letter(row, paired_prefixes=("mass_difference", "charge")):
    """
    For a row with ion1/ion2 (e.g., 'y3', 'b7', 'bi(2-4)'):
    - Put the 'earlier' ion (by first letter) in 'ion_first', the other in 'ion_second'.
    - For each prefix in paired_prefixes, move the matching *1/*2 values into
      '<prefix>_first' and '<prefix>_second' in the same order as the ions.
    """
    ion1, ion2 = row["ion1"], row["ion2"]

    # decide if we need to swap based only on first letter
    swap = ion_key_by_first_letter(ion1) > ion_key_by_first_letter(ion2)

    out = {
        "ion_first":  ion2 if swap else ion1,
        "ion_second": ion1 if swap else ion2,
    }

    # remap any number of paired columns like mass_difference1/2, charge1/2, etc.
    for pref in paired_prefixes:
        v1 = row.get(f"{pref}1", np.nan)
        v2 = row.get(f"{pref}2", np.nan)
        out[f"{pref}_first"]  = v2 if swap else v1
        out[f"{pref}_second"] = v1 if swap else v2

    return pd.Series(out)

paired = ("mass_difference", "charge", 'loss')  # add more prefixes if you have them

new_cols = df.apply(
    lambda r: canonicalize_row_by_first_letter(r, paired_prefixes=paired),
    axis=1
)

## !!! switch the ion order to let b and a ion before y ion

df = pd.concat([df, new_cols], axis=1)
#print(df[['ion1','ion_first','ion_second','mass_difference1', 'mass_difference_first', 'charge1', 'charge_first', 'loss1', 'loss_first']])

def create_mass_conserve_line(row):
    ion1,ion2 = row['ion_first'], row['ion_second']
    loss1,loss2 = row['loss_first'], row['loss_second']
    if pd.isna(loss1):
        loss1 = ''
    if pd.isna(loss2):
        loss2 = ''
        
        
    #print((type(ion1) == str) and (ion1[0] == 'a' and ion1[:2] != 'ai'))
    #print((type(ion2) == str) and (ion2[0] == 'a' and ion2[:2] != 'ai')and (loss1 == '' and loss2 == ''))
    if (type(ion1) == str) and (ion1[0] == 'a' and ion1[:2] != 'ai') and (loss1 == '' and loss2 == ''):
        combined_loss = 'a'
    elif (type(ion2) == str) and (ion2[0] == 'a' and ion2[:2] != 'ai') and (loss1 == '' and loss2 == ''):
        combined_loss = 'a'
    
    if (type(ion1) == str) and (ion1[0] == 'a' and ion1[:2] != 'ai'):
        loss1 = 'a' + '+' + loss1 if loss1 != '' else 'a'
        
    if (type(ion2) == str) and (ion2[0] == 'a' and ion2[:2] != 'ai'):
        loss2 = 'a' + '+' + loss2 if loss2 != '' else 'a'
        
    elif loss1 != '' and loss2 != '':
        #combined_loss = loss1 + ' + ' +loss2
        combined_loss = loss1 + '-' +loss2

    else:
        combined_loss = loss1 + loss2
    if combined_loss == '':
        combined_loss = 'Parent'
    return combined_loss.replace(' ', '')

df['conserve_line'] = df.apply(create_mass_conserve_line, axis=1)

ion_pair_mass_dict = {}

def peptie_arrange(length):
    result = []
    for i in range(1, length):
        result.append(f'b{i}y{length - i}')
        each_b_mass = round(pep.ion_mass(f'b{i}'),2)
        each_y_mass = round(pep.ion_mass(f'y{length - i}'),2)
        ion_pair_mass_dict[f'b{i}y{length - i}'] = f'({each_b_mass}, {each_y_mass})'
        
    return result

#rows = ['Parent','NH3','H2O', 'NH3 + H2O','H2O + NH3', 'a']

columns = peptie_arrange(the_length)
#df['loss_first_m'] = df['loss_first'].fillna('no loss')
#df['loss_second_m'] = df['loss_second'].fillna('no loss')

df['loss_first_m'] = df['loss_first'].fillna(df['ion_first']).fillna('undefined')
df['loss_second_m'] = df['loss_second'].fillna(df['ion_second']).fillna('undefined')

print(ion_pair_mass_dict)

reuslt_df = pd.DataFrame(index=rows, columns=columns)


#high_light = []
highlight_data = {}

total_count = {i:0 for i in rows}
unexplained_count = {i:0 for i in rows}
unexplained_list = {i:'' for i in rows}
abs_mass_difference = {i:[0,0] for i in rows}


for index, each_row in df.iterrows():
    
    rounded_sum = abs((round( abs(each_row['mass_difference1']) + abs(each_row['mass_difference2']), 2)))
    the_column = -1
    if (type(each_row['ion1']) == str and type(each_row['ion2']) == str):
        the_column = [each_row['ion1'], each_row['ion2']]
        the_column.sort()
        the_column = the_column[0] + the_column[1]
        the_column = the_column.replace('a', 'b')
    
    for i in conserve_line_mass_dict:
        if each_row['chosen_sum'] < conserve_line_mass_dict[i] + 0.5 and each_row['chosen_sum'] > conserve_line_mass_dict[i] - 0.5:
            total_count[i] += 1
            if rounded_sum >= 1 and the_column in reuslt_df.columns:
                
                unexplained_list[i] += f"{round(each_row['chosen_sum'] - conserve_line_mass_dict[i], 2)}({int(each_row['ranking'])}) \n"
                unexplained_count[i] += 1 
            elif each_row['ion1'] == '???' or each_row['ion2'] == '???':
                unexplained_list[i] += f"{round(each_row['chosen_sum'] - conserve_line_mass_dict[i], 2)}({int(each_row['ranking'])}) \n"
                unexplained_count[i] += 1 
        
        
    if (each_row['conserve_line'] in reuslt_df.index) and (the_column in reuslt_df.columns) and rounded_sum < 1:
        
        print(each_row['mass_difference1'], each_row['mass_difference2'],rounded_sum, )
        
        #print(each_row['conserve_line'])
        reuslt_df.at[each_row['conserve_line'], the_column] = f"({each_row['loss_first_m']},{each_row['loss_second_m']})" + ' \n ' +\
        f"({each_row['charge_first']} , {each_row['charge_second']})" +' \n ' + f"({str(round(each_row['mass_difference1'], 2))}, {round(each_row['mass_difference2'], 2)})" + '\n' + f"{str(int(each_row['ranking']))}"
        
        #high_light.append(int(each_row['ranking']))
        highlight_data[int(each_row['ranking'])] = each_row['conserve_line']
        abs_mass_difference[each_row['conserve_line']][0] += 1
        abs_mass_difference[each_row['conserve_line']][1] += abs(each_row['mass_difference1'])
        abs_mass_difference[each_row['conserve_line']][1] += abs(each_row['mass_difference2'])
        
reuslt_df = reuslt_df.fillna('--')

reuslt_df['Row_Count'] = (reuslt_df != '--').sum(axis=1)

##calculate the mean of abs value of each mass deviatoin:
for i in abs_mass_difference:
    if abs_mass_difference[i][0] > 0:
        abs_mass_difference[i] = round(abs_mass_difference[i][1] / abs_mass_difference[i][0], 2)
    else:
        abs_mass_difference[i] = 0

# Count valid entries in each column
reuslt_df.loc['Col_Count'] = (reuslt_df != '--').sum(axis=0)



total_valid = reuslt_df['Row_Count'][:-1].sum()
reuslt_df.loc['Col_Count', 'Row_Count'] = total_valid



print(total_count)
print(unexplained_count)

index_labels = ['Parent', 'NH3', 'H2O', 'NH3 + H2O', 'H2O + NH3', 'a']


# 2. Your dictionary mapping row index to color 🎨
color_map = {
    'Parent': '#7f7f7f',        # neutral gray
    '(H2O)': '#1f77b4',            # blue (water)
    '2(H2O)': '#aec7e8',         # lighter blue (double water)
    '(NH3)': '#2ca02c',            # green (ammonia)
    '(NH3)-(H2O)': '#98df8a',        # lighter green (mixed loss)
    '(H2O)-(NH3)': '#98df8a',        # same as above for symmetry
    'CH3-NH2': '#ff7f0e',        # same as above (alternate notation)
    '2(NH3)': '#ffbb78',     # light orange (combined loss)
    'a': '#CD5C5C'

}


def combine_rows_inplace(df, idx1, idx2, keep='first'):
    """
    Combine two rows in place:
      - If one cell is '--', keep the non-'--' value
      - If both are valid strings, keep the first by default
      - If both are '--', keep '--'
    The merged result replaces the row specified by `keep`.
    """
    # Ensure both indices exist
    if idx1 not in df.index or idx2 not in df.index:
        raise ValueError("One or both row indices not found in DataFrame.")

    row1, row2 = df.loc[idx1], df.loc[idx2]

    # Combine logic
    combined = row1.where(row1 != '--', row2)
    combined = combined.where(combined != '--', row1)  # covers double '--'

    # Assign result back to DataFrame
    if keep == 'first':
        df.loc[idx1] = combined
        df.drop(index=idx2, inplace=True)
    elif keep == 'second':
        df.loc[idx2] = combined
        df.drop(index=idx1, inplace=True)
    else:
        raise ValueError("keep must be 'first' or 'second'")
    return df

    # (Optional) reset index if you want continuous numbering
    # df.reset_index(drop=True, inplace=True)


reuslt_df = combine_rows_inplace(reuslt_df, '(NH3)-(H2O)', '(H2O)-(NH3)')
unexplained_count['Col_Count'] = sum(unexplained_count[v] for v in reuslt_df.index[:-1])
abs_mass_difference['Col_Count'] = round(np.mean([abs_mass_difference[i] for i in reuslt_df.index[:-1]]), 2)

reuslt_df['Unexplained Count'] = pd.Series(unexplained_count)
reuslt_df['Abs Average Mass Difference'] = pd.Series(abs_mass_difference)
reuslt_df['Unexplained Pairs'] = pd.Series(unexplained_list)
reuslt_df.loc['Ion Mass'] = ion_pair_mass_dict
reuslt_df.loc['Ion Mass'] = reuslt_df.loc['Ion Mass'].fillna(0)

reuslt_df['Unexplained Count'] = (
    reuslt_df['Unexplained Count'].fillna(0).astype(int)
)
reuslt_df['Row_Count'] = (
    reuslt_df['Row_Count'].fillna(0).astype(int)
)


print(reuslt_df)
reuslt_df = reuslt_df.map(lambda x: x.replace('\n', '<br>') if isinstance(x, str) else x)



df_display = reuslt_df.copy().astype(str)
known_components = reuslt_df.index.tolist()

def ion_pair_binary(df, exclude_cols=None):
    """
    Given a dataframe, returns a binary array (list of 0/1)
    where 1 means there is a pair (non-empty) in that ion column.
    
    Parameters:
        df (pd.DataFrame): input dataframe
        exclude_cols (list): optional list of column names to ignore at the end

    Returns:
        np.ndarray: binary array (0/1)
    """
    # Determine which columns represent ion pairs
    if exclude_cols is None:
        exclude_cols = ['Row_Count', 'Unexplained Count', 'Abs Average Mass Difference',
                        'Unexplained Pairs', 'Ion Mass']

    ion_cols = [c for c in df.columns if c not in exclude_cols]

    # Create binary array: 1 if non-empty / not NaN / not '--'
    binary = df[ion_cols].applymap(
        lambda x: 0 if pd.isna(x) or str(x).strip() in ['--', '', 'nan'] else 1
    ).values

    return binary

binary_explained_array = ion_pair_binary(reuslt_df.iloc[:1], exclude_cols = ['Row_Count', 'Unexplained Count', 'Abs Average Mass Difference', 'Unexplained Pairs'])[0]
binary_explained_array = [int(i) for i in binary_explained_array]
#binary_explained_array.append(1) if binary_explained_array[-1] == 1 else binary_explained_array.append(0)
binary_explained_array = "".join(str(item) for item in binary_explained_array)


def get_components(label_string):
    """Uses regex to find all known components within a string."""
    # Create a regex pattern like '(NH3|H2O|Parent|...)'
    pattern = '|'.join(re.escape(c) for c in known_components if c.strip())
    return re.findall(pattern, label_string)

for row_index, row_series in reuslt_df.iterrows():
    row_color = color_map.get(row_index)
    if not row_color:
        continue
        
    # Get the fundamental components for THIS row
    components_to_color = get_components(row_index)
    
    # 💡 Sort by length (descending) to replace 'H2O+NH3' before 'H2O'
    components_to_color.sort(key=len, reverse=True)
    
    current_row_data = df_display.loc[row_index]
    for component in components_to_color:
        replacement_html = f"<span style='color: {row_color}; font-weight: bold;'>{component}</span>"
        current_row_data = current_row_data.str.replace(
            component, replacement_html, regex=False
        )
    df_display.loc[row_index] = current_row_data
        
def color_index_series(index_labels_series):
    return index_labels_series.map(
        lambda label: f"color: {color_map.get(label, 'black')}; font-weight: bold;"
    )

# 5. Chain the styling methods and export
styled_df = df_display.style.apply_index(color_index_series, axis="index") \
                          .set_properties(**{
                              'border': '1px solid black',
                              'text-align': 'center'
                          })

html_output = styled_df.to_html(escape=False)

with open(f"vis/temp/{data}.html", "w") as f:
    f.write(html_output)
    
    
hti = Html2Image()
hti.output_path = "vis/temp/"
hti.screenshot(html_file=f"vis/temp/{data}.html", save_as=f'{data}_graph3.png')

'''

def plot_numbered_dots(highlight_array):
    """
    Generate a scatter plot of 50 dots labeled 1–50.
    Dots in highlight_array are drawn larger.
    """
    # Create 50 x positions spaced evenly
    x = np.arange(1, 51)
    y = np.zeros_like(x)  # place all dots on the same horizontal line

    # Create size array — larger for highlighted numbers
    sizes = np.where(np.isin(x, highlight_array), 200, 80)

    # Scatter plot
    plt.figure(figsize=(15, 3))
    plt.scatter(x, y, s=sizes, color='skyblue', edgecolor='black')

    # Add labels under each dot
    for i, val in enumerate(x):
        plt.text(x[i], -0.05, str(val), ha='center', va='top', fontsize=8)

    # Styling
    plt.axis('off')
    #plt.title("Numbered Dots (Highlighted if in given array)", fontsize=14, pad=20)
    plt.ylim(-0.2, 0.2)
    plt.xlim(0, 51)

    plt.savefig(f'vis/temp/{data}_graph4.png', bbox_inches='tight', transparent=True)
    plt.close()
'''

def plot_numbered_dots(highlight_data, color_map, data_variable_name):
    """
    Generate a scatter plot of 50 dots labeled 1–50.
    Dots in highlight_data are drawn larger and colored
    based on their conservation line via color_map.
    """
    # Create 50 x positions spaced evenly
    x = np.arange(1, 51)
    y = np.zeros_like(x)  # place all dots on the same horizontal line

    # --- New logic for sizes and colors ---
    plot_sizes = []
    plot_colors = []
    default_color = 'lightgrey'  # Color for non-highlighted dots
    fallback_color = 'black'     # Color if line is missing from color_map (for debugging)

    # Loop from 1 to 50 and decide the size and color for each dot
    for i in x:
        if i in highlight_data:
            # This dot IS highlighted
            plot_sizes.append(200)  # Larger size
            
            # Find its conservation line
            conserve_line = highlight_data[i]
            
            # Get the color from your map
            # .get() safely returns a fallback color if the line isn't in the map
            color = color_map.get(conserve_line, fallback_color)
            plot_colors.append(color)
            
        else:
            # This dot is NOT highlighted
            plot_sizes.append(80)  # Default size
            plot_colors.append(default_color)  # Default color
    # --- End of new logic ---

    # Scatter plot
    plt.figure(figsize=(15, 3))
    # Use the lists we just built for size and color
    plt.scatter(x, y, s=plot_sizes, color=plot_colors, edgecolor='black')

    # Add labels under each dot
    for i, val in enumerate(x):
        plt.text(x[i], -0.05, str(val), ha='center', va='top', fontsize=8)

    # Styling
    plt.axis('off')
    plt.ylim(-0.2, 0.2)
    plt.xlim(0, 51)
    
    
    legend_elements = [mpatches.Patch(color=default_color, label='Not Highlighted')]

    # Sort the color_map by key (the line name) to ensure a consistent legend order
    sorted_color_map = sorted(color_map.items())

    # Now, add a patch for every item in your color_map
    for line_name, color in sorted_color_map:
        legend_elements.append(mpatches.Patch(color=color, label=line_name))

    # Draw the legend
    # We place it horizontally, centered, and just below the plot
    plt.legend(
        handles=legend_elements,
        loc='upper center',          # Anchor point for the legend
        bbox_to_anchor=(0.5, -0.05), # Place it at 50% width, just below the axes
        ncol=len(legend_elements),   # Make the legend horizontal
        frameon=False,               # Remove the border
        fontsize='medium'            # Adjust font size as needed
    )

    # Use the passed data variable for the filename
    plt.savefig(f'vis/temp/{data_variable_name}_graph4.png', bbox_inches='tight', transparent=True)
    plt.close()



#plot_numbered_dots(high_light)
plot_numbered_dots(highlight_data, color_map, data)
print("Numbered dots plot saved as numbered_dots.png")

print(data, ':', sequence)
print('binary array:',binary_explained_array)
print('total count:',reuslt_df.iloc[0]['Row_Count'] + reuslt_df.iloc[0]['Unexplained Count'])
print('unexplained count:',reuslt_df.iloc[0]['Unexplained Count'])