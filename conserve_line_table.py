
## This script help to generate the table graph that the row is the conserve line and the column is the combination of b and y ion. 


import pandas as pd
import numpy as np
import re
import peptide
import dataframe_image as dfi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import HTML
from html2image import Html2Image

num = 7

df = pd.read_csv(f'data/data_table/data_sheet{num}.csv')
df = df.rename(columns={'Unnamed: 0': 'ranking'})
df = df[df.columns[:25]]

rows = ['Parent','NH3','H2O', 'NH3 + H2O','H2O + NH3', 'a']

the_list = []
the_y_list = []
data_loc = f'data/data{num}.txt'

with open(data_loc, "r", encoding="utf-8") as f:
    lines = [ln.rstrip("\n") for ln in f if ln.strip()]
peptide_header = lines[0]

the_length = len(peptide.Pep(peptide_header).AA_array)



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
        combined_loss = loss1 + ' + ' +loss2

    else:
        combined_loss = loss1 + loss2
    if combined_loss == '':
        combined_loss = 'Parent'
    return combined_loss


df['conserve_line'] = df.apply(create_mass_conserve_line, axis=1)

def peptie_arrange(length):
    result = []
    for i in range(1, length):
        result.append(f'b{i}y{length - i}')
    return result

#rows = ['Parent','NH3','H2O', 'NH3 + H2O','H2O + NH3', 'a']

columns = peptie_arrange(the_length)
#df['loss_first_m'] = df['loss_first'].fillna('no loss')
#df['loss_second_m'] = df['loss_second'].fillna('no loss')

df['loss_first_m'] = df['loss_first'].fillna(df['ion_first']).fillna('undefined')
df['loss_second_m'] = df['loss_second'].fillna(df['ion_second']).fillna('undefined')

reuslt_df = pd.DataFrame(index=rows, columns=columns)


high_light = []


for index, each_row in df.iterrows():
    the_column = -1
    if (type(each_row['ion1']) == str and type(each_row['ion2']) == str):
        the_column = [each_row['ion1'], each_row['ion2']]
        the_column.sort()
        the_column = the_column[0] + the_column[1]
        the_column = the_column.replace('a', 'b')
    if (each_row['conserve_line'] in reuslt_df.index) and (the_column in reuslt_df.columns):
        #print(each_row['conserve_line'])
        reuslt_df.at[each_row['conserve_line'], the_column] = f"({each_row['loss_first_m']},{each_row['loss_second_m']})" + ' \n ' +\
        f"({each_row['charge_first']} , {each_row['charge_second']})" +' \n ' + str(round(each_row['mass_difference1'] + each_row['mass_difference2'], 2)) + ' ' + f"({str(each_row['ranking'])})"
        
        high_light.append(int(each_row['ranking']))
        
reuslt_df = reuslt_df.fillna('--')

reuslt_df['Row_Count'] = (reuslt_df != '--').sum(axis=1)

# Count valid entries in each column
reuslt_df.loc['Col_Count'] = (reuslt_df != '--').sum(axis=0)

total_valid = reuslt_df['Row_Count'][:-1].sum()
reuslt_df.loc['Col_Count', 'Row_Count'] = total_valid


print(reuslt_df)


index_labels = ['Parent', 'NH3', 'H2O', 'NH3 + H2O', 'H2O + NH3', 'a']


# 2. Your dictionary mapping row index to color 🎨
color_map = {
    'Parent': '#7f7f7f',        # neutral gray
    'H2O': '#1f77b4',            # blue (water)
    '2(H2O)': '#aec7e8',         # lighter blue (double water)
    'NH3': '#2ca02c',            # green (ammonia)
    'NH3 + H2O': '#98df8a',        # lighter green (mixed loss)
    'H2O + NH3': '#98df8a',        # same as above for symmetry
    'CH3NH2': '#ff7f0e',         # orange (methylamine)
    'CH3-NH2': '#ff7f0e',        # same as above (alternate notation)
    'CH3NH2-NH3': '#ffbb78',     # light orange (combined loss)
    'HCOH-H2O': '#8c564b',       # brown (formaldehyde + water)
    'HN=C=N-CH3': '#9467bd',     # purple (complex nitrogen loss)
    'HN=C=NH-2(H2O)': '#c5b0d5', # light lavender (related nitrogen loss)
    'G': '#17becf',               # cyan (internal acid or glycine fragment)
    'a': '#CD5C5C'

}



# DOWN BELOW is the code to generate the color tale

#reuslt_df = reuslt_df.map(lambda x: x.replace('\n', '<br>') if isinstance(x, str) else x)

reuslt_df = reuslt_df.map(lambda x: x.replace('\n', '<br>') if isinstance(x, str) else x)
#reuslt_df.to_html('test.html',escape=False)
#hti = Html2Image()
# You can convert an HTML file directly
#hti.screenshot(html_file='test.html', save_as='output.png')



df_display = reuslt_df.copy().astype(str)

known_components = reuslt_df.index.tolist()

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
with open(f"data/conserve_line_pic/colored_table_and_index{num}.html", "w") as f:
    f.write(html_output)

hti = Html2Image()
hti.output_path = "data/conserve_line_pic"
hti.screenshot(html_file=f"data/conserve_line_pic/colored_table_and_index{num}.html", save_as=f'colored_table_and_index{num}.png')




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

    plt.savefig(f'data/conserve_line_pic/numbered_dots{num}.png', bbox_inches='tight', transparent=True)
    plt.close()


plot_numbered_dots(high_light)
print("Numbered dots plot saved as numbered_dots.png")

'''

# 3. A function to apply the styles
def color_rows(row):
    # Get the color from the map based on the row's name (its index)
    color = color_map.get(row.name, 'black') # Default to black if not found
    # Return a style string for each cell in the row
    return [f'color: {color}'] * len(row)

styled_df = reuslt_df.style.apply(color_rows, axis=1)

# 5. Export the STYLED DataFrame
dfi.export(styled_df, f'data/conserve_line_pic/colored_table{num}.png')

print("DataFrame has been saved as my_colored_table.png")
'''