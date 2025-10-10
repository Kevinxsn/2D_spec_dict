## This code is try to generate the second graph of the pdf, include the dataframe generation based on the data graph
## drawing function for the second graph. 

import re
import pandas as pd
import numpy as np
import peptide
def ion_data_organizer_d(df, seq):
    df['loss1'] = df['loss1'].str.replace(' ', '', regex=False)
    df['loss2'] = df['loss2'].str.replace(' ', '', regex=False)
    loss_list = list(df['loss1']) + list(df['loss2'])
    loss_set = set(loss_list)
    loss_include = []
    for i in loss_set:
        if loss_list.count(i) >=2:
            loss_include.append(i)

    for i in range(len(loss_include)):
        if type(loss_include[i]) == float:
            loss_include[i] = 'No Loss'
    df['loss1_m'] = df['loss1'].fillna('No Loss')
    df['loss2_m'] = df['loss2'].fillna('No Loss')
    pep = peptide.Pep(seq)
    pep_len = len(pep.AA_array)
    ion_list = [f'b{i}' for i in range(1, pep_len+1)]
    df_generate = pd.DataFrame(columns=ion_list, index = loss_include)
    
    for index, each_row in df.iterrows():
        #if each_row['ion1'] in df_generate.columns and each_row['loss1_m'] in df_generate.index and type(each_row['charge2']) is str and type(each_row['charge1']) is str:
        if (
                
                each_row['loss1_m'] in df_generate.index
                and isinstance(each_row['ion1'], str)
                and isinstance(each_row['charge1'], str)
                and isinstance(each_row['charge2'], str)
                and (
                    each_row['ion1'] in df_generate.columns or
                    re.match(r'^a\d+(-[A-Za-z0-9]+)?$', each_row['ion1'])
                )
            ):
            
            if each_row['loss1_m'] != 'No Loss':
                df_generate.at[each_row['loss1_m'], each_row['ion1'].replace('a', 'b')] = each_row['ion1'] + \
                '-' + each_row['loss1_m'] + f' ({round(each_row["mass_difference1"], 2)})' + \
                f' ({each_row["charge1"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge2"].replace("+", "")})'
            else: 
                df_generate.at[each_row['loss1_m'], each_row['ion1'].replace('a', 'b')] = each_row['ion1'] + \
                f' ({round(each_row["mass_difference1"], 2)})' + \
                f' ({each_row["charge1"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge2"].replace("+", "")})'
    
    for index, each_row in df.iterrows():
        #if each_row['ion2'] in df_generate.columns and each_row['loss2_m'] in df_generate.index and type(each_row['charge2']) is str and type(each_row['charge1']) is str:
        if (
                each_row['loss2_m'] in df_generate.index
                and isinstance(each_row['charge1'], str)
                and isinstance(each_row['charge2'], str)
                and   (
                    each_row['ion2'] in df_generate.columns or
                    re.match(r'^a\d+(-[A-Za-z0-9]+)?$', each_row['ion2'])
                )
            ):
            if each_row['loss2_m'] != 'No Loss':
                df_generate.at[each_row['loss2_m'], each_row['ion2'].replace('a', 'b')] = each_row['ion2'] + \
                '-' + each_row['loss2_m'] + f' ({round(each_row["mass_difference2"], 2)})' + \
                f' ({each_row["charge2"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge1"].replace("+", "")})'
            else: 
                df_generate.at[each_row['loss2_m'], each_row['ion2'].replace('a', 'b')] = each_row['ion2'] + \
                f' ({round(each_row["mass_difference2"], 2)})' + \
                f' ({each_row["charge2"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge1"].replace("+", "")})'
    df_generate = df_generate.dropna(axis=1, how='all')
    df_generate = df_generate.dropna(axis=0, how='all')
    return df_generate


def ion_data_organizer_y(df, seq):
    df['loss1'] = df['loss1'].str.replace(' ', '', regex=False)
    df['loss2'] = df['loss2'].str.replace(' ', '', regex=False)
    loss_list = list(df['loss1']) + list(df['loss2'])
    loss_set = set(loss_list)
    loss_include = []
    for i in loss_set:
        if loss_list.count(i) >=2:
            loss_include.append(i)

    for i in range(len(loss_include)):
        if type(loss_include[i]) == float:
            loss_include[i] = 'No Loss'
    df['loss1_m'] = df['loss1'].fillna('No Loss')
    df['loss2_m'] = df['loss2'].fillna('No Loss')
    pep = peptide.Pep(seq)
    pep_len = len(pep.AA_array)
    ion_list = [f'y{i}' for i in range(1, pep_len+1)]
    df_generate = pd.DataFrame(columns=ion_list, index = loss_include)
    
    for index, each_row in df.iterrows():
        if each_row['ion1'] in df_generate.columns and each_row['loss1_m'] in df_generate.index and type(each_row['charge2']) is str and type(each_row['charge1']) is str:
            
            if each_row['loss1_m'] != 'No Loss':
                df_generate.at[each_row['loss1_m'], each_row['ion1']] = each_row['ion1'] + \
                '-' + each_row['loss1_m'] + f' ({round(each_row["mass_difference1"], 2)})' + \
                f' ({each_row["charge1"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge2"].replace("+", "")})'
            else: 
                df_generate.at[each_row['loss1_m'], each_row['ion1']] = each_row['ion1'] + \
                f' ({round(each_row["mass_difference1"], 2)})' + \
                f' ({each_row["charge1"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge2"].replace("+", "")})'
    
    for index, each_row in df.iterrows():
        if each_row['ion2'] in df_generate.columns and each_row['loss2_m'] in df_generate.index and type(each_row['charge2']) is str and type(each_row['charge1']) is str:

            if each_row['loss2_m'] != 'No Loss':
                df_generate.at[each_row['loss2_m'], each_row['ion2']] = each_row['ion2'] + \
                '-' + each_row['loss2_m'] + f' ({round(each_row["mass_difference2"], 2)})' + \
                f' ({each_row["charge2"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge1"].replace("+", "")})'
            else: 
                df_generate.at[each_row['loss2_m'], each_row['ion2']] = each_row['ion2'] + \
                f' ({round(each_row["mass_difference2"], 2)})' + \
                f' ({each_row["charge2"].replace("+", "")}' + ' ,' + \
                f' {each_row["charge1"].replace("+", "")})'
    df_generate = df_generate.dropna(axis=1, how='all')
    df_generate = df_generate.dropna(axis=0, how='all')
    return df_generate


import matplotlib.pyplot as plt
import re
import pandas as pd
from matplotlib.patches import Patch # Import Patch for the legend

def parse_sequence(sequence):
    """
    Parses a peptide sequence that may contain modified residues.
    """
    residues = re.findall('[A-Z][a-z]*', sequence)
    return residues

def generate_b_ion_fragments(residues):
    """
    Generates a list of cumulative b-ion fragment sequences from the N-terminus.
    """
    fragments = []
    current_fragment = ""
    for residue in residues:
        current_fragment += residue
        fragments.append(current_fragment)
    return fragments

def plot_peptide_fragmentation(sequence, annotations=None, y_line_annotations=None, internal_peptides=None, color_map=None, show=True, save_path=None):
    """
    Generates and displays a visualization of peptide fragmentation with custom string annotations and a legend.

    Args:
        sequence (str): The peptide sequence to visualize.
        annotations (list of lists of tuples, optional): 
            Annotations for ABOVE the b-line nodes. Format: [(string, color), ...].
        y_line_annotations (list of lists of tuples, optional):
            Annotations for BELOW the y-line nodes. Format: [(string, color), ...].
        internal_peptides (list of tuples, optional): Data for internal peptide lines.
                                                      Format: [(start, end, label), ...].
        color_map (dict, optional): A dictionary mapping molecule/loss names to colors for the legend.
        show (bool): Whether to display the plot.
        save_path (str, optional): Path to save the figure.
    """
    residues = parse_sequence(sequence)
    b_ion_fragments = generate_b_ion_fragments(residues)
    num_fragments = len(residues)

    fig, ax = plt.subplots(figsize=(max(12, num_fragments * 1.2), 7))
    y_b_line, y_y_line = 1.0, 0.0
    line_color = '#333333'
    line_width = 1.5

    # Drawing main b/y lines and arrows (unchanged)
    ax.axhline(y=y_b_line, color=line_color, lw=line_width, zorder=1)
    ax.axhline(y=y_y_line, color=line_color, lw=line_width, zorder=1)
    arrow_length = 0.3
    ax.arrow(num_fragments + 0.5, y_b_line, arrow_length, 0,
             head_width=0.08, head_length=0.15, fc=line_color, ec=line_color, lw=line_width, zorder=1)
    ax.arrow(num_fragments + 0.5, y_y_line, arrow_length, 0,
             head_width=0.08, head_length=0.15, fc=line_color, ec=line_color, lw=line_width, zorder=1)

    # Drawing vertical fragment lines and annotations
    for i, fragment in enumerate(b_ion_fragments):
        x_pos = i + 1
        ax.plot([x_pos, x_pos], [y_y_line, y_b_line], color='#673ab7', lw=1.5, zorder=2)
        ax.plot(x_pos, y_b_line, 'o', ms=12, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        ax.plot(x_pos, y_y_line, 'o', ms=12, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        label_text = f"{i+1} {fragment}"
        ax.text(x_pos, y_b_line + 0.15, label_text, ha='left', va='bottom', rotation=60, fontsize=11)

        # --- Annotations ABOVE the b-line node ---
        if annotations and i < len(annotations) and annotations[i]:
            y_text_start = y_b_line + 1.4
            y_text_step = 0.15
            for j, annotation_tuple in enumerate(annotations[i]):
                annotation_string, color = annotation_tuple
                y_text = y_text_start + (j * y_text_step)
                ax.text(x_pos, y_text, annotation_string, ha='center', va='bottom',
                        color=color, fontsize=9, fontweight='bold')
        
        # --- Annotations BELOW the y-line node ---
        if y_line_annotations and i < len(y_line_annotations) and y_line_annotations[i]:
            y_text_start = y_y_line - 0.15
            y_text_step = 0.15
            for j, annotation_tuple in enumerate(y_line_annotations[i]):
                annotation_string, color = annotation_tuple
                y_text = y_text_start - (j * y_text_step)
                ax.text(x_pos, y_text, annotation_string, ha='center', va='top',
                        color=color, fontsize=9, fontweight='bold')

    # --- Logic for plotting internal peptides (unchanged) ---
    if internal_peptides:
        y_internal_start, y_internal_step = -0.5, -0.16
        lane_ends = [-1.0] * 20
        sorted_peptides = sorted(internal_peptides, key=lambda p: (p[0], p[1] - p[0]))
        for start, end, label in sorted_peptides:
            placed = False
            for lane_idx in range(len(lane_ends)):
                if start > lane_ends[lane_idx] + 0.5:
                    y_pos = y_internal_start + (lane_idx * y_internal_step)
                    ax.plot([start, end], [y_pos, y_pos], color='black', linewidth=2.5, solid_capstyle='butt')
                    ax.text(end + 0.2, y_pos, str(label), ha='left', va='center', color='green', fontsize=6, fontweight='bold')
                    lane_ends[lane_idx] = float(end)
                    placed = True
                    break
            if not placed:
                print(f"Warning: Could not place internal peptide {label} ({start}-{end}).")

    # Adjust plot limits and appearance
    ax.set_xlim(0.5, num_fragments + 1.5)
    ax.set_ylim(-3.0, 3.0) 
    ax.set_yticks([y_y_line, y_b_line])
    ax.set_yticklabels(['y-line', 'b-line'], fontsize=14, fontweight='bold')
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='y', length=0)
    plt.title(f'Fragmentation Diagram for: {sequence}', fontsize=16, pad=20)

    # --- NEW: Adding the legend ---
    if color_map:
        # Find which colors are actually used to avoid a cluttered legend.
        used_colors = set()
        if annotations:
            for site_annotations in annotations:
                for _, color in site_annotations:
                    used_colors.add(color)
        if y_line_annotations:
             for site_annotations in y_line_annotations:
                for _, color in site_annotations:
                    used_colors.add(color)
        
        # Create legend elements only for the colors that are present on the graph.
        legend_elements = [Patch(facecolor=color, edgecolor=color, label=label)
                           for label, color in color_map.items() if color in used_colors]

        if legend_elements:
            ax.legend(handles=legend_elements,
                      title="Neutral Loss",
                      bbox_to_anchor=(1.02, 1), # Position legend outside the plot
                      loc='upper left',
                      borderaxespad=0.)

    plt.tight_layout()

    if save_path:
        # Use bbox_inches='tight' to ensure the legend is included when saving.
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    
def create_annotation_list_from_df(df, peptide_length, color_map):
    """
    Processes a DataFrame for a SINGLE ion type (b or y) to create an annotation list.

    Args:
        df (pd.DataFrame): DataFrame with loss types as index and one type of ion (e.g., only 'y' columns) as columns.
        peptide_length (int): The total number of residues in the peptide.
        color_map (dict): A dictionary mapping loss types (DataFrame index) to colors.

    Returns:
        list: A single annotation list ready for the plotting function.
    """
    annotation_list = [[] for _ in range(peptide_length)]
    default_color = 'black'

    for loss_type, row in df.iterrows():
        color = color_map.get(loss_type, default_color)
        for ion_name, cell_value in row.items():
            if pd.notna(cell_value) and isinstance(cell_value, str):
                # --- MODIFIED: Extract a more detailed, specific label ---
                # This regex looks for a pattern like: y3 (-0.32) (1, 1) or b5-NH3-H2O (-0.06) (1, 1)
                # It captures the full ion name, the first parenthesis, and the first number from the second parenthesis.
                detailed_match = re.search(r'([aby]\d+[\w\-=\(\)]*?)\s+(\([^)]+\))\s*\((\d+)[^)]*\)', cell_value.strip())
                if detailed_match:
                    ion_part = detailed_match.group(1)
                    first_paren = detailed_match.group(2)
                    second_paren_num = detailed_match.group(3)
                    
                    if len(ion_part.split('-')) > 1:
                        ion_part = ion_part.split('-')[0]
                    #annotation_label = f"{ion_part} {first_paren} ({second_paren_num})"
                    annotation_label = f"{ion_part} {first_paren} {second_paren_num}"
                else:
                    # Fallback to the simpler version if the detailed format doesn't match
                    short_label_match = re.match(r'([aby]\d+)', cell_value.strip())
                    if short_label_match:
                        annotation_label = short_label_match.group(1)
                    else:
                        # Fallback in case the format is completely unexpected
                        annotation_label = cell_value 
                
                annotation_tuple = (annotation_label, color)
                # --- END MODIFICATION ---

                # Use regex to find ion type and number from the column header
                ion_match = re.match(r'([by])(\d+)', ion_name.strip())
                if ion_match:
                    ion_type = ion_match.group(1)
                    ion_number = int(ion_match.group(2))
                    
                    if ion_type == 'y':
                        # y-ions are indexed from the C-terminus
                        list_index = peptide_length - ion_number - 1
                        if 0 <= list_index < peptide_length:
                             annotation_list[list_index].append(annotation_tuple)
                    elif ion_type == 'b':
                        # b-ions are indexed from the N-terminus
                        list_index = ion_number - 1
                        if 0 <= list_index < peptide_length:
                            annotation_list[list_index].append(annotation_tuple)

    return annotation_list




neutral_loss_colors = {
    'No Loss': '#7f7f7f',        # neutral gray
    'H2O': '#1f77b4',            # blue (water)
    '2(H2O)': '#aec7e8',         # lighter blue (double water)
    'NH3': '#2ca02c',            # green (ammonia)
    'NH3-H2O': '#98df8a',        # lighter green (mixed loss)
    'H2O-NH3': '#98df8a',        # same as above for symmetry
    'CH3NH2': '#ff7f0e',         # orange (methylamine)
    'CH3-NH2': '#ff7f0e',        # same as above (alternate notation)
    'CH3NH2-NH3': '#ffbb78',     # light orange (combined loss)
    'HCOH-H2O': '#8c564b',       # brown (formaldehyde + water)
    'HN=C=N-CH3': '#9467bd',     # purple (complex nitrogen loss)
    'HN=C=NH-2(H2O)': '#c5b0d5', # light lavender (related nitrogen loss)
    'G': '#17becf'               # cyan (internal acid or glycine fragment)
}


for num in [7]:
    the_list = []
    the_y_list = []
    data_loc = f'data/data{num}.txt'

    with open(data_loc, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    peptide_header = lines[0]
    peptide_header
    the_df = pd.read_csv(f'data/data_table/data_sheet{num}.csv')
    #print(the_df)
    #the_df = the_df.dropna()
    #print(the_df)
    df_x = ion_data_organizer_d(the_df, peptide_header)
    df_y = ion_data_organizer_y(the_df, peptide_header)
    the_length = len(peptide.Pep(peptide_header).AA_array)
    b_list = create_annotation_list_from_df(df_x, the_length, neutral_loss_colors)
    y_list = create_annotation_list_from_df(df_y, the_length, neutral_loss_colors)
    df_x.to_csv(f'data/data_table_ion/data_ion_b_{num}.csv')
    df_y.to_csv(f'data/data_table_ion/data_ion_y_{num}.csv')

    save_graph_path = f'data/graph_ion/graph_ion_{num}.png'

    plot_peptide_fragmentation(peptide.Pep(peptide_header).seq, annotations=b_list, y_line_annotations = y_list, color_map=neutral_loss_colors, save_path = save_graph_path, show=False)