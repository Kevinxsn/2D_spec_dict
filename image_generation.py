import matplotlib.pyplot as plt
import re

def parse_sequence(sequence):
    """
    Parses a peptide sequence that may contain modified residues.
    Modified residues are identified as a capital letter followed by lowercase letters.
    For example, 'RMe' is treated as a single modified residue.

    Args:
        sequence (str): The peptide sequence string.

    Returns:
        list: A list of amino acid/modified residue strings.
    """
    residues = re.findall('[A-Z][a-z]*', sequence)
    return residues

def generate_b_ion_fragments(residues):
    """
    Generates a list of cumulative b-ion fragment sequences from the N-terminus.

    Args:
        residues (list): A list of parsed residues.

    Returns:
        list: A list of strings, where each string is a b-ion fragment.
    """
    fragments = []
    current_fragment = ""
    for residue in residues:
        current_fragment += residue
        fragments.append(current_fragment)
    return fragments

# --- MODIFIED FUNCTION ---
def plot_peptide_fragmentation(sequence, annotations=None, show = True, save_path = None):
    """
    Generates and displays a visualization of peptide fragmentation,
    showing b-line and y-line markers with b-ion labels and custom annotations.

    Args:
        sequence (str): The peptide sequence to visualize.
        annotations (list of lists, optional): A list where each inner list contains
                                               the annotations for the corresponding vertical line.
                                               Defaults to None.
    """
    residues = parse_sequence(sequence)
    b_ion_fragments = generate_b_ion_fragments(residues)
    num_fragments = len(residues)

    # --- NEW: Define colors for annotations ---
    annotation_colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown']

    fig, ax = plt.subplots(figsize=(max(12, num_fragments * 1.2), 5))
    y_b_line = 1.0
    y_y_line = 0.0
    line_color = '#333333'
    line_width = 1.5

    ax.axhline(y=y_b_line, color=line_color, lw=line_width, zorder=1)
    ax.axhline(y=y_y_line, color=line_color, lw=line_width, zorder=1)

    arrow_length = 0.3
    ax.arrow(num_fragments + 0.5, y_b_line, arrow_length, 0,
             head_width=0.08, head_length=0.15, fc=line_color, ec=line_color, lw=line_width, zorder=1)
    ax.arrow(num_fragments + 0.5, y_y_line, arrow_length, 0,
             head_width=0.08, head_length=0.15, fc=line_color, ec=line_color, lw=line_width, zorder=1)

    for i, fragment in enumerate(b_ion_fragments):
        x_pos = i + 1

        ax.plot([x_pos, x_pos], [y_y_line, y_b_line], color='#673ab7', lw=1.5, zorder=2)
        ax.plot(x_pos, y_b_line, 'o', ms=12, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        ax.plot(x_pos, y_y_line, 'o', ms=12, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, zorder=3)

        label_text = f"{i+1} {fragment}"
        ax.text(x_pos, y_b_line + 0.15, label_text,
                ha='left', va='bottom', rotation=60, fontsize=11)

        # --- NEW: Logic to add annotations ---
        if annotations and i < len(annotations) and annotations[i]:
            # Start position for the first annotation number
            y_text_start = y_b_line - 0.2
            y_text_step = 0.15 # Spacing between numbers

            for j, annotation_val in enumerate(annotations[i]):
                color = annotation_colors[j % len(annotation_colors)] # Cycle through colors
                y_text = y_text_start - (j * y_text_step)
                ax.text(x_pos + 0.1, y_text, str(annotation_val),
                        ha='left', va='center', color=color, fontsize=10, fontweight='bold')


    ax.set_xlim(0.5, num_fragments + 1.2)
    ax.set_ylim(-0.8, 2.5)
    ax.set_yticks([y_y_line, y_b_line])
    ax.set_yticklabels(['y-line', 'b-line'], fontsize=14, fontweight='bold')
    ax.set_xticks([])
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0)
    plt.title(f'Fragmentation Diagram for: {sequence}', fontsize=16, pad=20)
    plt.tight_layout()
    
    if show == True:
        plt.show()
    
    if save_path is not None:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')

# --- Main execution block ---
if __name__ == '__main__':
    peptide_sequence = "GGNFSGRGGFGGSR"

    # --- NEW: Define your annotations here ---
    # This is a list of lists. The outer list corresponds to each vertical line from left to right.
    # Each inner list contains the numbers you want to annotate that line with.
    # An empty list [] means no annotations for that line.
    # I've populated this with numbers from your example image.
    custom_annotations = [
        [],                          # 1: G
        [],                          # 2: GG
        [23],                        # 3: GGN
        [5, 10, 29],                 # 4: GGNF
        [3, 7, 12, 19, 20, 25, 26, 28], # 5: GGNFS
        [2, 4, 9],                   # 6: GGNFSG
        [1, 36, 46],                 # 7: GGNFSGR*
        [6, 14, 18, 33],             # 8: GGNFSGR*G
        [16, 35],                    # 9: GGNFSGR*GG
        [11, 39],                    # 10: GGNFSGR*GGF
        [],                          # 11: GGNFSGR*GGFG
        [15, 21, 44, 48],            # 12: GGNFSGR*GGFGG
        [34],                        # 13: GGNFSGR*GGFGGS
        []                           # 14: GGNFSGR*GGFGGSR (This is the last residue, typically no line after it)
    ]


    # Generate and display the plot with the new annotations
    plot_peptide_fragmentation(peptide_sequence, annotations=custom_annotations)
