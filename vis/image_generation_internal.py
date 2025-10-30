import matplotlib.pyplot as plt
import re

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

def plot_peptide_fragmentation(sequence, annotations=None, internal_peptides=None, show=True, save_path=None):
    """
    Generates and displays a visualization of peptide fragmentation with improved
    layout for internal peptide annotations.

    Args:
        sequence (str): The peptide sequence to visualize.
        annotations (list of lists, optional): Annotations for the vertical lines.
        internal_peptides (list of tuples, optional): Data for internal peptide lines.
                                                      Format: [(start, end, label), ...].
        show (bool): Whether to display the plot.
        save_path (str, optional): Path to save the figure.
    """
    residues = parse_sequence(sequence)
    b_ion_fragments = generate_b_ion_fragments(residues)
    num_fragments = len(residues)

    annotation_colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'black']

    # --- MODIFIED: Increased figure height for more vertical space ---
    fig, ax = plt.subplots(figsize=(max(12, num_fragments * 1.2), 7))
    y_b_line = 1.0
    y_y_line = 0.0
    line_color = '#333333'
    line_width = 1.5

    # --- Drawing main b/y lines and arrows (unchanged) ---
    ax.axhline(y=y_b_line, color=line_color, lw=line_width, zorder=1)
    ax.axhline(y=y_y_line, color=line_color, lw=line_width, zorder=1)
    arrow_length = 0.3
    ax.arrow(num_fragments + 0.5, y_b_line, arrow_length, 0,
             head_width=0.08, head_length=0.15, fc=line_color, ec=line_color, lw=line_width, zorder=1)
    ax.arrow(num_fragments + 0.5, y_y_line, arrow_length, 0,
             head_width=0.08, head_length=0.15, fc=line_color, ec=line_color, lw=line_width, zorder=1)

    # --- Drawing vertical fragment lines and annotations (unchanged) ---
    for i, fragment in enumerate(b_ion_fragments):
        x_pos = i + 1
        ax.plot([x_pos, x_pos], [y_y_line, y_b_line], color='#673ab7', lw=1.5, zorder=2)
        ax.plot(x_pos, y_b_line, 'o', ms=12, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        ax.plot(x_pos, y_y_line, 'o', ms=12, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        label_text = f"{i+1} {fragment}"
        ax.text(x_pos, y_b_line + 0.15, label_text, ha='left', va='bottom', rotation=60, fontsize=11)

        if annotations and i < len(annotations) and annotations[i]:
            y_text_start = y_b_line - 0.2
            y_text_step = 0.15
            for j, annotation_val in enumerate(annotations[i]):
                color = annotation_colors[j % len(annotation_colors)]
                y_text = y_text_start - (j * y_text_step)
                ax.text(x_pos + 0.1, y_text, str(annotation_val), ha='left', va='center', color=color, fontsize=10, fontweight='bold')

    # --- NEW: Improved logic for plotting internal peptides ---
    if internal_peptides:
        y_internal_start = -0.5  # Starting y-position for the first lane
        y_internal_step = -0.16   # Vertical distance between lanes
        
        # Tracks the ending x-coordinate of the last peptide in each lane.
        # Initialize with a low value. Max 20 lanes supported.
        lane_ends = [-1.0] * 20

        # Sort peptides by start position, then by length, for better packing
        sorted_peptides = sorted(internal_peptides, key=lambda p: (p[0], p[1] - p[0]))

        for start, end, label in sorted_peptides:
            placed = False
            # Find the first available lane where the new peptide doesn't overlap
            for i in range(len(lane_ends)):
                # Check for space (add a small 0.5 gap for clarity)
                if start > lane_ends[i] + 0.5:
                    y_pos = y_internal_start + (i * y_internal_step)
                    
                    # Draw the horizontal line for the internal peptide
                    ax.plot([start, end], [y_pos, y_pos], color='black', linewidth=2.5, solid_capstyle='butt')
                    
                    # Add the label number at the end of the line
                    ax.text(end + 0.2, y_pos, str(label), ha='left', va='center', color='green', fontsize=9, fontweight='bold')
                    
                    # Update the end position for this lane
                    lane_ends[i] = float(end)
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place internal peptide {label} ({start}-{end}). Consider increasing max lanes.")

    # --- MODIFIED: Adjust plot limits and appearance ---
    ax.set_xlim(0.5, num_fragments + 1.5)
    ax.set_ylim(-3.0, 3.0) # Expanded y-limits
    ax.set_yticks([y_y_line, y_b_line])
    ax.set_yticklabels(['y-line', 'b-line'], fontsize=14, fontweight='bold')
    ax.set_xticks([])
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0)
    plt.title(f'Fragmentation Diagram for: {sequence}', fontsize=16, pad=20)
    plt.tight_layout()

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


# --- Example Usage (using data from your image) ---
if __name__ == '__main__':
    peptide_sequence = "VTIMPKDIQLAR"

    # Annotations for the vertical b/y lines
    custom_annotations = [
        [], [], ['6', '46'], ['20', '26', '36'], ['15', '27', '35'], ['10', '18', '21', '48'],
        ['31', '38', '47'], ['5', '17', '40', '49'], [], ['13'], ['19', '54'], []
    ]

    # Internal peptide data: list of (start_residue, end_residue, label)
    # I've estimated these from your provided image
    internal_peptides_data = [
        (5, 7, '2'), (5, 10, '4'), (5, 9, '7'), (5, 11, '11'),
        (5, 12, '16'), (6, 12, '23'), (6, 11, '25'), (7, 8, '28'),
        (7, 10, '29'), (8, 11, '30'), (10, 12, '32'), (9, 10, '34'),
        (9, 11, '37'), (8, 10, '40'), (8, 9, '47'), (4, 5, '51'),
        (11, 12, '52'), (10, 11, '56'), (11, 12, '39') # Added 39
    ]

    plot_peptide_fragmentation(
        peptide_sequence,
        annotations=custom_annotations,
        internal_peptides=internal_peptides_data
    )


'''
if __name__ == '__main__':
    peptide_sequence = "GGNFSGRGGFGGSR"

    custom_annotations = [
        [], [], [23], [5, 10, 29], [3, 7, 12, 19, 20, 25, 26, 28],
        [2, 4, 9], [1, 36, 46], [6, 14, 18, 33], [16, 35], [11, 39],
        [], [15, 21, 44, 48], [34], []
    ]

    internal_lines_data = [
        (3, 5, 50),
        (5, 6, 49),
        (6, 7, 47),
        (7, 8, 30),
        (8, 9, 8)
    ]

    plot_peptide_fragmentation(
        peptide_sequence,
        annotations=custom_annotations,
        internal_peptides=internal_lines_data
    )
'''