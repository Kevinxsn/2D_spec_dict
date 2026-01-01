import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import img2pdf

def visualize_array_index(data_array, target_index, save_path=None):
    """
    Visualizes an array of strings, coloring elements based on their 
    position relative to a target index.
    
    Args:
        data_array (list of str): The input list of strings.
        target_index (int): The index to highlight.
    """
    
    # 1. Validation
    if not (0 <= target_index < len(data_array)):
        print(f"Error: Index {target_index} is out of bounds for array of length {len(data_array)}")
        return

    # 2. Setup Plot
    # Adjust figure size based on array length to prevent squishing
    fig_width = max(6, len(data_array) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 2))
    
    # Define colors
    COLOR_BEFORE = '#87CEEB'  # Sky Blue
    COLOR_TARGET = '#FFD700'  # Gold
    COLOR_AFTER  = '#90EE90'  # Light Green
    
    # 3. iterate and Draw
    for i, content in enumerate(data_array):
        # Determine color based on index comparison
        if i < target_index:
            face_color = COLOR_BEFORE
            label_text = "Before" if i == 0 else "" # Label logic for legend-like effect (optional)
        elif i == target_index:
            face_color = COLOR_TARGET
            label_text = "Target"
        else:
            face_color = COLOR_AFTER
            label_text = "After" if i == target_index + 1 else ""

        # Draw the rectangle box for the element
        # (x, y), width, height
        rect = patches.Rectangle((i, 0), 1, 1, 
                                 facecolor=face_color, 
                                 edgecolor='black',
                                 linewidth=1.5)
        ax.add_patch(rect)
        
        # Add the string text in the center of the box
        ax.text(i + 0.5, 0.5, str(content), 
                ha='center', va='center', 
                fontsize=12, fontweight='bold', color='#333333')
        
        # Add small index number below the box
        ax.text(i + 0.5, -0.2, str(i), 
                ha='center', va='top', 
                fontsize=9, color='gray')

    # 4. Final Formatting
    # Set plot limits
    ax.set_xlim(0, len(data_array))
    ax.set_ylim(-0.5, 1.5)
    
    # Remove standard axes and ticks
    ax.axis('off')
    
    # Add a title
    plt.title(f"Array Visualization (Target Index: {target_index})", pad=20)
    
    # Create a custom legend manually
    legend_elements = [
        patches.Patch(facecolor=COLOR_BEFORE, edgecolor='black', label='Before Index'),
        patches.Patch(facecolor=COLOR_TARGET, edgecolor='black', label='At Index'),
        patches.Patch(facecolor=COLOR_AFTER,  edgecolor='black', label='After Index')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.3))

    # Show the plot
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Graph saved successfully to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
        
        
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def visualize_array_range(data_array, target_indices, save_path=None):
    """
    Visualizes an array of strings, coloring elements based on their 
    position relative to a target index or list of consecutive indices.
    
    Args:
        data_array (list of str): The input list of strings.
        target_indices (int or list of int): The index (or list of indices) to highlight.
    """
    
    # --- 1. Input Normalization & Validation ---
    
    # Ensure target_indices is a list, even if a single int is passed
    if isinstance(target_indices, int):
        targets = [target_indices]
    else:
        targets = sorted(target_indices) # Sort to ensure we get true min/max

    if not targets:
        print("Error: Target list is empty.")
        return

    min_target = targets[0]
    max_target = targets[-1]

    # Check bounds
    if min_target < 0 or max_target >= len(data_array):
        print(f"Error: Target range {targets} contains indices out of bounds for array length {len(data_array)}")
        return

    # --- 2. Setup Plot ---
    # Adjust figure size based on array length
    fig_width = max(6, len(data_array) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 2))
    
    # Define colors
    COLOR_BEFORE = '#87CEEB'  # Sky Blue
    COLOR_TARGET = '#FFD700'  # Gold
    COLOR_AFTER  = '#90EE90'  # Light Green
    
    # --- 3. Iterate and Draw ---
    for i, content in enumerate(data_array):
        # Determine color based on range comparison
        if i < min_target:
            face_color = COLOR_BEFORE
        elif i > max_target:
            face_color = COLOR_AFTER
        else:
            # This covers the range [min_target, max_target]
            face_color = COLOR_TARGET

        # Draw the rectangle box
        rect = patches.Rectangle((i, 0), 1, 1, 
                                 facecolor=face_color, 
                                 edgecolor='black',
                                 linewidth=1.5)
        ax.add_patch(rect)
        
        # Add the string text in the center
        ax.text(i + 0.5, 0.5, str(content), 
                ha='center', va='center', 
                fontsize=12, fontweight='bold', color='#333333')
        
        # Add index number below
        ax.text(i + 0.5, -0.2, str(i), 
                ha='center', va='top', 
                fontsize=9, color='gray')

    # --- 4. Final Formatting ---
    ax.set_xlim(0, len(data_array))
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    
    # Dynamic Title
    if len(targets) > 1:
        title_str = f"Target Range: [{min_target} - {max_target}]"
    else:
        title_str = f"Target Index: {min_target}"
        
    plt.title(f"Array Visualization ({title_str})", pad=20)
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor=COLOR_BEFORE, edgecolor='black', label='Before Range'),
        patches.Patch(facecolor=COLOR_TARGET, edgecolor='black', label='Target Range'),
        patches.Patch(facecolor=COLOR_AFTER,  edgecolor='black', label='After Range')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.3))

    # Output
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Graph saved successfully to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()



def find_conserved_numbers(list_of_arrays, all_numbers):
    """
    Identifies which numbers have identical connections across all provided arrays.
    
    Args:
    list_of_arrays (list): A list containing lists of tuples. 
                           Example: [[(1,2)], [(2,1)]]
    all_numbers (set): A set containing all numbers to analyze.

    Returns:
    tuple: (set of conserved numbers, set of non-conserved numbers)
    """
    
    # Step 1: Parse each array into an "Adjacency Map"
    # This converts the list of tuples into a format that is easy to compare.
    # Structure: { 1: {2, 5}, 2: {1, 3} } -> Number 1 is connected to 2 and 5.
    parsed_maps = []

    for current_array in list_of_arrays:
        # We use a default dict so we don't get errors if a number has no pairs
        adj_map = defaultdict(set)
        
        for num1, num2 in current_array:
            # Add relationships both ways so order (x,y) vs (y,x) doesn't matter
            adj_map[num1].add(num2)
            adj_map[num2].add(num1)
            
        parsed_maps.append(adj_map)

    # Step 2: Check for conservation
    conserved = set()
    not_conserved = set()

    for num in all_numbers:
        # Get the partners of this number in the first array (reference)
        # If the number isn't in the array, it returns an empty set due to defaultdict
        reference_partners = parsed_maps[0][num]
        
        is_conserved = True
        
        # Compare against all other arrays
        for i in range(1, len(parsed_maps)):
            current_partners = parsed_maps[i][num]
            
            # If the set of partners is not exactly the same, it's not conserved
            if reference_partners != current_partners:
                is_conserved = False
                break
        
        if is_conserved:
            conserved.add(num)
        else:
            not_conserved.add(num)

    return conserved, not_conserved

def visualize_sets(universal_set, set_one, set_two, save_path=None):
    """
    Visualizes elements of a universal set, coloring them based on membership 
    in set_one or set_two.
    """
    
    # 0. Round all numbers to 1 decimal digit as requested
    # We update the sets themselves so the membership checks match the rounded values
    universal_set = {round(x, 1) for x in universal_set}
    set_one = {round(x, 1) for x in set_one}
    set_two = {round(x, 1) for x in set_two}

    # 1. Sort the universal set to arrange numbers from smallest to largest
    sorted_universe = sorted(list(universal_set))
    
    # 2. Prepare lists for plotting
    x_vals = []
    colors = []
    
    # Define colors
    color_set_1 = '#3498db'  # Blue
    color_set_2 = '#e74c3c'  # Red
    
    #print(f"{'Number':<10} | {'Category'}")
    #print("-" * 25)

    # 3. Iterate through the sorted universe and determine category
    # We use 'enumerate' to get a simple index (0, 1, 2...) for the x-position
    
    for i, number in enumerate(sorted_universe):
        x_vals.append(i)
        
        if number in set_one:
            colors.append(color_set_1)
            #print(f"{number:<10} | Set 1")
        elif number in set_two:
            colors.append(color_set_2)
            #print(f"{number:<10} | Set 2")
        else:
            # Fallback in case of unexpected data, though user noted this won't happen
            colors.append('black') 
            #print(f"{number:<10} | Unknown")
    
    # 4. Create the Graph
    # Adjusted figsize height to 1.5 to make it very compact
    plt.figure(figsize=(12, 1.5))
    
    # Plot the main number line (subtle)
    plt.axhline(y=0, color='gray', linewidth=0.5, zorder=1)
    
    # Add labels (the numbers themselves) directly on the line
    for x, label, color in zip(x_vals, sorted_universe, colors):
        plt.text(x, 0, str(label), 
                 ha='center', va='center', 
                 fontsize=14, fontweight='bold', color=color,
                 bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

    # 5. Add Legend and Labels
    patch1 = mpatches.Patch(color=color_set_1, label='Conserved')
    patch2 = mpatches.Patch(color=color_set_2, label='Non-Conserved')
    
    # Place legend outside or compactly
    plt.legend(handles=[patch1, patch2], loc='upper left', frameon=False, fontsize=10)
    
    plt.title("Consrved Numbers")
    
    # Hide both axes
    plt.axis('off')
    
    # Add some padding to the x-axis limits
    if len(x_vals) > 0:
        plt.xlim(min(x_vals) - 1, max(x_vals) + 1)
        
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Graph saved successfully to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
        


def ion_simplify(the_input):
    if the_input is None:
        return 'spurious'
    elif the_input[0] == 'y' and ' - ' not in the_input:
        return 'y'
    elif the_input[0] == 'b'and len(the_input) > 1 and the_input[1] != 'i' and ' - ' not in the_input:
        return 'b'
    elif the_input == 'b':
        return 'b'
    else:
        return 'spurious'
    
def mass_b_y_indentification(input_list, paired_dict):
    
    result = []
    b_ion_mass = set([input_list[0][0]])
    y_ion_mass = set([input_list[0][1]])
    for i in input_list:
        b_ion_mass.add(i[0])
        y_ion_mass.add(i[1])
    b_ion_mass.remove(input_list[0][0])
    y_ion_mass.remove(input_list[0][1])
    b_ion_mass = list(b_ion_mass)
    y_ion_mass = list(y_ion_mass)
    
    for i in b_ion_mass:
        result.append((i, 'b'))
        result.append((paired_dict[i], 'y'))
    for j in y_ion_mass:
        result.append((j, 'y'))
        result.append((paired_dict[j], 'b'))
    return result


        
def mass_b_y_indentification_with_middle(input_list):
    result = []
    b_ion_mass = set([input_list[0][0]])
    y_ion_mass = set([input_list[0][1]])
    for i in input_list:
        b_ion_mass.add(i[0])
        y_ion_mass.add(i[1])
    b_ion_mass.remove(input_list[0][0])
    y_ion_mass.remove(input_list[0][1])
    b_ion_mass = list(b_ion_mass)
    y_ion_mass = list(y_ion_mass)
    
    for i in b_ion_mass:
        result.append((i, 'b'))
    for j in y_ion_mass:
        result.append((j, 'y'))
    
    return result, [input_list[-1][0], input_list[-1][1]]



def draw_aligned_comparison(ground_truth, other_lists, aa_converter=None, save_path=None):
    """
    Draws a comparison of mass spec lists with annotations for b-ions in Ground Truth.
    
    Args:
        ground_truth: List of tuples (value, label)
        other_lists: List of lists containing subsets
        aa_converter: A function that takes a label (e.g., 'b3') and returns the Amino Acid string (e.g., 'A')
        save_path: Optional path to save the image
    """
    
    # 1. DEFINE FIXED COLORS
    FIXED_COLORS = {
        'b': '#87CEEB',        # SkyBlue
        'y': '#90EE90',        # LightGreen
        'spurious': '#e74c3c', # Red/Gray
        'unknown': '#000000'   # Black
    }

    # 2. PRE-PROCESSING
    ground_truth.sort(key=lambda x: x[0])
    
    # Map rounded values to x-indices for alignment
    val_to_x_map = {round(item[0], 3): i for i, item in enumerate(ground_truth)}
    
    # 3. SETUP FIGURE
    total_rows = 1 + len(other_lists)
    # Increase height slightly to make room for annotations
    fig, ax = plt.subplots(figsize=(len(ground_truth) * 2.5, total_rows * 1.2))
    ax.axis('off')
    
    # 4. DRAWING FUNCTION
    def plot_row(data, row_index, label, is_ground_truth=False):
        """Helper to plot a single row of numbers"""
        y_pos = row_index
        
        # Draw Row Label
        ax.text(-1.0, y_pos, label, fontsize=12, fontweight='bold', ha='right', va='center', color='#333')
        
        for value, raw_cat in data:
            val_key = round(value, 3)
            
            # Only plot if this value aligns with Ground Truth columns
            if val_key in val_to_x_map:
                x_pos = val_to_x_map[val_key]
                
                simple_cat = ion_simplify(raw_cat)
                text_color = FIXED_COLORS.get(simple_cat, FIXED_COLORS['unknown'])
                
                
                # Draw the Mass Value
                ax.text(
                    x=x_pos,
                    y=y_pos,
                    s=str(val_key),
                    color=text_color,
                    fontsize=25,
                    fontweight='bold',
                    ha='center',
                    va='center'
                )

                # --- NEW: ANNOTATION LOGIC ---
                # If this is Ground Truth AND it's a b-ion AND we have a converter
                if is_ground_truth and simple_cat == 'b' and aa_converter:
                    try:
                        aa_label = aa_converter(raw_cat) # Convert 'b3' -> 'A'
                        
                        # Draw the Amino Acid letter above the number
                        ax.text(
                            x=x_pos,
                            y=y_pos + 0.35,  # Offset y to place above
                            s=aa_label,
                            color=FIXED_COLORS['b'], # Match the b-ion color
                            fontsize=25,
                            fontweight='bold',
                            ha='center',
                            va='bottom'      # Anchor bottom of text to the offset point
                        )
                    except Exception as e:
                        print(f"Error converting label {raw_cat}: {e}")

    # 5. EXECUTE PLOTTING
    # Plot Ground Truth (Top Row) - Set is_ground_truth=True
    plot_row(ground_truth, total_rows - 1, "Ground Truth", is_ground_truth=True)
    
    # Plot Subsets (Iterate downwards)
    for i, subset in enumerate(other_lists):
        row_y = (total_rows - 2) - i
        plot_row(subset, row_y, f"Peptide {i+1}", is_ground_truth=False)

    # 6. VISUAL POLISH
    ax.set_xlim(-1.5, len(ground_truth)) 
    ax.set_ylim(-0.5, total_rows + 0.2) # Added space at top for annotations

    legend_handles = [
        mpatches.Patch(color=FIXED_COLORS['b'], label='b-ion'),
        mpatches.Patch(color=FIXED_COLORS['y'], label='y-ion'),
        mpatches.Patch(color=FIXED_COLORS['spurious'], label='Spurious')
    ]
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=20)

    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Graph saved successfully to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
        
        



def draw_aligned_comparison_b_only(ground_truth, other_lists, aa_converter=None, save_path=None, include_y=True, width_multiplier=1.5):
    """
    Args:
        width_multiplier: Controls horizontal spacing. Lower = tighter. (Try 1.0 - 1.5)
    """
    if include_y == False:
        ground_truth_modified = []
        for i in ground_truth:
            if i[1][0] != 'y':
                ground_truth_modified.append(i)
        ground_truth = ground_truth_modified
        
    
    # 1. DEFINE FIXED COLORS
    FIXED_COLORS = {
        'b': '#87CEEB',        # SkyBlue
        'y': '#90EE90',        # LightGreen
        'spurious': '#e74c3c', # Red/Gray
        'unknown': '#000000'   # Black
    }
    
    # Font configuration
    MAIN_FONT_SIZE = 15  # Reduced from 25 to fit tighter spacing
    LABEL_FONT_SIZE = 14
    
    # 2. PRE-PROCESSING
    ground_truth.sort(key=lambda x: x[0])
    
    # Map rounded values to x-indices for alignment
    val_to_x_map = {round(item[0], 3): i for i, item in enumerate(ground_truth)}
    x_to_cat_map = {round(item[0], 3): ion_simplify(item[1]) for i, item in enumerate(ground_truth)}
    
    # 3. SETUP FIGURE
    total_rows = 1 + len(other_lists)
    
    # --- FIX: Calculate a more reasonable width ---
    # We use the width_multiplier to control space per item.
    calculated_width = len(ground_truth) * width_multiplier
    # Optional: Set a minimum width so it doesn't get too squished if data is small
    fig_width = max(10, calculated_width)
    
    fig, ax = plt.subplots(figsize=(fig_width, total_rows * 1.5))
    ax.axis('off')
    
    # 4. DRAWING FUNCTION
    def plot_row(data, row_index, label, is_ground_truth=False):
        """Helper to plot a single row of numbers"""
        y_pos = row_index
        
        # Draw Row Label
        ax.text(-0.5, y_pos, label, fontsize=LABEL_FONT_SIZE, fontweight='bold', ha='right', va='center', color='#333')
        
        for value, raw_cat in data:
            
            
            
            val_key = round(value, 3)
            
            # Only plot if this value aligns with Ground Truth columns
            if val_key in val_to_x_map:
                x_pos = val_to_x_map[val_key]
                
                #raw_cat = x_to_cat_map[val_key]
                simple_cat = ion_simplify(raw_cat)
                simple_cat
                

                text_color = FIXED_COLORS.get(simple_cat, FIXED_COLORS['unknown'])
                
                # Determine if we should draw the text based on your logic
                should_draw = False
                
                if is_ground_truth:
                    if include_y:
                        should_draw = True
                    elif simple_cat == 'b' or simple_cat == 'spurious':
                        should_draw = True
                else:
                    # Non-ground truth (subsets)
                    if simple_cat == 'b':
                        should_draw = True
                
                if should_draw:
                    ax.text(
                        x=x_pos,
                        y=y_pos,
                        s=f"{val_key:.3f}", # Ensure consistent formatting
                        #color=text_color,
                        color = FIXED_COLORS.get(x_to_cat_map[val_key], FIXED_COLORS['unknown']),
                        fontsize=MAIN_FONT_SIZE,
                        fontweight='bold',
                        ha='center',
                        va='center'
                    )

                # --- ANNOTATION LOGIC ---
                if is_ground_truth and simple_cat == 'b' and aa_converter:
                    try:
                        aa_label = aa_converter(raw_cat)
                        ax.text(
                            x=x_pos,
                            y=y_pos + 0.3, # Slightly reduced offset
                            s=aa_label,
                            color=FIXED_COLORS['b'],
                            fontsize=MAIN_FONT_SIZE + 2, # Slightly larger for AA
                            fontweight='bold',
                            ha='center',
                            va='bottom'
                        )
                    except Exception as e:
                        print(f"Error converting label {raw_cat}: {e}")

    # 5. EXECUTE PLOTTING
    plot_row(ground_truth, total_rows - 1, "Ground Truth", is_ground_truth=True)
    
    for i, subset in enumerate(other_lists):
        row_y = (total_rows - 2) - i
        plot_row(subset, row_y, f"Peptide {i+1}", is_ground_truth=False)

    # 6. VISUAL POLISH
    # Tighten the x-limits to remove empty space on sides
    ax.set_xlim(-1.5, len(ground_truth)) 
    ax.set_ylim(-0.5, total_rows + 0.5)

    legend_handles = [
        mpatches.Patch(color=FIXED_COLORS['b'], label='b-ion'),
        mpatches.Patch(color=FIXED_COLORS['y'], label='y-ion'),
        mpatches.Patch(color=FIXED_COLORS['spurious'], label='Spurious')
    ]
    
    # Adjusted legend font size
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=14)

    plt.tight_layout()
    
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Graph saved successfully to: {save_path}")
    else:
        plt.show()


def draw_sequence_with_middle_points(data_tuple, save_path=None):
    """
    Draws a sequence of numbers where:
    1. The main list (b/y ions) is sorted and displayed first.
    2. The 'middle points' are appended to the end of the visualization.
    3. Colors are assigned strictly to 'b', 'y', and 'middle point'.
    
    Args:
        data_tuple (tuple): (main_list, middle_points_list)
            e.g. ([(484.2, 'b'), ...], [742.3, 679.3])
    """
    # 1. Unpack the data
    main_data, middle_points_values = data_tuple
    
    # 2. Prepare the Data
    # Sort the main data by value (standard for mass spec visualization)
    main_data_sorted = sorted(main_data, key=lambda x: x[0])
    
    # Create tuples for the middle points with the specific category
    middle_data = [(val, 'middle point') for val in middle_points_values]
    
    # Combine: Main data first, then Middle points appended at the end
    full_sequence = main_data_sorted + middle_data
    
    # 3. Define Fixed Colors
    # b = Blue, y = Red, middle point = Green (or Purple/Orange as preferred)
    COLORS = {
        'b': '#87CEEB',         # Blue
        'y': '#90EE90',         # Red
        'middle point': '#FFD700' # Green (Distinct)
    }

    # 4. Setup Figure
    # Width depends on total number of items
    fig, ax = plt.subplots(figsize=(len(full_sequence) * 1.5, 2))
    ax.axis('off')
    
    # 5. Draw the Numbers
    for i, (value, category) in enumerate(full_sequence):
        # Round for display
        display_text = str(round(value, 3))
        
        # Determine Color
        # We simplify category to handle potential whitespace or casing, though inputs seem clean
        cat_key = category.strip() 
        text_color = COLORS.get(cat_key, '#000000') # Default to black if unknown
        
        # Plot Text
        ax.text(
            x=i, 
            y=0.5, 
            s=display_text, 
            color=text_color, 
            fontsize=14, 
            fontweight='bold', 
            ha='center', 
            va='center'
        )

    # 6. Visual Polish
    ax.set_xlim(-0.5, len(full_sequence) - 0.5)
    ax.set_ylim(0, 1)
    
    # Add Legend
    legend_handles = [
        mpatches.Patch(color=COLORS['b'], label='b-ion'),
        mpatches.Patch(color=COLORS['y'], label='y-ion'),
        mpatches.Patch(color=COLORS['middle point'], label='Middle Point')
    ]
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)
    
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Graph saved successfully to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
        

def get_corresponded_aa(the_peptide, the_ion_name):
    # Given a peptide sequence and an ion name (e.g., 'b3', 'y5'), return the corresponding amino acid.
    
    if the_ion_name is None or len(the_ion_name) < 2:
        return None
    ion_type = the_ion_name[0]  # 'b' or 'y'
    ion_index = int(the_ion_name[1:])
    if ion_type == 'b':
        if 1 <= ion_index <= len(the_peptide):
            return the_peptide.AA_array[ion_index - 1]  # b-ions are 1-indexed
    elif ion_type == 'y':
        if 1 <= ion_index <= len(the_peptide):
            return the_peptide.AA_array[-ion_index]  # y-ions count from the end
    