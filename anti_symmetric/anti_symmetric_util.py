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
    elif the_input[0] == 'y':
        return 'y'
    elif the_input[0] == 'b'and len(the_input) > 1 and the_input[1] != 'i':
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


def draw_aligned_comparison(ground_truth, other_lists, save_path=None):
    """
    Draws a comparison of mass spec lists with fixed colors for b, y, and spurious.
    """
    
    # 1. DEFINE FIXED COLORS
    # Map the simplified categories to specific colors
    FIXED_COLORS = {
        'b': '#87CEEB',        # Blue
        'y': '#90EE90',        # Red
        'spurious': '#e74c3c', # Gray
        'unknown': '#000000'   # Black (fallback)
    }

    # 2. PRE-PROCESSING
    # Sort Ground Truth by value
    ground_truth.sort(key=lambda x: x[0])
    
    # Create a map to ensure vertical alignment: { rounded_value : x_index }
    val_to_x_map = {round(item[0], 3): i for i, item in enumerate(ground_truth)}
    
    # 3. SETUP FIGURE
    total_rows = 1 + len(other_lists)
    fig, ax = plt.subplots(figsize=(len(ground_truth) * 2.5, total_rows * 1.0))
    ax.axis('off')
    
    # 4. DRAWING FUNCTION
    def plot_row(data, row_index, label):
        """Helper to plot a single row of numbers"""
        y_pos = row_index
        
        # Row Label
        ax.text(-1.0, y_pos, label, fontsize=12, fontweight='bold', ha='right', va='center', color='#333')
        
        for value, raw_cat in data:
            val_key = round(value, 3)
            
            # Only plot if this value exists in our Ground Truth Map
            if val_key in val_to_x_map:
                x_pos = val_to_x_map[val_key]
                
                # Determine color based on simplified category
                simple_cat = ion_simplify(raw_cat)
                text_color = FIXED_COLORS.get(simple_cat, FIXED_COLORS['unknown'])
                
                # Draw the number
                ax.text(
                    x=x_pos,
                    y=y_pos,
                    s=str(val_key),
                    color=text_color,
                    fontsize=14,
                    fontweight='bold',
                    ha='center',
                    va='center'
                )

    # 5. EXECUTE PLOTTING
    # Plot Ground Truth (Top Row)
    plot_row(ground_truth, total_rows - 1, "Ground Truth")
    
    # Plot Subsets (Iterate downwards)
    for i, subset in enumerate(other_lists):
        row_y = (total_rows - 2) - i
        plot_row(subset, row_y, f"Subset {i+1}")

    # 6. VISUAL POLISH
    ax.set_xlim(-1.5, len(ground_truth)) 
    ax.set_ylim(-0.5, total_rows - 0.5)

    # Manual Legend for the 3 fixed categories
    legend_handles = [
        mpatches.Patch(color=FIXED_COLORS['b'], label='b-ion'),
        mpatches.Patch(color=FIXED_COLORS['y'], label='y-ion'),
        mpatches.Patch(color=FIXED_COLORS['spurious'], label='Spurious')
    ]
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
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