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
    else:
        return 'spurious'


def draw_aligned_comparison(ground_truth, other_lists):
    """
    Draws a comparison of mass spec lists.
    - Ground Truth is on top.
    - Subsets are below.
    - Numbers align vertically based on the Ground Truth value.
    - Vertical lines removed.
    - Increased spacing between numbers.
    """
    
    # 1. PRE-PROCESSING
    # Sort Ground Truth by value so the graph flows naturally from low to high
    ground_truth.sort(key=lambda x: x[0])
    
    # Create a map to ensure vertical alignment: { rounded_value : x_index }
    # We use round(val, 3) to avoid floating point mismatch issues
    val_to_x_map = {round(item[0], 3): i for i, item in enumerate(ground_truth)}
    
    # Combine all lists to find all unique categories for the legend
    all_items = ground_truth + [item for sublist in other_lists for item in sublist]
    all_categories = sorted(list(set([ion_simplify(x[1]) for x in all_items])))
    
    # 2. SETUP FIGURE
    # Height = (Number of subsets + 1 for GT) * spacing
    # CHANGE: Increased width multiplier from 1.2 to 2.5 to separate numbers
    total_rows = 1 + len(other_lists)
    fig, ax = plt.subplots(figsize=(len(ground_truth) * 2.5, total_rows * 1.0))
    
    ax.axis('off')
    
    # 3. DEFINE COLORS
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c', '#34495e']
    cat_to_color = {cat: palette[i % len(palette)] for i, cat in enumerate(all_categories)}

    # 4. DRAWING FUNCTION
    def plot_row(data, row_index, label):
        """Helper to plot a single row of numbers"""
        y_pos = row_index
        
        # Add a label for the row (e.g., "Ground Truth", "List 1")
        ax.text(-1.0, y_pos, label, fontsize=12, fontweight='bold', ha='right', va='center', color='#333')
        
        for value, cat in data:
            val_key = round(value, 3)
            
            # Only plot if this value exists in our Ground Truth Map
            if val_key in val_to_x_map:
                x_pos = val_to_x_map[val_key]
                
                # Draw the number
                ax.text(
                    x=x_pos,
                    y=y_pos,
                    s=str(val_key),
                    color=cat_to_color[ion_simplify(cat)],
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
    # CHANGE: Removed ax.axvline loop (Vertical lines are gone)

    # Set limits
    ax.set_xlim(-1.5, len(ground_truth)) 
    ax.set_ylim(-0.5, total_rows - 0.5)

    # Legend
    legend_handles = [mpatches.Patch(color=cat_to_color[cat], label=cat) for cat in all_categories]
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(all_categories), frameon=False)
    
    plt.tight_layout()
    plt.show()