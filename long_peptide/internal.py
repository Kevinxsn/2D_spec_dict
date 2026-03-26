import numpy as np

def find_optimal_shift_and_virtual_ions(A, B, delta=0.1, overlap_threshold=3):
    """
    Finds the optimal shift s to align set A to set B.
    Returns: optimal_s, SIM score, and the list of virtual ions (DIFF).
    """
    intervals = []
    
    # 1. Create intervals for every possible pair (a, b)
    # Each interval represents the range of 's' that makes a + s match b
    for a in A:
        for b in B:
            intervals.append((b - a - delta, 1))  # Start of interval
            intervals.append((b - a + delta, -1)) # End of interval
            
    # 2. Sweep Line Algorithm
    # Sort by the shift value. If values are equal, process 'start' (1) before 'end' (-1)
    intervals.sort(key=lambda x: (x[0], -x[1]))
    
    max_matches = 0
    current_matches = 0
    best_s = 0
    
    for s_val, type in intervals:
        current_matches += type
        if current_matches > max_matches:
            max_matches = current_matches
            best_s = s_val  # This is a valid shift within the peak overlap
            
    # 3. Calculate SIM and DIFF using the best shift found
    sim_count = max_matches
    virtual_ions = []
    
    if sim_count >= overlap_threshold:
        # A match is found! Now find which elements in A+s DON'T match B
        for a in A:
            shifted_a = a + best_s
            # Check if shifted_a is close to ANY element in B
            is_match = any(abs(shifted_a - b) <= delta for b in B)
            if not is_match:
                virtual_ions.append(shifted_a)
                
    return best_s, sim_count, virtual_ions

# --- Example Usage (based on your peptide VEADIAGHGQEVLIR) ---
master_b_ladder = [5, 6, 7, 8] # Simplified example masses
internal_line_A = [1, 2, 3, 4, 5, 6]         # A detected snippet
delta_val = 0.001

s, sim, virtuals = find_optimal_shift_and_virtual_ions(internal_line_A, master_b_ladder, delta=delta_val)

#print(f"Optimal Shift (s): {s:.2f}")
#print(f"SIM Score: {sim}")
#print(f"Virtual Ions to add to Grandmaster Ladder: {virtuals}")