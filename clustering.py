
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import peptide

def find_mass_clusters(mass_sums, actual_peptide_mass, eps=0.5, min_samples=2):
    """
    Identifies all mass clusters in a single pass and calculates their
    difference from a given actual peptide mass.

    Args:
        mass_sums (list or np.array): A list of m1+m2 sums.
        actual_peptide_mass (float): The known mass of the peptide for comparison.
        eps (float): The mass tolerance (delta) for clustering.
        min_samples (int): The minimum number of ion pairs to form a cluster.

    Returns:
        pd.DataFrame: A DataFrame summarizing the identified clusters.
    """
    
    # Ensure data is in the correct format for DBSCAN (a 2D array)
    X = np.array(mass_sums).reshape(-1, 1)

    # Run DBSCAN to find all clusters at once
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    
    # This list will store the results for each cluster
    results = []

    # Get the unique cluster IDs. -1 is for noise, so we ignore it.
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    print(f"Found {len(unique_labels)} cluster(s) and {np.sum(labels == -1)} noise points.")

    # Loop through each found cluster
    for cluster_id in sorted(list(unique_labels)):
        # Find the points that belong to this cluster
        points_in_cluster = X[labels == cluster_id]
        
        # Calculate the required metrics
        cluster_size = len(points_in_cluster)
        median_mass = np.median(points_in_cluster)
        mass_difference = median_mass - actual_peptide_mass
        
        # Store the results
        results.append({
            "Cluster ID": cluster_id,
            "Cluster Size": cluster_size,
            "Median Mass (Da)": median_mass,
            "Difference from Actual Mass (Da)": mass_difference
        })
        
    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)

# --- Example Usage ---
'''
# 1. DEFINE YOUR INPUTS
PEPTIDE_MASS_ACTUAL = 1327.6224600000003

pep = peptide.Pep('[GGNFSGRMeGGFGGSR+2H]2+')
df = pd.read_csv('data/data_table/data_sheet1')
data = np.array(df['chosen_sum'])
data = data[np.isfinite(data)]

# Simulate a list of m1+m2 sums (replace this with your actual data)

# 2. RUN THE ANALYSIS
# The function directly returns the final table
results_df = find_mass_clusters(
    mass_sums=data,
    actual_peptide_mass=PEPTIDE_MASS_ACTUAL
)

# 3. PRINT THE FINAL RESULT
print("\n--- Cluster Analysis Results ---")
print(results_df.to_string())
'''




