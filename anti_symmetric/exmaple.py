import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.patches as patches
import anti_symmetric_util as ay_util
import img2pdf

# Directory where *this script* is located
current_dir = Path(__file__).resolve().parent

# Parent directory of the script
parent_dir = current_dir.parent

# Directories containing your modules
vis_dir = parent_dir / "vis"
connected_graphs_dir = parent_dir / "vis_connect"

# Put them at the front of sys.path so they are found first
sys.path.insert(0, str(vis_dir))
sys.path.insert(0, str(connected_graphs_dir))

import data_parse
import util
import peptide
import pandas as pd
import numpy as np
import connected_graph
import anti_symmetric_real_data as AS


amino_acid_masses_switch = {v: k for k, v in AS.amino_acid_masses_merge.items()}
sequence = 'AGVSTK'
pep = peptide.Pep('[AGVSTK+2H]2+')
noise = []
full_spec = [0, 71.037, 128.058, 128.095, 227.126, 229.143]

correct = [(0.0,0.0), (71.037,0.0), (128.058, 0.0), (128.058, 128.095), (227.126, 128.095), (227.126, 229.143)]
candidates = []

AS.visualize_all_paths(full_spec, spurious_masses=noise, 
                         candidate_paths=candidates, 
                         correct_path=correct, 
                         aa_map=amino_acid_masses_switch,
                         title = sequence,
                         pep_mass=543.301,
                         save_path=None
                         )