import data_calssify
import image_generation_internal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import re
import os
import util
import data_parse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
import peptide
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    
)
from reportlab.lib.styles import getSampleStyleSheet

csv_data = "ME14_2+.csv"
file_path = os.path.join(
    os.path.dirname(__file__),
    f"../data/Top_Correlations_At_Full_Num_Scans_PCov/annotated/{csv_data}"
)
file_path = os.path.abspath(file_path) 

## Store sequence into peptide class
sequence = util.name_ouput(csv_data)
print(sequence)
csv_data = file_path
pep = peptide.Pep(sequence)
df = pd.read_csv(csv_data)
df = df[df['Index'].notna()]
results = data_parse.process_ion_dataframe(df.head(50), pep)
results['classification'] = results.apply(data_parse.data_classify, args=(pep,), axis=1)
df = results

seq = pep.seq
custom_annotations_list = [[] for _ in range(len(seq))]

for index, each_row in df.iterrows():
    if each_row['classification'] == 'usable' or each_row['classification'] == 'rare_mod':
        if each_row['y_ion'] is not None:
            y_index = int(each_row['y_ion'][1:])
            custom_annotations_list[len(custom_annotations_list) - y_index - 1].append(each_row['n'])
        elif each_row['b_ion'] is not None:
            b_index = int(each_row['b_ion'][1:])
            custom_annotations_list[b_index - 1].append(each_row['n'])

graph_path = os.path.join(
    os.path.dirname(__file__),
    f"../data/graph_internal/pep_frag{csv_data}.png"
)
graph_path = os.path.abspath(graph_path)

internal_list = []
for index, each_row in df.iterrows():
    
    if type(each_row['ion1']) == str and each_row['ion1'][:2] == 'bi':
        
        the_index = peptide.Pep.extract_numbers(each_row['ion1'])
        the_index.sort()
        if len(the_index) == 2:
            #print(the_index)
            internal_list.append((the_index[0],the_index[1], each_row['n']))
    if type(each_row['ion2']) == str and each_row['ion2'][:2] == 'bi':
        the_index = peptide.Pep.extract_numbers(each_row['ion2'])
        the_index.sort()
        if len(the_index) == 2:
            internal_list.append((the_index[0],the_index[1], each_row['n']))
    
    if each_row['classification'] == 'non_complementary':
        ion1, ion2 = each_row['ion1'], each_row['ion2']
        if 'y' in ion1 and 'b' in ion2:
            internal_list.append((int(ion2[1:]), len(seq) - int(ion1[1:]),each_row['n']))
        if 'y' in ion2 and 'b' in ion1:
            internal_list.append((int(ion1[1:]), len(seq) - int(ion2[1:]),each_row['n']))
print(internal_list)
image_generation_internal.plot_peptide_fragmentation(seq, annotations=custom_annotations_list, internal_peptides=internal_list, show=False, save_path=graph_path)