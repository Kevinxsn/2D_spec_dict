import image_generation_internal
import b_y_graph
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

data = 'ME14_3+'
csv_data = f"{data}.csv"
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
peptide_header = sequence
df = pd.read_csv(csv_data)
df = df[df['Index'].notna()]
results = data_parse.process_ion_dataframe(df.head(50), pep)
results['classification'] = results.apply(data_parse.data_classify, args=(pep,), axis=1)
the_list = []
the_y_list = []

results['loss1'] = results['loss1'].replace({None: np.nan})
results['loss2'] = results['loss2'].replace({None: np.nan})


df_y = b_y_graph.ion_data_organizer_y(results, sequence)
df_x = b_y_graph.ion_data_organizer_d(results, sequence)


the_length = len(pep.AA_array)
b_list = b_y_graph.create_annotation_list_from_df(df_x, the_length, b_y_graph.neutral_loss_colors)
y_list = b_y_graph.create_annotation_list_from_df(df_y, the_length, b_y_graph.neutral_loss_colors)


df = results

df['n'] = df['Index'].astype(int)

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


graph1_path = f'vis/temp/{data}_graph1.png'
graph2_path = f'vis/temp/{data}_graph2.png'

image_generation_internal.plot_peptide_fragmentation(seq, annotations=custom_annotations_list, internal_peptides=internal_list, show=False, save_path=graph1_path)
b_y_graph.plot_peptide_fragmentation(pep.seq, annotations=b_list, y_line_annotations = y_list, color_map=b_y_graph.neutral_loss_colors, show=False, save_path=graph2_path)


def create_report_with_reportlab(pdf_path, image_path, select_data_frame):
    """
    Generates a PDF report using the reportlab library.

    Args:
        pdf_path (str): The file path for the output PDF.
        image_path (str): The file path for the graph image.
        csv_path (str): The file path for the CSV data.
    """
    # 1. SETUP THE DOCUMENT
    # SimpleDocTemplate handles the page layout and rendering.
    # We set right/left/top/bottom margins to 1 inch.
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter),
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)

    # The "story" is a list of "Flowable" objects that reportlab will place on pages.
    story = []
    styles = getSampleStyleSheet()

    # 2. ADD TITLE
    # Create a Paragraph object for the title with a specific style.
    title = Paragraph(peptide_header, styles['h1'])
    story.append(title)

    # Add a spacer for some vertical whitespace.
    story.append(Spacer(1, 0.2 * inch))

    # 3. ADD GRAPH
    # Create an Image object. We calculate the width to fit within the page margins.
    page_width = letter[0] - 0.5 * inch # Page width minus left/right margins
    img = Image(image_path, width=page_width, height=page_width/2) # Maintain 2:1 aspect ratio

    story.append(img)
    story.append(Spacer(1, 0.2 * inch))

    # 4. ADD CSV DATA AS A TABLE
    story.append(Paragraph("Detailed Data", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    # Read the CSV with pandas
    df = select_data_frame
    
    # Convert dataframe to a list of lists format that reportlab's Table requires
    # The first row is the header.
    table_data = [df.columns.values.tolist()] + df.values.tolist()
    
    # Create the Table object
    report_table = Table(table_data)

    # 5. STYLE THE TABLE
    # Create a TableStyle object to define the table's appearance
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')), # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Header font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#DCE6F1')), # Data row background
        ('GRID', (0, 0), (-1, -1), 1, colors.black), # Add grid lines
        ('FONTSIZE', (0, 0), (-1, -1), 8), 
        ('LEADING', (0, 0), (-1, -1), 10),
    ])
    report_table.setStyle(style)

    # Add the styled table to the story
    story.append(report_table)

    # 6. BUILD THE PDF
    # The build method takes the story and writes the PDF file.
    doc.build(story)
    print(f"Successfully created '{pdf_path}' with reportlab")
    print(f"Successfully created '{pdf_path}'")



    
    
    
def create_multi_item_report(pdf_path, peptide_header, image_paths, data_frames):
    """
    Generates a PDF report with multiple graphs and tables stacked vertically.

    Args:
        pdf_path (str): The file path for the output PDF.
        peptide_header (str): The main title for the report.
        image_paths (list): A list of file paths for the graph images.
        data_frames (list): A list of pandas DataFrames to be converted to tables.
    """
    # 1. SETUP THE DOCUMENT
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter),
                            rightMargin=inch * 0.3, leftMargin=inch*0.3,
                            topMargin=inch, bottomMargin=inch)
    
    story = []
    styles = getSampleStyleSheet()

    # 2. ADD TITLE
    title = Paragraph(peptide_header, styles['h1'])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    # 3. ADD GRAPHS (LOOPING THROUGH THE LIST)
    # The drawable width of the page (11 inches wide - 2 inches of margin)
    drawable_width = landscape(letter)[0] - 2 * inch
    for image_path in image_paths:
        img = Image(image_path, width=drawable_width, height=drawable_width / 2.5) # Maintain aspect ratio
        #img = Image(image_path, width=drawable_width)
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

    # 4. ADD DATA TABLES (LOOPING THROUGH THE LIST)
    # Define a reusable table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#DCE6F1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 5.5),
        ('LEADING', (0, 0), (-1, -1), 10),
    ])

    for i, df in enumerate(data_frames):
        # Add a sub-header for each table
        story.append(Paragraph(f"Detailed Data - Table {i+1}", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))

        # Convert dataframe to a list of lists
        table_data = [df.columns.values.tolist()] + df.values.tolist()
        
        # Create and style the Table object
        report_table = Table(table_data)
        report_table.setStyle(table_style)

        # Add the styled table to the story
        story.append(report_table)
        story.append(Spacer(1, 0.3 * inch)) # Add extra space after each table

    # 5. BUILD THE PDF
    doc.build(story)
    print(f"Successfully created '{pdf_path}'")
    


selected_df = df[['n','classification', 'ion1', 'loss1', 'mass1', 'correct_mass1','mass_difference1', 'ion2', 'loss2', 'mass2', 'correct_mass2', 'mass_difference2','chosen_sum']]
selected_df = selected_df.round(2)

list_of_image_paths = [graph1_path, graph2_path]
list_of_dataframes = [df_x, df_y, selected_df]

create_multi_item_report(
    pdf_path=f'vis/temp/{data}.pdf',
    peptide_header=peptide_header,
    image_paths=list_of_image_paths,
    data_frames=list_of_dataframes
)

for file in [f"vis/temp/{data}_graph1.png", f"vis/temp/{data}_graph2.png"]:
    if os.path.exists(file):
        os.remove(file)