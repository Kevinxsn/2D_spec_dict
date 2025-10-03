import data_calssify
import image_generation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import re
from fpdf import FPDF
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch


the_number = 1
data_number = f'data{the_number}'


data_loc = f'data/{data_number}.txt'
out = data_calssify.classify_msms_file(data_loc)
with open(f"data/data_classification/classified_msms_{data_number}.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["n", "classification", "line"])
    w.writeheader()
    for row in out:
        w.writerow(row)
    print(f"Wrote classified_msms_{data_number}.csv")
    
    
class_file = f"data/data_classification/classified_msms_{data_number}.csv"
data_frame = pd.read_csv(class_file)




def extract_sequence(peptide: str) -> str:
    """
    Remove modifications (e.g. Me), charge info (e.g. +2H, 2+),
    and keep only the plain amino acid sequence.
    
    Example:
    [GGNFSGRMeGGFGGSR+2H]2+  -->  GGNFSGRGGFGGSR
    """
    # 1. Remove surrounding brackets if present
    peptide = peptide.replace('(P)', '')
    peptide = peptide.replace('(nitro)', '')
    peptide = peptide.replace('(Me2)', '')
    peptide = peptide.replace('Me', '')
    peptide = peptide.replace('Ac', '')
    peptide = peptide.replace('(p)', '')
    peptide = peptide.replace('(NH2)', '')
    hydro_pattern = r'\+\d+H'
    peptide = re.sub(hydro_pattern, '', peptide)
    
    peptide = peptide.strip("[]")
    
    # 2. Remove modification tags like "Me", "Ox", "Ac" (assuming uppercase letters only are residues)
    # Keep only capital letters A–Z
    seq = re.sub(r'[^A-Z]', '', peptide)
    
    return seq


with open(data_loc, "r", encoding="utf-8") as f:
    lines = [ln.rstrip("\n") for ln in f if ln.strip()]
peptide_header = lines[0]
seq = extract_sequence(peptide_header)


def parse_mass_spec_line(line):
    """
    Parses a single line of mass spectrometry data to extract y and b ion information.

    Args:
        line (str): A string containing the mass spec data for one line.

    Returns:
        tuple: A tuple containing (y_ion, y_mz, b_ion, b_mz).
               Returns None for any value that cannot be identified.
    """
    y_ion, y_mz, b_ion, b_mz = None, None, None, None

    # This regex pattern is designed to find an ion type (specifically 'y' or 'b' followed by numbers)
    # and its associated m/z value.
    # - .*?       : Non-greedily matches any character at the start.
    # - ([yb]\d+) : Captures the ion group, which must start with 'y' or 'b' and be followed by one or more digits.
    # - .*?       : Non-greedily matches any characters between the ion and the m/z value (e.g., "(1+)", "-NH3", etc.).
    # - @\s* : Matches the '@' symbol, followed by any whitespace.
    # - ([\d.]+)  : Captures the m/z value, which consists of digits and decimal points.
    pattern = re.compile(r'.*?([yb]\d+).*?@\s*([\d.]+)')

    # The data for y and b ions are separated by an '&'
    parts = line.split('&')

    for part in parts:
        # Search for the pattern in each part of the line
        match = pattern.search(part)
        if match:
            ion_name = match.group(1)
            mz_value = float(match.group(2))

            # Assign the found values based on whether it's a 'y' or 'b' ion
            if ion_name.startswith('y'):
                y_ion = ion_name
                y_mz = mz_value
            elif ion_name.startswith('b'):
                b_ion = ion_name
                b_mz = mz_value

    return y_ion, y_mz, b_ion, b_mz

def process_mass_spec_dataframe(df):
    """
    Applies the mass spec parser to a DataFrame column and expands the results.

    This function takes a DataFrame with a column named 'line', applies the 
    parse_mass_spec_line function to each row in that column, and adds four new 
    columns ('y_ion', 'y_mz', 'b_ion', 'b_mz') with the extracted data.

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain a 'line' column.

    Returns:
        pd.DataFrame: The DataFrame with the four new columns added.
    """
    # Check if the required 'line' column exists
    if 'line' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'line' column.")
        
    # Apply the parsing function to the 'line' column.
    # The result is a list of tuples, which we convert into a new DataFrame.
    parsed_data = df['line'].apply(parse_mass_spec_line)
    
    # Create a DataFrame from the list of tuples with specified column names.
    new_cols_df = pd.DataFrame(
        parsed_data.tolist(), 
        index=df.index, 
        columns=['y_ion', 'y_mz', 'b_ion', 'b_mz']
    )
    
    # Concatenate the new columns with the original DataFrame.
    return pd.concat([df, new_cols_df], axis=1)

data_frame = process_mass_spec_dataframe(data_frame)

custom_annotations_list = [[] for _ in range(len(seq))]


for index, each_row in data_frame.iterrows():
    if each_row['classification'] == 'usable' or each_row['classification'] == 'rare_mod':
        if each_row['y_ion'] is not None:
            y_index = int(each_row['y_ion'][1:])
            custom_annotations_list[len(custom_annotations_list) - y_index - 1].append(each_row['n'])
        elif each_row['b_ion'] is not None:
            b_index = int(each_row['b_ion'][1:])
            custom_annotations_list[b_index - 1].append(each_row['n'])
            
graph_path = f"data/data_graph/pep_frag{data_number}.png"
#image_generation.plot_peptide_fragmentation(seq, annotations=custom_annotations_list, show=False, save_path=graph_path)
print(data_frame)




def create_report_with_reportlab(pdf_path, image_path, csv_path):
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
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
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
    page_width = letter[0] - 2 * inch # Page width minus left/right margins
    img = Image(image_path, width=page_width, height=page_width/3) # Maintain 2:1 aspect ratio

    story.append(img)
    story.append(Spacer(1, 0.2 * inch))

    # 4. ADD CSV DATA AS A TABLE
    story.append(Paragraph("Detailed Data", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    # Read the CSV with pandas
    df = pd.read_csv(csv_path)
    
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
        ('GRID', (0, 0), (-1, -1), 1, colors.black) # Add grid lines
    ])
    report_table.setStyle(style)

    # Add the styled table to the story
    story.append(report_table)

    # 6. BUILD THE PDF
    # The build method takes the story and writes the PDF file.
    doc.build(story)
    print(f"Successfully created '{pdf_path}' with reportlab")
    print(f"Successfully created '{pdf_path}'")

'''
create_report_with_reportlab(
    pdf_path=f'data/pdf/{data_number}.pdf',
    image_path=graph_path,
    csv_path=class_file
    )
'''