# %%
import re
import pandas as pd
import neutral_loss_mass
import peptide


# Define the regex pattern to capture the ion and its optional addition
# It handles cases like y7, b8-NH3, bi(2-4), and y10/b10

# %% 

ion_pattern = re.compile(
    r"([a-z]+(?:(?:(?:\(\d+-\d+\))|\d+)?(?:/[a-z]+\d+)?))\s*([+-][^\]\s\(\)]*)?"
)

_TOKEN = r'(?:\d*\([^\)]*\)|[^\]\s\(\)\+\-]+)'

PATTERN = re.compile(
    rf"""
    ^
    (?P<ion>
        [a-z]+
        (?:\d+(?:-\d+)?|\(\d+-\d+\))?                 # optional index or range (with/without parens)
        (?:/[a-z]+(?:\d+(?:-\d+)?|\(\d+-\d+\))?)?     # optional /other ion (e.g., b4/y4, bi(2-4)/y3)
    )
    (?P<loss>
        (?:[+\-]{_TOKEN})+                            # one or more +/- segments (e.g., -NH3-2(H2O)+H2O)
    )?
    $
    """,
    re.VERBOSE | re.IGNORECASE
)

def parse_ion_string(s):
    """
    Parses a single ion string (e.g., '[y9-CH3NH2–NH3] (1+)')
    Returns (ion_core, loss_string_without_leading_sign) or ('???', None) or (None, None).
    """
    s = s.strip()
    if s.startswith('???'):
        return '???', None

    # Normalize: remove spaces; map fancy dashes to ASCII '-'
    s = re.sub(r'\s+', '', s).replace('–', '-').replace('—', '-').replace('−', '-')

    # If in square brackets, take inside; otherwise keep as-is
    m = re.search(r'\[([^\]]+)\]', s)
    s_core = m.group(1) if m else s

    # Drop trailing charge like (1+), (2+)
    s_core = re.sub(r'\(\d\+\)$', '', s_core)
    
    ion = None
    
    s_no_space = re.sub(r'\s+', '', s)
    match = ion_pattern.search(s_no_space)
    if match:
        ion = match.group(1)

    m2 = PATTERN.fullmatch(s_core)
    if not m2:
        return ion, None

    ion = m2.group('ion')
    loss = m2.group('loss')
    if loss:
        # Remove the very first sign, keep interior signs: '-2(H2O)-NH3' -> '2(H2O)-NH3'
        loss = loss[1:] if loss[0] in '+-' else loss
    
    return ion, loss
# %%

# Process the data

with open('data/data1.txt', 'r', encoding='utf-8') as file:
    content = file.read()
#print(content)

# %%    

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



data = content
lines = data.strip().split('\n')[1:] # Skip the first header line
parsed_data = []
peptide_seq = data.strip().split('\n')[0]
peptide_seq = extract_sequence(peptide_seq)

for line in lines:
    # Clean up line number and split into two ion pairs
    line_content = re.sub(r'^\d+\.\s*', '', line.strip())
    parts = line_content.split('&')

    if len(parts) == 2:
        ion1_str, ion2_str = parts
        ion1, add1 = parse_ion_string(ion1_str)
        ion2, add2 = parse_ion_string(ion2_str)
        parsed_data.append([ion1, add1, ion2, add2])

# Create a pandas DataFrame for nice output
df = pd.DataFrame(parsed_data, columns=['ion1', 'ion1_addition', 'ion2', 'ion2_addition'])
df.index = df.index + 1 # Start index from 1 to match original data
#df['neutral_mass1'] = df['ion1_addition'].apply(neutral_loss_mass.mass_of_loss)
#df['neutral_mass2'] = df['ion2_addition'].apply(neutral_loss_mass.mass_of_loss)



# Display the result
print(df.to_string())
# %%
