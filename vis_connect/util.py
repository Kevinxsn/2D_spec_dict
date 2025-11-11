import peptide
import re

pep_name_dict = peptide.peptide_table

def peptide_name(short_name, charge):
    sequence = pep_name_dict[short_name]
    sequence = sequence + f'+{charge}H'
    sequence = '[' + sequence + ']' + f'{charge}+'
    return sequence


def name_ouput(name):
    pattern = re.compile(r'^([A-Za-z]+\d+)_([23])\+\.csv$', re.IGNORECASE)
    match = pattern.match(name)
    if match:
        base_name = match.group(1)
        charge = int(match.group(2))
        #print(base_name, charge)
        return peptide_name(base_name, charge)
    else:
        print("No match")
        

