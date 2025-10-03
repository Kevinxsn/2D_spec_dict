import re
from typing import List, Dict

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

# --- Utilities ---------------------------------------------------------------



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



def get_by_indices(line: str):
    """
    Return (b_indices, y_indices) found in the line, ignoring modifiers.
    Works for tokens like: b7, [b7-NH3], y10, [y10-CH3NH2], etc.
    """
    # Capture occurrences like 'b10' or '[b10' or 'b10(' etc.
    # We'll first strip spaces around letters to handle 'y6(1+)' forms
    # Pattern: start of token or '[', then b or y, then digits
    pattern = re.compile(r"(?i)(?:^|[\[\s,;&])([by])\s*([0-9]{1,3})")
    b_idx, y_idx = [], []
    for ion, idx in pattern.findall(line):
        if ion.lower() == 'b':
            b_idx.append(int(idx))
        elif ion.lower() == 'y':
            y_idx.append(int(idx))
    return b_idx, y_idx

def has_ambiguous_tokens(line: str) -> bool:
    """
    Ambiguity: explicit 'ambiguous' tag OR a b/y slash like 'y10/b10' or '[b4/y4'.
    """
    l = line.lower()
    if "ambiguous" in l:
        return True
    # heuristic: any / separating b and y token names
    if re.search(r"(?i)([by]\s*\d+)\s*/\s*([by]\s*\d+)", line):
        return True
    if re.search(r"(?i)\[?\s*[by]\s*\d+\s*[/]\s*[by]\s*\d+", line):
        return True
    return False

def has_unclear_tokens(line: str) -> bool:
    l = line.lower()
    return "unclear" in l or "???" in line

def has_rare_mod_tokens(line: str) -> bool:
    """
    Rare mods: explicit 'rare mod' tag OR exotic guanidino (Arg) side-chain losses.
    """
    l = line.lower()
    if "rare mod" in l:
        return True
    # exotic guanidino losses frequently written as HN=C=NH or HN=C=N-CH3
    if "hn=c=nh" in line.lower() or "hn=c=n-ch3" in line.lower():
        return True
    return False

def has_noncomplementary_tag(line: str) -> bool:
    return "non-complementary" in line.lower()

def is_complementary(b_idx: List[int], y_idx: List[int], pep_len: int) -> bool:
    """
    Complementarity rule for singly charged fragments:
    b_k complements y_{L-k}. We only test if exactly one b and one y were found.
    """
    if len(b_idx) == 1 and len(y_idx) == 1:
        k = b_idx[0]
        j = y_idx[0]
        return (k + j) == pep_len
    return False

def has_internal_acid(line: str) -> bool:
    """
    Detect 'internal acid' ions such as bi3-9, bi5-9, bi3-4, etc.
    Matches tokens like: biX-Y, where X and Y are integers.
    """
    # Look for patterns like bi3-9, bi5-9, bi3-4, etc.
    return bool(re.search(r"\bbi\d+-\d+\b", line.lower()))

def has_undefined(line: str) -> bool:
    """
    Detect lines that only contain m/z values with '&' in between,
    and no recognizable ion labels (b, y, a, bi, ???).
    Example: '503.73 & 498.42'
    """
    # Quick reject if there's any known ion label
    if re.search(r"\b[aby]\d", line.lower()) or "bi" in line.lower() or "???" in line:
        return False
    # Pattern: two floating-point numbers separated by '&'
    return bool(re.match(r"^\s*\d+(\.\d+)?\s*&\s*\d+(\.\d+)?\s*$", line.strip()))

# --- Parser for file lines ---------------------------------------------------

def parse_number_and_body(line: str):
    """
    From a line like '8. y6(1+) @ 580.43 & [b7-NH3] (1+) @ 673.04 non-complementary'
    return (8, 'y6(1+) @ 580.43 & [b7-NH3] (1+) @ 673.04 non-complementary')
    """
    m = re.match(r"\s*(\d+)\.\s*(.+)", line)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, line.strip()

# --- Classifier --------------------------------------------------------------

def classify_line(body: str, pep_len: int) -> str:
    """
    Return one of: 'internal_acid', 'ambiguous', 'rare_mod', 'unclear', 'non_complementary', 'usable'
    (priority order below is deliberate).
    """
    # 1) explicit / structural tags first
    if has_undefined(body):
        return "undefined"
    if has_internal_acid(body):
        return "internal_acid"
    if has_ambiguous_tokens(body):
        return "ambiguous"
    if has_rare_mod_tokens(body):
        return "rare_mod"
    if has_unclear_tokens(body):
        return "unclear"
    if has_noncomplementary_tag(body):
        return "non_complementary"

    # 2) if exactly one b and one y, check complementarity
    b_idx, y_idx = get_by_indices(body)
    if len(b_idx) == 1 and len(y_idx) == 1:
        if is_complementary(b_idx, y_idx, pep_len):
            return "usable"
        else:
            return "non_complementary"

    # 3) otherwise treat as usable (could be a-ions, internal ions, or singletons)
    return "usable"

def classify_msms_file(path: str) -> List[Dict]:
    """
    Reads your text file:
      - First non-empty line = peptide header
      - Subsequent numbered lines = ion pairs + optional tags
    Returns a list of dicts with number, body, and classification.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        return []

    peptide_header = lines[0]
    print('original_seq', peptide_header)
    
    seq = extract_sequence(peptide_header)
    
    print('seq', seq)
    pep_len = len(seq)
    
    print('pep_line:', pep_len)

    results = []
    for ln in lines[1:]:
        num, body = parse_number_and_body(ln)
        label = classify_line(body, pep_len)
        results.append({
            "n": num,
            "line": body,
            "classification": label
        })
    return results

# --- Example usage -----------------------------------------------------------
if __name__ == "__main__":
    # Replace 'example.txt' with your actual file path.
    infile = "data2.txt"
    out = classify_msms_file(infile)

    # Pretty print
    from pprint import pprint
    pprint(out)

    # Or write to CSV
    import csv
    with open("classified_msms.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "classification", "line"])
        w.writeheader()
        for row in out:
            w.writerow(row)
    print("Wrote classified_msms.csv")