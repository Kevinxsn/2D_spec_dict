import re
from collections import defaultdict

# Monoisotopic atomic masses (extend if needed)
AMU = {
    'H': 1.00782503223,
    'C': 12.00000000000,
    'N': 14.00307400443,
    'O': 15.99491461957,
    'S': 31.9720711744,
    'P': 30.97376199842,
    'G': 57.02146,
}

# ---- Core helpers ----

def _normalize_loss(s: str) -> str:
    """Normalize separators/dashes and strip spaces."""
    s = s.strip()
    # unify fancy dashes to ASCII '-'
    s = s.replace('–', '-').replace('—', '-').replace('−', '-')
    # remove spaces
    s = re.sub(r'\s+', '', s)
    # strip surrounding brackets like [ ... ]
    m = re.match(r'^\[?(.*?)]?$', s)
    s = m.group(1) if m else s
    # drop trailing charge like (1+)
    s = re.sub(r'\(\d\+\)$', '', s)
    return s

def _split_terms(loss: str):
    """
    Split a loss string into signed terms.
    E.g. '2(H2O)-NH3+CH3NH2' -> [('+','2(H2O)'), ('-','NH3'), ('+','CH3NH2')]
    If no explicit sign at start, assume '+'.
    """
    # Insert explicit '+' at the beginning if needed
    if not loss or loss[0] not in '+-':
        loss = '+' + loss
    # Tokenize on +/-, but keep the sign
    parts = re.findall(r'([+\-])([^+\-]+)', loss)
    return parts  # list of (sign, term)

def _parse_formula_to_counts(formula: str) -> dict:
    """
    Parse a chemical 'formula-like' string into element counts.
    Understands:
      - Element tokens: H, C, N, O, S, P (and 2-letter like 'Cl' if you add mass)
      - Counts after elements: e.g., H2, C6
      - Parentheses with multipliers: (H2O)2 or 2(H2O)
      - Bond/structure characters (= and -) are ignored for composition
    """
    # 1) Remove bond/structure markers that don't affect composition
    f = formula.replace('=', '').replace('-', '')

    # 2) Convert PREFIX multipliers like 2(H2O) into postfix (H2O)2
    #    Do this repeatedly until none remain
    while True:
        newf = re.sub(r'(\d+)\(([^()]+)\)', r'(\2)\1', f)
        if newf == f:
            break
        f = newf

    # 3) Now parse with a stack to handle nested parentheses and POSTFIX multipliers
    i = 0
    stack = [defaultdict(int)]

    def read_int(start):
        j = start
        while j < len(f) and f[j].isdigit():
            j += 1
        return (int(f[start:j]) if j > start else 1, j)

    def read_element(start):
        # One uppercase letter + optional lowercase (enables extension to e.g. 'Cl', 'Na' later)
        if start >= len(f) or not f[start].isalpha():
            return None, start
        # Accept only letters; composition is case-sensitive (H, C, N, O, S, P)
        # We’ll treat any letter run starting with uppercase as an element symbol possibly including lowercase
        j = start + 1
        if j < len(f) and f[j].islower():
            j += 1
        return f[start:j], j

    while i < len(f):
        ch = f[i]
        if ch == '(':
            stack.append(defaultdict(int))
            i += 1
        elif ch == ')':
            i += 1
            mult, i = read_int(i)
            group = stack.pop()
            for el, cnt in group.items():
                stack[-1][el] += cnt * mult
        elif ch.isalpha():
            el, i2 = read_element(i)
            if el is None:
                raise ValueError(f"Unexpected token at {i} in {f!r}")
            cnt, i3 = read_int(i2)
            stack[-1][el] += cnt
            i = i3
        elif ch.isdigit():
            # A bare leading digit here (not before '(') likely indicates a user typo like 'H20' instead of 'H2O'.
            # We'll accumulate it as count for the previous element if that happened; otherwise it's an error.
            # To be conservative, raise a helpful error:
            raise ValueError(
                f"Unexpected standalone digit at position {i} in {formula!r}. "
                f"Did you mean 'H2O' (with the letter 'O') instead of 'H20' (zero)?"
            )
        else:
            # Any other character (shouldn't happen after cleanup)
            raise ValueError(f"Unexpected character {ch!r} at {i} in {formula!r}")

    return dict(stack[-1])

def mass_of_formula(formula: str) -> float:
    """
    Monoisotopic mass of a single formula-like token, e.g. 'CH3NH2', 'HN=C=NH', '(H2O)2', '2(H2O)'.
    """
    counts = _parse_formula_to_counts(formula)
    mass = 0.0
    for el, n in counts.items():
        if el not in AMU:
            raise KeyError(f"Element '{el}' not in mass table. Add it to AMU if needed.")
        mass += AMU[el] * n
    return mass

def mass_of_loss(loss_str: str) -> float:
    """
    Monoisotopic mass for a full loss string with +/- segments, e.g.:
      '2(H2O)-NH3+CH3NH2'  -> sum(2*H2O) - NH3 + CH3NH2
      'HN=C=N-CH3'         -> treated as one combined composition (sum of parts)
      
    """
    
    if loss_str is None:
        return 0
    s = _normalize_loss(loss_str)
    # If there are explicit +/- segments, use them.
    # Otherwise, treat the whole thing as one combined formula where '-' and '=' were compositionally ignored.
    if any(sign in s for sign in '+-'):
        total = 0.0
        for sign, term in _split_terms(s):
            if not term:
                continue
            #total += (1 if sign == '+' else -1) * mass_of_formula(term)
            total +=  mass_of_formula(term)
        return total
    else:
        return mass_of_formula(s)

'''
# ---- Quick sanity checks (monoisotopic) ----
examples = [
    "H2O",
    'CH3NH2-NH3',
    "NH3",
    "CH3NH2",
    "HN=C=NH",          # CH2N2
    "HN=C=N-CH3",       # combined composition
    "2(H2O)-NH3",
    "(H2O)2-NH3",
    "[y4-2(H2O)-NH3](1+)",  # full annotation: the parser will ignore brackets/charge for mass
]

for e in examples:
    try:
        print(f"{e:20s} -> {mass_of_loss(e):.6f} Da")
    except Exception as ex:
        print(f"{e:20s} -> ERROR: {ex}")
'''