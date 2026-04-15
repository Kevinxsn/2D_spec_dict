from itertools import combinations_with_replacement

# Standard 20 amino acids (lexicographically ordered, single-letter codes)
AMINO_ACIDS = sorted(['A','R','N','D','C','E','Q','G','H','I',
                      'L','K','M','F','P','S','T','W','Y','V'])

# Monoisotopic residue masses (Da)
AA_MASSES = {
    'A':  71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G':  57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P':  97.05276, 'S':  87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V':  99.06841,
}


def extend(b_set, A):
    """
    B-SET ⨁ A: extend each multiset B in b_set by every a_i in A
    such that a_i >= max(B). Lexicographic order on A defines '>='.
    Returns the new collection of multisets (as tuples).
    """
    rank = {a: i for i, a in enumerate(A)}
    new_set = []
    for B in b_set:
        # max(B) under the lex order; empty B accepts anything
        start = rank[B[-1]] if B else 0
        for a in A[start:]:
            new_set.append(B + (a,))
    return new_set


def compositions_of_length(n, A=AMINO_ACIDS):
    """Recursively build all length-n multisets over A."""
    comp = [()]  # Composition_0
    for _ in range(n):
        comp = extend(comp, A)
    return comp


def composition_mass(B):
    return sum(AA_MASSES[a] for a in B)


def assign_compositions(target_mass, max_length=10, tol=0.02, A=AMINO_ACIDS):
    """
    Find all amino acid multisets (up to max_length residues) whose
    monoisotopic mass matches target_mass within tol (Da).
    """
    hits = []
    comp = [()]
    for n in range(1, max_length + 1):
        comp = extend(comp, A)
        for B in comp:
            m = composition_mass(B)
            if abs(m - target_mass) <= tol:
                hits.append((B, m))
    return hits


if __name__ == "__main__":
    # Quick sanity check: how many length-3 compositions? Should be C(20+3-1, 3) = 1540
    c3 = compositions_of_length(3)
    print(f"Length-3 compositions: {len(c3)} (expected 1540)")
    assert len(c3) == 1540

    # Cross-check against itertools
    ref = list(combinations_with_replacement(AMINO_ACIDS, 3))
    assert sorted(c3) == sorted(ref)
    print("Matches itertools.combinations_with_replacement ✓")

    # Example: assign compositions to a fragment mass
    target = 357.17  # Da
    matches = assign_compositions(target, max_length=4, tol=0.02)
    print(f"\nMatches for {target} Da:")
    for B, m in matches[:10]:
        print(f"  {''.join(B)}  mass={m:.4f}  Δ={m-target:+.4f}")