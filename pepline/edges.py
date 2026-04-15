"""
Generate List_n = set of distinct compound masses of n amino acids,
rounded to 0.001 Da (3 decimal digits) precision.

Recurrence:  List_{n+1} = List_n + List_1
where A + B = { a_i + b_j : a in A, b in B } with duplicates removed.

These masses will be used as edge weights in a de novo sequencing graph.
"""

# Monoisotopic residue masses (Da)
AA_MASSES = {
    'A':  71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G':  57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P':  97.05276, 'S':  87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V':  99.06841,
}

PRECISION = 3  # 0.001 Da


def quantize(x, precision=PRECISION):
    """Round to fixed precision and return as int (for exact set membership)."""
    return round(x * 10**precision)


def minkowski_sum(A, B, precision=PRECISION):
    """
    A + B = { a + b : a in A, b in B }, duplicates removed at given precision.
    A, B are sets of ints (already quantized).
    """
    return {a + b for a in A for b in B}


def build_lists(max_n, precision=PRECISION):
    """
    Build List_1, List_2, ..., List_{max_n}.
    Returns a dict {n: set of quantized masses}.
    """
    # List_1: unique masses of the 20 amino acids at the given precision
    list1 = {quantize(m, precision) for m in AA_MASSES.values()}
    lists = {1: list1}
    current = list1
    for n in range(2, max_n + 1):
        current = minkowski_sum(current, list1, precision)
        lists[n] = current
    return lists


def main():
    MAX_N = 10
    lists = build_lists(MAX_N)

    print(f"Distinct compound masses at 0.001 Da precision")
    print(f"(these are the edges in the de novo sequencing graph)")
    print()
    print(f"{'n':>4} | {'|List_n|':>12} | {'cumulative':>12}")
    print("-" * 36)

    cumulative = set()
    for n in range(1, MAX_N + 1):
        size = len(lists[n])
        cumulative |= lists[n]
        print(f"{n:>4} | {size:>12,} | {len(cumulative):>12,}")

    print()
    print(f"Note: List_1 has {len(lists[1])} entries, not 20, because")
    print(f"I and L collide exactly (both 113.084 Da) at 0.001 Da precision.")
    print()

    # Peek at a few values from List_1 and List_2
    print("List_1 (sorted, Da):")
    print(" ", sorted(m / 1000 for m in lists[1]))
    print()
    print(f"List_2 first 10 (sorted, Da):")
    print(" ", sorted(m / 1000 for m in lists[2])[:10])


if __name__ == "__main__":
    main()