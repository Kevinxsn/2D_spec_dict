import sys

with open(sys.argv[1]) as f:
    for line in f:
        row = line.split()
        if len(row) >= 2 and row[0] == row[1]:
            print("\t".join(row))
