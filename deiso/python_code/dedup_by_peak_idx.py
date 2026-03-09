import sys
import csv

in_file, out_file = sys.argv[1], sys.argv[2]

best = {}  # PEAK_IDX -> row with max THEO_INTE

with open(in_file) as f:
    reader = csv.DictReader(f, delimiter='\t')
    fieldnames = reader.fieldnames
    for row in reader:
        key = row['PEAK_IDX']
        if key not in best or float(row['THEO_INTE']) > float(best[key]['THEO_INTE']):
            best[key] = row

with open(out_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(best.values())
