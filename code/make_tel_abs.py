#!/usr/bin/env python3
# make_tel_abs.py
# Identify significant telluric absorption regions in ave_{order}.txt
# Usage: python3 make_tel_abs.py m44 0.95

import sys
import os
import numpy as np
import utils

TXT_DIR = 'eigen_txt'
buffer_size = 3  # number of pixels to expand on both sides

# ------------------------------------------
# Argument parsing
# ------------------------------------------
if len(sys.argv) != 3:
    print('Usage: python3 {} [order (e.g., m44)] [flux threshold (e.g., 0.95)]'.format(sys.argv[0]), file=sys.stderr)
order = sys.argv[1]
threshold = float(sys.argv[2])
if order not in utils.orders:
    print(f'Error: Unknown order {order}', file=sys.stderr)
    sys.exit(1)

out_file = f'{TXT_DIR}/tel_abs_{order}.txt'
ave_file = f'{TXT_DIR}/ave_{order}.txt'
if not os.path.isfile(ave_file):
    print(f'Input file not found: {ave_file}', file=sys.stderr)
    sys.exit(1)

# ------------------------------------------
# Load and scan the average spectrum
# ------------------------------------------
data = np.loadtxt(ave_file)
waves, fluxes = data[:, 0], data[:, 1]
n = len(waves)

# Identify pixels below threshold
mask_abs = fluxes < threshold
abs_regions = []
i = 0
while i < n:
    if mask_abs[i]:
        # Start of an absorption region
        start = i
        while (i+1 < n) and (mask_abs[i+1]):
            i += 1
        end = i
        # Add buffer (ensuring indices remain in range)
        start_buffered = max([0, start-buffer_size])
        end_buffered = min([n-1, end+buffer_size])

        abs_regions.append((waves[start_buffered],waves[end_buffered]))
    i += 1

# Merge overlapping or adjacent regions
merged = []
for region in abs_regions:
    if not merged:
        merged.append(region)
    else:
        last_start, last_end = merged[-1]
        current_start, current_end = region
        if current_start <= last_end:
            # Overlapping or adjacent - merge
            merged[-1] = (last_start, max(last_end,current_end))
        else:
            merged.append(region)

# ------------------------------------------
# Save the result
# ------------------------------------------
with open(out_file, 'w') as fout:
    for w0, w1 in merged:
        print(f'{w0:.3f} {w1:.3f}', file=fout)
print('# Wrote {} absorption region(s) to {}'.format(len(merged),out_file))

