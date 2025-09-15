#!/usr/bin/env python3
# list_tel_abs.py
# Identify significant telluric absorption regions from ave_{order}_{vac_air}.txt
# and output tel_abs_{order}_{vac_air}.txt in both air and vacuum scales.

import sys
import os
import numpy as np
import utils
from mask_from_spectrum import mask_from_spectrum

buffer_size = 5  # number of pixels to expand on both sides

# ------------------------------------------------
# Argument parsing
# ------------------------------------------------
if len(sys.argv) != 4:
    print(f"Usage: python3 {sys.argv[0]} [setting_label] [order (e.g., m44)] [flux threshold (e.g., 0.95)]",
          file=sys.stderr)
    sys.exit(1)

setting = sys.argv[1]
order = sys.argv[2]
threshold = float(sys.argv[3])

setting_orders = utils.load_setting(setting)
if order not in setting_orders:
    print(f"Error: Unknown order {order} in setting_{setting}.txt", file=sys.stderr)
    sys.exit(1)

TXT_DIR = os.path.join(utils.HOME_DIR, "models_txt", setting)
ave_vac = os.path.join(TXT_DIR, f"ave_{order}_vac.txt")
ave_air = os.path.join(TXT_DIR, f"ave_{order}_air.txt")

if (not os.path.isfile(ave_vac)) or (not os.path.isfile(ave_air)):
    print(f"ERROR: ave_{order}_vac.txt or ave_{order}_air.txt not found in {TXT_DIR}", file=sys.stderr)
    sys.exit(1)

# ------------------------------------------------
# Process for both vac and air
# ------------------------------------------------
for vac_air, ave_file in [("vac", ave_vac), ("air", ave_air)]:
    sp = np.loadtxt(ave_file)
    masked_ranges = mask_from_spectrum(sp, flux_thresh=threshold, buffer_pix=3, invert=False)

    out_file = os.path.join(TXT_DIR, f"tel_abs_{order}_{vac_air}.txt")
    with open(out_file, 'w') as fout:
        for w0, w1 in masked_ranges:
            print(f"{w0:.3f} {w1:.3f}", file=fout)

    print(f"[{order}] {vac_air}: threshold={threshold:.2f}, {len(masked_ranges)} regions saved to {out_file}")

