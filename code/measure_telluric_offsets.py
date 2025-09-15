#!/usr/bin/env python3
# measure_telluric_offsets.py
# Measure wavelength offsets in telluric lines between observed spectra and PCA models 
# for all orders in a given setting.

import os, sys, argparse
import numpy as np
from scipy.interpolate import CubicSpline

from calc_sp_offset import calc_sp_offset
import utils
c_kms = utils.c_kms

# ---------------------------------------------
# Argument parsing
# ---------------------------------------------
parser = argparse.ArgumentParser(description="Measure offsets for all orders.")
parser.add_argument("pattern", type=str, default="./object_{order}.txt",
                    help="Filename pattern for observed spectra (default: object_{order}.txt)")
parser.add_argument("--vac_air", type=str, required=True, choices=["vac","air"],
                    help="Wavelength scale of the input object spectra")
parser.add_argument("-s","--setting", type=str, default=utils.default_setting,
                    help=f"Instrumental setting label (default: {utils.default_setting})")
parser.add_argument("-o", "--orders", type=str, default="",
                    help="Comma-separated list of orders to process (default: all orders in the setting)")
parser.add_argument("--vmin", type=float, default=-30, help="Min velocity shift (km/s)")
parser.add_argument("--vmax", type=float, default=30, help="Max velocity shift (km/s)")
parser.add_argument("--vstep", type=float, default=1, help="Step of velocity grid (km/s)")
parser.add_argument("--fmin", type=float, default=0.0, help="Minimum flux to include")
parser.add_argument("--fmax", type=float, default=1.5, help="Maximum flux to include")
args = parser.parse_args()

vac_air = args.vac_air
setting = args.setting
setting_orders = utils.load_setting(setting)
list_v = np.arange(args.vmin, args.vmax + args.vstep, args.vstep)
frange = (args.fmin, args.fmax)

# Decide which orders to analyze
if args.orders.strip() == "":
    target_orders = list(setting_orders.keys())
else:
    target_orders = [o.strip() for o in args.orders.split(",")]
    for o in target_orders:
        if o not in setting_orders:
            print(f"[ERROR] order={o} is not defined in setting_{args.setting}.txt", file=sys.stderr)
            sys.exit(1)

print(f"# Measuring telluric offsets for {len(target_orders)} orders in {args.setting} ({vac_air})", file=sys.stderr)
print(f"# File pattern: {args.pattern}")
print("# o    v[km/s]  n_pix  file", flush=True)

# ---------------------------------------------
# Loop over orders
# ---------------------------------------------
for order, oconf in setting_orders.items():
    if order not in target_orders:
        continue
    obj_filename = args.pattern.format(order=order)
    if not os.path.isfile(obj_filename):
        print(f"[{order}] {obj_filename} not found, skipping", file=sys.stderr)
        continue

    # Load observed spectrum
    sp_obs, *_ = utils.load_sp(obj_filename)
    if sp_obs is None or len(sp_obs)==0:
        print(f"[{order}] No data loaded from {obj_filename}", file=sys.stderr)
        continue

    # Load model average spectrum
    sp_ref = utils.load_model_ave(setting, order, vac_air=vac_air)
    if sp_ref is None or len(sp_ref)==0:
        print(f"[{order}] No average model found", file=sys.stderr)
        continue

    # Load telluric absorption regions (for use_ranges)
    tel_abs = utils.load_tel_abs(setting, order, vac_air=vac_air)
    if len(tel_abs)==0:
        print(f"[{order}] No tel_abs found, using full range", file=sys.stderr)
        use_ranges = [(np.min(sp_ref[:,0]), np.max(sp_ref[:,0]))]
    else:
        use_ranges = tel_abs

    # Call offset measurement
    xadj, yadj, n_used = calc_sp_offset(sp_obs, sp_ref, list_v=list_v, frange=frange, use_ranges=use_ranges, reject_ranges=[])

    if np.isnan(xadj):
        print(f"{order:5s}  NaN        0    {obj_filename}")
    else:
        print(f"{order:5s}  {xadj:7.3f}  {n_used:5d}  {obj_filename}")
