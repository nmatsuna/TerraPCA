#!/usr/bin/env python3
import os, sys, argparse
import numpy as np
import utils

frac_OK, frac_Partial = 0.9, 0.5

def coverage_fraction(file_min, file_max, set_min, set_max):
    """Return the fraction of expected range covered by the file."""
    overlap_min = max(file_min, set_min)
    overlap_max = min(file_max, set_max)
    overlap = max(0.0, overlap_max - overlap_min)
    expected_width = set_max - set_min
    return overlap / expected_width if expected_width > 0 else 0.0

def coverage_label(frac):
    """Classify coverage quality."""
    if frac >= frac_OK:
        return "OK"
    elif frac >= frac_Partial:
        return "Partial"
    else:
        return "NG"

def main():
    parser = argparse.ArgumentParser(
        description="List available spectra files per order and check wavelength coverage.")
    parser.add_argument("-s", "--setting", type=str, default=utils.default_setting,
                        help="Setting label (e.g., WINERED_WIDE, WINERED_HIRES_Y)")
    parser.add_argument("-o", "--orders", type=str, default="",
                    help="Comma-separated list of orders to process (default: all orders in the setting)")
    args = parser.parse_args()

    setting_orders = utils.load_setting(args.setting)

    # Decide which orders to analyze
    if args.orders.strip() == "":
        target_orders = list(setting_orders.keys())
    else:
        target_orders = [o.strip() for o in args.orders.split(",")]
        for o in target_orders:
            if o not in setting_orders:
                print(f"[ERROR] order={o} is not defined in setting_{args.setting}.txt", file=sys.stderr)
                sys.exit(1)

    print(f"# Setting: {args.setting}")
    for order, oconf in setting_orders.items():
        if order not in target_orders:
            continue
        set_wmin, set_wmax = oconf['wmin'], oconf['wmax']
        n_pix, n_base = oconf['n_pix'], oconf['n_base']
        print(f"[{order}] range={set_wmin:.1f}:{set_wmax:.1f}, n_pix={n_pix}, n_base={n_base}", file=sys.stderr)
        num = {'OK':0, 'Partial':0, 'NG':0}
        files = utils.list_telluric_files(args.setting, order)
        if not files:
            print("  (no files found)")
            num['NG'] += 1
            continue
        for f in files:
            sp, *_ = utils.load_sp(f)
            if sp is None or len(sp) == 0:
                print(f"  {os.path.basename(f)}  [FAILED TO LOAD]")
                continue
            fmin, fmax = np.min(sp[:,0]), np.max(sp[:,0])
            frac = coverage_fraction(fmin, fmax, set_wmin, set_wmax)
            label = coverage_label(frac)
            num[label] += 1
            print(f"  {os.path.basename(f):30s} "
                  f"{fmin:8.1f} - {fmax:8.1f} ({len(sp):4d} px)",
                  f"[{label}, coverage={frac*100:3.0f}%]")

        print('  -> N(OK)={:d}'.format(num['OK']))
        for label in ['Partial', 'NG']:
            if num[label] > 0:
                print('     N({})={:d}'.format(label,num[label]))

if __name__ == "__main__":
    main()

