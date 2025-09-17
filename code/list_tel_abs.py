#!/usr/bin/env python3
# list_tel_abs.py
# Identify significant telluric absorption regions from ave_{order}_{vac_air}.txt
# and output tel_abs_{order}_{vac_air}.txt in both air and vacuum scales.

import os, sys, argparse
import numpy as np
import utils
from mask_from_spectrum import mask_from_spectrum
import matplotlib.pyplot as plt

default_thresh = 0.95   # threshold in flux (continuum=1) to detect 
default_buffer_pix = 3  # number of pixels to expand on both sides
plot_vac_air = 'air'    # Only one of air or vac to be plotted 

# ------------------------------------------------
# Argument parsing
# ------------------------------------------------
parser = argparse.ArgumentParser(description='Create a list of wavelength regions of significant telluric absorption')
parser.add_argument('order',type=str,help='Order (eg. m44)')
parser.add_argument("-s","--setting", type=str, default=utils.default_setting,
                    help=f"Instrumental setting label (default: {utils.default_setting})")
parser.add_argument('-t', '--flux_thresh', type=float, default=default_thresh,
                    help=f'Flux threshold to detect telluric absorption (default: {default_thresh})')
parser.add_argument('-b', '--buffer_pix', type=int, default=default_buffer_pix,
                    help=f'Buffer region in pixels on each side (default: {default_buffer_pix})')
parser.add_argument("--no_plot", action="store_true", help="Avoid plotting outputs")

args = parser.parse_args()
setting = args.setting
order = args.order
flux_thresh = args.flux_thresh
buffer_pix = args.buffer_pix

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

if not args.no_plot:
    PLOT_DIR = os.path.join(utils.HOME_DIR, "models_plot", setting)
    os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------------------------
# Process for both vac and air
# ------------------------------------------------
for vac_air, ave_file in [("vac", ave_vac), ("air", ave_air)]:
    sp = np.loadtxt(ave_file)
    masked_ranges = mask_from_spectrum(sp, flux_thresh=flux_thresh, buffer_pix=buffer_pix, invert=False)

    out_file = os.path.join(TXT_DIR, f"tel_abs_{order}_{vac_air}.txt")
    with open(out_file, 'w') as fout:
        for w0, w1 in masked_ranges:
            print(f"{w0:.3f} {w1:.3f}", file=fout)

    print(f"[{order}] {vac_air}: threshold={flux_thresh:.2f}, {len(masked_ranges)} regions saved to {out_file}")

    if not args.no_plot:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        fmin, fmax = np.min(sp[:,1]), np.max(sp[:,1])
        ax.plot(sp[:,0], sp[:,1], color='black', lw=1)
        ax.axhline(flux_thresh,ls='--',color='deepskyblue')
        for w0, w1 in masked_ranges:
            ax.fill_between([w0, w1], fmax + 0.01*(fmax - fmin), fmax + 0.04*(fmax - fmin),
                            color='blue', alpha=0.5)
        ax.set_xlabel(f'{vac_air} wavelength')
        ax.set_ylabel('flux')
        ax.set_ylim(fmin - 0.05*(fmax - fmin), fmax + 0.1*(fmax - fmin))
        ax.set_title(f'Telluric ranges for {order} of {setting} (thresh={flux_thresh}, buffer={buffer_pix})')
        fig.tight_layout()
        plot_out = os.path.join(PLOT_DIR, f"tel_abs_{order}.png")
        fig.savefig(plot_out, dpi=200)
        plt.close(fig)

