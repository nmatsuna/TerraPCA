#!/usr/bin/env python3
"""
Convert a 1D FITS spectrum into a 2-column text file (wavelength, flux).
Optionally add the entire FITS header as comments, and/or plot the spectrum.
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Convert 1D FITS spectrum to 2-column text")
parser.add_argument("input_fits", help="Input FITS file")
parser.add_argument("output_txt", help="Output text file (wavelength flux)")
parser.add_argument("--plot", action="store_true", help="Show spectrum plot")
parser.add_argument("--write_header", action="store_true",
                    help="Write the full FITS header as comments in the text file")
args = parser.parse_args()

# -------------------------------
# Load FITS spectrum
# -------------------------------
if not os.path.isfile(args.input_fits):
    print(f"[ERROR] {args.input_fits} not found", file=sys.stderr)
    sys.exit(1)

hdul = fits.open(args.input_fits)
flux = hdul[0].data
header = hdul[0].header
hdul.close()

if flux.ndim != 1:
    print("[ERROR] This script only supports 1D spectra.", file=sys.stderr)
    sys.exit(1)

crval1 = header.get('CRVAL1')
crpix1 = header.get('CRPIX1', 1.0)
cdelt1 = header.get('CDELT1')

if (crval1 is None) or (cdelt1 is None):
    print("[ERROR] FITS header missing CRVAL1 or CDELT1 keywords.", file=sys.stderr)
    sys.exit(1)

num_pix = len(flux)
pixels = np.arange(1, num_pix+1)
waves = crval1 + (pixels - crpix1) * cdelt1

# -------------------------------
# Save as 2-column text
# -------------------------------
with open(args.output_txt, 'w') as fout:
    fout.write(f"# {args.input_fits}")
    if args.write_header:
        for card in header.cards:
            fout.write(f"# {card.keyword} = {card.value} / {card.comment}\n")
    for w, f in zip(waves, flux):
        fout.write(f"{w:14.6f} {f:12.6f}\n")

print(f"[INFO] Wrote {num_pix} points to {args.output_txt}", file=sys.stderr)

# -------------------------------
# Optional plot
# -------------------------------
if args.plot:
    plt.figure(figsize=(8,4))
    plt.plot(waves, flux, lw=1, color='black')
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title(os.path.basename(args.input_fits))
    plt.tight_layout()
    plt.show()
