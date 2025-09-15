#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

from utils import c_kms, merge_ranges, vac_air_conversion

default_rv_to_add = 0.
default_thresh = 0.95
default_buffer_pix = 0

def mask_from_spectrum(data, rv_to_add=0., flux_thresh=0.95, buffer_pix=0, invert=False, conversion=None):
    """
    Identifies wavelength regions with significant absorption (or emission) in a given spectrum.

    Parameters:
        data (np.ndarray or list of tuples): 2-column array (wavelength, normalized flux)
        rv_to_add (float): Radial velocity to redshift wavelengths (in km/s)
        flux_thresh (float): Flux threshold to detect absorption (or continuum/emission if invert=True)
        buffer_pix (int): Number of pixels to expand each region
        invert (bool): If True, mask regions where flux is ABOVE threshold
        conversion (str): Conversion "vac_to_air" or "air_to_vac" if necessary

    Returns:
        List of merged (start_wave, end_wave) tuples.
    """
    rvfactor = 1. + rv_to_add / c_kms
    data = np.array(data)
    wave_shifted = data[:, 0] * rvfactor
    if conversion is not None:
        wave_shifted = vac_air_conversion(wave_shifted,conversion=conversion)

    flux = data[:, 1]
    if invert:
        mask = flux > flux_thresh
    else:
        mask = flux < flux_thresh
    if not np.any(mask):
        return []

    regions = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = i
            while i < len(mask) and mask[i]:
                i += 1
            end = i
            start_idx = max(0, start - buffer_pix)
            end_idx = min(len(wave_shifted) - 1, end + buffer_pix)
            regions.append((wave_shifted[start_idx], wave_shifted[end_idx]))
        else:
            i += 1

    return merge_ranges(regions)


def _main():
    parser = argparse.ArgumentParser(description='Create a list of wavelength regions to mask in an input spectrum')
    parser.add_argument('file_in', type=str, help='Input text file with 2-column spectrum (wavelength flux)')
    parser.add_argument('-v', '--rv_to_add', type=float, default=default_rv_to_add,
                        help=f'RV (km/s) to redshift wavelengths (default: {default_rv_to_add})')
    parser.add_argument('-t', '--flux_thresh', type=float, default=default_thresh,
                        help=f'Flux threshold for masking condition (default: {default_thresh})')
    parser.add_argument('-i', '--invert', action='store_true',
                        help='Mask bright regions (flux > threshold) instead of absorption (flux < threshold)')
    parser.add_argument('-b', '--buffer_pix', type=int, default=default_buffer_pix,
                        help=f'Buffer region in pixels on each side (default: {default_buffer_pix})')
    parser.add_argument('-c', '--conversion', type=str, choices=['vac_to_air','air_to_vac'], default=None,
                        help='Convert wavelength scale before masking')
    parser.add_argument('-p', '--plot_out', type=str, default='NONE',
                        help='Plot output file (png, jpg, pdf)')

    args = parser.parse_args()
    if not os.path.isfile(args.file_in):
        print(f'[ERROR] File not found: {args.file_in}', file=sys.stderr)
        sys.exit(1)

    sp = np.loadtxt(args.file_in)
    masked_ranges = mask_from_spectrum(sp, rv_to_add=args.rv_to_add, flux_thresh=args.flux_thresh, buffer_pix=args.buffer_pix, invert=args.invert, conversion=args.conversion)
    print(','.join([f'{start:.3f}:{end:.3f}' for start, end in masked_ranges]))

    if args.plot_out != 'NONE':
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        wave_shifted = sp[:, 0] * (1 + args.rv_to_add / c_kms)
        if args.conversion is not None:
            wave_shifted = vac_air_conversion(wave_shifted,conversion=args.conversion)
        flux = sp[:, 1]
        fmin, fmax = np.min(flux), np.max(flux)
        ax.plot(wave_shifted, flux, color='black', lw=1)
        for w0, w1 in masked_ranges:
            ax.fill_between([w0, w1], fmax + 0.01*(fmax - fmin), fmax + 0.04*(fmax - fmin),
                            color='blue', alpha=0.5)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.set_ylim(fmin - 0.05*(fmax - fmin), fmax + 0.1*(fmax - fmin))
        if args.invert:
            ax.set_title(r'Pixel ranges with $f > {}$ (buffer_pix={})'.format(str(args.flux_thresh),str(args.buffer_pix)))
        else:
            ax.set_title(r'Pixel ranges with $f < {}$ (buffer_pix={})'.format(str(args.flux_thresh),str(args.buffer_pix)))
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=175)
        plt.close(fig)


if __name__ == "__main__":
    _main()

