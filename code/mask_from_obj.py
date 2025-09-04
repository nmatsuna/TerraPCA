#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

default_rv_to_add = 0.
default_depth_thresh = 0.95
default_buffer_pix = 3

def mask_from_obj(file_in, rv_to_add=0., depth_thresh=0.97, buffer_pix=3):
    """
    Identifies wavelength regions where the object's expected spectrum shows significant absorption,
    redshifted and thresholded. Returns a list of (start_wave, end_wave) tuples, merged for overlap.

    Parameters:
        file_in (str): Path to the input spectrum (2-column: wavelength, normalized flux)
        rv_to_add (float): RV to add to the input wavelengths
        depth_thresh (float): Mask wavelengths where flux < thresh
        buffer_pix (int): Buffer region (in pixels) to include on each side

    Returns:
        List of wavelength range to reject (w1a:w1b,w2a:w2b,...) : Merged list of masked wavelength ranges
    """
    if not os.path.isfile(file_in):
        print(f'Failed to fine {file_in}', file=sys.stderr)
        return []
    redshift = rv_to_add/300000.
    data = np.loadtxt(file_in)
    wave_obj = data[:, 0] * (1 + redshift)
    flux_obj = data[:, 1]

    mask = flux_obj < depth_thresh
    if not np.any(mask):
        return []

    abs_regions = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = i
            while i < len(mask) and mask[i]:
                i += 1
            end = i

            # Apply buffer
            start_idx = max(0, start - buffer_pix)
            end_idx = min(len(wave_obj) - 1, end + buffer_pix)
            abs_regions.append((wave_obj[start_idx], wave_obj[end_idx]))
        else:
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
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append(region)

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a list of the wavelength ranges to mask the absorption in an input spectrum')
    parser.add_argument('file_in',type=str,help='Input file (text)')
    parser.add_argument('-v','--rv_to_add',type=float,default=default_rv_to_add,help=f'RV (km/s) to add to the input spectrum (default: {default_rv_to_add})')
    parser.add_argument('-d','--depth_thresh',type=float,default=default_depth_thresh,help=f'The threshold in flux to search for the absorption (default: {default_depth_thresh})')
    parser.add_argument('-b','--buffer_pix',type=float,default=default_buffer_pix,help=f'Buffer region (in pixels) to include on each side of the absorption (default: {default_buffer_pix})')
    parser.add_argument('-p','--plot_out',type=str,default='NONE',help='Plot output file (png, jpg, pdf)')

    args = parser.parse_args()
    file_in = args.file_in
    rv_to_add = args.rv_to_add
    depth_thresh = args.depth_thresh
    buffer_pix = args.buffer_pix
    plot_out = args.plot_out

    masked_ranges = mask_from_obj(file_in, rv_to_add=rv_to_add, depth_thresh=depth_thresh, buffer_pix=buffer_pix)

    print(','.join([f'{start:.3f}:{end:.3f}' for start, end in masked_ranges]))
    if plot_out != 'NONE':
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(111)
        sp = np.loadtxt(file_in)
        fmin, fmax = np.min(sp[:,1]), np.max(sp[:,1])
        ymin = fmin - 0.1*(fmax-fmin)
        ymax = fmax + 0.15*(fmax-fmin)
        redshift = 1 + rv_to_add/300000.
        ax.plot(sp[:,0]*redshift,sp[:,1], color='red', alpha=0.8)
        for w0, w1 in masked_ranges:
            ax.plot([w0,w1],[fmax+0.05*(fmax-fmin),fmax+0.05*(fmax-fmin)], color='blue', alpha=0.8)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.tick_params(direction='in',length=8)
        fig.tight_layout()
        fig.savefig(plot_out,dpi=175)
        plt.close(fig)

