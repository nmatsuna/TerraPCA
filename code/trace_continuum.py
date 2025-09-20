#!/usr/bin/env python3
"""
continuum.py
Estimate the continuum of a spectrum using Weighted Asymmetric Least Squares (AsLS).

The continuum is traced smoothly while downweighting absorption/emission features.
User-specified masks (reject_ranges and mask_obj) can further refine
the regions included/excluded from the fit.

Output:
    - Text file with two columns: wavelength, normalized flux
    - Optional plot showing observed spectrum, fitted continuum, and masks
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

import utils
from mask_from_spectrum import mask_from_spectrum

# --------------------------------------------------
# Weighted Asymmetric Least Squares baseline fitting
# --------------------------------------------------

def _build_D2(n):
    """Second-difference operator (n-2) x n."""
    # rows i correspond to second diff at positions i+2 (0-based)
    diags = [np.ones(n), -2*np.ones(n), np.ones(n)]
    return sparse.diags(diags, offsets=[0, -1, -2], shape=(n-2, n))

def asls_baseline(y, lam=1e5, p=0.05, niter=10, pad_mode='reflect', pad_frac=0.10, mask=None):
    """
    Weighted Asymmetric Least Squares baseline fitting.
    y : flux array (1D)
    lam : smoothness parameter (higher = smoother baseline)
    p : asymmetry parameter (0 < p < 1; smaller -> baseline above points)
    niter : number of iterations
    pad_mode : {'reflect', 'linear', None}
        Padding strategy before fitting; None disables padding.
    pad_frac : float
        Fraction of array length to pad on each side (e.g. 0.05 = 5%).
    mask : 1D ndarray of bool, optional
        If given, True = use pixel, False = reject pixel.
        Rejected pixels get *lower initial weights*.
    Returns baseline array of same shape as y.
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    pad = int(max(2, round(pad_frac*L))) if pad_mode else 0
    
    # build padding
    if (pad > 0) and (pad_mode == 'reflect'):
        left = y[1:pad+1][::-1]
        right = y[-pad-1:-1][::-1]
        y_pad = np.concatenate([left, y, right])
    elif (pad > 0) and (pad_mode == 'linear'):
        # linear trend extrapolation from first/last few points
        k = min(10, L)  # small window
        # left fit
        xL = np.arange(k)
        aL, bL = np.polyfit(xL, y[:k], 1)
        left = (aL*np.arange(-pad, 0) + (bL)).astype(float)
        # right fit
        xR = np.arange(k)
        aR, bR = np.polyfit(xR, y[-k:], 1)
        right = (aR*np.arange(1, pad+1) + (bR + aR*(k-1))).astype(float)
        y_pad = np.concatenate([left, y, right])
    else:
        y_pad = y
        pad = 0

    L2 = len(y_pad)

    # --- initial weights from mask ---
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != L:
            raise ValueError("mask must have same length as y")
        if pad > 0:
            mask_pad = np.concatenate([mask[1:pad+1][::-1], mask, mask[-pad-1:-1][::-1]])
        else:
            mask_pad = mask
        w = np.where(mask_pad, 1.0, 1e-2)  # low weight for rejected pixels
    else:
        w = np.ones(L2)

    # --- second-derivative penalty ---
    D2 = _build_D2(L2)
    S = lam * (D2.T @ D2)

    # --- iterative reweighting ---
    z = y_pad.copy()
    for _ in range(niter):
        W = sparse.diags(w, 0)
        eps = 1e-6 * np.mean(y_pad**2)
        A = W + S + eps * sparse.identity(L2) # adding a small ridge term to ensure invertibility
        z = spsolve(A, w * y_pad) 
        # z = lsmr(A, w * y_pad)[0]  # more robust solver
        # update weights (upper envelope: small p)
        w = p * (y_pad > z) + (1 - p) * (y_pad < z)
        if mask is not None:
            # enforce low weight for rejected pixels
            w *= np.where(mask_pad, 1.0, 1e-3)

    # crop padding
    if pad > 0:
        z = z[pad:-pad]
    return z

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Estimate continuum with AsLS")
    parser.add_argument("file_in", help="Input spectrum (txt or FITS)")
    parser.add_argument("file_out", help="Output text (wavelength, normalized flux)")
    parser.add_argument("--reject_ranges", type=str, default="", help="Wavelength ranges to reject")
    parser.add_argument("--mask_obj", type=str, default="", help="Optional object model file to mask features")
    parser.add_argument("--obj_rv_add", type=float, default=0.0, help="RV shift (km/s) for mask_obj")
    parser.add_argument("--obj_thresh", type=float, default=0.95, help="Flux threshold for mask_obj")
    parser.add_argument("--asls_lam", type=float, default=1e5, help="AsLS Smoothness parameter (default=1e5)")
    parser.add_argument("--asls_p", type=float, default=0.97, help="AsLS Asymmetry parameter (default=0.97)")
    parser.add_argument("--niter", type=int, default=15, help="Number of iterations (default=10)")
    parser.add_argument("-p","--plot_out", default="NONE", help="Optional plot file (png/pdf/jpg) or NONE")
    args = parser.parse_args()

    # Load spectrum
    sp, header = utils.load_sp(args.file_in)
    if (sp is None) or (len(sp) == 0):
        print(f"[ERROR] Failed to load the spectrum from {args.file_in}", file=sys.stderr)
        sys.exit(1)
    waves, flux = sp[:,0], sp[:,1]

    # Masks
    reject_ranges = utils.parse_ranges_string(args.reject_ranges)
    if args.mask_obj and os.path.isfile(args.mask_obj):
        mask_obj_ranges = mask_from_spectrum(
            np.loadtxt(args.mask_obj),
            rv_to_add=args.obj_rv_add,
            flux_thresh=args.obj_thresh,
            invert=False
        )
        reject_ranges += mask_obj_ranges
    mask = np.ones_like(waves, dtype=bool)
    for w0, w1 in reject_ranges:
        mask[(waves >= w0) & (waves <= w1)] = False

    # Construct weight array
    w = np.ones_like(flux)
    for w0, w1 in reject_ranges:
        w[(waves >= w0) & (waves <= w1)] = 0.0

    # Apply AsLS baseline to fit continuum
    flux_norm = flux / np.median(flux[mask])
    baseline = asls_baseline(flux_norm, lam=args.asls_lam, p=args.asls_p, niter=args.niter, mask=mask)
    baseline *= np.median(flux[mask])
    flux_out = flux / baseline

    # Save
    np.savetxt(args.file_out, np.column_stack([waves, flux_out]), fmt="%.6f")
    print(f"[INFO] Wrote normalized spectrum to {args.file_out}", file=sys.stderr)

    # Plot
    if args.plot_out != "NONE":
        fmin, fmax = np.min(flux), np.max(flux)
        ymin, ymax = fmin - 0.1*(fmax-fmin), fmax + 0.1*(fmax-fmin)
        fig, (ax_flux, ax_out) = plt.subplots(2,1,figsize=(12,8))
        ax_flux.step(waves, flux, color="black", lw=1, label="Observed")
        ax_flux.step(waves, baseline, color="red", lw=1.2, label="AsLS continuum")
        for w0, w1 in reject_ranges:
            ax_flux.fill_between([w0, w1],y1=0.07*ymin+0.93*ymax, y2=0.04*ymin+0.96*ymax, color="orange",alpha=0.5)
        ax_flux.set_ylim([ymin, ymax])
        ax_flux.set_xlabel("Wavelength")
        ax_flux.set_ylabel("Flux")
        ax_flux.set_title(f"AsLS Continuum: {os.path.basename(args.file_in)}")
        ax_flux.legend(loc="lower right")
        ax_out.axhline(1, color='darkgray', ls='dotted', lw=1)
        ax_out.step(waves,flux_out)
        ax_out.set_xlabel("Wavelength")
        ax_out.set_ylabel("Normalized Flux")
        ax_out.set_ylim([0,1.2])
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=150)
        plt.close(fig)
        print(f"[INFO] Plot saved to {args.plot_out}", file=sys.stderr)

if __name__ == "__main__":
    main()
