#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from utils import c_kms, parse_ranges_string, combine_use_and_reject, apply_use_and_reject_ranges

def calc_sp_offset(sp_obj, sp_ref, 
                   list_v=np.arange(-30, 30, 1), frange=(0.0, 1.5),
                   use_ranges=[], reject_ranges=[]):
    """
    Estimate wavelength and continuum offset between two spectra
    Returns offsets in wavelength (given by redshift in km/s) and flux,
            which can be added to sp_obj in the following analysis).
    """
    xadj, yadj, chi2_min, n_used = np.nan, np.nan, np.nan, 0
    wmin = np.max([np.min(sp_obj[:,0]), np.min(sp_ref[:,0])])
    wmax = np.min([np.max(sp_obj[:,0]), np.max(sp_ref[:,0])])
    func_ref = CubicSpline(sp_ref[:, 0], sp_ref[:, 1])
    for v in list_v:
        rv_factor = 1.+v/c_kms
        sp_obj_tmp = apply_use_and_reject_ranges(sp_obj, rv=v, use_ranges=use_ranges, reject_ranges=reject_ranges)
        residuals = []
        for w, fobs in sp_obj_tmp:
            if (wmin < w < wmax) and (frange[0] <= fobs <= frange[1]):
                residuals.append(fobs - func_ref(w * rv_factor))
        if len(residuals) == 0:
            continue
        std = np.std(residuals)
        if np.isnan(chi2_min) or (std < chi2_min):
            chi2_min = std
            n_used = len(residuals)
            xadj = v
            yadj = -np.median(residuals)
    return xadj, yadj, n_used

def plot_comparison(sp_obj, sp_ref, used_ranges, xadj, yadj, plot_out):
    """Plot redshifted and flux-shifted object spectrum over reference."""
    rv_factor = 1 + xadj / c_kms
    sp_obj_plot = np.copy(sp_obj)
    sp_obj_plot[:, 0] *= rv_factor
    sp_obj_plot[:, 1] += yadj

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.plot(sp_ref[:, 0], sp_ref[:, 1], color='black', lw=1.2, label='Ref')
    ax.plot(sp_obj_plot[:, 0], sp_obj_plot[:, 1], color='red', lw=1.0, alpha=0.7, label=f'Obj ({xadj:.3f} km/s shifted)')
    for w0, w1 in used_ranges:
        ax.axvspan(w0, w1, ymin=0.96, ymax=0.98, color='blue', alpha=0.25)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    ax.set_title("Comparison of Reference and Shifted Object Spectrum")
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(plot_out, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate offsets between obj/ref spectra.")
    parser.add_argument("file_obs", type=str, help="Observed spectrum (2-column text)")
    parser.add_argument("file_ref", type=str, help="Reference spectrum (2-column text)")
    parser.add_argument("--wmin", type=float, default=np.nan, help="Min wavelength to use (optional)")
    parser.add_argument("--wmax", type=float, default=np.nan, help="Max wavelength to use (optional)")
    parser.add_argument("--vmin", type=float, default=-30, help="Min velocity shift (km/s)")
    parser.add_argument("--vmax", type=float, default=30, help="Max velocity shift (km/s)")
    parser.add_argument("--vstep", type=float, default=1, help="Step of velocity grid (km/s)")
    parser.add_argument("--fmin", type=float, default=0.0, help="Minimum flux to include")
    parser.add_argument("--fmax", type=float, default=1.5, help="Maximum flux to include")
    parser.add_argument("--use_ranges", type=str, default="", help="Use ranges in wavelength (start:end,...)")
    parser.add_argument("--reject_ranges", type=str, default="", help="Rejection ranges in wavelength (start:end,...)")
    parser.add_argument("-p", "--plot_out", type=str, default="NONE", help="Output filename to save comparison plot")

    args = parser.parse_args()

    sp_obs = np.loadtxt(args.file_obs)
    sp_ref = np.loadtxt(args.file_ref)
    wmin_ref, wmax_ref = np.min(sp_ref[:,0]), np.max(sp_ref[:,0])
    list_v = np.arange(args.vmin, args.vmax + args.vstep, args.vstep)
    frange = (args.fmin, args.fmax)

    reject_ranges = parse_ranges_string(args.reject_ranges)
    if args.use_ranges.strip() == "":
        use_ranges = [(wmin_ref, wmax_ref)]
    else:
        use_ranges = parse_ranges_string(args.use_ranges)
    used_ranges = combine_use_and_reject(use_ranges, reject_ranges)

    xadj, yadj, n_used = calc_sp_offset(sp_obs[:,0:2], sp_ref[:,0:2], list_v=list_v, frange=frange, use_ranges=use_ranges, reject_ranges=reject_ranges)
                                     
    print(f"{xadj:.3f} {yadj:.4f} {n_used:d}")

    if args.plot_out != "NONE":
        plot_comparison(sp_obs[:, 0:2], sp_ref[:, 0:2], used_ranges, xadj, yadj, args.plot_out)
